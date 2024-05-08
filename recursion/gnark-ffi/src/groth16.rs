use std::{
    env,
    fs::File,
    io::{Read, Write},
    panic,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::Duration,
};

use crate::witness::GnarkWitness;
use crossbeam::channel::{bounded, Sender};
use rand::Rng;
use reqwest::{blocking::Client, StatusCode};
use serde::{Deserialize, Serialize};
use sp1_recursion_compiler::{
    constraints::Constraint,
    ir::{Config, Witness},
};
use std::process::exit;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

/// A prover that can generate proofs with the Groth16 protocol using bindings to Gnark.
#[derive(Debug, Clone)]
pub struct Groth16Prover {
    // The port to use for the Gnark server.
    port: String,
    // JoinHandle for threads that do not return values.
    thread_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    // Sender for cancellation requests.
    cancel_sender: Sender<()>,
}

/// A zero-knowledge proof generated by the Groth16 protocol with a solidity proof representation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SolidityGroth16Proof {
    pub public_inputs: [String; 2],
    pub solidity_proof: String,
}

/// A zero-knowledge proof generated by the Groth16 protocol with a Base64 encoded gnark groth16 proof.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Groth16Proof {
    pub public_inputs: [String; 2],
    pub encoded_proof: String,
}

impl Groth16Prover {
    /// Starts up the Gnark server using Groth16 on the given port and waits for it to be ready.
    pub fn new(build_dir: PathBuf) -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let gnark_dir = manifest_dir.join("../gnark");
        let port = env::var("HOST_PORT").unwrap_or_else(|_| generate_random_port().to_string());
        let port_clone = port.clone();

        // Create a channel for cancellation
        let (cancel_sender, cancel_receiver) = bounded(1);

        // Run `make`.
        make_go_bindings(&gnark_dir);

        // Catch panics and attempt to terminate the child process if main thread panics
        let child_process = Arc::new(Mutex::new(None::<Child>));
        let child_handle = child_process.clone();
        panic::set_hook(Box::new(move |_info| {
            let mut child = child_handle.lock().unwrap();
            if let Some(ref mut child) = *child {
                child.kill().unwrap();
            }
        }));

        // Spawn a thread to run the Go command and panic on errors
        let thread_handle = thread::spawn(move || {
            let cwd = std::env::current_dir().unwrap();
            let data_dir = cwd.join(build_dir);
            let data_dir_str = data_dir.to_str().unwrap();

            let child = Command::new("go")
                .args([
                    "run",
                    "main.go",
                    "serve",
                    "--data",
                    data_dir_str,
                    "--type",
                    "groth16",
                    "--port",
                    &port,
                ])
                .current_dir(gnark_dir)
                .stderr(Stdio::inherit())
                .stdout(Stdio::inherit())
                .stdin(Stdio::inherit())
                .spawn()
                .unwrap();

            *child_process.lock().unwrap() = Some(child);

            loop {
                if cancel_receiver.try_recv().is_ok() {
                    let mut child = child_process.lock().unwrap();
                    if let Some(ref mut child) = *child {
                        child.kill().unwrap();
                    }
                    break;
                }

                let mut child = child_process.lock().unwrap();
                if let Some(ref mut child) = *child {
                    if let Ok(Some(exit_status)) = child.try_wait() {
                        if !exit_status.success() {
                            println!("Gnark server exited with an error: {:?}", exit_status);
                            exit(1);
                        }
                        break;
                    }
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        let prover = Self {
            port: port_clone,
            thread_handle: Arc::new(Mutex::new(Some(thread_handle))),
            cancel_sender,
        };

        prover.wait_for_healthy_server().unwrap();

        prover
    }

    /// Checks if the server is ready to accept requests.
    fn wait_for_healthy_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        let client = Client::new();
        let url = format!("http://localhost:{}/healthz", self.port);

        log::debug!("Waiting for Gnark server to be healthy...");

        loop {
            match client.get(&url).send() {
                Ok(response) => {
                    if response.status() == StatusCode::OK {
                        log::debug!("Gnark server is healthy!");
                        return Ok(());
                    } else {
                        log::debug!(
                            "Gnark server is not healthy, code: {:?} message: {:?}",
                            response.status(),
                            response.text()
                        );
                    }
                }
                Err(_) => {
                    log::debug!("Gnark server is not ready yet");
                }
            }

            thread::sleep(Duration::from_secs(3));
        }
    }

    /// Executes the prover in testing mode with a circuit definition and witness.
    pub fn test<C: Config>(constraints: Vec<Constraint>, witness: Witness<C>) {
        let serialized = serde_json::to_string(&constraints).unwrap();
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let gnark_dir = manifest_dir.join("../gnark");

        // Write constraints.
        let mut constraints_file = tempfile::NamedTempFile::new().unwrap();
        constraints_file.write_all(serialized.as_bytes()).unwrap();

        // Write witness.
        let mut witness_file = tempfile::NamedTempFile::new().unwrap();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        // Run `make`.
        make_go_bindings(&gnark_dir);

        let result = Command::new("go")
            .args([
                "test",
                "-tags=release_checks",
                "-v",
                "-timeout",
                "100000s",
                "-run",
                "^TestMain$",
                "github.com/succinctlabs/sp1-recursion-gnark",
            ])
            .current_dir(gnark_dir)
            .env("WITNESS_JSON", witness_file.path().to_str().unwrap())
            .env(
                "CONSTRAINTS_JSON",
                constraints_file.path().to_str().unwrap(),
            )
            .stderr(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stdin(Stdio::inherit())
            .output()
            .unwrap();

        if !result.status.success() {
            panic!("failed to run test circuit");
        }
    }

    pub fn build<C: Config>(constraints: Vec<Constraint>, witness: Witness<C>, build_dir: PathBuf) {
        let serialized = serde_json::to_string(&constraints).unwrap();
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let gnark_dir = manifest_dir.join("../gnark");
        let cwd = std::env::current_dir().unwrap();

        // Write constraints.
        let constraints_path = build_dir.join("constraints_groth16.json");
        let mut file = File::create(constraints_path).unwrap();
        file.write_all(serialized.as_bytes()).unwrap();

        // Write witness.
        let witness_path = build_dir.join("witness_groth16.json");
        let gnark_witness = GnarkWitness::new(witness);
        let mut file = File::create(witness_path).unwrap();
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        file.write_all(serialized.as_bytes()).unwrap();

        // Run `make`.
        make_go_bindings(&gnark_dir);

        // Run the build script.
        run_gnark_ffi_command(
            &gnark_dir,
            "build-groth16".to_string(),
            vec![
                "--data".to_string(),
                cwd.join(build_dir).to_str().unwrap().to_string(),
            ],
        )
    }

    /// Generates a Groth16 proof by sending a request to the Gnark server.
    pub fn prove<C: Config>(&self, witness: Witness<C>) -> Groth16Proof {
        let url = format!("http://localhost:{}/groth16/prove", self.port);

        let gnark_witness = GnarkWitness::new(witness);

        // Proof generation times out after 1 hour.
        let response = Client::new()
            .post(url)
            .timeout(Duration::from_secs(60 * 60))
            .json(&gnark_witness)
            .send()
            .unwrap();

        // Deserialize the JSON response to a Groth16Proof instance
        let response = response.text().unwrap();
        println!("response: {}", response);
        let proof: Groth16Proof = serde_json::from_str(&response).expect("deserializing the proof");

        proof
    }

    pub fn shutdown(&mut self) {
        let mut handle_opt = self.thread_handle.lock().unwrap();
        if let Some(handle) = handle_opt.take() {
            let _ = self.cancel_sender.send(());
            let _ = handle.join();
        }
    }
}

/// Generates a Groth16 proof by sending a request to the Gnark server.
pub fn verify(proof: Groth16Proof, build_dir: &PathBuf) -> bool {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let gnark_dir = manifest_dir.join("../gnark");
    let cwd = std::env::current_dir().unwrap();
    let data_dir = cwd.join(build_dir);
    let data_dir_str = data_dir.to_str().unwrap();

    // Run `make`.
    make_go_bindings(&gnark_dir);

    // Run the verify script.
    run_gnark_ffi_command(
        &gnark_dir,
        "verify-groth16".to_string(),
        vec![
            "--data".to_string(),
            data_dir_str.to_string(),
            "--encoded-proof".to_string(),
            proof.encoded_proof.to_string(),
            "--vkey-hash".to_string(),
            proof.public_inputs[0].to_string(),
            "--commited-values-digest".to_string(),
            proof.public_inputs[1].to_string(),
        ],
    );

    true
}

/// Generates a Groth16 proof by sending a request to the Gnark server.
pub fn convert(proof: Groth16Proof, build_dir: &PathBuf) -> SolidityGroth16Proof {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let gnark_dir = manifest_dir.join("../gnark");
    let cwd = std::env::current_dir().unwrap();
    let data_dir = cwd.join(build_dir);
    let data_dir_str = data_dir.to_str().unwrap();

    // Run `make`.
    make_go_bindings(&gnark_dir);

    // Run the convert script.
    run_gnark_ffi_command(
        &gnark_dir,
        "convert-groth16".to_string(),
        vec![
            "--data".to_string(),
            data_dir_str.to_string(),
            "--encoded-proof".to_string(),
            proof.encoded_proof.to_string(),
            "--vkey-hash".to_string(),
            proof.public_inputs[0].to_string(),
            "--commited-values-digest".to_string(),
            proof.public_inputs[1].to_string(),
        ],
    );

    // Read solidity_proof.json into SolidityGroth16Proof
    let mut file = File::open(data_dir.join("solidity_proof.json")).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let solidity_proof: SolidityGroth16Proof = serde_json::from_str(&contents).unwrap();

    solidity_proof
}

impl Drop for Groth16Prover {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Generate a random port.
fn generate_random_port() -> u16 {
    let mut rng = rand::thread_rng();
    rng.gen_range(1024..49152)
}

impl Default for Groth16Prover {
    fn default() -> Self {
        Self::new(PathBuf::from("build"))
    }
}

/// Runs the `make` command to generate the Go bindings for the Gnark library for FFI.
fn make_go_bindings(gnark_dir: &PathBuf) {
    let make = Command::new("make")
        .current_dir(gnark_dir)
        .stderr(Stdio::inherit())
        .stdin(Stdio::inherit())
        .output()
        .unwrap();
    if !make.status.success() {
        panic!("failed to run make");
    }
}

/// Runs the FFI command to interface with the Gnark library. Command is one of the commands
/// defined in recursion/gnark/main.go.
fn run_gnark_ffi_command(gnark_dir: &PathBuf, command: String, args: Vec<String>) {
    let mut command_args = vec!["run".to_string(), "main.go".to_string(), command.clone()];

    command_args.extend(args);

    let result = Command::new("go")
        .args(command_args)
        .current_dir(gnark_dir)
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stdin(Stdio::inherit())
        .output()
        .unwrap();

    if !result.status.success() {
        panic!(
            "failed to run script for {:?}: {:?}",
            command, result.status
        );
    }
}
