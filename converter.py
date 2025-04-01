import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and print its output."""
    try:
        print(f"Running: {' '.join(command)}")
        subprocess.check_call(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command {' '.join(command)} failed with exit code {e.returncode}.")
        sys.exit(1)

def main():
    # Step 1: Clone the TensorFlow Models Repository
    if not os.path.exists("models"):
        run_command(["git", "clone", "https://github.com/tensorflow/models.git"])
    else:
        print("TensorFlow Models repository already exists. Skipping clone step.")

    # Step 2: Navigate to the research directory
    os.chdir("models/research")
    print("Changed working directory to models/research.")

    # Step 3: Install dependencies using setup.py
    print("Installing TensorFlow Object Detection API...")
    run_command([sys.executable, "setup.py", "build"])
    run_command([sys.executable, "setup.py", "install"])

    # Step 4: Install additional dependencies
    print("Installing additional dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "tensorflow-addons"])

    # Step 5: Compile Protobuf files
    if not os.path.exists("object_detection/protos"):
        print("Error: Protos directory not found. Ensure you've cloned the repository correctly.")
        sys.exit(1)

    print("Compiling Protobuf files...")
    run_command(["protoc", "object_detection/protos/*.proto", "--python_out=."])

    # Step 6: Add research directory to PYTHONPATH
    python_path = os.environ.get("PYTHONPATH", "")
    research_path = os.getcwd()
    slim_path = os.path.join(research_path, "slim")
    
    if research_path not in python_path or slim_path not in python_path:
        new_python_path = f"{python_path}:{research_path}:{slim_path}"
        os.environ["PYTHONPATH"] = new_python_path
        print("Updated PYTHONPATH environment variable.")
    else:
        print("PYTHONPATH already contains the required paths.")

    print("Setup complete! You can now use the TensorFlow Object Detection API.")

if __name__ == "__main__":
    main()
