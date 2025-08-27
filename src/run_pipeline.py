import subprocess

def run_step(script_name):
    print(f"\n Running {script_name}...")
    result = subprocess.run(["python", f"src/{script_name}"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f" {script_name} finished successfully.")
        print(result.stdout)
    else:
        print(f" {script_name} failed.")
        print(result.stderr)

if __name__ == "__main__":
    # Étape 1 : Data preprocessing
    run_step("data_preprocessing.py")

    # Étape 2 : Training
    run_step("train.py")

    # Étape 3 : Evaluation
    run_step("evaluate.py")

    print("\n Pipeline executes with sucess")
