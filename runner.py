import subprocess
import os

# List of modules and their parameters to run
scripts = [
    ("Contrastive.Contrastive_5_train_EuclideanDistance",[]),
    ("Classification.Classification_embeddings", ["Models/Contrastive_Models/Contrastive_b0_128_96_5_EuclideanDistance1.pth"]),
    ("tests.test_embeddings", ["Models/Contrastive_Models/Contrastive_b0_128_96_5_EuclideanDistance1.pth", 
                               'Models/Classification_Models/Classification_embeddings_128_96_5_EuclideanDistance1.pth']),
]

# Function to run the scripts
def run_scripts():
    for module, params in scripts:
        try:
            # The command now uses '-m' to run the module
            command = ["python", "-m", module] + params
            print(f"Running {module} with arguments {params}...")
            subprocess.run(command, check=True, cwd=os.path.abspath(os.path.dirname(__file__)))  # Executes the module with parameters
            print(f"{module} finished successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error while running {module}: {e}\n")

if __name__ == "__main__":
    run_scripts()