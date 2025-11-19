import subprocess
import os
import gc
import torch

# List of modules and their parameters to run
scripts = [
    # ("Classification.Classification_embeddings",["Models/Contrastive_Models/Contrastive_b0_128_96_8_EuclideanDistance1.pth"]),
    # ("tests.embeddings_creator",["Models/Contrastive_Models/Contrastive_b0_128_96_5_EuclideanDistance1.pth"]),
    # ("Contrastive.Contrastive_8_train_SupConLoss",[]),
    # ("tests.embeddings_creator",["Models/Contrastive_Models/Contrastive_b0_128_182_5_GLIDE_ADM_wukong_SupConLoss.pth"]),
    
    # ("tests.test_EuclideanDistance",["Models/Contrastive_Models/Contrastive_b0_128_96_8_EuclideanDistance1.pth","False"]),
    ("tests.test_KNN_SOURCE_ATTRIBUTION_SYNTH_TRAINED_Few_Shot_Mamba_New_ES1",["Models/Contrastive_NEW_ES1.pth"]),
    ("tests.test_KNN_SOURCE_ATTRIBUTION_SYNTH_TRAINED_Few_Shot_Mamba_New_ES2",["Models/Contrastive_NEW_ES2.pth"]),
    ("tests.test_KNN_SOURCE_ATTRIBUTION_SYNTH_TRAINED_Few_Shot_Mamba_New_ES3",["Models/Contrastive_NEW_ES3.pth"]),
    ("Contrastive.Contrastive_train_SupConLoss_MiniBatch_ForenSynths_Mamba",[])
    
    # ("tests.test_KNN",["Models/Contrastive_Models/Contrastive_b0_128_96_8_EuclideanDistance1.pth"])
]

# Function to run the scripts
def run_scripts():
    for module, params in scripts:
        gc.collect()
        torch.cuda.empty_cache() 
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