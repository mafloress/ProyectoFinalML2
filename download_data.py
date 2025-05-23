import kagglehub
import os

try:
    print("Attempting to download dataset...")
    path = kagglehub.dataset_download("wordsforthewise/lending-club")
    print(f"Dataset path: {path}")
    
    print("\nFiles in dataset directory:")
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            if filename.endswith('.csv') or filename.endswith('.csv.gz'):
                # Try to read the head of CSV or gzipped CSV files
                file_path_to_read = os.path.join(dirname, filename)
                print(f"\n--- Head of {file_path_to_read} ---")
                try:
                    if filename.endswith('.csv.gz'):
                        import gzip
                        import pandas as pd
                        with gzip.open(file_path_to_read, 'rt') as f:
                            df = pd.read_csv(f, nrows=5)
                            print(df.head().to_string())
                    elif filename.endswith('.csv'):
                        import pandas as pd
                        df = pd.read_csv(file_path_to_read, nrows=5)
                        print(df.head().to_string())
                except Exception as e:
                    print(f"Could not read head of {file_path_to_read}: {e}")
                print(f"--- End of {file_path_to_read} ---\n")

except Exception as e:
    print(f"An error occurred: {e}")
    # Attempt to install kagglehub if it's a ModuleNotFoundError
    if "No module named 'kagglehub'" in str(e) or "kagglehub.api" in str(e) : # Second condition for kaggle older versions
        print("kagglehub not found. Attempting to install...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            print("kagglehub installed successfully. Please re-run the subtask.")
        except subprocess.CalledProcessError as install_error:
            print(f"Failed to install kagglehub: {install_error}")
    elif "NewKaggleApi" in str(e) or "Unauthorized" in str(e) or "401" in str(e):
         print("This seems to be an authentication error. The worker might need Kaggle API credentials (kaggle.json).")
         print("Please ensure that a valid kaggle.json file is present in ~/.kaggle/ or that KAGGLE_USERNAME and KAGGLE_KEY environment variables are set.")
