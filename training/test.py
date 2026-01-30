import os
import glob
import pandas as pd
from typing import List

class EEGFileTester:
    def __init__(self, data_dir, csv_path=None):
        self.data_dir = data_dir
        
        # Load CSV only if provided
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None

    def _get_eeg_files(self) -> List[str]:
        files = []

        if self.df is not None and 'File' in self.df.columns:
            print("Using CSV file list...")
            for csv_file in self.df['File']:
                file_path = str(csv_file).strip()

                if os.path.isabs(file_path):
                    if os.path.exists(file_path):
                        files.append(file_path)
                    else:
                        filename = os.path.basename(file_path)
                        alt_path = os.path.join(self.data_dir, filename)
                        if os.path.exists(alt_path):
                            files.append(alt_path)
                else:
                    rel_path = os.path.join(self.data_dir, file_path)
                    if os.path.exists(rel_path):
                        files.append(rel_path)
                    else:
                        filename = os.path.basename(file_path)
                        alt_path = os.path.join(self.data_dir, filename)
                        if os.path.exists(alt_path):
                            files.append(alt_path)

        if not files:
            print("CSV not used or no files found — falling back to scanning directory")
            files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))

        return files


# ======================
# 🔧 CHANGE THESE PATHS
# ======================
DATA_DIR = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\nmt\data\data_processed\results0"
CSV_PATH = r"C:\Users\S.S.T\Documents\VsCode\eeg models\results\nmt\data\data_processed\results0\data.csv" # or r"/path/to/metadata.csv"

tester = EEGFileTester(DATA_DIR, CSV_PATH)
files = tester._get_eeg_files()

print("\n📂 EEG files found:")
for f in files:
    print(f)

print(f"\nTotal files found: {len(files)}")
