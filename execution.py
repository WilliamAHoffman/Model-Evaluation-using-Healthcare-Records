from testing import run_experiments
from visualizations import generate_visualizations
import os

from pathlib import Path
OUTPUT_DIR = Path("outputs")

if os.path.exists(OUTPUT_DIR):
    for file in OUTPUT_DIR.iterdir():
        if file.is_file():
            file.unlink()
else:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df, y = run_experiments()

generate_visualizations(results_df, y)