from pathlib import Path
import pandas as pd
import sys

SRC = Path(__file__).parent / "go24-full.csv"
DST = Path(__file__).parent / "go24-benchmark.csv"

if not SRC.exists():
    raise FileNotFoundError(f"Source file not found: {SRC}")

df = pd.read_csv(SRC)

if "Puzzles" not in df.columns:
    raise ValueError("Source CSV must contain a 'Puzzles' column")

selected = df.tail(100)  # last 100 entries by Rank
puzzles = selected["Puzzles"].astype(str).tolist()

# create a single-row CSV where each cell is one puzzle (no header, no index)
out_df = pd.DataFrame([puzzles])
out_df.to_csv(DST, index=False, header=False)