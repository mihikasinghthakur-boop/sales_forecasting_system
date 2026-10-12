"""
generate_sample.py
Create data/sample_data.csv — a small, real Favorita subset
suitable for the Dashboard demo on Streamlit Cloud.

Rows: ~3,650 (5 stores x 4 families x 365 days in 2017)
Size target: < 300 KB so it can be committed to GitHub.
"""
from pathlib import Path
import pandas as pd
from src.data_loader import preprocess

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# -- Load source files --------------------------------------------------------
train = pd.read_csv(
    DATA / "train.csv",
    parse_dates=["date"],
    usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
)
stores = pd.read_csv(DATA / "stores.csv")
oil = pd.read_csv(DATA / "oil.csv", parse_dates=["date"])
txns = pd.read_csv(DATA / "transactions.csv", parse_dates=["date"])
holidays = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])

# -- Stratified sample: 5 stores x 4 families, full year 2017 ----------------
TARGET_STORES = [1, 2, 3, 4, 5]
TARGET_FAMILIES = ["BEVERAGES", "GROCERY I", "PRODUCE", "CLEANING"]

mask = (
    (train["date"].dt.year == 2017)
    & (train["store_nbr"].isin(TARGET_STORES))
    & (train["family"].isin(TARGET_FAMILIES))
)
sample = train[mask].copy()

# -- Merge store metadata -----------------------------------------------------
sample = sample.merge(stores[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")

# -- Merge oil prices ---------------------------------------------------------
oil = oil.sort_values("date")
sample = sample.merge(oil[["date", "dcoilwtico"]], on="date", how="left")
sample["dcoilwtico"] = sample["dcoilwtico"].ffill().bfill()

# -- Merge transactions -------------------------------------------------------
sample = sample.merge(txns[["date", "store_nbr", "transactions"]], on=["date", "store_nbr"], how="left")
sample["transactions"] = sample["transactions"].fillna(0).astype(int)

# -- Merge holidays -----------------------------------------------------------
h = holidays[holidays["transferred"] == False][["date", "type"]].copy()
h["is_holiday"] = 1
h = h.drop_duplicates("date")
sample = sample.merge(h[["date", "is_holiday"]], on="date", how="left")
sample["is_holiday"] = sample["is_holiday"].fillna(0).astype(int)

# -- Run standard preprocess --------------------------------------------------
sample = preprocess(sample)

# -- Final cleanup & save -----------------------------------------------------
sample = sample.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")

out_path = DATA / "sample_data.csv"
sample.to_csv(out_path, index=False)

size_kb = out_path.stat().st_size / 1024
print(f"[OK] sample_data.csv written: {len(sample):,} rows, {size_kb:.1f} KB")
print(f"   Stores: {sorted(sample['store_nbr'].unique())}")
print(f"   Families: {sorted(sample['family'].unique())}")
print(f"   Date range: {sample['date'].min()} -> {sample['date'].max()}")
print(f"   Columns: {list(sample.columns)}")
