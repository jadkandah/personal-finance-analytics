import re
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/expenses_raw.csv")
CLEAN_PATH = Path("data/processed/expenses_clean.csv")


def to_snake(s: str) -> str:
    """Convert a column name to clean snake_case."""
    s = s.strip().lower()
    s = s.replace("&", "and").replace("/", "_")
    s = re.sub(r"[^\w]+", "_", s)     # non-word -> underscore
    s = re.sub(r"_+", "_", s)         # collapse multiple underscores
    return s.strip("_")


def clean_data() -> None:
    # Load
    df = pd.read_csv(RAW_PATH)

    # Standardize column names
    df.columns = [to_snake(c) for c in df.columns]

    # Rename key columns (based on your dataset)
    rename_map = {
        "month": "date",
        "totalexpenditure": "total_expenditure",
        "emiloans": "emi_loans",
        "diningndentertainment": "dining_and_entertainment",
        "shoppingndwants": "shopping_and_wants",
    }
    df = df.rename(columns=rename_map)

    # Validate required columns exist
    required = ["date", "income", "savings", "investments", "total_expenditure"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Columns: {df.columns.tolist()}")

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Sort + dedupe
    df = df.sort_values("date").drop_duplicates()

    # Feature engineering (avoid divide-by-zero)
    df["savings_rate"] = df["savings"] / df["income"].replace(0, pd.NA)
    df["investment_rate"] = df["investments"] / df["income"].replace(0, pd.NA)
    df["expense_ratio"] = df["total_expenditure"] / df["income"].replace(0, pd.NA)

    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month

    # Save
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print("Clean data saved to:", CLEAN_PATH)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    clean_data()
