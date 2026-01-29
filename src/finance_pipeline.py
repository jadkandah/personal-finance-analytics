# =========================
# src/finance_pipeline.py
# =========================
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Config
# =========================

DEFAULT_DATE_COL = "date"
DEFAULT_INCOME_COL = "income"

EXPENSE_COMPONENT_COLS = [
    "groceries",
    "rent",
    "transportation",
    "gym",
    "utilities",
    "healthcare",
    "emi_loans",
    "dining_and_entertainment",
    "shopping_and_wants",
]

DERIVED_COLS = ["total_expenditure", "expense_ratio"]

# Flexible detection patterns
DATE_PATTERNS = [
    r"\bdate\b",
    r"\bmonth\b",
    r"\bperiod\b",
    r"\btimestamp\b",
]
INCOME_PATTERNS = [
    r"\bincome\b",
    r"\bsalary\b",
    r"\bwage\b",
    r"\brevenue\b",
    r"\bearning\b",
    r"\bearnings\b",
]
TOTAL_EXP_PATTERNS = [
    r"\btotal\b.*\bexpend",
    r"\btotal\b.*\bexpense",
    r"\btotal\s*spend\b",
]
# map normalized internal names -> patterns for detection in user CSV
COMPONENT_PATTERNS = {
    "groceries": [r"\bgrocer"],
    "rent": [r"\brent\b", r"\bhousing\b"],
    "transportation": [r"\btransport", r"\btravel\b", r"\bcommut"],
    "gym": [r"\bgym\b", r"\bfitness\b"],
    "utilities": [r"\butilit", r"\bbills?\b"],
    "healthcare": [r"\bhealth", r"\bmedical\b", r"\bclinic\b", r"\bpharma\b"],
    "emi_loans": [r"\bemi\b", r"\bloan\b", r"\bdebt\b"],
    "dining_and_entertainment": [r"\bdining\b", r"\brestaurant\b", r"\bentertain"],
    "shopping_and_wants": [r"\bshopping\b", r"\bwants?\b", r"\bmisc\b", r"\bluxur"],
}

# =========================
# Helpers
# =========================

def _normalize_col(s: str) -> str:
    """
    Lowercase, strip currency/symbols, replace non-alphanum with underscore.
    Example: 'Income (₹)' -> 'income'
    """
    s = s.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)            # remove (...) blocks
    s = re.sub(r"[₹$€£,%]", "", s)           # remove common symbols
    s = re.sub(r"[^a-z0-9]+", "_", s)        # non-alphanum -> _
    s = re.sub(r"_+", "_", s).strip("_")     # collapse underscores
    return s

def _best_match_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """
    Returns the original column name with best pattern match after normalization.
    """
    norm_map = {c: _normalize_col(c) for c in df.columns}

    # Score: count pattern hits, prefer exact-ish matches
    best_col = None
    best_score = -1

    for orig, norm in norm_map.items():
        score = 0
        for p in patterns:
            if re.search(p, norm):
                score += 1
        # small boost if normalized equals a common token exactly
        if any(norm == re.sub(r"\\b", "", p).replace("\\", "").replace(".*", "") for p in patterns):
            score += 1
        if score > best_score:
            best_score = score
            best_col = orig

    return best_col if best_score > 0 else None

def _to_month_start(dt: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=dt.year, month=dt.month, day=1)

def _next_month_start(dt: pd.Timestamp) -> pd.Timestamp:
    y, m = dt.year, dt.month
    if m == 12:
        return pd.Timestamp(year=y + 1, month=1, day=1)
    return pd.Timestamp(year=y, month=m + 1, day=1)

def _risk_level_from_ratio(r: float) -> str:
    if r < 0.60:
        return "Excellent"
    if r < 0.75:
        return "Stable"
    if r < 0.85:
        return "Warning"
    return "Critical"

# =========================
# Loading + Validation + Cleaning
# =========================

def load_finance_csv(
    data_path: str,
    date_col: str = DEFAULT_DATE_COL,
    income_col: str = DEFAULT_INCOME_COL,
) -> Tuple[pd.DataFrame, str, str]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")

    df = pd.read_csv(data_path)

    # Detect date column if needed
    if date_col not in df.columns:
        detected_date = _best_match_column(df, DATE_PATTERNS)
        if detected_date is None:
            raise ValueError(
                f"Missing date column '{date_col}' and couldn't auto-detect one.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Fix: call run(..., date_col='YOUR_DATE_COL')"
            )
        date_col = detected_date

    # Detect income column if needed
    if income_col not in df.columns:
        detected_income = _best_match_column(df, INCOME_PATTERNS)
        if detected_income is None:
            raise ValueError(
                f"Missing income column '{income_col}' and couldn't auto-detect one.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Fix: call run(..., income_col='YOUR_INCOME_COL')"
            )
        income_col = detected_income

    # Standardize internal names for pipeline
    df = df.copy()
    df.rename(columns={date_col: "date", income_col: "income"}, inplace=True)
    date_col, income_col = "date", "income"

    # Parse date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError("Some date values could not be parsed. Please check your date/month column format.")

    df[date_col] = df[date_col].apply(_to_month_start)

    # Map expense components if user columns are named differently
    # Only rename if the canonical name doesn't already exist
    for canon, pats in COMPONENT_PATTERNS.items():
        if canon in df.columns:
            continue
        found = _best_match_column(df, pats)
        if found is not None and found in df.columns:
            df.rename(columns={found: canon}, inplace=True)

    # Detect total_expenditure if exists
    if "total_expenditure" not in df.columns:
        found_total = _best_match_column(df, TOTAL_EXP_PATTERNS)
        if found_total is not None:
            df.rename(columns={found_total: "total_expenditure"}, inplace=True)

    # Coerce numerics
    for c in df.columns:
        if c != date_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort + dedupe
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

    # Compute total_expenditure if missing
    if "total_expenditure" not in df.columns:
        present_components = [c for c in EXPENSE_COMPONENT_COLS if c in df.columns]
        if not present_components:
            raise ValueError(
                "Missing total expenditure and could not find expense component columns to compute it.\n"
                f"Expected one of: total_expenditure OR components like {EXPENSE_COMPONENT_COLS}\n"
                f"Available columns: {list(df.columns)}"
            )
        df["total_expenditure"] = df[present_components].sum(axis=1)

    # Validate income
    if df["income"].isna().any():
        raise ValueError("Income has missing values after parsing. Fix your income column.")
    if df["income"].le(0).any():
        raise ValueError("Income contains non-positive values. Expense ratio would be invalid.")

    # Compute expense_ratio
    df["expense_ratio"] = df["total_expenditure"] / df["income"]

    # Time features
    df["year"] = df[date_col].dt.year.astype(int)
    df["month_num"] = df[date_col].dt.month.astype(int)

    # Final sanity
    if df["total_expenditure"].isna().any():
        raise ValueError("total_expenditure has missing values after computation/coercion.")
    if df["expense_ratio"].isna().any():
        raise ValueError("expense_ratio has missing values after computation.")

    return df, date_col, income_col

# =========================
# Feature Engineering (lag-only; no leakage)
# =========================

def make_lag_features(
    df: pd.DataFrame,
    target: str,
    lag_cols: Optional[List[str]] = None,
    lags: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    if lag_cols is None:
        candidates = ["income", "total_expenditure", "expense_ratio"] + EXPENSE_COMPONENT_COLS
        lag_cols = [c for c in candidates if c in df.columns]

    df2 = df.copy()

    feature_cols: List[str] = []
    for c in lag_cols:
        for k in range(1, lags + 1):
            col_name = f"{c}_lag{k}"
            df2[col_name] = df2[c].shift(k)
            feature_cols.append(col_name)

    feature_cols += ["year", "month_num"]

    df2 = df2.dropna().reset_index(drop=True)

    if target not in df2.columns:
        raise ValueError(f"Target '{target}' missing from dataframe.")
    return df2, feature_cols

# =========================
# Model selection (backtest last 20%)
# =========================

@dataclass
class ModelResult:
    name: str
    model: Any
    mae: float
    rmse: float
    r2: float

def train_and_select_model(
    X: pd.DataFrame,
    y: pd.Series,
    split_ratio: float = 0.8,
    random_state: int = 42,
) -> ModelResult:
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    candidates: List[Tuple[str, Any]] = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(
            n_estimators=600,
            max_depth=10,
            random_state=random_state
        )),
    ]

    best: Optional[ModelResult] = None

    for name, model in candidates:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)

        mr = ModelResult(name=name, model=model, mae=mae, rmse=rmse, r2=r2)

        if best is None or (mr.rmse < best.rmse) or (mr.rmse == best.rmse and mr.mae < best.mae):
            best = mr

    assert best is not None
    return best

# =========================
# Spending habits
# =========================

def spending_habits_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    comps = [c for c in EXPENSE_COMPONENT_COLS if c in df.columns]
    if comps:
        shares = (df[comps].mean() / df["total_expenditure"].mean()).sort_values(ascending=False)
        out["avg_category_share"] = shares.round(4).to_dict()
        out["top_categories"] = shares.head(3).index.tolist()
    else:
        out["avg_category_share"] = {}
        out["top_categories"] = []

    first_income = float(df["income"].iloc[0])
    last_income = float(df["income"].iloc[-1])
    first_exp = float(df["total_expenditure"].iloc[0])
    last_exp = float(df["total_expenditure"].iloc[-1])

    out["income_change_pct"] = round((last_income / first_income - 1) * 100, 2)
    out["expense_change_pct"] = round((last_exp / first_exp - 1) * 100, 2)

    out["avg_expense_ratio"] = round(float(df["expense_ratio"].mean()), 4)
    out["latest_expense_ratio"] = round(float(df["expense_ratio"].iloc[-1]), 4)
    return out

def _history_payload(df: pd.DataFrame, date_col: str = "date", tail: int = 24) -> Dict[str, Any]:
    h = df[[date_col, "total_expenditure", "expense_ratio", "income"]].copy()
    h = h.tail(tail).reset_index(drop=True)
    return {
        "dates": [str(d.date()) for d in h[date_col]],
        "total_expenditure": [float(x) for x in h["total_expenditure"]],
        "expense_ratio": [float(x) for x in h["expense_ratio"]],
        "income": [float(x) for x in h["income"]],
    }

# =========================
# Main pipeline
# =========================

def run(
    data_path: str,
    date_col: str = DEFAULT_DATE_COL,
    income_col: str = DEFAULT_INCOME_COL,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    One-call pipeline:
      - Load + validate data (auto-detect date/income column names)
      - Summarize spending habits
      - Forecast next-month total_expenditure (lag-only, no leakage)
      - Forecast next-month expense_ratio (lag-only, no leakage)
      - Convert ratio to risk label
      - Return results with small history payload for plotting
    """
    df, date_col, income_col = load_finance_csv(data_path=data_path, date_col=date_col, income_col=income_col)

    habits = spending_habits_summary(df)

    # ---- Forecast next-month total expenditure ----
    lag_cols = ["income", "total_expenditure", "expense_ratio"] + [c for c in EXPENSE_COMPONENT_COLS if c in df.columns]

    df_exp, exp_features = make_lag_features(df=df, target="total_expenditure", lag_cols=lag_cols, lags=1)
    X_exp = df_exp[exp_features]
    y_exp = df_exp["total_expenditure"]

    best_exp = train_and_select_model(X_exp, y_exp)
    best_exp.model.fit(X_exp, y_exp)

    last = df.iloc[-1]
    next_month_date = _next_month_start(last[date_col])

    next_row_exp = {f"{c}_lag1": float(last[c]) for c in lag_cols}
    next_row_exp["year"] = int(next_month_date.year)
    next_row_exp["month_num"] = int(next_month_date.month)

    X_next_exp = pd.DataFrame([next_row_exp], columns=exp_features)
    next_exp_pred = float(best_exp.model.predict(X_next_exp)[0])

    # ---- Forecast next-month expense ratio ----
    df_ratio, ratio_features = make_lag_features(df=df, target="expense_ratio", lag_cols=lag_cols, lags=1)
    X_r = df_ratio[ratio_features]
    y_r = df_ratio["expense_ratio"]

    best_ratio = train_and_select_model(X_r, y_r)
    best_ratio.model.fit(X_r, y_r)

    next_row_r = {f"{c}_lag1": float(last[c]) for c in lag_cols}
    next_row_r["year"] = int(next_month_date.year)
    next_row_r["month_num"] = int(next_month_date.month)

    X_next_r = pd.DataFrame([next_row_r], columns=ratio_features)
    next_ratio_pred = float(best_ratio.model.predict(X_next_r)[0])
    next_risk = _risk_level_from_ratio(next_ratio_pred)

    results: Dict[str, Any] = {
        "data": {
            "rows": int(len(df)),
            "start_date": str(df[date_col].min().date()),
            "end_date": str(df[date_col].max().date()),
            "latest_month": str(df[date_col].iloc[-1].date()),
            "next_month": str(next_month_date.date()),
        },
        "spending_habits": habits,
        "models": {
            "expense_forecast": {
                "selected_model": best_exp.name,
                "backtest": {"rmse": round(best_exp.rmse, 2), "r2": round(best_exp.r2, 3)},
            },
            "expense_ratio_forecast": {
                "selected_model": best_ratio.name,
                "backtest": {"rmse": round(best_ratio.rmse, 4), "r2": round(best_ratio.r2, 3)},
            },
        },
        "next_month_forecast": {
            "pred_total_expenditure": round(next_exp_pred, 2),
            "pred_expense_ratio": round(next_ratio_pred, 4),
            "pred_risk_level": next_risk,
        },
        "history": _history_payload(df, date_col=date_col, tail=24),
    }

    # Short output only (important info)
    if verbose:
        nm = results["next_month_forecast"]
        print(f"Next month: {results['data']['next_month']}")
        print(f"- Predicted spending: {nm['pred_total_expenditure']}")
        print(f"- Predicted expense ratio: {nm['pred_expense_ratio']} ({nm['pred_risk_level']})")

    return results
