"""
analytics.py: business metrics for the E-Presso dashboard.

Rules for every function here:
  * No Streamlit code, so each function can be tested with pytest.
  * Inputs are passed explicitly (a DataFrame and filter values). Nothing
    reads hidden app state.
  * Nothing is hardcoded: every number comes from the data passed in.

The input `sales` is the enriched table from data_prep.build_model(): one row
per store, product, and month, with Channel, Store, Category, and Product
columns already joined on.
"""

from __future__ import annotations

import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_sales(sales: pd.DataFrame, years=None, channels=None, stores=None,
                 categories=None) -> pd.DataFrame:
    """Keep rows matching every selected filter. None means 'no filter'.
    An empty list means 'nothing selected' and returns no rows."""
    mask = pd.Series(True, index=sales.index)
    for col, chosen in (("Year", years), ("Channel", channels),
                        ("Store", stores), ("Category", categories)):
        if chosen is not None:
            mask &= sales[col].isin(list(chosen))
    return sales[mask]


def filter_options(sales: pd.DataFrame) -> dict[str, list]:
    """Filter choices built from the loaded data, never typed into the code."""
    return {
        "years": sorted(int(y) for y in sales["Year"].dropna().unique()),
        "channels": sorted(sales["Channel"].dropna().unique()),
        "stores": sorted(sales["Store"].dropna().unique()),
        "categories": sorted(sales["Category"].dropna().unique()),
    }


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
def yoy_growth(current: float, previous: float | None) -> float | None:
    """Year-over-year growth as a percentage. None when there is no prior
    year or the prior year is zero (growth is undefined, not infinite)."""
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return (current - previous) / previous * 100


def kpis(df: pd.DataFrame, all_years_df: pd.DataFrame | None = None) -> dict:
    """Headline numbers for the filtered rows.
    all_years_df: the same filters EXCEPT year, so growth for the latest
    selected year can be compared with the year before it even when only one
    year is selected. If omitted, growth uses df alone."""
    total = float(df["Revenue"].sum())
    by_year = df.groupby("Year")["Revenue"].sum().sort_index()
    latest_year = int(by_year.index.max()) if len(by_year) else None
    ref = all_years_df if all_years_df is not None else df
    ref_by_year = ref.groupby("Year")["Revenue"].sum()
    prev = ref_by_year.get(latest_year - 1) if latest_year is not None else None
    web = float(df.loc[df["Channel"] == "Website", "Revenue"].sum())
    return {
        "total_revenue": total,
        "website_revenue": web,
        "store_revenue": float(df.loc[df["Channel"] == "Brick & Mortar", "Revenue"].sum()),
        "website_share_pct": web / total * 100 if total else None,
        "latest_year": latest_year,
        "latest_year_revenue": float(by_year.iloc[-1]) if len(by_year) else 0.0,
        "latest_year_growth_pct": yoy_growth(by_year.iloc[-1], prev) if len(by_year) else None,
        "rows_used": int(df["Revenue"].notna().sum()),
        "rows_blank_revenue": int(df["Revenue"].isna().sum()),
    }


# ---------------------------------------------------------------------------
# Breakdowns used by the charts
# ---------------------------------------------------------------------------
def revenue_by_year_channel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["Year", "Channel"], as_index=False)["Revenue"].sum()
    out["Year"] = out["Year"].astype(int).astype(str)  # category axis, not 2,025.5
    return out


def revenue_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue and each month's share of THAT YEAR's revenue
    (denominator = the same year's total under the same filters)."""
    out = df.groupby(["Year", "Month"], as_index=False)["Revenue"].sum()
    out["Share of year %"] = out["Revenue"] / out.groupby("Year")["Revenue"].transform("sum") * 100
    out["Month name"] = out["Month"].astype(int).map(lambda m: MONTH_NAMES[m - 1])
    out["Year"] = out["Year"].astype(int).astype(str)
    return out.sort_values(["Year", "Month"])


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("Category", as_index=False)["Revenue"].sum()
    total = out["Revenue"].sum()
    out["Share %"] = out["Revenue"] / total * 100 if total else 0.0
    return out.sort_values("Revenue", ascending=False)


def revenue_by_store(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue per store plus change from the first to the last year in the
    selection (in dollars and percent)."""
    tot = df.groupby(["Store", "Channel"], as_index=False)["Revenue"].sum()
    by_sy = df.groupby(["Store", "Year"])["Revenue"].sum().unstack("Year").sort_index(axis=1)
    if by_sy.shape[1] >= 2:
        first, last = by_sy.columns[0], by_sy.columns[-1]
        change = (by_sy[last] - by_sy[first]).rename("Change $")
        pct = ((by_sy[last] - by_sy[first]) / by_sy[first].where(by_sy[first] != 0) * 100
               ).rename("Change %")
        tot = tot.merge(change, on="Store", how="left").merge(pct, on="Store", how="left")
        tot["Change period"] = f"{int(first)} to {int(last)}"
    return tot.sort_values("Revenue", ascending=False)


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    # Grouped by Product ID label: three different products are named "Base Model".
    out = df.groupby(["Product label", "Category"], as_index=False)["Revenue"].sum()
    return out.sort_values("Revenue", ascending=False).head(n)


def satisfaction_vs_revenue(df: pd.DataFrame, store_year: pd.DataFrame) -> pd.DataFrame:
    """Store-year revenue (from the filtered rows) joined to survey scores.
    Website rows are dropped because the website has no survey."""
    rev = df.groupby(["Store ID", "Store", "Channel", "Year"], as_index=False)["Revenue"].sum()
    scores = store_year[["Store ID", "Year", "Satisfaction", "Interest"]]
    out = rev.merge(scores, on=["Store ID", "Year"], how="left", validate="one_to_one")
    out = out[out["Channel"] != "Website"].dropna(subset=["Satisfaction"])
    out["Year"] = out["Year"].astype(int).astype(str)
    return out


def correlation(x: pd.Series, y: pd.Series) -> tuple[float | None, int]:
    """Pearson correlation and the number of pairs used. Association only."""
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pairs) < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return None, len(pairs)
    return float(pairs["x"].corr(pairs["y"])), len(pairs)


# ---------------------------------------------------------------------------
# Kept from the course starter repo. The app uses the merge checks in
# data_prep.build_model(), which also validates keys and counts unmatched rows.
# ---------------------------------------------------------------------------
def check_merge(left: pd.DataFrame, right: pd.DataFrame, on, value_col: str, how: str = 'left'):
    """Merge, and report whether the join changed the row count or the revenue sum.
    `value_col` must be a numeric column in `left`. Returns (merged, report)."""
    before_rows = len(left)
    before_sum = float(pd.to_numeric(left[value_col], errors='coerce').sum())
    merged = left.merge(right, on=on, how=how)
    after_rows = len(merged)
    after_sum = float(pd.to_numeric(merged[value_col], errors='coerce').sum())
    fan_out = after_rows > before_rows or abs(after_sum - before_sum) > 1e-9
    report = {
        'rows_before': before_rows, 'rows_after': after_rows,
        'sum_before': before_sum, 'sum_after': after_sum,
        'changed': fan_out,
        'warning': ('The join changed the row count or the total. Check the grain and keys '
                    'of both tables before using this result.') if fan_out else '',
    }
    return merged, report
