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


# ===========================================================================
# TOOL FUNCTIONS (Stage 6)
#
# These are the only functions the analytical assistant may call. Each one:
#   * takes explicit parameters (never reads the sidebar or other app state),
#   * checks every parameter against values that exist in the loaded data and
#     returns an error result for unknown values instead of guessing,
#   * returns the tool contract: tool, value, definition, scope, rows_used,
#     caveats (and error, when a parameter is invalid),
#   * returns plain numbers, strings, lists and dicts so the result can be
#     shown in the tool-call log and sent to a language model as JSON.
# ===========================================================================
REVENUE_NOT_PROFIT = "Revenue is not profit: the data has no costs, so profitability cannot be calculated."

CHANNEL_ALIASES = {
    "website": "Website", "web": "Website", "online": "Website", "ecommerce": "Website",
    "e-commerce": "Website", "internet": "Website",
    "brick & mortar": "Brick & Mortar", "brick and mortar": "Brick & Mortar",
    "brick-and-mortar": "Brick & Mortar", "stores": "Brick & Mortar", "store": "Brick & Mortar",
    "physical": "Brick & Mortar", "physical stores": "Brick & Mortar", "in-store": "Brick & Mortar",
    "retail": "Brick & Mortar",
}


def _plain(x):
    """Convert numpy/pandas values to plain Python so results are JSON-safe."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    if hasattr(x, "item"):          # numpy scalar
        x = x.item()
    if isinstance(x, float):
        return round(x, 2)
    return x


def _norm(v) -> str:
    return str(v).strip().lower()


def _resolve(value, options, kind, aliases=None):
    """Match a requested value to one that exists in the data.
    Returns (matched_value_or_None, error_or_None). None/'All' means no filter."""
    if value is None or _norm(value) in ("", "all", "any", "none"):
        return None, None
    v = _norm(value)
    if aliases and v in aliases and aliases[v] in options:
        return aliases[v], None
    for o in options:
        if _norm(o) == v:
            return o, None
    partial = [o for o in options if v in _norm(o)]
    if len(partial) == 1:                 # e.g. "Chicago" or "SH11" -> "SH11 Chicago, IL"
        return partial[0], None
    if partial:
        return None, f"'{value}' matches more than one {kind}: {', '.join(map(str, partial))}."
    return None, f"Unknown {kind} '{value}'. Available: {', '.join(map(str, options))}."


def _resolve_year(value, years):
    if value is None or _norm(value) in ("", "all", "any", "none"):
        return None, None
    try:
        y = int(str(value).strip().lstrip("Yy"))
    except ValueError:
        return None, f"Year must be a number like 2027, not '{value}'."
    if y not in years:
        return None, f"No data for year {y}. Available years: {', '.join(map(str, years))}."
    return y, None


def _scope(sales, year=None, channel=None, store=None, category=None):
    """Validate parameters and filter. Returns (df, scope, error)."""
    opts = filter_options(sales)
    y, e1 = _resolve_year(year, opts["years"])
    c, e2 = _resolve(channel, opts["channels"], "channel", CHANNEL_ALIASES)
    s, e3 = _resolve(store, opts["stores"], "store")
    g, e4 = _resolve(category, opts["categories"], "category")
    scope = {"year": y if y is not None else "All", "channel": c or "All",
             "store": s or "All", "category": g or "All"}
    errors = [e for e in (e1, e2, e3, e4) if e]
    if errors:
        return None, scope, " ".join(errors)
    df = filter_sales(sales, [y] if y is not None else None, [c] if c else None,
                      [s] if s else None, [g] if g else None)
    return df, scope, None


def _base_caveats(df) -> list[str]:
    cav = [REVENUE_NOT_PROFIT]
    blank = int(df["Revenue"].isna().sum())
    if blank:
        cav.append(f"{blank} rows with blank revenue were excluded.")
    unknown = int((df.get("Channel", pd.Series(dtype=str)) == "Unknown").sum())
    if unknown:
        cav.append(f"{unknown} rows have a store not found in the Stores table (channel 'Unknown').")
    return cav


def _result(tool, value, definition, scope, rows_used, caveats, error=None):
    out = {"tool": tool, "value": _plain(value), "definition": definition,
           "scope": _plain(scope), "rows_used": int(rows_used), "caveats": list(caveats)}
    if error:
        out["error"] = error
    return out


def _error(tool, scope, msg):
    return _result(tool, None, "", scope, 0, [], error=msg)


def _empty_check(tool, df, scope, definition):
    if df is None or df.empty or df["Revenue"].notna().sum() == 0:
        return _result(tool, None, definition, scope, 0,
                       ["No sales rows match this scope, so there is nothing to calculate."])
    return None


# ---------------------------------------------------------------------------
def get_total_revenue(sales, year=None, channel=None, store=None, category=None) -> dict:
    tool = "get_total_revenue"
    definition = "Sum of Revenue over all sales rows in the scope (one row = one store, one product, one month)."
    df, scope, err = _scope(sales, year, channel, store, category)
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    return _result(tool, float(df["Revenue"].sum()), definition, scope,
                   df["Revenue"].notna().sum(), _base_caveats(df))


def get_revenue_by_channel(sales, year=None, store=None, category=None) -> dict:
    tool = "get_revenue_by_channel"
    definition = ("Sum of Revenue grouped by sales channel (Stores.StoreType). "
                  "Share % = channel revenue / total revenue in the scope.")
    df, scope, err = _scope(sales, year, None, store, category)
    scope.pop("channel")
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    by = df.groupby("Channel")["Revenue"].sum()
    total = by.sum()
    value = {ch: {"revenue": float(v), "share_pct": float(v / total * 100) if total else None}
             for ch, v in by.items()}
    return _result(tool, value, definition, scope, df["Revenue"].notna().sum(), _base_caveats(df))


def get_revenue_by_year(sales, channel=None, store=None, category=None) -> dict:
    tool = "get_revenue_by_year"
    definition = ("Sum of Revenue for each calendar year (from the sales Date). "
                  "YoY growth % = (this year - previous year) / previous year.")
    df, scope, err = _scope(sales, None, channel, store, category)
    scope.pop("year")
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    by = df.groupby("Year")["Revenue"].sum().sort_index()
    months = df.groupby("Year")["Month"].nunique()
    growth = {}
    for y in by.index:
        growth[str(int(y))] = yoy_growth(by[y], by.get(y - 1))
    first_y, last_y = int(by.index[0]), int(by.index[-1])
    value = {"revenue_by_year": {str(int(y)): float(v) for y, v in by.items()},
             "yoy_growth_pct": growth,
             "first_year": first_y, "last_year": last_y,
             "change_first_to_last_pct": yoy_growth(by.iloc[-1], by.iloc[0])
             if first_y != last_y else None}
    cav = _base_caveats(df)
    partial = [f"{int(y)} ({int(n)} months)" for y, n in months.items() if n < 12]
    if partial:
        cav.append("Some years have fewer than 12 months of data, so their totals are not "
                   "comparable to full years: " + ", ".join(partial) + ".")
    cav.append("Growth is undefined (null) for the first year because there is no prior year.")
    return _result(tool, value, definition, scope, df["Revenue"].notna().sum(), cav)


def get_store_trend(sales, start_year=None, end_year=None, channel=None, category=None) -> dict:
    tool = "get_store_trend"
    definition = ("For each store: revenue in the end year minus revenue in the start year, "
                  "in dollars and as % of start-year revenue.")
    opts = filter_options(sales)
    sy, e1 = _resolve_year(start_year, opts["years"])
    ey, e2 = _resolve_year(end_year, opts["years"])
    sy = sy if sy is not None else (opts["years"][0] if opts["years"] else None)
    ey = ey if ey is not None else (opts["years"][-1] if opts["years"] else None)
    df, scope, err = _scope(sales, None, channel, None, category)
    scope.pop("year"); scope.pop("store")
    scope.update({"start_year": sy, "end_year": ey})
    err = " ".join(e for e in (e1, e2, err) if e)
    if err:
        return _error(tool, scope, err)
    if sy is None or ey is None or sy >= ey:
        return _error(tool, scope, "The start year must be earlier than the end year.")
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    piv = (df[df["Year"].isin([sy, ey])].groupby(["Store", "Channel", "Year"])["Revenue"]
           .sum().unstack("Year"))
    rows, skipped = [], []
    for (store, ch), r in piv.iterrows():
        start, end = r.get(sy), r.get(ey)
        if pd.isna(start) or pd.isna(end) or start == 0:
            skipped.append(store)
            continue
        rows.append({"store": store, "channel": ch, "start_revenue": float(start),
                     "end_revenue": float(end), "change": float(end - start),
                     "change_pct": float((end - start) / start * 100)})
    rows.sort(key=lambda d: d["change_pct"])
    declining = [r for r in rows if r["change"] < 0]
    value = {"stores": rows,
             "stores_declining": len(declining),
             "stores_growing": len([r for r in rows if r["change"] > 0]),
             "largest_pct_decline": declining[0] if declining else None,
             "largest_pct_growth": rows[-1] if rows and rows[-1]["change"] > 0 else None}
    cav = _base_caveats(df)
    if not declining:
        cav.append(f"No store declined between {sy} and {ey}, so there is no 'greatest decline'.")
    if skipped:
        cav.append("Excluded because they have no revenue in the start or end year: "
                   + ", ".join(skipped) + ".")
    cav.append("Only the start and end years are compared; changes in between are not shown.")
    return _result(tool, value, definition, scope,
                   df[df["Year"].isin([sy, ey])]["Revenue"].notna().sum(), cav)


def get_revenue_by_store(sales, year=None, channel=None, category=None) -> dict:
    tool = "get_revenue_by_store"
    definition = ("Sum of Revenue for each store, ranked highest to lowest. "
                  "Share % = store revenue / total revenue in the scope.")
    df, scope, err = _scope(sales, year, channel, None, category)
    scope.pop("store")
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    by = df.groupby(["Store", "Channel"], as_index=False)["Revenue"].sum()
    by = by.sort_values("Revenue", ascending=False)
    total = by["Revenue"].sum()
    value = [{"rank": i + 1, "store": r["Store"], "channel": r["Channel"],
              "revenue": float(r["Revenue"]),
              "share_pct": float(r["Revenue"] / total * 100) if total else None}
             for i, (_, r) in enumerate(by.iterrows())]
    cav = _base_caveats(df)
    if channel is None and (by["Channel"] == "Website").any():
        cav.append("The Website (SH00) is a sales channel, not a physical store; it is "
                   "ranked here for completeness.")
    return _result(tool, value, definition, scope, df["Revenue"].notna().sum(), cav)


def get_top_products(sales, year=None, channel=None, store=None, category=None,
                     n: int = 5) -> dict:
    tool = "get_top_products"
    definition = ("Sum of Revenue for each product, ranked highest to lowest. Products are "
                  "identified by Product ID because three products share the name 'Base Model'.")
    df, scope, err = _scope(sales, year, channel, store, category)
    try:
        n = max(1, min(20, int(n)))
    except (TypeError, ValueError):
        n = 5
    scope["top_n"] = n
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    total = df["Revenue"].sum()
    tp = top_products(df, n)
    value = [{"rank": i + 1, "product": r["Product label"], "category": r["Category"],
              "revenue": float(r["Revenue"]),
              "share_pct": float(r["Revenue"] / total * 100) if total else None}
             for i, (_, r) in enumerate(tp.iterrows())]
    return _result(tool, value, definition, scope, df["Revenue"].notna().sum(), _base_caveats(df))


def get_category_performance(sales, year=None, channel=None, store=None) -> dict:
    tool = "get_category_performance"
    definition = ("Sum of Revenue for each product category (Sales.Category ID joined to "
                  "Categories). Share % = category revenue / total revenue in the scope.")
    df, scope, err = _scope(sales, year, channel, store, None)
    scope.pop("category")
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    cat = revenue_by_category(df)
    value = [{"category": r["Category"], "revenue": float(r["Revenue"]),
              "share_pct": float(r["Share %"])} for _, r in cat.iterrows()]
    return _result(tool, value, definition, scope, df["Revenue"].notna().sum(), _base_caveats(df))


def get_monthly_revenue_share(sales, year=None, channel=None, store=None, category=None) -> dict:
    tool = "get_monthly_revenue_share"
    definition = ("Revenue for each month and its share of THAT YEAR's revenue "
                  "(denominator = the same year's total in the same scope).")
    df, scope, err = _scope(sales, year, channel, store, category)
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    m = revenue_by_month(df)
    value = {}
    for y, g in m.groupby("Year"):
        value[y] = {r["Month name"]: {"revenue": float(r["Revenue"]),
                                      "share_of_year_pct": float(r["Share of year %"])}
                    for _, r in g.iterrows()}
    cav = _base_caveats(df)
    short = [y for y, g in m.groupby("Year") if len(g) < 12]
    if short:
        cav.append("These years have fewer than 12 months, so shares are of a partial year: "
                   + ", ".join(short) + ".")
    return _result(tool, value, definition, scope, df["Revenue"].notna().sum(), cav)


def get_satisfaction_revenue_relationship(sales, store_year, year=None, store=None) -> dict:
    tool = "get_satisfaction_revenue_relationship"
    definition = ("Pearson correlation between each physical store's annual revenue and its "
                  "survey Satisfaction score (0 to 5), one point per store-year.")
    df, scope, err = _scope(sales, year, None, store, None)
    scope.pop("channel"); scope.pop("category")
    if err:
        return _error(tool, scope, err)
    if (empty := _empty_check(tool, df, scope, definition)):
        return empty
    pairs = satisfaction_vs_revenue(df, store_year)
    all_sy = df[df["Channel"] != "Website"].groupby(["Store ID", "Year"]).ngroups
    r, n = correlation(pairs["Satisfaction"], pairs["Revenue"])
    strength = None
    if r is not None:
        a = abs(r)
        strength = ("very weak or none" if a < 0.1 else "weak" if a < 0.3
                    else "moderate" if a < 0.5 else "strong")
        strength += " positive" if r > 0 else " negative" if r < 0 else ""
    value = {"correlation_r": r, "strength": strength, "store_years_used": n,
             "average_satisfaction": float(pairs["Satisfaction"].mean()) if n else None}
    cav = [
        "Correlation shows association only. It does not show that satisfaction causes "
        "revenue (or the reverse); store size, location, and market could drive both.",
        "The Website has no survey scores and is excluded.",
    ]
    if all_sy - n > 0:
        cav.append(f"{all_sy - n} physical store-years have no satisfaction score and were excluded.")
    if r is None:
        cav.append(f"Only {n} store-years with scores: too few to calculate a correlation.")
    elif n < 30:
        cav.append(f"Small sample ({n} store-years): the correlation is unreliable.")
    return _result(tool, value, definition, scope, n, cav)


def get_data_quality_report(dq: pd.DataFrame, merge_log: pd.DataFrame | None = None) -> dict:
    tool = "get_data_quality_report"
    definition = ("Data-quality findings from the loaded files (blank values, duplicates, "
                  "unmatched keys, out-of-range scores) and the merge checks.")
    findings = [{"table": r["Table"], "check": r["Check"], "count": int(r["Count"]),
                 "severity": r["Severity"], "detail": r["Detail"]} for _, r in dq.iterrows()]
    merges_ok = None
    if merge_log is not None and not merge_log.empty:
        merges_ok = bool((merge_log["Status"] != "Error").all())
    value = {"findings": findings,
             "errors": sum(f["severity"] == "Error" for f in findings),
             "warnings": sum(f["severity"] == "Warning" for f in findings),
             "all_merges_kept_rows_and_revenue": merges_ok}
    return _result(tool, value, definition, {"data": "All loaded files"}, 0,
                   ["Findings are reported, not corrected. Some (like the Website having no "
                    "survey) are structural rather than errors."])


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
