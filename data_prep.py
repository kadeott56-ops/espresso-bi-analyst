"""
data_prep.py: Stage 1 of the E-Presso app.

Loads the five E-Presso tables, identifies each file by its columns (not its
file name), converts data types, and reports data-quality issues.

No Streamlit code lives here, so every function can be tested with pytest.
Nothing in this file contains revenue totals or other E-Presso results;
everything is calculated from whatever data is loaded.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Table specifications
#    The guide says column names will not change in updated data, so we use
#    them to recognise which table a file holds.
# ---------------------------------------------------------------------------
TABLE_SPECS: dict[str, dict] = {
    "sales": {
        "label": "Sales",
        "required": ["Sales ID", "Store ID", "StoreYear", "Category ID",
                     "Product ID", "Date", "Revenue"],
        "key": "Sales ID",
        "grain": "Revenue for one store and one product in one month "
                 "(Date = last day of the month).",
        "source": "Support_EX_B_Sales.accdb, table Sales (exported)",
    },
    "categories": {
        "label": "Categories",
        "required": ["Category ID", "Category"],
        "key": "Category ID",
        "grain": "One product category.",
        "source": "Support_EX_B_Sales.accdb, table Categories (exported)",
    },
    "products": {
        "label": "Products",
        "required": ["Product ID", "Product"],
        "key": "Product ID",
        "grain": "One product (names are not unique; use Product ID).",
        "source": "Support_EX_B_Sales.accdb, table Products (exported)",
    },
    "stores": {
        "label": "Stores",
        "required": ["ID", "City", "State", "StoreType", "Market", "Location"],
        "key": "ID",
        "grain": "One store or sales channel (SH00 is the Website).",
        "source": "Support_EX_B_Stores.csv (tab separated)",
    },
    "survey": {
        "label": "Survey Results",
        "required": ["Survey ID", "Store ID", "Year", "Satisfaction", "Interest"],
        "key": "Survey ID",
        "grain": "One store in one year.",
        "source": "Support_EX_B_Survey.xlsx, sheet Survey Results",
    },
}

# Check order matters: tables with more specific column sets are tried first,
# so a Sales file is never mistaken for something smaller.
_DETECT_ORDER = ["sales", "survey", "stores", "products", "categories"]
_HEADER_SCAN_ROWS = 15  # how far down a sheet we look for the header row


@dataclass
class LoadResult:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)   # table -> file name
    messages: list[str] = field(default_factory=list)       # plain-language notes


# ---------------------------------------------------------------------------
# 2. Reading files
# ---------------------------------------------------------------------------
def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _promote_header(raw: pd.DataFrame, required: list[str]) -> pd.DataFrame | None:
    """Find the row that holds the column names (the Survey sheet has title
    rows above its header) and rebuild the table from there."""
    if set(required).issubset({str(c).strip() for c in raw.columns}):
        return _clean_headers(raw)
    for i in range(min(_HEADER_SCAN_ROWS, len(raw))):
        row = [str(v).strip() for v in raw.iloc[i].tolist()]
        if set(required).issubset(row):
            out = raw.iloc[i + 1:].copy()
            out.columns = row
            out = out.loc[:, [c for c in out.columns if c and c != "nan"]]
            return out.dropna(how="all").reset_index(drop=True)
    return None


def _read_frames(name: str, data: bytes) -> list[tuple[str, pd.DataFrame]]:
    """Return every sheet (Excel) or the single table (CSV/TXT) in a file,
    read as text so nothing is silently converted yet."""
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=object)
        return list(sheets.items())
    if lower.endswith((".csv", ".txt", ".tsv")):
        text = data.decode("utf-8-sig", errors="replace")
        # sep=None lets pandas detect tabs vs commas; the Stores ".csv" uses tabs.
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python",
                         dtype=str, keep_default_na=True)
        return [("(csv)", df)]
    raise ValueError(f"{name}: unsupported file type. Use .xlsx or .csv.")


def identify_table(name: str, data: bytes) -> tuple[str, pd.DataFrame, str] | None:
    """Work out which E-Presso table a file holds. Returns
    (table_key, dataframe, sheet_name) or None if nothing matches."""
    for sheet, raw in _read_frames(name, data):
        for key in _DETECT_ORDER:
            df = _promote_header(raw, TABLE_SPECS[key]["required"])
            if df is not None:
                return key, df, sheet
    return None


def load_files(files: dict[str, bytes]) -> LoadResult:
    """files = {file_name: file_bytes}. Later files replace earlier ones for
    the same table, so an upload overrides the default copy."""
    result = LoadResult()
    for name, data in files.items():
        try:
            found = identify_table(name, data)
        except Exception as exc:  # unreadable file: report it, don't crash
            result.messages.append(f"Could not read {name}: {exc}")
            continue
        if found is None:
            result.messages.append(
                f"{name} did not match any E-Presso table. Check that its "
                "column names are unchanged.")
            continue
        key, df, sheet = found
        result.tables[key] = convert_types(key, df)
        where = f"{name}" + (f" (sheet '{sheet}')" if sheet != "(csv)" else "")
        result.sources[key] = where

    for key, spec in TABLE_SPECS.items():
        if key not in result.tables:
            result.messages.append(
                f"Missing table: {spec['label']}. Needed columns: "
                + ", ".join(spec["required"]))
    return result


# ---------------------------------------------------------------------------
# 3. Data types
#    Values that cannot be converted become blank (NaN/NaT) and are counted in
#    the data-quality report instead of crashing the app.
# ---------------------------------------------------------------------------
def _strip_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = df[c].replace("", pd.NA)
    return df


_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
                 "%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%y", "%m/%d/%Y"]


def parse_dates(s: pd.Series) -> pd.Series:
    """Try the known export formats one at a time, so a date is never read
    with day and month swapped. mdbtools writes 01/31/25 00:00:00; Excel and
    our own exports write 2025-01-31."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    text = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in _DATE_FORMATS:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        out[todo] = pd.to_datetime(text[todo], format=fmt, errors="coerce")
    return out


def convert_types(key: str, df: pd.DataFrame) -> pd.DataFrame:
    df = _strip_text(df)
    if key == "sales":
        df["Revenue_raw"] = df["Revenue"]
        df["Revenue"] = pd.to_numeric(
            df["Revenue"].astype(str).str.replace(r"[$,]", "", regex=True)
            .replace({"nan": None, "<NA>": None, "None": None}),
            errors="coerce")
        df["Date"] = parse_dates(df["Date"])
    elif key == "survey":
        # Year arrives as text like "Y2025"; keep the number only.
        df["Year"] = pd.to_numeric(
            df["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
        ).astype("Int64")
        for c in ("Satisfaction", "Interest"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 4. Profiling
# ---------------------------------------------------------------------------
def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, blanks, distinct values, example value."""
    rows = []
    for c in df.columns:
        if c.endswith("_raw"):
            continue
        s = df[c]
        rows.append({
            "Column": c,
            "Type": str(s.dtype),
            "Blank values": int(s.isna().sum()),
            "Distinct values": int(s.nunique(dropna=True)),
            "Example": None if s.dropna().empty else str(s.dropna().iloc[0]),
        })
    return pd.DataFrame(rows)


def grain_table(result: LoadResult) -> pd.DataFrame:
    rows = []
    for key, spec in TABLE_SPECS.items():
        df = result.tables.get(key)
        rows.append({
            "Table": spec["label"],
            "Source file / sheet": result.sources.get(key, "not loaded"),
            "One row =": spec["grain"],
            "Key column(s)": spec["key"],
            "Row count": None if df is None else len(df),
            "Contains revenue?": "Yes" if key == "sales" else "No",
        })
    out = pd.DataFrame(rows)
    out["Row count"] = out["Row count"].astype("Int64")  # whole numbers, blank if not loaded
    return out


# ---------------------------------------------------------------------------
# 5. Data-quality checks (reported, never silently fixed)
# ---------------------------------------------------------------------------
def _issue(table, check, count, detail, severity="Warning"):
    return {"Table": table, "Check": check, "Count": int(count),
            "Detail": detail, "Severity": severity}


def data_quality_report(result: LoadResult) -> pd.DataFrame:
    t = result.tables
    issues: list[dict] = []

    for key, df in t.items():
        label = TABLE_SPECS[key]["label"]
        keycol = TABLE_SPECS[key]["key"]
        visible = df[[c for c in df.columns if not c.endswith("_raw")]]

        dup_rows = int(visible.duplicated().sum())
        if dup_rows:
            issues.append(_issue(label, "Exact duplicate rows", dup_rows,
                                 "Identical rows; could double count.", "Error"))
        dup_keys = int(df[keycol].dropna().duplicated().sum())
        if dup_keys:
            issues.append(_issue(label, f"Duplicate {keycol}", dup_keys,
                                 "Key should be unique; joins on it would duplicate rows.",
                                 "Error"))
        for c in visible.columns:
            n = int(visible[c].isna().sum())
            if n:
                issues.append(_issue(label, f"Blank {c}", n,
                                     f"{n} of {len(df)} rows blank.", "Info"))

    if "sales" in t:
        s = t["sales"]
        bad_rev = s["Revenue"].isna() & s["Revenue_raw"].notna()
        if bad_rev.any():
            examples = ", ".join(map(str, s.loc[bad_rev, "Revenue_raw"].unique()[:5]))
            issues.append(_issue("Sales", "Revenue not a number", bad_rev.sum(),
                                 f"Excluded from totals. Examples: {examples}", "Error"))
        neg = int((s["Revenue"] < 0).sum())
        if neg:
            issues.append(_issue("Sales", "Negative revenue", neg,
                                 "Could be returns; confirm meaning.", "Warning"))
        zero = int((s["Revenue"] == 0).sum())
        if zero:
            issues.append(_issue("Sales", "Zero revenue", zero, "Check if valid.", "Info"))
        yr_from_key = pd.to_numeric(
            s["StoreYear"].astype(str).str.extract(r"_(\d{2})$")[0], errors="coerce") + 2000
        mismatch = int(((yr_from_key != s["Date"].dt.year)
                        & yr_from_key.notna() & s["Date"].notna()).sum())
        if mismatch:
            issues.append(_issue("Sales", "StoreYear disagrees with Date year", mismatch,
                                 "Filtering by year would depend on which column is used.",
                                 "Error"))
        key_prefix = s["StoreYear"].astype(str).str.split("_").str[0]
        prefix_bad = int((key_prefix != s["Store ID"].astype(str)).sum())
        if prefix_bad:
            issues.append(_issue("Sales", "StoreYear store differs from Store ID",
                                 prefix_bad, "Survey join would attach the wrong store.",
                                 "Error"))

        # Grain check: one row per store, product, and month.
        month = s["Date"].dt.to_period("M")
        grain_dups = int(pd.DataFrame({"st": s["Store ID"], "p": s["Product ID"],
                                       "m": month}).dropna().duplicated().sum())
        if grain_dups:
            issues.append(_issue("Sales", "More than one row per store, product, month",
                                 grain_dups, "Breaks the stated grain; revenue may be "
                                 "double counted.", "Error"))
        not_eom = s["Date"].notna() & ~s["Date"].dt.is_month_end
        if not_eom.any():
            months = ", ".join(sorted(s.loc[not_eom, "Date"].dt.strftime("%Y-%m-%d").unique())[:5])
            issues.append(_issue("Sales", "Date is not the last day of its month",
                                 not_eom.sum(), f"Dates: {months}. Monthly totals are "
                                 "unaffected; only the day is off.", "Info"))

        # A product should belong to exactly one category.
        cats_per_prod = s.groupby("Product ID")["Category ID"].nunique()
        multi = cats_per_prod[cats_per_prod > 1]
        if len(multi):
            issues.append(_issue("Sales", "Product appears in more than one category",
                                 len(multi), ", ".join(multi.index[:5]), "Error"))

        # Orphan checks: keys in Sales with no match in the lookup table.
        for lookup, s_col, l_col in [("stores", "Store ID", "ID"),
                                     ("categories", "Category ID", "Category ID"),
                                     ("products", "Product ID", "Product ID")]:
            if lookup in t:
                missing = ~s[s_col].isin(t[lookup][l_col]) & s[s_col].notna()
                if missing.any():
                    vals = ", ".join(map(str, s.loc[missing, s_col].unique()[:5]))
                    issues.append(_issue(
                        "Sales", f"{s_col} not found in {TABLE_SPECS[lookup]['label']}",
                        missing.sum(),
                        f"Revenue on these rows would be lost in an inner join: {vals}",
                        "Error"))

        # Category consistency: does Sales' Category ID match the product's?
        if "products" in t and "Category ID" in t["products"].columns:
            m = s[["Product ID", "Category ID"]].merge(
                t["products"][["Product ID", "Category ID"]].drop_duplicates("Product ID"),
                on="Product ID", how="left", suffixes=("", "_product"))
            diff = int((m["Category ID"] != m["Category ID_product"])
                       .where(m["Category ID_product"].notna(), False).sum())
            if diff:
                issues.append(_issue("Sales", "Category ID differs from product's category",
                                     diff, "Category totals depend on which path you join.",
                                     "Error"))

    if "products" in t:
        names = t["products"]["Product"].dropna()
        dup_names = names[names.duplicated(keep=False)]
        for name, ids in t["products"].loc[dup_names.index].groupby("Product")["Product ID"]:
            issues.append(_issue("Products", "Same product name, different IDs", len(ids),
                                 f"'{name}' is {', '.join(ids)}. Group by Product ID, "
                                 "not name.", "Warning"))

    if "stores" in t:
        st_ = t["stores"]
        types = sorted(st_["StoreType"].dropna().unique())
        issues.append(_issue("Stores", "StoreType values (channels)", len(types),
                             ", ".join(types), "Info"))
        # Same city listed under different Location regions.
        loc = st_.dropna(subset=["City", "State"]).groupby(["City", "State"])["Location"].nunique()
        for (city, state), n in loc[loc > 1].items():
            issues.append(_issue("Stores", "Same city, different Location", n,
                                 f"{city}, {state} appears in {n} regions.", "Warning"))

    if "survey" in t and "stores" in t:
        sv = t["survey"]
        web_ids = set(t["stores"].loc[t["stores"]["StoreType"].str.lower()
                                      .eq("website"), "ID"])
        blank_web = int((sv["Store ID"].isin(web_ids) & sv["Satisfaction"].isna()).sum())
        if blank_web:
            issues.append(_issue("Survey Results", "Website has no survey score", blank_web,
                                 "Structural, not an error: no store visit to survey. "
                                 "Exclude from satisfaction analysis.", "Info"))
        out_range = int(((sv["Satisfaction"] < 0) | (sv["Satisfaction"] > 5)).sum()
                        + ((sv["Interest"] < 0) | (sv["Interest"] > 100)).sum())
        if out_range:
            issues.append(_issue("Survey Results", "Score outside its scale", out_range,
                                 "Satisfaction is 0 to 5, Interest is 0 to 100.", "Error"))

    cols = ["Table", "Check", "Count", "Detail", "Severity"]
    return pd.DataFrame(issues, columns=cols)


# ---------------------------------------------------------------------------
# 6. Joining the tables (Stage 3)
#
#    Relationships (verified on the original data):
#      Stores.ID           1 --< Sales.Store ID      (gives the sales channel)
#      Categories.Cat. ID  1 --< Sales.Category ID
#      Products.Product ID 1 --< Sales.Product ID
#      Survey (store, year) 1 -- 1 store-year revenue table
#
#    Rules:
#      * Every lookup join is a LEFT join from Sales, so no revenue row can be
#        dropped. Rows with no match get the label "Unknown" and are counted.
#      * validate="many_to_one" makes pandas stop if a lookup key repeats,
#        which is exactly the situation that duplicates revenue.
#      * The survey is joined to revenue summed by store and year, never to
#        individual Sales rows (different grain).
# ---------------------------------------------------------------------------
UNKNOWN = "Unknown"


@dataclass
class Model:
    sales: pd.DataFrame | None = None        # Sales + store, category, product
    store_year: pd.DataFrame | None = None   # revenue by store and year + survey
    merge_log: pd.DataFrame = field(default_factory=pd.DataFrame)


def _revenue(df: pd.DataFrame) -> float:
    return float(df["Revenue"].sum()) if "Revenue" in df.columns else 0.0


def check_merge(log: list, name: str, before: pd.DataFrame, after: pd.DataFrame,
                join: str, unmatched: int = 0, note: str = "") -> None:
    """Record rows and revenue sum before and after a merge. If either
    changes, the merge duplicated or dropped data and is marked Error."""
    rb, ra = len(before), len(after)
    vb, va = _revenue(before), _revenue(after)
    ok = rb == ra and round(vb, 2) == round(va, 2)
    explanation = "Rows and revenue unchanged." if ok else \
        "Rows or revenue changed: the join duplicated or dropped data."
    if unmatched:
        explanation += f" {unmatched:,} rows had no match and are labelled '{UNKNOWN}'."
    if note:
        explanation += " " + note
    log.append({"Merge": name, "Join": join, "Rows before": rb, "Rows after": ra,
                "Revenue sum before": vb, "Revenue sum after": va,
                "Status": "OK" if ok else "Error", "Explanation": explanation})


def _safe_lookup(lookup: pd.DataFrame, key: str, log: list, label: str) -> pd.DataFrame:
    """A lookup table must have one row per key. If uploaded data breaks that,
    keep the first row per key so revenue is not duplicated, and log it."""
    dups = int(lookup[key].duplicated().sum())
    if dups:
        log.append({"Merge": f"Prepare {label}", "Join": "n/a",
                    "Rows before": len(lookup), "Rows after": len(lookup) - dups,
                    "Revenue sum before": None, "Revenue sum after": None,
                    "Status": "Warning",
                    "Explanation": f"{dups} repeated {key} value(s) in {label}; kept the "
                                   "first row of each so Sales rows are not duplicated."})
        lookup = lookup.drop_duplicates(key, keep="first")
    return lookup


def _left_lookup(sales: pd.DataFrame, lookup: pd.DataFrame | None, s_key: str,
                 l_key: str, cols: list[str], label: str, log: list) -> pd.DataFrame:
    if lookup is None:
        out = sales.copy()
        for c in cols:
            out[c] = UNKNOWN
        log.append({"Merge": f"Sales + {label}", "Join": "skipped",
                    "Rows before": len(sales), "Rows after": len(out),
                    "Revenue sum before": _revenue(sales), "Revenue sum after": _revenue(out),
                    "Status": "Warning",
                    "Explanation": f"{label} table not loaded; columns set to '{UNKNOWN}'."})
        return out
    keep = [l_key] + [c for c in cols if c in lookup.columns]
    lk = _safe_lookup(lookup[keep], l_key, log, label)
    if l_key != s_key:
        lk = lk.rename(columns={l_key: s_key})
    out = sales.merge(lk, on=s_key, how="left", validate="many_to_one", indicator=True)
    unmatched = int((out["_merge"] == "left_only").sum())
    out = out.drop(columns="_merge")
    for c in cols:
        if c not in out.columns:
            out[c] = UNKNOWN
        out[c] = out[c].fillna(UNKNOWN)
    check_merge(log, f"Sales + {label}", sales, out, f"left, many-to-one on {s_key}",
                unmatched)
    return out


def build_model(result: LoadResult) -> Model:
    t = result.tables
    log: list[dict] = []
    if "sales" not in t:
        return Model(merge_log=pd.DataFrame(log))

    sales = t["sales"].drop(columns=["Revenue_raw"], errors="ignore").copy()
    sales["Year"] = sales["Date"].dt.year.astype("Int64")
    sales["Month"] = sales["Date"].dt.month.astype("Int64")

    stores = t.get("stores")
    if stores is not None:
        stores = stores.copy()
        # Readable store name for filters and charts, e.g. "SH09 Phoenix, AZ".
        place = (stores["City"].fillna("") + ", " + stores["State"].fillna("")).str.strip(", ")
        stores["Store"] = stores["ID"] + " " + place.where(place != "", stores["StoreType"])
        stores = stores.rename(columns={"StoreType": "Channel"})
    sales = _left_lookup(sales, stores, "Store ID", "ID",
                         ["Store", "Channel", "City", "State", "Location", "Market"],
                         "Stores", log)
    sales = _left_lookup(sales, t.get("categories"), "Category ID", "Category ID",
                         ["Category"], "Categories", log)
    sales = _left_lookup(sales, t.get("products"), "Product ID", "Product ID",
                         ["Product"], "Products", log)
    # Product names repeat (three "Base Model" products), so label with the ID.
    sales["Product label"] = sales["Product ID"].astype(str) + " " + sales["Product"].astype(str)

    # Store-year table: sum revenue first, THEN join the survey 1:1.
    sy = (sales.dropna(subset=["Year"])
          .groupby(["Store ID", "Store", "Channel", "Year"], as_index=False)["Revenue"].sum())
    survey = t.get("survey")
    if survey is not None:
        sv = survey[["Store ID", "Year", "Satisfaction", "Interest"]].dropna(subset=["Year"])
        dups = int(sv.duplicated(["Store ID", "Year"]).sum())
        if dups:
            sv = sv.drop_duplicates(["Store ID", "Year"], keep="first")
        joined = sy.merge(sv, on=["Store ID", "Year"], how="left",
                          validate="one_to_one", indicator=True)
        no_score = int((joined["_merge"] == "left_only").sum())
        joined = joined.drop(columns="_merge")
        missing_scores = int(joined["Satisfaction"].isna().sum())
        note = (f"Joined at store-year grain, not to individual Sales rows. "
                f"{missing_scores} store-years have no Satisfaction score.")
        if dups:
            note += f" {dups} repeated survey store-years were dropped."
        check_merge(log, "Store-year revenue + Survey", sy, joined,
                    "left, one-to-one on Store ID + Year", no_score, note)
        sy = joined
    else:
        sy["Satisfaction"] = pd.NA
        sy["Interest"] = pd.NA

    # Final reconciliation: the enriched table must still equal raw Sales.
    raw = t["sales"]
    log.append({"Merge": "Final check: enriched Sales vs raw Sales", "Join": "n/a",
                "Rows before": len(raw), "Rows after": len(sales),
                "Revenue sum before": _revenue(raw), "Revenue sum after": _revenue(sales),
                "Status": "OK" if len(raw) == len(sales)
                and round(_revenue(raw), 2) == round(_revenue(sales), 2) else "Error",
                "Explanation": "Revenue in the dashboard table equals revenue in the "
                               "uploaded Sales file."})
    cols = ["Merge", "Join", "Rows before", "Rows after", "Revenue sum before",
            "Revenue sum after", "Status", "Explanation"]
    return Model(sales=sales, store_year=sy, merge_log=pd.DataFrame(log, columns=cols))
