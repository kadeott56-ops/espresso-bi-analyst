"""
streamlit_app.py: layout only. Calculations live in data_prep.py (and later
analytics.py), so the screen and the math can be tested separately.

Stage 1: load the E-Presso tables and show a data profile.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import data_prep as dp

st.set_page_config(page_title="E-Presso BI Analyst", layout="wide")
DATA_DIR = Path(__file__).parent / "data"


# The cache key is the file contents (bytes). Upload a changed file and the
# app recalculates; upload the same file and it reuses the earlier result.
@st.cache_data(show_spinner="Loading E-Presso data...")
def load(files: dict[str, bytes]) -> dp.LoadResult:
    return dp.load_files(files)


def default_files() -> dict[str, bytes]:
    """The supplied E-Presso exports in data/, used until the user uploads."""
    return {p.name: p.read_bytes() for p in sorted(DATA_DIR.glob("*"))
            if p.suffix.lower() in {".xlsx", ".csv", ".txt", ".tsv"}}


# ---------------- Sidebar: data source ----------------
st.sidebar.header("Data")
uploads = st.sidebar.file_uploader(
    "Upload updated E-Presso files (.xlsx or .csv)",
    type=["xlsx", "csv", "txt", "tsv"], accept_multiple_files=True,
    help="Upload one or more tables. Any table you don't upload uses the "
         "supplied copy. Files are recognised by their column names.")

files = default_files()
if uploads:
    # Uploaded files are added last, so they replace the default copy.
    files.update({f"uploaded: {u.name}": u.getvalue() for u in uploads})

if not files:
    st.title("E-Presso BI Analyst")
    st.info("Upload the E-Presso data files in the sidebar to begin. "
            "Files are recognised by their column names, so they can be "
            "uploaded in any order.")
    st.dataframe(pd.DataFrame(
        [{"Table": s["label"], "Required columns": ", ".join(s["required"]),
          "Original source": s["source"]} for s in dp.TABLE_SPECS.values()]),
        hide_index=True, use_container_width=True)
    st.stop()

result = load(files)

st.sidebar.caption("Using: " + ("uploaded files + defaults" if uploads else "supplied data"))
for key, src in result.sources.items():
    st.sidebar.caption(f"**{dp.TABLE_SPECS[key]['label']}**: {src}")

# ---------------- Main page ----------------
st.title("E-Presso BI Analyst")
st.subheader("Data profile")

for msg in result.messages:
    st.warning(msg)

if not result.tables:
    st.error("No E-Presso tables loaded. Upload files in the sidebar.")
    st.stop()

# Headline numbers, all calculated from the loaded data.
sales = result.tables.get("sales")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tables loaded", f"{len(result.tables)} of {len(dp.TABLE_SPECS)}")
if sales is not None:
    c2.metric("Sales rows", f"{len(sales):,}")
    c3.metric("Revenue (all valid rows)", f"${sales['Revenue'].sum():,.2f}")
    d = sales["Date"].dropna()
    c4.metric("Date range", f"{d.min():%b %Y} to {d.max():%b %Y}" if len(d) else "n/a")

st.markdown("#### Grain statements")
st.caption("What one row of each table represents. Row counts are calculated "
           "from the loaded files.")
st.dataframe(dp.grain_table(result), hide_index=True, use_container_width=True)

st.markdown("#### Table relationships and merge checks")
model = dp.build_model(result)
st.markdown(
    "- **Stores → Sales** on `Store ID` (one store, many sales rows). Gives the sales channel.\n"
    "- **Categories → Sales** on `Category ID` (one category, many rows).\n"
    "- **Products → Sales** on `Product ID` (one product, many rows).\n"
    "- **Survey → store-year revenue** on `Store ID` + `Year` (one to one). The survey is "
    "joined *after* revenue is summed by store and year, never to individual sales rows.\n\n"
    "All lookups are left joins, so no revenue row can be dropped. Rows and revenue are "
    "recorded before and after every merge.")
if model.merge_log.empty:
    st.info("Load the Sales table to build the data model.")
else:
    bad = model.merge_log["Status"].eq("Error").sum()
    (st.error if bad else st.success)(
        f"{bad} merge(s) changed rows or revenue." if bad
        else "Every merge kept the same rows and revenue.")
    st.dataframe(model.merge_log, hide_index=True, use_container_width=True,
                 column_config={
                     "Revenue sum before": st.column_config.NumberColumn(format="$%,.2f"),
                     "Revenue sum after": st.column_config.NumberColumn(format="$%,.2f")})

st.markdown("#### Data quality")
dq = dp.data_quality_report(result)
if dq.empty:
    st.success("No data-quality issues found.")
else:
    errors = int((dq["Severity"] == "Error").sum())
    st.caption(f"{len(dq)} findings, {errors} marked Error. Issues are reported, "
               "not silently fixed.")
    st.dataframe(dq, hide_index=True, use_container_width=True)

st.markdown("#### Tables")
tabs = st.tabs([dp.TABLE_SPECS[k]["label"] for k in result.tables])
for tab, (key, df) in zip(tabs, result.tables.items()):
    with tab:
        shown = df[[c for c in df.columns if not c.endswith("_raw")]]
        st.caption(f"{len(shown):,} rows x {shown.shape[1]} columns. "
                   f"Source: {result.sources[key]}")
        st.markdown("**Column profile**")
        st.dataframe(dp.profile_table(df), hide_index=True, use_container_width=True)
        st.markdown("**First 20 rows**")
        st.dataframe(shown.head(20), hide_index=True, use_container_width=True)
