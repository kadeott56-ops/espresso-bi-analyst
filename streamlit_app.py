"""
streamlit_app.py: layout only. Calculations live in data_prep.py (loading,
joins, data quality) and analytics.py (metrics), so the screen and the math
can be tested separately.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import analytics as an
import data_prep as dp

st.set_page_config(page_title="E-Presso BI Analyst", layout="wide")
DATA_DIR = Path(__file__).parent / "data"
CHANNEL_COLORS = {"Website": "#2a78b5", "Brick & Mortar": "#c7792b", dp.UNKNOWN: "#999999"}


def money(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"${x:,.0f}"


def dollar_axis(fig, axis="y"):
    fig.update_layout(**{f"{axis}axis_tickprefix": "$", f"{axis}axis_tickformat": ",.0f"},
                      margin=dict(t=50, b=10), legend_title_text="")
    return fig


# The cache key is the file contents (bytes). Upload a changed file and the
# app recalculates; upload the same file and it reuses the earlier result.
@st.cache_data(show_spinner="Loading E-Presso data...")
def load(files: dict[str, bytes]) -> dp.LoadResult:
    return dp.load_files(files)


def default_files() -> dict[str, bytes]:
    """Data files in data/, if any. The hosted app normally starts empty and
    uses uploaded files."""
    return {p.name: p.read_bytes() for p in sorted(DATA_DIR.glob("*"))
            if p.suffix.lower() in {".xlsx", ".csv", ".txt", ".tsv"}}


# ======================= Sidebar: data =======================
st.sidebar.header("1. Data")
uploads = st.sidebar.file_uploader(
    "Upload E-Presso files (.xlsx or .csv)",
    type=["xlsx", "csv", "txt", "tsv"], accept_multiple_files=True,
    help="Upload the five tables in any order. Files are recognised by their "
         "column names. Upload a changed file to recalculate everything.")

files = default_files()
if uploads:
    files.update({f"uploaded: {u.name}": u.getvalue() for u in uploads})

st.title("E-Presso BI Analyst")

if not files:
    st.info("Upload the E-Presso data files in the sidebar to begin.")
    st.dataframe(pd.DataFrame(
        [{"Table": s["label"], "Required columns": ", ".join(s["required"]),
          "Original source": s["source"]} for s in dp.TABLE_SPECS.values()]),
        hide_index=True, use_container_width=True)
    st.stop()

result = load(files)
model = dp.build_model(result)

for key, src in result.sources.items():
    st.sidebar.caption(f"**{dp.TABLE_SPECS[key]['label']}**: {src}")
for msg in result.messages:
    st.warning(msg)

dash_tab, data_tab = st.tabs(["📊 Dashboard", "🗂️ Data, joins & quality"])

# ======================= Dashboard tab =======================
with dash_tab:
    if model.sales is None:
        st.info("Upload the Sales table to see the dashboard.")
    else:
        sales = model.sales
        opts = an.filter_options(sales)

        # ---- Sidebar filters (choices come from the loaded data) ----
        st.sidebar.header("2. Filters")
        st.sidebar.caption("Leave a filter blank to include everything.")
        f_years = st.sidebar.multiselect("Year", opts["years"], placeholder="All years")
        f_channels = st.sidebar.multiselect("Sales channel", opts["channels"],
                                            placeholder="All channels")
        f_stores = st.sidebar.multiselect("Store", opts["stores"], placeholder="All stores")
        f_cats = st.sidebar.multiselect("Product category", opts["categories"],
                                        placeholder="All categories")

        def chosen(x):  # blank multiselect = no filter
            return x or None

        filtered = an.filter_sales(sales, chosen(f_years), chosen(f_channels),
                                   chosen(f_stores), chosen(f_cats))
        # Same filters except year: used for "growth vs prior year".
        all_years = an.filter_sales(sales, None, chosen(f_channels),
                                    chosen(f_stores), chosen(f_cats))

        def scope_part(name, sel, all_label):
            return f"{name}: " + (", ".join(map(str, sel)) if sel else all_label)

        scope = " · ".join([scope_part("Years", f_years, "all"),
                            scope_part("Channels", f_channels, "all"),
                            scope_part("Stores", f_stores, "all"),
                            scope_part("Categories", f_cats, "all")])
        st.caption(f"**Scope:** {scope}")

        if filtered.empty:
            st.warning("No sales match these filters. Remove a filter to see results.")
            st.stop()

        # ---- KPIs ----
        k = an.kpis(filtered, all_years)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total revenue", money(k["total_revenue"]),
                  help="Sum of Revenue for all rows in the current scope.")
        c2.metric("Website revenue", money(k["website_revenue"]),
                  f"{k['website_share_pct']:.1f}% of total" if k["website_share_pct"] is not None
                  else None, delta_color="off")
        c3.metric("Brick & mortar revenue", money(k["store_revenue"]))
        g = k["latest_year_growth_pct"]
        c4.metric(f"{k['latest_year']} revenue", money(k["latest_year_revenue"]),
                  f"{g:+.1f}% vs {k['latest_year'] - 1}" if g is not None else "no prior year",
                  delta_color="normal" if g is not None else "off",
                  help="Latest year in the scope compared with the year before, "
                       "using the same channel, store, and category filters.")
        if k["rows_blank_revenue"]:
            st.caption(f"{k['rows_blank_revenue']} rows with blank revenue are excluded "
                       "from totals (see Data quality).")
        st.caption("Revenue is not profit: this data has no costs, so profitability "
                   "cannot be calculated.")

        # ---- Chart 1: annual revenue by channel ----
        left, right = st.columns(2)
        with left:
            yc = an.revenue_by_year_channel(filtered)
            fig = px.bar(yc, x="Year", y="Revenue", color="Channel", barmode="group",
                         color_discrete_map=CHANNEL_COLORS,
                         title="Annual revenue by sales channel")
            st.plotly_chart(dollar_axis(fig), use_container_width=True)

        # ---- Chart 2: monthly pattern ----
        with right:
            m = an.revenue_by_month(filtered)
            as_share = st.toggle("Show months as % of each year's revenue", value=False)
            ycol = "Share of year %" if as_share else "Revenue"
            fig = px.line(m, x="Month name", y=ycol, color="Year", markers=True,
                          category_orders={"Month name": an.MONTH_NAMES},
                          title="Monthly revenue pattern",
                          labels={"Month name": "Month"})
            if as_share:
                fig.update_layout(yaxis_ticksuffix="%", margin=dict(t=50, b=10))
            else:
                dollar_axis(fig)
            st.plotly_chart(fig, use_container_width=True)
            if as_share:
                st.caption("Denominator: that year's revenue under the same filters.")

        # ---- Chart 3: category ----
        left, right = st.columns(2)
        with left:
            cat = an.revenue_by_category(filtered)
            fig = px.bar(cat, x="Revenue", y="Category", orientation="h",
                         text=cat["Share %"].map(lambda v: f"{v:.1f}%"),
                         title="Revenue by product category (share of scope)")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(dollar_axis(fig, "x"), use_container_width=True)

        # ---- Chart 4: top products ----
        with right:
            tp = an.top_products(filtered, 10)
            fig = px.bar(tp, x="Revenue", y="Product label", color="Category",
                         orientation="h", title="Top 10 products by revenue",
                         labels={"Product label": "Product"})
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(dollar_axis(fig, "x"), use_container_width=True)

        # ---- Chart 5: stores ----
        st.markdown("#### Store performance")
        stores_df = an.revenue_by_store(filtered)
        hide_web = st.checkbox("Hide the Website so physical stores are easier to compare",
                               value=False)
        plot_df = stores_df[stores_df["Channel"] != "Website"] if hide_web else stores_df
        fig = px.bar(plot_df, x="Revenue", y="Store", color="Channel", orientation="h",
                     color_discrete_map=CHANNEL_COLORS, title="Revenue by store",
                     height=max(300, 28 * len(plot_df) + 80))
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(dollar_axis(fig, "x"), use_container_width=True)
        if "Change %" in stores_df.columns:
            st.caption(f"Change compares the first and last year in scope "
                       f"({stores_df['Change period'].iloc[0]}). Blank means the store "
                       "had no revenue in the first year.")
            st.dataframe(
                stores_df.drop(columns=["Change period"]), hide_index=True,
                use_container_width=True,
                column_config={
                    "Revenue": st.column_config.NumberColumn(format="$%,.0f"),
                    "Change $": st.column_config.NumberColumn(format="$%,.0f"),
                    "Change %": st.column_config.NumberColumn(format="%.1f%%")})
        else:
            st.caption("Select two or more years to see each store's change over time.")

        # ---- Chart 6: satisfaction vs revenue ----
        st.markdown("#### Customer satisfaction and revenue")
        sv = an.satisfaction_vs_revenue(filtered, model.store_year)
        if sv.empty:
            st.info("No physical-store survey scores in this scope.")
        else:
            r, n = an.correlation(sv["Satisfaction"], sv["Revenue"])
            fig = px.scatter(sv, x="Satisfaction", y="Revenue", color="Year",
                             hover_data=["Store", "Interest"],
                             title="Store-year revenue vs. satisfaction score (0 to 5)")
            st.plotly_chart(dollar_axis(fig), use_container_width=True)
            st.caption(
                (f"Correlation r = {r:.2f} across {n} store-years. " if r is not None
                 else f"Too few store-years ({n}) for a correlation. ")
                + "This shows association only, not that satisfaction causes revenue. "
                  "Each point is one physical store in one year; the Website has no survey "
                  "and is excluded. Revenue reflects the current filters.")

# ======================= Data tab =======================
with data_tab:
    sales_raw = result.tables.get("sales")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tables loaded", f"{len(result.tables)} of {len(dp.TABLE_SPECS)}")
    if sales_raw is not None:
        c2.metric("Sales rows", f"{len(sales_raw):,}")
        c3.metric("Revenue (all valid rows)", f"${sales_raw['Revenue'].sum():,.2f}")
        d = sales_raw["Date"].dropna()
        c4.metric("Date range", f"{d.min():%b %Y} to {d.max():%b %Y}" if len(d) else "n/a")

    st.markdown("#### Grain statements")
    st.caption("What one row of each table represents. Row counts are calculated "
               "from the loaded files.")
    st.dataframe(dp.grain_table(result), hide_index=True, use_container_width=True)

    st.markdown("#### Table relationships and merge checks")
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
    if result.tables:
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
