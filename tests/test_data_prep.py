"""Stage 1 tests on tiny tables whose answers you can check by hand.
Run with:  pytest -q
"""
import io

import pandas as pd

import data_prep as dp


def _xlsx(df: pd.DataFrame, title_rows: int = 0, sheet="Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=sheet, startrow=title_rows)
    return buf.getvalue()


STORES_TSV = (
    "ID\tCity\tState\tPostalCode\tPhone\tStoreType\tMarket\tLocation\tManager\n"
    "SH00\t\t\t\t1-800\tWebsite\t5-Global\tWebsite\tA\n"
    "SH01\tOrlando\tFL\t1\tx\tBrick & Mortar\t3-Medium\tSouth\tB\n"
    "SH02\tOrlando\tFL\t2\tx\tBrick & Mortar\t1-Small\tEast\tC\n"
).encode()

SALES = pd.DataFrame({
    "Sales ID":    ["S1", "S2", "S3", "S4", "S5"],
    "Store ID":    ["SH00", "SH01", "SH01", "SH02", "SH99"],
    "StoreYear":   ["SH00_25", "SH01_25", "SH01_26", "SH02_26", "SH99_26"],
    "Category ID": ["C1", "C1", "C2", "C2", "C1"],
    "Product ID":  ["P01", "P01", "P02", "P02", "P01"],
    "Date":        ["2025-01-15", "2025-02-01", "2026-03-01", "2025-04-01", "2026-05-01"],
    "Revenue":     [100, 200.5, "abc", 50, 10],
})
CATS = pd.DataFrame({"Category ID": ["C1", "C2"], "Category": ["Coffee", "Syrups"]})
PRODS = pd.DataFrame({"Product ID": ["P01", "P02"], "Product": ["Beans", "Vanilla"],
                      "Category ID": ["C1", "C2"]})
SURVEY = pd.DataFrame({"Survey ID": ["SH00_25", "SH01_25"], "Store ID": ["SH00", "SH01"],
                       "Year": ["Y2025", "Y2025"], "Satisfaction": [None, 4.1],
                       "Interest": [None, 89]})


def _load():
    return dp.load_files({
        "Stores.csv": STORES_TSV,
        "Sales.xlsx": _xlsx(SALES),
        "Categories.xlsx": _xlsx(CATS),
        "Products.xlsx": _xlsx(PRODS),
        "Survey.xlsx": _xlsx(SURVEY, title_rows=3, sheet="Survey Results"),
    })


def test_all_tables_identified_by_columns():
    r = _load()
    assert set(r.tables) == {"sales", "categories", "products", "stores", "survey"}


def test_tab_separated_csv_is_read():
    assert list(_load().tables["stores"]["ID"]) == ["SH00", "SH01", "SH02"]


def test_survey_header_below_title_rows_and_year_parsed():
    sv = _load().tables["survey"]
    assert len(sv) == 2
    assert list(sv["Year"]) == [2025, 2025]


def test_revenue_total_excludes_unparseable_value():
    # By hand: 100 + 200.5 + 50 + 10 = 360.5 ("abc" is excluded)
    assert _load().tables["sales"]["Revenue"].sum() == 360.5


def test_quality_checks_find_planted_problems():
    dq = dp.data_quality_report(_load())
    checks = dict(zip(dq["Check"], dq["Count"]))
    assert checks["Revenue not a number"] == 1          # "abc"
    assert checks["Store ID not found in Stores"] == 1  # SH99
    assert checks["StoreYear disagrees with Date year"] == 1  # S4: _26 vs 2025
    assert checks["Same city, different Location"] == 2       # Orlando
    assert checks["Website has no survey score"] == 1


def test_mdbtools_date_format_parsed():
    # mdbtools writes 01/31/25 00:00:00; must be Jan 31 2025, not Jan 25 2031.
    out = dp.parse_dates(pd.Series(["01/31/25 00:00:00", "2025-02-28"]))
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2025-01-31", "2025-02-28"]


def test_duplicate_store_product_month_is_flagged():
    dup = pd.concat([SALES, SALES.iloc[[1]].assign(**{"Sales ID": "S9"})])
    r = dp.load_files({"Sales.xlsx": _xlsx(dup), "Stores.csv": STORES_TSV})
    checks = set(dp.data_quality_report(r)["Check"])
    assert "More than one row per store, product, month" in checks


def test_missing_table_is_reported_not_crashing():
    r = dp.load_files({"Stores.csv": STORES_TSV})
    assert any("Missing table: Sales" in m for m in r.messages)
    dp.data_quality_report(r)  # must not raise


def test_unrecognised_file_gives_plain_message():
    r = dp.load_files({"junk.csv": b"a,b\n1,2\n"})
    assert any("did not match" in m for m in r.messages)


# ---------------- Stage 3: joins ----------------
def test_joins_keep_rows_and_revenue():
    m = dp.build_model(_load())
    # By hand: 5 sales rows; revenue 360.5 ("abc" is blank, not dropped).
    assert len(m.sales) == 5
    assert m.sales["Revenue"].sum() == 360.5
    assert set(m.merge_log["Status"]) <= {"OK", "Warning"}  # no Error


def test_unmatched_store_is_labelled_not_dropped():
    m = dp.build_model(_load())
    row = m.sales[m.sales["Store ID"] == "SH99"].iloc[0]
    assert row["Channel"] == dp.UNKNOWN and row["Revenue"] == 10


def test_channel_totals_add_up():
    m = dp.build_model(_load())
    by_ch = m.sales.groupby("Channel")["Revenue"].sum().to_dict()
    # Website SH00 = 100; Brick & Mortar SH01+SH02 = 200.5 + 50; Unknown SH99 = 10
    assert by_ch == {"Website": 100, "Brick & Mortar": 250.5, dp.UNKNOWN: 10}


def test_duplicate_lookup_key_does_not_duplicate_revenue():
    cats = pd.concat([CATS, CATS.iloc[[0]]])          # C1 appears twice
    r = _load()
    r.tables["categories"] = cats
    m = dp.build_model(r)
    assert m.sales["Revenue"].sum() == 360.5          # not inflated
    assert "Warning" in set(m.merge_log["Status"])


def test_survey_joined_at_store_year_grain():
    m = dp.build_model(_load())
    sy = m.store_year
    # SH01 has two sales rows but in different years -> two store-years.
    assert len(sy[sy["Store ID"] == "SH01"]) == 2
    sh01_25 = sy[(sy["Store ID"] == "SH01") & (sy["Year"] == 2025)].iloc[0]
    assert sh01_25["Revenue"] == 200.5 and sh01_25["Satisfaction"] == 4.1


def test_model_without_lookups_still_builds():
    r = dp.load_files({"Sales.xlsx": _xlsx(SALES)})
    m = dp.build_model(r)
    assert m.sales["Revenue"].sum() == 360.5
    assert (m.sales["Channel"] == dp.UNKNOWN).all()
