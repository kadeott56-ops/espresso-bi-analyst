"""Tests for analytics.py on a tiny table you can check by hand.
Run with:  pytest -q
"""
import pandas as pd

import analytics as an

# 6 rows. Hand totals:
#   all = 100+200+50+150+300+200 = 1000
#   Website = 100+200+300 = 600 ; Brick & Mortar = 50+150+200 = 400
#   2025 = 100+50+300 = 450 ; 2026 = 200+150+200 = 550  -> growth 22.2%
SALES = pd.DataFrame({
    "Store ID": ["SH00", "SH00", "SH01", "SH01", "SH00", "SH02"],
    "Store":    ["SH00 Website", "SH00 Website", "SH01 Orlando, FL", "SH01 Orlando, FL",
                 "SH00 Website", "SH02 Boulder, CO"],
    "Channel":  ["Website", "Website", "Brick & Mortar", "Brick & Mortar", "Website",
                 "Brick & Mortar"],
    "Category": ["Beans", "Beans", "Beans", "Syrups", "Syrups", "Syrups"],
    "Product label": ["P01 French Roast", "P01 French Roast", "P01 French Roast",
                      "P06 Vanilla Bean", "P06 Vanilla Bean", "P06 Vanilla Bean"],
    "Year":     [2025, 2026, 2025, 2026, 2025, 2026],
    "Month":    [1, 1, 12, 12, 12, 1],
    "Revenue":  [100.0, 200.0, 50.0, 150.0, 300.0, 200.0],
})
STORE_YEAR = pd.DataFrame({
    "Store ID": ["SH00", "SH00", "SH01", "SH01", "SH02"],
    "Year": [2025, 2026, 2025, 2026, 2026],
    "Satisfaction": [None, None, 4.0, 3.0, 5.0],
    "Interest": [None, None, 80, 85, 90],
})


def test_kpis_by_hand():
    k = an.kpis(SALES)
    assert k["total_revenue"] == 1000
    assert k["website_revenue"] == 600 and k["store_revenue"] == 400
    assert k["website_share_pct"] == 60
    assert k["latest_year"] == 2026 and k["latest_year_revenue"] == 550
    assert round(k["latest_year_growth_pct"], 1) == 22.2


def test_filters_combine_and_update_totals():
    f = an.filter_sales(SALES, years=[2025], channels=["Brick & Mortar"])
    assert f["Revenue"].sum() == 50
    assert an.filter_sales(SALES, categories=["Syrups"])["Revenue"].sum() == 650


def test_empty_selection_returns_no_rows():
    assert an.filter_sales(SALES, years=[]).empty


def test_filter_options_come_from_data():
    o = an.filter_options(SALES)
    assert o["years"] == [2025, 2026]
    assert o["channels"] == ["Brick & Mortar", "Website"]


def test_growth_undefined_without_prior_year():
    assert an.kpis(an.filter_sales(SALES, years=[2026]))["latest_year_growth_pct"] is None
    assert an.yoy_growth(100, 0) is None


def test_growth_uses_prior_year_from_same_non_year_filters():
    web = an.filter_sales(SALES, channels=["Website"])
    web_2026 = an.filter_sales(web, years=[2026])
    k = an.kpis(web_2026, all_years_df=web)
    # Website 2026 = 200 ; 2025 = 100 + 300 = 400 -> -50%
    assert k["latest_year_growth_pct"] == -50


def test_monthly_share_uses_same_year_denominator():
    m = an.revenue_by_month(SALES)
    dec25 = m[(m["Year"] == "2025") & (m["Month"] == 12)].iloc[0]
    # Dec 2025 = 50 + 300 = 350 of 450
    assert dec25["Revenue"] == 350
    assert round(dec25["Share of year %"], 1) == 77.8
    assert m.groupby("Year")["Share of year %"].sum().round(6).eq(100).all()


def test_channel_and_category_breakdowns_add_to_total():
    assert an.revenue_by_year_channel(SALES)["Revenue"].sum() == 1000
    c = an.revenue_by_category(SALES).set_index("Category")["Revenue"]
    assert c["Beans"] == 350 and c["Syrups"] == 650


def test_store_change_first_to_last_year():
    s = an.revenue_by_store(SALES).set_index("Store")
    # SH01: 2025 = 50, 2026 = 150 -> +100, +200%
    assert s.loc["SH01 Orlando, FL", "Change $"] == 100
    assert s.loc["SH01 Orlando, FL", "Change %"] == 200
    # SH02 has no 2025 revenue: change is blank, not a made-up number
    assert pd.isna(s.loc["SH02 Boulder, CO", "Change %"])


def test_satisfaction_excludes_website_and_blanks():
    sv = an.satisfaction_vs_revenue(SALES, STORE_YEAR)
    assert set(sv["Channel"]) == {"Brick & Mortar"}
    assert len(sv) == 3  # SH01 2025, SH01 2026, SH02 2026


def test_correlation_needs_enough_pairs():
    r, n = an.correlation(pd.Series([1, 2]), pd.Series([3, 4]))
    assert r is None and n == 2
