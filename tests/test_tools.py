"""Tests for the assistant's tool functions (Stage 6).
Uses the same 6-row table as test_analytics.py, so every answer can be
checked by hand. Run with:  pytest -q
"""
import json

import pandas as pd

import analytics as an
from test_analytics import SALES, STORE_YEAR

CONTRACT_KEYS = {"tool", "value", "definition", "scope", "rows_used", "caveats"}


def test_every_tool_returns_the_contract_and_is_json_safe():
    dq = pd.DataFrame([{"Table": "Stores", "Check": "Blank City", "Count": 1,
                        "Detail": "x", "Severity": "Info"}])
    results = [
        an.get_total_revenue(SALES),
        an.get_revenue_by_channel(SALES),
        an.get_revenue_by_year(SALES),
        an.get_store_trend(SALES),
        an.get_category_performance(SALES),
        an.get_monthly_revenue_share(SALES),
        an.get_satisfaction_revenue_relationship(SALES, STORE_YEAR),
        an.get_data_quality_report(dq),
    ]
    for r in results:
        assert CONTRACT_KEYS <= set(r), r["tool"]
        json.dumps(r)  # must not raise: results go to the log and the model


def test_total_revenue_with_filters():
    # Website 2025 = 100 + 300 = 400
    r = an.get_total_revenue(SALES, year=2025, channel="Website")
    assert r["value"] == 400 and r["rows_used"] == 2
    assert r["scope"] == {"year": 2025, "channel": "Website", "store": "All", "category": "All"}
    assert an.REVENUE_NOT_PROFIT in r["caveats"]


def test_parameters_are_matched_flexibly_but_never_guessed():
    assert an.get_total_revenue(SALES, channel="online")["value"] == 600
    assert an.get_total_revenue(SALES, channel="brick and mortar")["value"] == 400
    assert an.get_total_revenue(SALES, store="Boulder")["value"] == 200     # partial name
    assert an.get_total_revenue(SALES, year="Y2026")["value"] == 550


def test_unknown_values_return_an_error_not_a_number():
    for r in (an.get_total_revenue(SALES, store="Chicago"),
              an.get_total_revenue(SALES, year=2031),
              an.get_total_revenue(SALES, category="Tea")):
        assert r["value"] is None and "error" in r


def test_channel_shares():
    v = an.get_revenue_by_channel(SALES)["value"]
    assert v["Website"]["revenue"] == 600 and v["Website"]["share_pct"] == 60
    assert v["Brick & Mortar"]["share_pct"] == 40


def test_revenue_by_year_and_growth():
    v = an.get_revenue_by_year(SALES)["value"]
    assert v["revenue_by_year"] == {"2025": 450, "2026": 550}
    assert v["yoy_growth_pct"]["2025"] is None
    assert v["yoy_growth_pct"]["2026"] == 22.22


def test_store_trend_largest_decline_and_skipped_stores():
    r = an.get_store_trend(SALES)
    v = r["value"]
    # SH00 Website: 2025 = 400, 2026 = 200 -> -50%. SH01: 50 -> 150 -> +200%.
    assert v["largest_pct_decline"]["store"] == "SH00 Website"
    assert v["largest_pct_decline"]["change_pct"] == -50
    assert v["largest_pct_growth"]["store"] == "SH01 Orlando, FL"
    # SH02 has no 2025 revenue, so it cannot have a % change.
    assert any("SH02" in c for c in r["caveats"])


def test_store_trend_says_so_when_no_store_declined():
    r = an.get_store_trend(SALES, channel="Brick & Mortar")
    assert r["value"]["largest_pct_decline"] is None
    assert any("No store declined" in c for c in r["caveats"])


def test_store_trend_rejects_backwards_years():
    assert "error" in an.get_store_trend(SALES, start_year=2026, end_year=2025)


def test_category_performance():
    v = an.get_category_performance(SALES)["value"]
    assert v[0] == {"category": "Syrups", "revenue": 650, "share_pct": 65}


def test_monthly_share_denominator_is_same_year():
    v = an.get_monthly_revenue_share(SALES, year=2025)["value"]
    # 2025: Jan = 100, Dec = 350, total 450
    assert v["2025"]["Dec"]["share_of_year_pct"] == 77.78
    assert v["2025"]["Jan"]["share_of_year_pct"] == 22.22


def test_satisfaction_tool_warns_about_causation_and_small_sample():
    r = an.get_satisfaction_revenue_relationship(SALES, STORE_YEAR)
    assert r["value"]["store_years_used"] == 3
    assert any("does not show that satisfaction causes" in c for c in r["caveats"])
    assert any("Website" in c for c in r["caveats"])


def test_empty_scope_returns_no_value_with_explanation():
    r = an.get_total_revenue(SALES, year=2025, store="Boulder")  # SH02 only sold in 2026
    assert r["value"] is None and "error" not in r
    assert "No sales rows match" in r["caveats"][0]
