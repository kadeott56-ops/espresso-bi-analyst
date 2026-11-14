"""Tests for the rules-based assistant (Stage 7): routing, the dispatcher's
safety rules, and the answer format. Uses the 6-row table from
test_analytics.py so answers can be checked by hand. Run with:  pytest -q
"""
import pandas as pd

import agent as ag
import analytics as an
from test_analytics import SALES, STORE_YEAR

DQ = pd.DataFrame([{"Table": "Stores", "Check": "Blank City", "Count": 1,
                    "Detail": "1 of 3 rows blank.", "Severity": "Info"}])
CTX = ag.Context(SALES, STORE_YEAR, DQ, pd.DataFrame())
OPTS = an.filter_options(SALES)


def tools_for(q, defaults=None):
    return [name for name, _ in ag.route(q, OPTS, defaults).calls]


def test_required_agent_test_questions_route_to_the_right_tools():
    assert tools_for("How much revenue did the website generate in 2026?") == ["get_total_revenue"]
    assert tools_for("Which store experienced the greatest percentage revenue decline?") \
        == ["get_store_trend"]
    assert tools_for("Does higher customer satisfaction cause higher revenue?") \
        == ["get_satisfaction_revenue_relationship"]
    assert ag.route("Which E-Presso store is the most profitable?", OPTS).intent == "profit"


def test_more_question_types():
    assert tools_for("Which channel grew faster?") == ["get_revenue_by_year"] * 2
    assert tools_for("Compare website and brick and mortar revenue") == ["get_revenue_by_channel"]
    assert tools_for("What share of 2025 revenue came in December?") == ["get_monthly_revenue_share"]
    assert tools_for("Which category sells the most?") == ["get_category_performance"]
    assert tools_for("What are the top 3 products?") == ["get_top_products"]
    assert tools_for("Which store had the highest revenue?") == ["get_revenue_by_store"]
    assert tools_for("Are there any data quality problems?") == ["get_data_quality_report"]


def test_questions_outside_the_data_run_no_tools():
    for q in ("Who is the manager of the Boulder store?", "Who manages the Boulder store?",
              "Who runs the Boulder location?", "What is the average transaction size?",
              "What will revenue be in 2031?", "What's the weather like?"):
        assert tools_for(q) == [], q


def test_parameters_come_from_the_question():
    plan = ag.route("How much did Boulder sell online in 2026?", OPTS)
    assert plan.calls == [("get_total_revenue",
                           {"year": 2026, "channel": "Website", "store": "Boulder"})]


def test_sidebar_defaults_fill_gaps_and_question_wins():
    plan = ag.route("What was total revenue?", OPTS, {"year": 2025})
    assert plan.calls[0][1] == {"year": 2025} and plan.sources["year"] == "sidebar filter"
    plan = ag.route("What was revenue in 2026?", OPTS, {"year": 2025})
    assert plan.calls[0][1] == {"year": 2026} and plan.sources["year"] == "question"


def test_answer_has_scope_findings_and_numbers_from_the_tool():
    answer, log, _ = ag.answer_with_rules("How much revenue did the website make in 2025?",
                                          CTX, OPTS)
    assert answer.startswith("*Scope: year 2025, channel Website")
    assert "**Findings**" in answer and "**Next steps**" in answer
    assert "$400" in answer                       # 100 + 300, by hand
    assert log[0]["result"]["value"] == 400       # the same number is in the tool log


def test_scope_line_marks_sidebar_values():
    answer, _, _ = ag.answer_with_rules("What was total revenue?", CTX, OPTS, {"year": 2026})
    assert "year 2026 (from the sidebar filter)" in answer


def test_profit_question_explains_why_it_cannot_be_answered():
    answer, _, _ = ag.answer_with_rules("Which store is most profitable?", CTX, OPTS)
    assert "Profit cannot be calculated" in answer
    assert "not necessarily the most profitable" in answer


def test_causation_question_is_answered_as_association():
    answer, _, _ = ag.answer_with_rules("Does satisfaction cause revenue?", CTX, OPTS)
    assert "association, not proof of cause" in answer


def test_decision_question_gives_evidence_but_no_recommendation():
    answer, log, _ = ag.answer_with_rules("Should E-Presso close the Boulder store?", CTX, OPTS)
    assert "No recommendation is supported by this data" in answer
    assert {c["tool"] for c in log} == {"get_total_revenue", "get_store_trend"}


def test_dispatcher_only_runs_registered_tools():
    log = []
    r = ag.run_tool("os.system", {"cmd": "rm -rf /"}, CTX, log)
    assert "not an approved tool" in r["error"] and len(log) == 1


def test_dispatcher_drops_arguments_a_tool_does_not_accept():
    log = []
    ag.run_tool("get_total_revenue", {"year": 2025, "sql": "DROP TABLE"}, CTX, log)
    assert log[0]["arguments"] == {"year": 2025}


def test_answers_change_when_the_data_changes():
    changed = SALES.copy()
    changed.loc[0, "Revenue"] += 1000                 # SH00, 2025
    ctx2 = ag.Context(changed, STORE_YEAR, DQ, pd.DataFrame())
    q = "How much revenue did the website make in 2025?"
    before, _, _ = ag.answer_with_rules(q, CTX, OPTS)
    after, _, _ = ag.answer_with_rules(q, ctx2, OPTS)
    assert "$400" in before and "$1,400" in after


def test_ambiguous_store_says_nothing_was_calculated():
    two = SALES.copy()
    two.loc[two["Store ID"] == "SH02", "Store"] = "SH02 Orlando, FL"
    ctx = ag.Context(two, STORE_YEAR, DQ, pd.DataFrame())
    answer, _, _ = ag.answer_with_rules("How much did Orlando make?", ctx, an.filter_options(two))
    assert "Scope: not calculated" in answer
    assert "SH01 Orlando, FL" in answer and "SH02 Orlando, FL" in answer
    assert "Ask again using one of the valid values" in answer
