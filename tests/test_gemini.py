"""Tests for the Gemini router (Stage 8) using a scripted fake Gemini, so they
run without an API key or network. They check that:
  * the app, not the model, runs the tools, and logs every call;
  * an answer with a number not found in any tool result is rejected;
  * quota errors and missing models fall back cleanly.
Run with:  pytest -q
"""
import sys
import types as pytypes

import pandas as pd
import pytest

import agent as ag
import analytics as an
from test_analytics import SALES, STORE_YEAR

CTX = ag.Context(SALES, STORE_YEAR, pd.DataFrame(columns=["Table", "Check", "Count",
                                                          "Detail", "Severity"]),
                 pd.DataFrame())
OPTS = an.filter_options(SALES)


# ---------------- a minimal fake of the google-genai SDK ----------------
class FC:
    def __init__(self, name, args):
        self.name, self.args, self.id = name, args, f"id-{name}"


class Resp:
    def __init__(self, calls=None, text=None):
        self.function_calls = calls or None
        self.text = text
        self.candidates = [pytypes.SimpleNamespace(content=("model-turn", calls))]


def install_fake(monkeypatch, script, seen):
    """script: model name -> list of Resp objects or an Exception to raise."""
    types = pytypes.SimpleNamespace(
        Content=lambda role, parts: (role, parts),
        Part=type("Part", (), {
            "__init__": lambda self, text=None: setattr(self, "text", text),
            "from_function_response": staticmethod(
                lambda name, response, id=None: ("fn-response", name, response))}),
        Tool=lambda function_declarations: function_declarations,
        GenerateContentConfig=lambda **kw: kw,
        AutomaticFunctionCallingConfig=lambda disable: {"disable": disable},
    )

    class Models:
        def generate_content(self, model, contents, config):
            seen.append(model)
            step = script[model]
            if isinstance(step, Exception):
                raise step
            return step.pop(0)

    genai = pytypes.SimpleNamespace(Client=lambda api_key: pytypes.SimpleNamespace(models=Models()))
    google = pytypes.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    genai.types = types


Q = "How much did the website make in 2025?"
GOOD = ("*Scope: year 2025, channel Website, all stores, all categories*\n\n"
        "**Findings**\n- Website revenue was $400.\n\n"
        "**Next steps**\n- No recommendation is supported by this data.")


def test_gemini_chooses_tool_app_runs_it_and_answer_is_accepted(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": [
        Resp(calls=[FC("get_total_revenue", {"year": 2025, "channel": "Website"})]),
        Resp(text=GOOD)]}, seen)
    answer, log, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1"])
    assert how.startswith("Answered by: Gemini (m1)")
    assert log[0]["tool"] == "get_total_revenue" and log[0]["result"]["value"] == 400
    assert "$400" in answer


def test_invented_number_is_rejected_and_rules_answer_shown(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": [
        Resp(calls=[FC("get_total_revenue", {"year": 2025, "channel": "Website"})]),
        Resp(text=GOOD.replace("$400", "$450"))]}, seen)
    answer, log, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1"])
    assert "rules-based router" in how and "$450" in how
    assert "$400" in answer                     # the rules answer, from the tool


def test_answer_without_any_tool_call_cannot_contain_numbers(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": [Resp(text="*Scope: all*\n\nRevenue was $9,999.")]}, seen)
    _, _, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1"])
    assert "not found in any tool result" in how


def test_profit_answer_without_numbers_is_allowed(monkeypatch):
    text = ("*Scope: no calculation was run.*\n\n**Findings**\n- Profit cannot be calculated: "
            "the data has no costs.\n\n**Next steps**\n- No recommendation is supported by this data.")
    seen = []
    install_fake(monkeypatch, {"m1": [Resp(text=text)]}, seen)
    answer, log, how = ag.answer("Which store is most profitable?", CTX, OPTS,
                                 api_key="fake", models=["m1"])
    assert how.startswith("Answered by: Gemini") and log == []


def test_quota_error_tries_next_model(monkeypatch):
    seen = []
    install_fake(monkeypatch, {
        "m1": Exception("429 RESOURCE_EXHAUSTED quota"),
        "m2": [Resp(calls=[FC("get_total_revenue", {"year": 2025, "channel": "Website"})]),
               Resp(text=GOOD)]}, seen)
    _, _, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1", "m2"])
    assert seen[0] == "m1" and how.startswith("Answered by: Gemini (m2)")


def test_all_models_failing_falls_back_to_rules(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": Exception("404 NOT_FOUND"),
                               "m2": Exception("429 RESOURCE_EXHAUSTED")}, seen)
    answer, log, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1", "m2"])
    assert "rules-based router" in how and "quota" in how
    assert "$400" in answer and log[0]["tool"] == "get_total_revenue"


def test_bad_key_falls_back_with_plain_message(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": Exception("400 API_KEY_INVALID")}, seen)
    _, _, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1"])
    assert "the API key is invalid" in how


def test_model_cannot_call_unapproved_function(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": [Resp(calls=[FC("run_python", {"code": "print(1)"})]),
                                      Resp(text="*Scope: none*\n\nI could not run that.")]}, seen)
    answer, log, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1"])
    assert "not an approved tool" in log[0]["result"]["error"]


def test_missing_scope_line_is_added(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": [
        Resp(calls=[FC("get_total_revenue", {"year": 2025})]),
        Resp(text="**Findings**\n- Revenue was $450.")]}, seen)
    answer, _, _ = ag.answer("Revenue in 2025?", CTX, OPTS, api_key="fake", models=["m1"])
    assert answer.startswith("*Scope: year 2025")


def test_no_key_uses_rules():
    _, _, how = ag.answer(Q, CTX, OPTS, api_key=None)
    assert how == "Answered by: rules-based router (no Gemini API key configured)"


@pytest.mark.parametrize("text,ok", [
    ("$6,339,640", True), ("$6.3 million", True), ("$6.4 million", False),
    ("64.5%", True), ("65%", True), ("70%", False), ("-$360,610", True)])
def test_number_check_respects_written_precision(text, ok):
    log = [{"result": {"value": {"a": 6339640.0, "b": 64.54, "c": -360610.0}, "rows_used": 1}}]
    assert (ag.unsupported_numbers(text, log) == []) is ok


def test_declarations_only_offer_values_in_the_data():
    decls = {d["name"]: d for d in ag.tool_declarations(OPTS)}
    assert set(decls) == set(ag.TOOLS)
    store_enum = decls["get_total_revenue"]["parameters"]["properties"]["store"]["enum"]
    assert store_enum == OPTS["stores"]


def test_timeout_falls_back_quickly(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": TimeoutError("Request timed out"),
                               "m2": [Resp(text=GOOD)]}, seen)
    answer, log, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1", "m2"])
    assert "took too long" in how and seen == ["m1"]      # does not keep trying
    assert "$400" in answer


def test_time_budget_stops_trying(monkeypatch):
    seen = []
    install_fake(monkeypatch, {"m1": [Resp(text=GOOD)]}, seen)
    monkeypatch.setattr(ag, "TIME_BUDGET_S", -1)
    _, _, how = ag.answer(Q, CTX, OPTS, api_key="fake", models=["m1"])
    assert "took too long" in how and seen == []
