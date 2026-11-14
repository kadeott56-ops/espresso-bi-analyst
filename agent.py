"""
agent.py: the analytical assistant's tool registry, dispatcher, and routers.

How a question is answered:
  1. A ROUTER reads the question and picks one or more tools from TOOLS, with
     parameters. Stage 7 uses a rules-based (keyword) router. Stage 8 adds a
     language-model router that chooses from the same TOOLS.
  2. The DISPATCHER (run_tool) runs the chosen function on the currently
     loaded data and logs the call: tool name, arguments, raw result.
  3. The ANSWER WRITER turns the tool results into Findings, Interpretation,
     and Next steps. Every number in Findings comes from a tool result.

What the assistant can never do:
  * run code or SQL written by a model or a user (no eval, exec, or
    DataFrame agents): only the named functions in TOOLS can run;
  * change the loaded data: every tool only reads it;
  * state a number that is not in a tool result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import analytics as an

MAX_QUESTIONS = 15
FILTER_PARAMS = ("year", "channel", "store", "category")

# ---------------------------------------------------------------------------
# 1. Tool registry: the ONLY functions the assistant may call.
#    "params" lists the arguments each tool accepts; anything else is dropped.
#    "needs" says which loaded objects the dispatcher must pass in.
# ---------------------------------------------------------------------------
TOOLS: dict[str, dict] = {
    "get_total_revenue": {
        "fn": an.get_total_revenue, "needs": ["sales"],
        "params": ["year", "channel", "store", "category"],
        "description": "Total revenue for an optional year, sales channel, store, and/or "
                       "product category.",
    },
    "get_revenue_by_channel": {
        "fn": an.get_revenue_by_channel, "needs": ["sales"],
        "params": ["year", "store", "category"],
        "description": "Website vs. brick-and-mortar revenue and each channel's share.",
    },
    "get_revenue_by_year": {
        "fn": an.get_revenue_by_year, "needs": ["sales"],
        "params": ["channel", "store", "category"],
        "description": "Revenue for each year and year-over-year growth %.",
    },
    "get_store_trend": {
        "fn": an.get_store_trend, "needs": ["sales"],
        "params": ["start_year", "end_year", "channel", "category"],
        "description": "Each store's revenue change between a start and end year, with the "
                       "largest % decline and largest % growth.",
    },
    "get_revenue_by_store": {
        "fn": an.get_revenue_by_store, "needs": ["sales"],
        "params": ["year", "channel", "category"],
        "description": "Stores ranked by revenue, with each store's share.",
    },
    "get_category_performance": {
        "fn": an.get_category_performance, "needs": ["sales"],
        "params": ["year", "channel", "store"],
        "description": "Revenue and share for each product category.",
    },
    "get_top_products": {
        "fn": an.get_top_products, "needs": ["sales"],
        "params": ["year", "channel", "store", "category", "n"],
        "description": "Top products by revenue (n = how many, default 5).",
    },
    "get_monthly_revenue_share": {
        "fn": an.get_monthly_revenue_share, "needs": ["sales"],
        "params": ["year", "channel", "store", "category"],
        "description": "Revenue by month and each month's share of that year's revenue "
                       "(seasonality).",
    },
    "get_satisfaction_revenue_relationship": {
        "fn": an.get_satisfaction_revenue_relationship, "needs": ["sales", "store_year"],
        "params": ["year", "store"],
        "description": "Correlation between physical stores' customer satisfaction scores "
                       "and their annual revenue (association only).",
    },
    "get_data_quality_report": {
        "fn": an.get_data_quality_report, "needs": ["dq", "merge_log"],
        "params": [],
        "description": "Data-quality findings and whether every table join kept rows and "
                       "revenue unchanged.",
    },
}

CAPABILITIES = [
    "total revenue for any year, channel, store, or category",
    "website vs. brick-and-mortar revenue",
    "revenue by year and year-over-year growth",
    "which stores grew or declined the most between two years",
    "store revenue rankings",
    "category performance and top products",
    "monthly patterns and seasonality",
    "how customer satisfaction relates to revenue",
    "data-quality issues in the loaded files",
]


# ---------------------------------------------------------------------------
# 2. Dispatcher
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """Everything a tool may read. Built fresh from the loaded data on every
    app run, so answers always reflect the current files."""
    sales: object
    store_year: object
    dq: object
    merge_log: object


def run_tool(name: str, args: dict, ctx: Context, log: list) -> dict:
    """Run one registered tool and append {tool, arguments, result} to log."""
    if name not in TOOLS:
        result = {"tool": name, "value": None, "error": f"'{name}' is not an approved tool.",
                  "definition": "", "scope": {}, "rows_used": 0, "caveats": []}
        log.append({"tool": name, "arguments": args, "result": result})
        return result
    spec = TOOLS[name]
    clean = {k: v for k, v in (args or {}).items()
             if k in spec["params"] and v not in (None, "", "All", "all")}
    inputs = [getattr(ctx, n) for n in spec["needs"]]
    try:
        result = spec["fn"](*inputs, **clean)
    except Exception as exc:  # a tool failure must not crash the chat
        result = {"tool": name, "value": None, "error": f"The calculation failed: {exc}",
                  "definition": "", "scope": clean, "rows_used": 0, "caveats": []}
    log.append({"tool": name, "arguments": clean, "result": result})
    return result


# ---------------------------------------------------------------------------
# 3. Rules-based router (Option B). Reads keywords and picks tools.
# ---------------------------------------------------------------------------
PROFIT_WORDS = r"\b(profit\w*|margin\w*|cost\w*|expense\w*|earnings|net income|ebitda|roi|" \
               r"return on investment|break[- ]?even)\b"
DECISION_WORDS = r"\b(should|close|closing|shut|open a|open new|invest|expand|recommend\w*|" \
                 r"worth keeping)\b"
NOT_IN_DATA = {
    r"\bmanag(er|ers|es|ed|ing|ement)\b|\bwho runs\b|\bin charge\b|\bphone\b|\baddress\b|"
    r"\bzip\b|\bpostal\b|\bcontact\b": "store contact or manager details",
    r"\bemployees?\b|\bstaff\w*\b|\bwages?\b|\bsalar\w*\b|\bpayroll\b": "staffing or wages",
    r"\bprices?\b|\bpricing\b|\bunits?\b|\bquantit\w*\b|\bvolume\b": "prices or units sold",
    r"\bcustomers? (count|numbers?)\b|\bhow many customers\b|\bfoot traffic\b|\bvisits?\b":
        "customer counts or traffic",
    r"\baverage (sale|order|transaction|ticket|basket)\b|\btransactions?\b|\border size\b":
        "individual transactions (each Sales row is a monthly total for one store and product)",
    r"\binventory\b|\bstock levels?\b": "inventory",
}
FORECAST_WORDS = r"\b(forecast\w*|predict\w*|projection|next year|will .* be)\b"
WEBSITE_WORDS = r"\b(website|online|web|e-?commerce|internet)\b"
STORE_CH_WORDS = r"\b(brick|physical|in-store|retail locations?|brick-and-mortar)\b"
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


@dataclass
class Plan:
    intent: str                                  # e.g. "store_trend", "profit"
    calls: list = field(default_factory=list)    # [(tool_name, args)]
    sources: dict = field(default_factory=dict)  # param -> "question" | "sidebar filter"
    month: str | None = None                     # month the user asked about
    notes: list = field(default_factory=list)


def extract_entities(question: str, opts: dict) -> dict:
    q = question.lower()
    ent: dict = {}
    years = sorted({int(y) for y in re.findall(r"\b(20\d\d)\b", q)})
    if years:
        ent["years"] = years
    web, phys = re.search(WEBSITE_WORDS, q), re.search(STORE_CH_WORDS, q)
    if web and not phys:
        ent["channel"] = "Website"
    elif phys and not web:
        ent["channel"] = "Brick & Mortar"
    elif web and phys:
        ent["both_channels"] = True
    # Store: by ID (SH11) or by city name taken from the loaded Stores table.
    m = re.search(r"\bsh\s?(\d{2})\b", q)
    if m:
        ent["store"] = f"SH{m.group(1)}"
    else:
        cities = {}
        for s in opts.get("stores", []):
            parts = s.split(" ", 1)
            if len(parts) == 2 and "," in parts[1]:
                cities.setdefault(parts[1].split(",")[0].strip().lower(), []).append(s)
        for city in sorted(cities, key=len, reverse=True):
            if re.search(rf"\b{re.escape(city)}\b", q):
                ent["store"] = city.title()   # ambiguous cities make the tool list options
                break
    for c in opts.get("categories", []):
        stem = c.lower().rstrip("s")
        if re.search(rf"\b{re.escape(stem)}", q):
            ent["category"] = c
            break
    for name in MONTHS:
        if re.search(rf"\b({name}|{name[:3]})\b", q):
            ent["month"] = name[:3].title()
            break
    n = re.search(r"\btop\s+(\d{1,2})\b", q)
    if n:
        ent["n"] = int(n.group(1))
    return ent


def route(question: str, opts: dict, defaults: dict | None = None) -> Plan:
    """Pick tools for a question. `defaults` are single-value sidebar filters
    used for any parameter the question does not mention."""
    q = question.lower()
    ent = extract_entities(question, opts)
    defaults = {k: v for k, v in (defaults or {}).items() if v is not None}

    def args_for(tool: str, **explicit) -> tuple[dict, dict]:
        params = TOOLS[tool]["params"]
        args, src = {}, {}
        given = {"year": ent["years"][-1] if len(ent.get("years", [])) == 1 else None,
                 "channel": ent.get("channel"), "store": ent.get("store"),
                 "category": ent.get("category"), "n": ent.get("n")}
        given.update(explicit)
        for p in params:
            if given.get(p) is not None:
                args[p], src[p] = given[p], "question"
            elif p in defaults:
                args[p], src[p] = defaults[p], "sidebar filter"
        return args, src

    def plan(intent, tool, **explicit):
        args, src = args_for(tool, **explicit)
        return Plan(intent, [(tool, args)], src, ent.get("month"))

    # --- questions the data cannot answer ---
    for pattern, what in NOT_IN_DATA.items():
        if re.search(pattern, q):
            p = Plan("not_in_data")
            p.notes.append(what)
            return p
    if re.search(PROFIT_WORDS, q):
        p = Plan("profit", sources={}, month=None)
        if ent.get("store") or re.search(r"\bstores?\b", q):
            a, s = args_for("get_revenue_by_store")
            p.calls, p.sources = [("get_revenue_by_store", a)], s
            p.notes.append("revenue shown as context only")
        return p
    if re.search(FORECAST_WORDS, q) or any(y > max(opts.get("years", [0]) or [0])
                                           for y in ent.get("years", [])):
        return Plan("forecast")

    # --- business decisions: show the evidence, do not recommend ---
    if re.search(DECISION_WORDS, q):
        calls, src = [], {}
        if ent.get("store"):
            a, s = args_for("get_total_revenue")
            calls.append(("get_total_revenue", a)); src.update(s)
        a2, _ = args_for("get_store_trend")
        calls.append(("get_store_trend", a2))
        return Plan("decision", calls, src, None)

    if re.search(r"satisf|survey|happy|happiness|rating|customer experience", q):
        return plan("satisfaction", "get_satisfaction_revenue_relationship")
    if re.search(r"data quality|quality|missing|blank|duplicate|null|error|problem|issue|"
                 r"clean|trust the data", q):
        return Plan("data_quality", [("get_data_quality_report", {})])
    growth_words = (r"declin|decreas|drop|fell|fall|grew|grow|growth|increas|improv|trend|"
                    r"change|worst|best performing|biggest gain|faster|slower")
    if re.search(growth_words, q) and (ent.get("both_channels") or re.search(r"\bchannels?\b", q)):
        calls = []
        for ch in opts.get("channels", []):
            if ch == "Unknown":
                continue
            a, _ = args_for("get_revenue_by_year", channel=ch)
            calls.append(("get_revenue_by_year", a))
        return Plan("channel_growth", calls, {}, None)
    if re.search(growth_words, q) and re.search(r"\bstores?\b|\blocations?\b|\bshops?\b", q):
        yrs = ent.get("years", [])
        explicit = {}
        if len(yrs) >= 2:
            explicit = {"start_year": yrs[0], "end_year": yrs[-1]}
        p = plan("store_trend", "get_store_trend", **explicit)
        if len(yrs) >= 2:
            p.sources.update(start_year="question", end_year="question")
        return p
    if re.search(r"\bmonth|monthly|season|holiday|peak|busiest|slowest", q) or ent.get("month"):
        return plan("monthly", "get_monthly_revenue_share")
    if re.search(r"\bproducts?\b|\bitems?\b|best[- ]?sell", q):
        return plan("products", "get_top_products")
    if re.search(r"categor|product line|product type", q) or \
            (ent.get("category") is None and re.search(r"what sells|which sells", q)):
        return plan("category", "get_category_performance")
    if re.search(r"\b(rank|ranking|top|highest|lowest|most|least|largest|smallest|best|"
                 r"worst)\b", q) and re.search(r"\bstores?\b|\blocations?\b", q) \
            and not ent.get("store"):
        return plan("store_rank", "get_revenue_by_store")
    if ent.get("both_channels") or re.search(r"\bchannels?\b|website vs|online vs|"
                                              r"versus|compare.*(website|online)", q):
        return plan("channel", "get_revenue_by_channel")
    if re.search(r"each year|per year|by year|annual|yearly|year over year|yoy|over time|"
                 r"growth|grow|trend", q) or len(ent.get("years", [])) >= 2:
        return plan("by_year", "get_revenue_by_year")
    if re.search(r"revenue|sales|sell|sold|make|made|earn|generate|bring in|how much|total",
                 q) or ent:
        return plan("total", "get_total_revenue")
    return Plan("unsupported")


# ---------------------------------------------------------------------------
# 4. Answer writer: Findings (numbers from tool results only), Interpretation,
#    Next steps. Starts with the scope.
# ---------------------------------------------------------------------------
def money(x) -> str:
    if x is None:
        return "n/a"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def pct(x, signed=False) -> str:
    return "n/a" if x is None else (f"{x:+.1f}%" if signed else f"{x:.1f}%")


def scope_line(results: list[dict], sources: dict) -> str:
    if not results:
        return "Scope: no calculation was run."
    sc = dict(results[0].get("scope", {}))
    chans = {r.get("scope", {}).get("channel") for r in results}
    if len(results) > 1 and len(chans) > 1 and "channel" in sc:
        sc["channel"] = "All"  # one call per channel: overall scope covers every channel
    parts = []
    labels = {"year": "year", "channel": "channel", "store": "store", "category": "category",
              "start_year": "from", "end_year": "to"}
    for k in ("year", "start_year", "end_year", "channel", "store", "category"):
        if k not in sc:
            continue
        v = sc[k]
        if v in (None, "All"):
            plural = {"year": "all years", "channel": "all channels", "store": "all stores",
                      "category": "all categories"}.get(k, f"all {k}s")
            parts.append(plural)
        else:
            tag = f" (from the {sources[k]})" if sources.get(k) == "sidebar filter" else ""
            parts.append(f"{labels[k]} {v}{tag}")
    if "data" in sc:
        parts.append(sc["data"].lower())
    return "Scope: " + ", ".join(parts) if parts else "Scope: all data"


def _section(title, lines):
    if not lines:
        return ""
    body = "\n".join(f"- {l}" for l in lines)
    return f"**{title}**\n{body}"


def _findings_for(r: dict, month: str | None) -> tuple[list[str], list[str]]:
    """Return (findings, interpretation) for one tool result."""
    t, v = r["tool"], r.get("value")
    if r.get("error"):
        return [f"I could not run {t}: {r['error']}"], []
    if v is None:
        return [r["caveats"][0] if r.get("caveats") else "No data matched."], []
    F, I = [], []
    if t == "get_total_revenue":
        F.append(f"Total revenue: **{money(v)}** ({r['rows_used']:,} sales rows).")
    elif t == "get_revenue_by_channel":
        for ch, d in sorted(v.items(), key=lambda kv: -kv[1]["revenue"]):
            F.append(f"{ch}: **{money(d['revenue'])}** ({pct(d['share_pct'])} of revenue).")
        if len(v) > 1:
            top = max(v, key=lambda c: v[c]["revenue"])
            I.append(f"{top} is the larger channel in this scope.")
    elif t == "get_revenue_by_year":
        rev, g = v["revenue_by_year"], v["yoy_growth_pct"]
        for y, amt in rev.items():
            F.append(f"{y}: **{money(amt)}**" + (f" ({pct(g[y], True)} vs {int(y) - 1})"
                                                  if g.get(y) is not None else ""))
        if v.get("change_first_to_last_pct") is not None:
            I.append(f"Revenue changed {pct(v['change_first_to_last_pct'], True)} from "
                     f"{v['first_year']} to {v['last_year']}.")
            gs = {y: x for y, x in g.items() if x is not None}
            if gs:
                best = max(gs, key=gs.get)
                I.append(f"The fastest growth was in {best} ({pct(gs[best], True)}).")
    elif t == "get_store_trend":
        sy, ey = r["scope"]["start_year"], r["scope"]["end_year"]
        d, gr = v["largest_pct_decline"], v["largest_pct_growth"]
        if d:
            F.append(f"Largest % decline: **{d['store']}**, {money(d['start_revenue'])} in {sy} "
                     f"to {money(d['end_revenue'])} in {ey} (**{pct(d['change_pct'], True)}**, "
                     f"{money(d['change'])}).")
        else:
            F.append(f"No store declined between {sy} and {ey}.")
        if gr:
            F.append(f"Largest % growth: **{gr['store']}** ({pct(gr['change_pct'], True)}, "
                     f"{money(gr['start_revenue'])} to {money(gr['end_revenue'])}).")
        else:
            F.append(f"No store grew between {sy} and {ey}.")
        F.append(f"{v['stores_declining']} store(s) declined and {v['stores_growing']} grew.")
        growers = [x for x in v["stores"] if x["change"] > 0]
        if 0 < len(growers) <= 5:
            F.append("Stores that grew: " + "; ".join(
                f"{x['store']} ({pct(x['change_pct'], True)})" for x in reversed(growers)) + ".")
        I.append("A revenue decline alone does not show whether a store is unprofitable or "
                 "should change; costs and local factors are not in this data.")
    elif t == "get_revenue_by_store":
        for row in v[:5]:
            F.append(f"#{row['rank']} {row['store']} ({row['channel']}): **{money(row['revenue'])}**"
                     f" ({pct(row['share_pct'])}).")
        if len(v) > 5:
            last = v[-1]
            F.append(f"Lowest: {last['store']}: {money(last['revenue'])} "
                     f"({pct(last['share_pct'])}).")
    elif t == "get_category_performance":
        for row in v:
            F.append(f"{row['category']}: **{money(row['revenue'])}** ({pct(row['share_pct'])}).")
        I.append(f"{v[0]['category']} is the largest category, at {pct(v[0]['share_pct'])} "
                 "of revenue in this scope.")
    elif t == "get_top_products":
        for row in v:
            F.append(f"#{row['rank']} {row['product']} ({row['category']}): "
                     f"**{money(row['revenue'])}** ({pct(row['share_pct'])}).")
    elif t == "get_monthly_revenue_share":
        for y, months in v.items():
            if month and month in months:
                d = months[month]
                F.append(f"{month} {y}: **{money(d['revenue'])}**, "
                         f"**{pct(d['share_of_year_pct'])}** of {y} revenue.")
            else:
                hi = max(months, key=lambda mo: months[mo]["revenue"])
                lo = min(months, key=lambda mo: months[mo]["revenue"])
                F.append(f"{y}: peak month {hi} ({money(months[hi]['revenue'])}, "
                         f"{pct(months[hi]['share_of_year_pct'])} of the year); lowest {lo} "
                         f"({pct(months[lo]['share_of_year_pct'])}).")
        I.append("Shares use each year's own total as the denominator, so years of different "
                 "size can be compared.")
    elif t == "get_satisfaction_revenue_relationship":
        if v["correlation_r"] is None:
            F.append(f"Not enough store-years with scores ({v['store_years_used']}) to "
                     "calculate a correlation.")
        else:
            F.append(f"Correlation between satisfaction and annual store revenue: "
                     f"**r = {v['correlation_r']:.2f}** ({v['strength']}), across "
                     f"{v['store_years_used']} store-years.")
            F.append(f"Average satisfaction score: {v['average_satisfaction']:.2f} out of 5.")
        I.append("This is an association, not proof of cause. Larger or better-located stores "
                 "could have both higher revenue and happier customers, and revenue could "
                 "influence satisfaction rather than the reverse.")
    elif t == "get_data_quality_report":
        F.append(f"{v['errors']} error(s) and {v['warnings']} warning(s) found.")
        for f in [f for f in v["findings"] if f["severity"] in ("Error", "Warning")][:6]:
            F.append(f"{f['table']}: {f['check']} ({f['count']}). {f['detail']}")
        if v["all_merges_kept_rows_and_revenue"] is not None:
            F.append("Every table join kept the same rows and revenue."
                     if v["all_merges_kept_rows_and_revenue"]
                     else "At least one table join changed rows or revenue.")
    return F, I


def write_answer(plan: Plan, results: list[dict]) -> str:
    findings, interp, nexts = [], [], []

    if plan.intent == "profit":
        findings.append("Profit cannot be calculated: the E-Presso data contains revenue only, "
                        "with no product costs, wages, rent, or other expenses.")
        for r in results:
            f, _ = _findings_for(r, None)
            findings += [f"Revenue (for context, not profit): {x}" for x in f[:5]]
        interp.append("The highest-revenue store is not necessarily the most profitable; a "
                      "large store can also have the highest costs.")
        nexts.append("To measure profitability, E-Presso would need cost of goods sold, "
                     "operating expenses, and lease costs by store.")
        nexts.append("I can show revenue by store, store trends, or channel revenue instead.")
    elif plan.intent == "not_in_data":
        findings.append(f"No calculation was run: the E-Presso data does not include "
                        f"{plan.notes[0]}, and none of my approved functions report it.")
        nexts.append("I can answer questions about: " + "; ".join(CAPABILITIES) + ".")
    elif plan.intent == "channel_growth":
        rows = []
        for r in results:
            if r.get("error") or r.get("value") is None:
                findings += _findings_for(r, None)[0]
                continue
            v = r["value"]
            ch = r["scope"]["channel"]
            rev = v["revenue_by_year"]
            findings.append(f"{ch}: {money(rev[str(v['first_year'])])} in {v['first_year']} to "
                            f"{money(rev[str(v['last_year'])])} in {v['last_year']} "
                            f"(**{pct(v['change_first_to_last_pct'], True)}**).")
            if v["change_first_to_last_pct"] is not None:
                rows.append((v["change_first_to_last_pct"], ch))
        if len(rows) >= 2:
            rows.sort()
            interp.append(f"{rows[-1][1]} grew faster than {rows[0][1]} over this period.")
        nexts.append("No recommendation is supported by this data alone; channel growth does "
                     "not show which channel is more profitable.")
    elif plan.intent == "forecast":
        findings.append("No calculation was run: I can describe the loaded data, but I do not "
                        "forecast future values.")
        nexts.append("Ask for revenue by year or store trends to see past patterns.")
    elif plan.intent == "unsupported":
        findings.append("I can't answer that from the E-Presso data with my approved functions.")
        nexts.append("I can answer questions about: " + "; ".join(CAPABILITIES) + ".")
    else:
        for r in results:
            f, i = _findings_for(r, plan.month)
            findings += f
            interp += i
        if plan.intent == "decision":
            # Report the asked-about store's own change, not just the overall extremes.
            store = next((r["scope"].get("store") for r in results
                          if r.get("scope", {}).get("store") not in (None, "All")), None)
            trend = next((r for r in results if r["tool"] == "get_store_trend"
                          and r.get("value")), None)
            if store and trend:
                rows = trend["value"]["stores"]
                mine = next((x for x in rows if x["store"] == store), None)
                if mine:
                    sy, ey = trend["scope"]["start_year"], trend["scope"]["end_year"]
                    rank = [x["store"] for x in rows].index(store) + 1
                    findings.insert(1, f"{store} changed **{pct(mine['change_pct'], True)}** "
                                       f"from {sy} to {ey} ({money(mine['start_revenue'])} to "
                                       f"{money(mine['end_revenue'])}), rank {rank} of "
                                       f"{len(rows)} stores from steepest decline.")
            interp.append("Revenue evidence alone does not justify closing, opening, or "
                          "investing in a store.")
            nexts.append("No recommendation is supported by this data. A decision would also "
                         "need store costs and profitability, lease terms, staffing, and local "
                         "market and customer information.")
        if results and all(r.get("error") for r in results):
            nexts.append("Ask again using one of the valid values listed above.")
        elif not nexts:
            nexts.append("No recommendation is supported by this data alone; use these figures "
                         "as a starting point for questions about costs and causes.")

    caveats = []
    for r in results:
        for c in r.get("caveats", []):
            if c not in caveats and c != an.REVENUE_NOT_PROFIT:
                caveats.append(c)
    if any(an.REVENUE_NOT_PROFIT in r.get("caveats", []) for r in results) \
            and plan.intent != "profit":
        caveats.append(an.REVENUE_NOT_PROFIT)

    if results and all(r.get("error") for r in results):
        scope = "Scope: not calculated, because a value in the question did not match the data."
    else:
        scope = scope_line(results, plan.sources)
    parts = [f"*{scope}*",
             _section("Findings", findings),
             _section("Interpretation", interp),
             _section("Caveats", caveats),
             _section("Next steps", nexts)]
    return "\n\n".join(p for p in parts if p)


def answer_with_rules(question: str, ctx: Context, opts: dict,
                      defaults: dict | None = None) -> tuple[str, list[dict], Plan]:
    """Option B: route with keywords, run tools, write the answer."""
    plan = route(question, opts, defaults)
    log: list[dict] = []
    results = [run_tool(name, args, ctx, log) for name, args in plan.calls]
    return write_answer(plan, results), log, plan
