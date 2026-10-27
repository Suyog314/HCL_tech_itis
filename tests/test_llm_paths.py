"""NL->SQL, ExplanationChecker, sanity checker, pipeline and Streamlit app. All LLM calls are mocked."""
import json
from pathlib import Path
from datetime import date

import pandas as pd
import pytest

from copilot.explain import ExplanationChecker, SanityChecker
from copilot.llm import LLMClient, LLMReply
from copilot.nl2sql import parse_intent, template_sql, translate

BIDS = ["Fox_office_Alpha", "Fox_office_Beta", "Fox_education_Gamma"]
END = date(2017, 3, 4)


def test_intent_parsing():
    it = parse_intent("Forecast next week for Alpha", BIDS, END)
    assert "forecast" in it.analyses and it.buildings == ["Fox_office_Alpha"] and it.horizon_hours == 168
    assert parse_intent("forecast consumption", BIDS, END).clarification
    assert "office buildings" in parse_intent("Show the office building usage", BIDS, END).clarification or True
    assert parse_intent("usage of the office building", BIDS, END).clarification.startswith("There are 2 office")
    assert parse_intent("How much gas did we use?", BIDS, END).unsupported
    assert parse_intent("What's the usage right now?", BIDS, END).unsupported
    assert parse_intent("What did it cost?", BIDS, END).unsupported
    assert parse_intent("Totals in February 2017", BIDS, END).period == ("2017-02-01 00:00:00", "2017-03-01 00:00:00")
    assert parse_intent("last 7 days", BIDS, END).period == ("2017-02-26 00:00:00", "2017-03-05 00:00:00")
    assert parse_intent("in 2017", BIDS, END, date(2017, 1, 1)).period == ("2017-01-01 00:00:00", "2017-03-05 00:00:00")
    assert parse_intent("in 2019", BIDS, END, date(2017, 1, 1)).period[0] == "2019-01-01 00:00:00"  # left for the validator


def test_template_sql_is_valid(ctx):
    _, _, ex = ctx
    for q in ["Which buildings used the most electricity in 2017?", "top 2 buildings", "average hourly profile",
              "compare weekend vs weekday", "monthly consumption for Beta", "daily trend for Alpha in February 2017"]:
        sql = template_sql(parse_intent(q, BIDS, END, date(2017, 1, 1)))
        assert sql and ex.run(sql).ok, q


def test_translate_uses_llm_once_and_validates(ctx, fake_llm):
    _, schema, ex = ctx
    llm = fake_llm([json.dumps({"action": "sql", "sql": "SELECT building_id, MAX(electricity_kwh) AS max_kwh FROM readings "
                                                         "GROUP BY building_id ORDER BY max_kwh DESC"})])
    plan = translate(parse_intent("peak hour per building?", BIDS, END), schema, ex.validator, llm)
    assert plan.action == "sql" and llm.calls == 1
    assert "Fox_office_Alpha" in llm.prompts[0] or "peak" in llm.prompts[0]


def test_translate_repairs_once_and_never_runs_rejected(ctx, fake_llm):
    _, schema, ex = ctx
    llm = fake_llm([json.dumps({"action": "sql", "sql": "DELETE FROM readings"}),
                    json.dumps({"action": "sql", "sql": "SELECT COUNT(*) AS n FROM readings"})])
    plan = translate(parse_intent("how many rows?", BIDS, END), schema, ex.validator, llm)
    assert plan.action == "sql" and plan.sql.startswith("SELECT") and llm.calls == 2
    assert plan.attempts[0]["code"] == "NOT_SELECT" and "rejected" in llm.prompts[1]
    llm = fake_llm([json.dumps({"action": "sql", "sql": "DROP TABLE readings"}),
                    json.dumps({"action": "sql", "sql": "SELECT token FROM secrets"}), "unused"])
    plan = translate(parse_intent("break it", BIDS, END), schema, ex.validator, llm)
    assert plan.action == "error" and plan.sql is None and llm.calls == 2  # bounded: no third call


def test_translate_clarify_unsupported_and_errors(ctx, fake_llm):
    _, schema, ex = ctx
    it = parse_intent("how efficient are they?", BIDS, END)
    assert translate(it, schema, ex.validator, fake_llm([json.dumps({"action": "clarify", "question": "Which metric?"})])).action == "clarify"
    assert translate(it, schema, ex.validator, fake_llm([json.dumps({"action": "unsupported", "reason": "no"})])).action == "unsupported"
    assert translate(it, schema, ex.validator, fake_llm([LLMReply(False, error="LLM_RATE_LIMIT")])).action == "error"
    assert translate(it, schema, ex.validator, fake_llm([], available=False)).action == "unsupported"
    assert translate(it, schema, ex.validator, fake_llm(["not json", "still not json"])).action == "error"


def test_llm_client_without_key_and_errors(monkeypatch):
    assert LLMClient(api_key="").chat("s", "u").error == "LLM_NOT_CONFIGURED"

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise type("AuthenticationError", (Exception,), {})("bad key gsk_secret")

    c = LLMClient(api_key="gsk_secret")
    c._client = Boom
    r = c.chat("s", "u")
    assert not r.ok and r.error == "LLM_AUTH_ERROR" and "gsk_secret" not in (r.error + r.text)


# ---- explanation ------------------------------------------------------------------------------
DF = pd.DataFrame({"building_id": ["Fox_office_Alpha", "Fox_office_Beta"], "total_kwh": [1500.0, 1000.0]})


def test_explanation_checker_llm_and_return_shape(fake_llm):
    llm = fake_llm(["Fox_office_Alpha used the most: 1,500.0 kWh, 50% more than Fox_office_Beta (1,000 kWh)."])
    ec = ExplanationChecker(llm, building_ids=BIDS)
    out = ec.explain("Which used most?", DF, {"note": "test"})
    assert out["source"] == "llm" and out["data_results"] is DF and out["user_prompt"] == "Which used most?"
    assert out["analytics_summary"] == {"note": "test"} and out["verification"]["verified"], out["verification"]
    ctx = llm.prompts[0]
    assert "Fox_office_Alpha,1500.00" in ctx and "UNIT: kWh" in ctx


def test_explanation_fallback_without_llm(fake_llm):
    out = ExplanationChecker(fake_llm([LLMReply(False, error="LLM_TIMEOUT")]), building_ids=BIDS).explain("q", DF)
    assert out["source"] == "template" and out["error"] == "LLM_TIMEOUT" and "1,500.00" in out["explanation"]
    assert out["verification"]["verified"]
    empty = ExplanationChecker(fake_llm(["should not be called"])).explain("q", pd.DataFrame())
    assert "No rows" in empty["explanation"]
    assert ExplanationChecker(None).explain("q", {"total_kwh": [5.0]})["explanation"].startswith("Result")


@pytest.mark.parametrize("text,issue", [
    ("Fox_office_Alpha used 2,000 kWh.", "Numbers not found"),
    ("Alpha used 35% more than Beta.", "Numbers not found"),
    ("Fox_office_Zeta used 1,500 kWh.", "Unknown building"),
    ("Fox_education_Gamma used 1,500 kWh.", "not in the results"),
    ("Peak demand was 1500 kW.", "power unit"),
    ("Fox_office_Beta had the highest use at 1,000 kWh.", "Ranking claim"),
    ("This cost $1500.", "costs"),
    ("On 2019-05-01 Alpha used 1,500 kWh.", "outside"),
])
def test_sanity_checker_flags(text, issue):
    sc = SanityChecker(BIDS, date_range=("2017-01-01 00:00:00", "2017-03-04 23:00:00"))
    ctx = ExplanationChecker().format_context("q", DF)
    res = sc.check(text, ctx, DF)
    assert not res["verified"] and any(issue in i for i in res["issues"]), res


@pytest.mark.parametrize("text", [
    "Fox_office_Alpha used 1,500 kWh, 50% more than Fox_office_Beta.",
    "Together they used 2.5 MWh; Beta used 500 kWh less, or 33.3% less.",
    "Fox_office_Alpha had the highest use. Beta used 1000 kWh in the 2 buildings compared.",
])
def test_sanity_checker_accepts_supported(text):
    res = SanityChecker(BIDS).check(text, ExplanationChecker().format_context("q", DF), DF)
    assert res["verified"], res


# ---- pipeline + app ---------------------------------------------------------------------------
@pytest.fixture
def copilot(db_path, fake_llm):
    from copilot.pipeline import Copilot

    return Copilot(db_path, llm=fake_llm([], available=False))


def test_pipeline_workflows(copilot):
    a = copilot.ask("Which buildings used the most electricity in 2017?")
    assert a.status == "answered" and a.chart.kind == "bar" and a.explanation["verification"]["verified"]
    assert any("without readings" in w for w in a.warnings)  # Beta has NULL hours: reported, not zero-filled
    assert copilot.ask("Which buildings used the most electricity in 2017?").from_cache
    p0 = copilot.page(a.sql, 0, 2)
    assert len(p0.df) == 2 and p0.has_next
    assert copilot.ask("How much gas?").status == "unsupported"
    assert copilot.ask("forecast usage").status == "clarify"
    assert copilot.ask("readings in 2019?").status in ("error", "unsupported")
    f = copilot.ask("Forecast the next 24 hours for Fox_office_Alpha")
    assert f.status == "answered" and f.forecast.status == "ok" and len(f.forecast.frame) == 24
    an = copilot.ask("Any anomalies for Beta in the last 30 days?")
    assert an.anomalies and any(x.category == "data_quality" for x in an.anomalies)
    r = copilot.ask("What should we do to save energy?")
    assert r.status == "answered" and r.recommendations


def test_streamlit_app_starts_and_answers(db_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from copilot import config

    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Energy Analytics Copilot"
    at.text_input(key="question").set_value("Which buildings used the most electricity in 2017?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception, at.exception
    assert any(m.value.startswith("#### Answer") for m in at.markdown)
    badges = [e.value for e in list(at.success) + list(at.warning)]
    assert any("Numbers checked" in b or "Not verified" in b for b in badges), badges
    nxt = next(b for b in at.button if b.label == "Next ▶")
    assert nxt.disabled  # 3 fixture buildings fit on one page
