"""Energy Analytics Copilot - Streamlit UI. Run: streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import config
from copilot.anomaly import AnomalyDetector
from copilot.charts import forecast_figure, render, select_chart
from copilot.nl2sql import Intent
from copilot.pipeline import Copilot
from copilot.recommend import to_frame

st.set_page_config(page_title="Energy Analytics Copilot", page_icon="⚡", layout="wide")

EXAMPLES = [
    "Which buildings used the most electricity in 2017?",
    "Show monthly consumption for Fox_office_Clayton",
    "What is the average hourly profile in July 2017?",
    "Compare weekend vs weekday usage",
    "Any anomalies in the last 30 days?",
    "Forecast the next 7 days for the campus total",
    "Expected vs actual for Fox_education_Ray in June 2017",
    "What should we do to save energy in office buildings?",
]
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


@st.cache_resource(show_spinner="Opening the energy database...")
def get_copilot() -> Copilot:
    return Copilot()


@st.cache_data(show_spinner=False, max_entries=64)
def cached_anomalies(_cp: Copilot, version: str, buildings: tuple, start: str, end: str):
    return _cp.run_anomalies(Intent("anomalies", buildings=list(buildings), period=(start, end)))[0]


@st.cache_data(show_spinner=False, max_entries=64)
def cached_forecast(_cp: Copilot, version: str, target: str, horizon: int, as_of: str | None):
    period = (f"{as_of} 00:00:00", str(pd.Timestamp(as_of) + pd.Timedelta(hours=horizon))) if as_of else None
    it = Intent("expected" if as_of else "forecast", buildings=[] if target == "Campus total" else [target],
                campus=target == "Campus total", period=period, horizon_hours=horizon)
    return _cp.run_forecast(it)


def show_recommendations(recs):
    if not recs:
        return
    st.subheader("Recommendations")
    for r in recs:
        with st.expander(f"{r.building_id}: {r.finding}  ·  confidence {r.confidence}"):
            st.markdown(f"**Evidence (measured):** {r.evidence}")
            st.markdown(f"**Possible explanation (hypothesis):** {r.possible_explanation}")
            st.markdown(f"**Suggested action:** {r.suggested_action}")
            st.caption(f"Limitations: {r.limitations}")


def show_anomalies(items, limit=15):
    if not items:
        st.info("No anomalies or data-quality issues found in this window.")
        return
    real = [a for a in items if a.category == "anomaly"]
    dq = [a for a in items if a.category == "data_quality"]
    c1, c2 = st.columns(2)
    c1.metric("Consumption anomalies", len(real))
    c2.metric("Data-quality issues", len(dq))
    for title, group in (("Consumption anomalies", real), ("Data-quality issues (not necessarily real consumption)", dq)):
        if not group:
            continue
        st.markdown(f"**{title}**")
        for a in group[:limit]:
            dev = f" · {a.deviation_pct:+.1f}%" if a.deviation_pct is not None else ""
            obs = f"observed {a.observed:,.1f} {a.unit}" if a.observed is not None else ""
            exp = f" vs expected {a.expected:,.1f} {a.unit}" if a.expected is not None else ""
            st.markdown(f"{SEV_ICON[a.severity]} **{a.building_id}** · {a.kind.replace('_', ' ')} · {a.start}"
                        f"{' to ' + a.end if a.end != a.start else ''}{dev}  \n{obs}{exp}. {a.detail}")
            for w in a.warnings:
                st.caption(f"⚠️ {w}")
    with st.expander("All results as a table"):
        st.dataframe(AnomalyDetector.to_frame(items), width="stretch", hide_index=True)


def show_forecast(fc):
    if fc is None:
        return
    st.subheader(f"Forecast / expected consumption: {fc.target}")
    if fc.status == "insufficient_history":
        st.warning("Not enough history to produce a forecast.")
        return
    st.plotly_chart(forecast_figure(fc.history, fc.frame, f"{fc.target}: expected hourly electricity", fc.unit),
                    width="stretch")
    s = fc.summary
    c = st.columns(3)
    c[0].metric("Expected total", f"{s['expected_total_kwh']:,.0f} {fc.unit}")
    c[1].metric("80% interval", f"{s['interval_total_kwh'][0]:,.0f} – {s['interval_total_kwh'][1]:,.0f}")
    if s.get("actual_total_kwh") is not None:
        c[2].metric("Actual", f"{s['actual_total_kwh']:,.0f} {fc.unit}", f"{s['deviation_pct']:+.1f}% vs expected", delta_color="inverse")
    m = fc.metrics
    st.caption(f"Model: {fc.model} · chosen by chronological backtest (MAE/RMSE/MAPE): "
               + "; ".join(f"{k}: MAE {v['mae']}, RMSE {v['rmse']}, MAPE {v['mape']}%" for k, v in m.items() if isinstance(v, dict)))


def show_answer(cp: Copilot, ans):
    if ans.status == "clarify":
        st.info(f"❓ {ans.message}")
        return
    if ans.status == "unsupported":
        st.warning(f"ℹ️ {ans.message}")
        return
    if ans.status == "error":
        st.error(ans.message)
        return
    e = ans.explanation
    if e:
        v = e.get("verification", {})
        st.markdown(f"#### Answer\n{e['explanation']}")
        if v.get("verified"):
            st.success(f"✔ Numbers checked against the data ({v.get('numbers_checked', 0)} values).", icon=None)
        else:
            st.warning("⚠ Not verified: " + " ".join(v.get("issues", [])))
        if e.get("error"):
            st.caption(f"Language model unavailable ({e['error']}); showing a deterministic summary.")
    elif ans.message:
        st.info(ans.message)
    for w in ans.warnings:
        st.caption(f"⚠️ {w}")
    if ans.from_cache:
        st.caption("Served from cache.")

    if ans.data is not None:
        if ans.chart and ans.chart.kind == "metric":
            cols = st.columns(min(3, len(ans.data.columns)))
            for c, col in zip(cols, ans.data.columns):
                val = ans.data.iloc[0][col]
                c.metric(col.replace("_", " "), f"{val:,.2f}" if isinstance(val, (int, float)) else str(val))
        else:
            fig = render(ans.chart, ans.data) if ans.chart else None
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            elif ans.chart and ans.chart.notes:
                st.caption(" ".join(ans.chart.notes))
        show_table(cp, ans)
    show_forecast(ans.forecast)
    if ans.anomalies:
        st.subheader("Anomaly alerts")
        show_anomalies(ans.anomalies)
    show_recommendations(ans.recommendations)
    if ans.sql:
        with st.expander("Query details (validated, read-only)"):
            st.code(ans.sql, language="sql")
            st.caption("Validator notes: " + "; ".join(ans.validation_notes))
            for o in ans.optimization:
                st.caption(f"Rewrite {o['rule']}: {'kept' if o.get('keep') else 'rejected'} — {o['reason']}"
                           + (f" ({o['median_ms_before']} → {o['median_ms_after']} ms)" if "median_ms_before" in o else ""))


def show_table(cp: Copilot, ans):
    key = f"page::{ans.sql}"
    page = st.session_state.get(key, 0)
    res = cp.page(ans.sql, page)
    if not res.ok:
        st.error(res.error_text)
        return
    st.markdown(f"**Results** · page {page + 1}")
    st.dataframe(res.df, width="stretch", hide_index=True)
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("◀ Previous", disabled=page == 0, key=key + "prev"):
        st.session_state[key] = page - 1
        st.rerun()
    if c2.button("Next ▶", disabled=not res.has_next, key=key + "next"):
        st.session_state[key] = page + 1
        st.rerun()


def sidebar(cp: Copilot):
    o = cp.dataset_overview()
    st.sidebar.title("⚡ Energy Copilot")
    st.sidebar.markdown(f"**Campus:** BDG2 site *{o['site']}* · {len(o['buildings'])} buildings  \n"
                        f"**Coverage:** {o['start'][:10]} → {o['end'][:10]} (hourly)  \n**Unit:** {o['unit']} (electricity)")
    st.sidebar.caption("Historical dataset, not live monitoring. Timestamps are site-local time.")
    st.sidebar.caption(f"Database opened read-only: {o['read_only']} · dataset version {o['dataset_version']}")
    llm_on = cp.llm.available
    st.sidebar.caption(f"Language model: {'configured (' + config.LLM_MODEL + ')' if llm_on else 'not configured: example questions still work'}")
    s = cp.cache.stats()
    st.sidebar.caption(f"Cache: {s['hits']} hits / {s['misses']} misses (in-memory, resets on restart)")


def tab_ask(cp: Copilot):
    st.markdown("Ask a question about campus electricity use. Answers come from the database; numbers are checked.")
    cols = st.columns(4)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 4].button(q, key=f"ex{i}", width="stretch"):
            st.session_state["question"] = q
    with st.form("ask", clear_on_submit=False):
        q = st.text_input("Your question", key="question", placeholder="e.g. Which building had the highest weekend use in 2016?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted or (q and st.session_state.get("last_q") != q):
        st.session_state["last_q"] = q
        with st.spinner("Analysing..."):
            st.session_state["answer"] = cp.ask(q)
    ans = st.session_state.get("answer")
    if ans is not None:
        show_answer(cp, ans)


def tab_anomalies(cp: Copilot):
    b = cp.schema.buildings["building_id"].tolist()
    c1, c2, c3 = st.columns([2, 1, 1])
    sel = c1.multiselect("Buildings", b, default=[], placeholder="All buildings")
    lo, hi = pd.Timestamp(cp.schema.min_ts).date(), pd.Timestamp(cp.schema.max_ts).date()
    start = c2.date_input("From", max(lo, hi - pd.Timedelta(days=89)), min_value=lo, max_value=hi)
    end = c3.date_input("To", pd.Timestamp(cp.schema.max_ts).date(), min_value=pd.Timestamp(cp.schema.min_ts).date(),
                        max_value=pd.Timestamp(cp.schema.max_ts).date())
    if start > end:
        st.error("'From' must be before 'To'.")
        return
    with st.spinner("Scanning readings..."):
        items = cached_anomalies(cp, cp.schema.dataset_version, tuple(sel), f"{start} 00:00:00",
                                 f"{end + pd.Timedelta(days=1)} 00:00:00")
    show_anomalies(items, limit=25)


def tab_forecast(cp: Copilot):
    b = ["Campus total"] + cp.schema.buildings["building_id"].tolist()
    c1, c2, c3 = st.columns([2, 1, 1])
    target = c1.selectbox("Building", b)
    horizon = c2.radio("Horizon", ["24 hours", "7 days"], horizontal=True)
    mode = c3.radio("Start", ["After last reading", "Backtest a past date"], horizontal=False)
    as_of = None
    if mode != "After last reading":
        lo, hi = pd.Timestamp(cp.schema.min_ts).date(), pd.Timestamp(cp.schema.max_ts).date()
        as_of = st.date_input("Start date", min(max(lo, pd.Timestamp("2017-06-05").date()), hi), min_value=lo, max_value=hi)
    with st.spinner("Fitting models..."):
        fc = cached_forecast(cp, cp.schema.dataset_version, target, 24 if horizon == "24 hours" else 168,
                             str(as_of) if as_of else None)
    show_forecast(fc)
    for w in fc.warnings:
        st.caption(f"⚠️ {w}")


def tab_dataset(cp: Copilot):
    o = cp.dataset_overview()
    st.markdown(f"**Building selection:** {o['selection_rule']}.  \n**Note:** {o['note']}.")
    st.dataframe(o["buildings"], width="stretch", hide_index=True)
    with st.expander("Schema and semantic rules"):
        st.code(cp.schema.compact_schema())
        st.markdown("\n".join(f"- {r}" for r in cp.schema.semantic_rules))
        st.caption("Indexes: " + "; ".join(i for v in o["indexes"].values() for i in v))


def tab_upload():
    st.markdown("Upload a CSV to chart it and scan for anomalies. Columns are detected from the file "
                "(for anomaly scanning include a building/meter id, a timestamp and a numeric reading).")
    f = st.file_uploader("CSV file", type=["csv"])
    if not f:
        return
    try:
        df = pd.read_csv(f, nrows=200_000)
    except Exception as e:
        st.error(f"Could not read the CSV ({type(e).__name__}).")
        return
    if df.empty:
        st.warning("The file has no rows.")
        return
    q = st.text_input("What do you want to see?", "trend over time")
    spec = select_chart(q, df)
    fig = render(spec, df)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(" ".join(spec.notes) or "No suitable chart for this data.")
    st.dataframe(df.head(10), width="stretch", hide_index=True)
    cols = df.columns.tolist()
    with st.expander("Scan for anomalies"):
        c = st.columns(3)
        bid = c[0].selectbox("ID column", cols)
        ts = c[1].selectbox("Timestamp column", cols, index=min(1, len(cols) - 1))
        val = c[2].selectbox("Reading column", cols, index=min(2, len(cols) - 1))
        if st.button("Scan"):
            try:
                items = AnomalyDetector().detect(df.rename(columns={bid: "building_id", ts: "ts", val: "kwh"}))
                show_anomalies(items)
            except Exception as e:
                st.error(f"Could not scan these columns ({type(e).__name__}). Check that the timestamp column holds dates.")


def main():
    try:
        cp = get_copilot()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    sidebar(cp)
    st.title("Energy Analytics Copilot")
    tabs = st.tabs(["💬 Ask", "🚨 Anomalies", "📈 Forecast", "🏢 Dataset", "📤 Upload CSV"])
    with tabs[0]:
        tab_ask(cp)
    with tabs[1]:
        tab_anomalies(cp)
    with tabs[2]:
        tab_forecast(cp)
    with tabs[3]:
        tab_dataset(cp)
    with tabs[4]:
        tab_upload()


main()
