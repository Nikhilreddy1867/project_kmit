"""
streamlit_app.py
================
Local Streamlit dashboard for the AI Governance Platform (Phase 6).

Run (with the API already running on port 8000):

    streamlit run dashboard/streamlit_app.py

Data contract
-------------
**Every value displayed here is fetched over HTTP from the governance API.** This
module opens no files, imports nothing from ``src/`` or ``app/``, and performs no
metric arithmetic -- charts and tables render the numbers exactly as the API
served them. Where the interface needs prose (interpretation caveats, the
four-fifths framing, provenance, limitations), that text is also pulled from the
API rather than restated here, so the dashboard cannot drift from the audits.

The only client-side computation is chart layout: bar positions and axis ranges.
Notably the confusion matrix is displayed as **raw counts only** -- no row
normalisation, no derived percentages -- because deriving a rate here would be
recomputing a metric the audit already owns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run dashboard/streamlit_app.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import (  # noqa: E402
    DEFAULT_BASE_URL,
    ApiError,
    ApiMalformed,
    ApiNotFound,
    ApiServerError,
    ApiUnavailable,
    GovernanceApiClient,
)

PAGES = [
    "Overview",
    "Model Performance",
    "Fairness Audit",
    "Explainability",
    "Governance Decision & Risks",
    "Agent Review",
    "Model Registry",
]

st.set_page_config(
    page_title="AI Governance Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Chart styling
#
# Reuses the same validated categorical palette as the static Phase 2/3 figures,
# so a chart means the same thing wherever it appears. Slots 1-3 were checked
# with the palette validator: worst all-pairs CVD deltaE 9.2 (deutan),
# normal-vision 24.0, in both light and dark modes. Every bar carries a direct
# value label, which is also the contrast relief the aqua slot requires.
# --------------------------------------------------------------------------- #
C_SELECTION = "#2a78d6"   # slot 1, blue
C_TPR = "#eb6834"         # slot 2, orange
C_FPR = "#1baf7a"         # slot 3, aqua
C_POSITIVE = "#2a78d6"    # diverging pole: pushes toward >50K
C_NEGATIVE = "#e34948"    # diverging pole: pushes toward <=50K
C_THRESHOLD = "#d03b3b"   # status: critical (reference lines only)
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12)


def _style(fig: go.Figure, height: int = 420, x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply the shared chart chrome: recessive grid, no chart junk, one axis."""
    fig.update_layout(
        height=height,
        font=FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font=FONT),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=BASELINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=BASELINE)
    return fig


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a number for display without altering it. None -> 'n/a'."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------- #
# Cached API access
#
# Cached on (base_url, args) so switching pages does not re-hit the API. The
# sidebar "Refresh data" button clears the cache when an audit has been re-run.
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60, show_spinner=False)
def fetch_health(base_url: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).health()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models(base_url: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).models()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_performance(base_url: str, model: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).performance(model)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fairness(base_url: str, model: str, attribute: str | None) -> dict[str, Any]:
    return GovernanceApiClient(base_url).fairness(model, attribute=attribute)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_explainability(base_url: str, model: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).explainability(model)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_decision(base_url: str, include_markdown: bool = False) -> dict[str, Any]:
    return GovernanceApiClient(base_url).decision(include_markdown=include_markdown)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_risks(
    base_url: str, overall_risk: str | None, category: str | None, status: str | None
) -> dict[str, Any]:
    return GovernanceApiClient(base_url).risks(
        overall_risk=overall_risk, category=category, status=status
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_card(base_url: str, sections_only: bool = False) -> dict[str, Any]:
    return GovernanceApiClient(base_url).model_card(sections_only=sections_only)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_agents(base_url: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).agents()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_agent_review(base_url: str, model: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).agent_review(model)


# Registry fetches use a short TTL: integrity is a live check, so a stale cached
# "verified" would be actively misleading.
@st.cache_data(ttl=30, show_spinner=False)
def fetch_registry_runs(base_url: str, status: str | None = None) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_runs(status=status)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_registry_run(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_run(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_registry_integrity(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_integrity(run_id)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_registry_timeline(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_timeline(run_id)


# --------------------------------------------------------------------------- #
# Error rendering
# --------------------------------------------------------------------------- #
def show_api_error(exc: ApiError, context: str = "") -> None:
    """Render a typed API failure as a specific, actionable message."""
    prefix = f"**{context}** — " if context else ""
    if isinstance(exc, ApiUnavailable):
        st.error(f"{prefix}API unavailable. {exc.message}", icon="🔌")
    elif isinstance(exc, ApiNotFound):
        st.warning(f"{prefix}Not available. {exc.message}", icon="🚫")
        if exc.available:
            st.caption("Available instead: " + ", ".join(f"`{a}`" for a in exc.available))
    elif isinstance(exc, ApiServerError):
        st.error(
            f"{prefix}The API could not serve this (HTTP {exc.status_code}). {exc.message}",
            icon="⚠️",
        )
    elif isinstance(exc, ApiMalformed):
        st.error(f"{prefix}Unexpected response shape. {exc.message}", icon="🧩")
    else:  # pragma: no cover - defensive
        st.error(f"{prefix}{exc.message}", icon="⚠️")
    if exc.hint:
        st.caption(exc.hint)


def split_markdown_sections(markdown: str) -> dict[str, str]:
    """
    Split API-supplied Markdown into ``{section heading: body}`` on level-2 rules.

    Presentation only -- no content is altered, nothing is computed. Used to show
    model-card sections in expanders and to surface the provenance/limitations
    sections in the interface.
    """
    sections: dict[str, str] = {}
    title, buffer = None, []
    for line in (markdown or "").splitlines():
        if line.startswith("## "):
            if title:
                sections[title] = "\n".join(buffer).strip()
            title, buffer = line[3:].strip(), []
        elif title:
            buffer.append(line)
    if title:
        sections[title] = "\n".join(buffer).strip()
    return sections


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> tuple[str, str]:
    st.sidebar.title("⚖️ AI Governance Platform")
    st.sidebar.caption("Phase 6 dashboard · read-only view of the Adult Income audit")

    page = st.sidebar.radio("Page", PAGES, key="page")

    st.sidebar.divider()
    base_url = st.sidebar.text_input(
        "API base URL",
        value=DEFAULT_BASE_URL,
        key="api_base_url",
        help="Where the Phase 5 governance API is running.",
    ).strip() or DEFAULT_BASE_URL

    if st.sidebar.button("↻ Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    # Connection badge -- every page depends on the API, so surface it once here.
    try:
        health = fetch_health(base_url)
        if health.get("status") == "ok":
            st.sidebar.success(
                f"API connected · v{health.get('version', '?')}\n\n"
                f"{health.get('artifacts_present')}/{health.get('artifacts_expected')} "
                "audit artefacts present",
                icon="✅",
            )
        else:
            st.sidebar.warning(
                f"API degraded — {len(health.get('missing_artifacts') or [])} artefact(s) "
                "missing. Some pages will be incomplete.",
                icon="⚠️",
            )
    except ApiError as exc:
        st.sidebar.error("API not reachable", icon="🔌")
        st.sidebar.caption(exc.message)

    st.sidebar.divider()
    st.sidebar.caption(
        "All figures are served by the API from the committed audit artefacts. "
        "This dashboard reads no files and recalculates nothing."
    )
    return page, base_url


# --------------------------------------------------------------------------- #
# Shared blocks
# --------------------------------------------------------------------------- #
def render_decision_banner(decision: dict[str, Any]) -> None:
    """The governance decision, rendered from the API's own status fields."""
    research = decision.get("research_use")
    deployment = decision.get("real_world_deployment")

    left, right = st.columns(2)
    with left:
        if research == "conditionally_approved":
            st.success(
                "#### ✅ Conditionally approved for research/education only\n"
                "Subject to the conditions listed below.",
                icon="✅",
            )
        else:
            st.info(f"#### Research use: {research or 'undetermined'}")
    with right:
        if deployment == "blocked":
            st.error(
                "#### ⛔ Blocked from real-world deployment\n"
                "Not for any decision about a real person.",
                icon="⛔",
            )
        else:
            st.info(f"#### Deployment: {deployment or 'undetermined'}")

    if decision.get("disclaimer"):
        st.caption(f"ℹ️ {decision['disclaimer']}")


def render_caveats(items: list[str] | None, title: str, icon: str = "⚠️") -> None:
    """Render an API-supplied caveat list. Always expanded: these are not optional."""
    if not items:
        return
    with st.expander(f"{icon} {title}", expanded=True):
        for item in items:
            st.markdown(f"- {item}")


def render_provenance(base_url: str) -> None:
    """
    'Data provenance and limitations' — required in the interface, and sourced
    from the model card over the API rather than restated here.
    """
    st.divider()
    st.subheader("📚 Data provenance and limitations")
    try:
        card = fetch_model_card(base_url)
    except ApiError as exc:
        show_api_error(exc, "Model card")
        return

    sections = split_markdown_sections(card.get("content") or "")
    wanted = {
        "provenance": next((k for k in sections if "provenance" in k.lower()), None),
        "limitations": next(
            (k for k in sections if k.lower().startswith(("10.", "limitations"))
             and "limitation" in k.lower()), None
        ),
        "non_intended": next((k for k in sections if "non-intended" in k.lower()), None),
    }

    st.caption(
        "Dataset: UCI Adult / Census Income (id 2), a **1994 US Census** extract — "
        f"48,842 records. Source of this text: `{card.get('source')}` via the API."
    )
    tabs = st.tabs(["Provenance & 1994 context", "Limitations", "Non-intended uses"])
    for tab, key in zip(tabs, ("provenance", "limitations", "non_intended")):
        with tab:
            heading = wanted[key]
            if heading:
                st.markdown(sections[heading])
            else:
                st.info("This section is not present in the model card served by the API.")


# --------------------------------------------------------------------------- #
# Page 1 — Overview
# --------------------------------------------------------------------------- #
def page_overview(base_url: str) -> None:
    st.title("AI Governance Platform")
    st.markdown(
        "A local, end-to-end governance audit of a machine-learning baseline: "
        "**model performance → fairness → explainability → risk assessment**. "
        "The subject is an income classifier trained on the **UCI Adult / Census "
        "Income** dataset, a 1994 US Census extract, predicting whether a person's "
        "recorded annual income exceeds **\\$50,000**."
    )
    st.caption(
        "The model exists to be *audited*, not deployed. Its purpose here is to "
        "demonstrate a governance workflow end to end on a realistic, imperfect model."
    )

    # --- API health -------------------------------------------------------- #
    st.subheader("API health")
    try:
        health = fetch_health(base_url)
    except ApiError as exc:
        show_api_error(exc, "Health check")
        return

    cols = st.columns(4)
    cols[0].metric("Status", str(health.get("status", "?")).upper())
    cols[1].metric("Mode", str(health.get("mode", "?")))
    cols[2].metric(
        "Artefacts present",
        f"{health.get('artifacts_present')}/{health.get('artifacts_expected')}",
    )
    cols[3].metric("API version", str(health.get("version", "?")))

    missing = health.get("missing_artifacts") or []
    if missing:
        st.warning(
            "The API reports missing audit artefacts, so some pages will be "
            f"incomplete: {', '.join(f'`{m}`' for m in missing)}",
            icon="⚠️",
        )

    # --- governance decision (prominent) ----------------------------------- #
    st.divider()
    st.subheader("Governance decision")
    try:
        decision = fetch_decision(base_url)
        render_decision_banner(decision)
    except ApiError as exc:
        show_api_error(exc, "Governance decision")
        decision = None

    # --- primary model headline metrics ------------------------------------ #
    st.divider()
    st.subheader("Primary model — XGBoost")
    try:
        models = fetch_models(base_url)
    except ApiError as exc:
        show_api_error(exc, "Model list")
        return

    entries = models.get("models") or []
    primary = next((m for m in entries if m.get("is_primary")), None)
    if primary is None:
        st.warning("The API did not flag a primary model.", icon="⚠️")
        return

    st.caption(
        f"Selected baseline: **{primary.get('model_name')}** — best held-out "
        f"performance of {models.get('count')} evaluated models. "
        f"Test set: {_fmt(models.get('test_set_rows'))} held-out rows · "
        f"positive class `{models.get('positive_class')}` · "
        f"decision threshold {models.get('decision_threshold')}."
    )

    metrics = primary.get("metrics") or {}
    cols = st.columns(5)
    for col, key, label in zip(
        cols,
        ("accuracy", "precision", "recall", "f1", "roc_auc"),
        ("Accuracy", "Precision", "Recall / TPR", "F1", "ROC-AUC"),
    ):
        col.metric(label, _fmt(metrics.get(key)))

    st.info(
        "**Read accuracy with care.** The majority-class floor on this test set is "
        "0.7607 — a model predicting `<=50K` for everyone would score that. F1 and "
        "ROC-AUC are the metrics to judge on, and only ROC-AUC is independent of the "
        "0.5 decision threshold.",
        icon="ℹ️",
    )

    if decision:
        risk = decision.get("risk_profile") or {}
        counts = risk.get("counts_by_overall_risk") or {}
        if counts:
            st.divider()
            st.subheader("Risk profile")
            cols = st.columns(len(counts) + 1)
            cols[0].metric("Total risks", _fmt(risk.get("total_risks")))
            for col, (rating, count) in zip(cols[1:], counts.items()):
                col.metric(rating, _fmt(count))

    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Page 2 — Model Performance
# --------------------------------------------------------------------------- #
def page_performance(base_url: str) -> None:
    st.title("Model performance")
    st.caption("Held-out test-set metrics as recorded by the Phase 1 audit.")

    try:
        models = fetch_models(base_url)
    except ApiError as exc:
        show_api_error(exc, "Model list")
        return

    names = [str(m.get("model_name")) for m in models.get("models") or []]
    if not names:
        st.warning("The API returned no evaluated models.", icon="⚠️")
        return

    model = st.selectbox("Model", names, key="perf_model")

    try:
        perf = fetch_performance(base_url, model)
    except ApiError as exc:
        show_api_error(exc, f"Performance for {model}")
        return

    if perf.get("is_primary"):
        st.success("This is the selected primary baseline.", icon="⭐")

    st.caption(
        f"Test rows: {_fmt(perf.get('n_test'))} · positive class "
        f"`{perf.get('positive_class')}` · threshold {perf.get('decision_threshold')} · "
        f"source `{perf.get('source')}`"
    )

    metrics = perf.get("metrics") or {}
    cols = st.columns(5)
    for col, key, label in zip(
        cols,
        ("accuracy", "precision", "recall", "f1", "roc_auc"),
        ("Accuracy", "Precision", "Recall / TPR", "F1", "ROC-AUC"),
    ):
        col.metric(label, _fmt(metrics.get(key)))

    left, right = st.columns([1, 1])

    # --- confusion matrix: RAW COUNTS ONLY --------------------------------- #
    with left:
        st.subheader("Confusion matrix")
        cm = perf.get("confusion_matrix") or {}
        tn, fp = cm.get("true_negatives"), cm.get("false_positives")
        fn, tp = cm.get("false_negatives"), cm.get("true_positives")

        grid = [[tn or 0, fp or 0], [fn or 0, tp or 0]]
        labels = [[_fmt(tn), _fmt(fp)], [_fmt(fn), _fmt(tp)]]
        fig = go.Figure(
            go.Heatmap(
                z=grid,
                x=["Predicted <=50K", "Predicted >50K"],
                y=["Actual <=50K", "Actual >50K"],
                text=labels,
                texttemplate="%{text}",
                textfont=dict(size=16),
                colorscale=[[0, "#eaf2fd"], [1, C_SELECTION]],
                showscale=False,
                hovertemplate="%{y} · %{x}<br>count = %{text}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(_style(fig, height=330), width="stretch")
        st.caption(
            "Raw counts exactly as served by the API. No percentages are derived "
            "here — normalising would mean recomputing a metric the audit owns."
        )

    # --- error pattern ------------------------------------------------------ #
    with right:
        st.subheader("Error pattern")
        err = perf.get("error_analysis") or {}
        c1, c2 = st.columns(2)
        c1.metric("False negatives", _fmt(err.get("false_negatives")),
                  help="True high earners the model missed.")
        c2.metric("False positives", _fmt(err.get("false_positives")),
                  help="Low earners wrongly flagged as >50K.")

        fig = go.Figure(
            go.Bar(
                x=[err.get("false_negatives") or 0, err.get("false_positives") or 0],
                y=["False negatives", "False positives"],
                orientation="h",
                marker_color=[C_NEGATIVE, C_TPR],
                text=[_fmt(err.get("false_negatives")), _fmt(err.get("false_positives"))],
                textposition="outside",
                hovertemplate="%{y}: %{x:,}<extra></extra>",
            )
        )
        st.plotly_chart(_style(fig, height=210, x_title="Count"), width="stretch")

        for note in err.get("notes") or []:
            st.markdown(f"- {note}")

    render_caveats(perf.get("caveats"), "Interpretation caveats (supplied by the API)")
    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Page 3 — Fairness Audit
# --------------------------------------------------------------------------- #
def page_fairness(base_url: str) -> None:
    st.title("Fairness audit")
    st.caption(
        "Phase 2 group metrics, disaggregated by sensitive attribute. Every value "
        "is served by the API from the committed fairness audit."
    )

    try:
        models = fetch_models(base_url)
    except ApiError as exc:
        show_api_error(exc, "Model list")
        return

    audited = [
        str(m.get("model_name"))
        for m in models.get("models") or []
        if m.get("has_fairness_audit")
    ]
    if not audited:
        st.warning("No model has a fairness audit available from the API.", icon="⚠️")
        return

    col_a, col_b = st.columns(2)
    model = col_a.selectbox("Model", audited, key="fair_model")

    try:
        full = fetch_fairness(base_url, model, None)
    except ApiError as exc:
        show_api_error(exc, f"Fairness for {model}")
        return

    attributes = full.get("sensitive_attributes") or []
    if not attributes:
        st.warning("The API returned no sensitive attributes for this model.", icon="⚠️")
        return
    attribute = col_b.selectbox("Sensitive attribute", attributes, key="fair_attribute")

    try:
        data = fetch_fairness(base_url, model, attribute)
    except ApiError as exc:
        show_api_error(exc, f"Fairness for {model} / {attribute}")
        return

    groups = data.get("groups") or []
    summary = data.get("summary") or []
    if not groups:
        st.warning("No group rows were returned for this selection.", icon="⚠️")
        return

    # --- headline disparity ------------------------------------------------- #
    if summary:
        s = summary[0]
        st.subheader(f"Headline disparity — {attribute}")
        cols = st.columns(4)
        cols[0].metric("Reference group", str(s.get("reference_group")),
                       help="The largest group by sample count.")
        cols[1].metric("Lowest disparate-impact ratio",
                       _fmt(s.get("disparate_impact_ratio_min")),
                       help=f"Worst group: {s.get('disparate_impact_ratio_worst_group')}")
        cols[2].metric("Equal-opportunity difference",
                       _fmt(s.get("equal_opportunity_difference_max_abs")),
                       help=f"Worst group: {s.get('equal_opportunity_worst_group')}")
        cols[3].metric("Groups below 0.80 screen",
                       _fmt(s.get("groups_failing_four_fifths")))

    st.info(f"**Reference group rule.** {data.get('reference_group_rule')}", icon="📐")

    # --- rates chart -------------------------------------------------------- #
    st.subheader("Selection rate, TPR and FPR by group")
    names = [str(g.get("group")) for g in groups]
    tick_labels = [
        f"{g.get('group')}{' (ref)' if g.get('is_reference') else ''}"
        f"<br>n = {_fmt(g.get('n_samples'))}"
        f"{'<br>⚠ small n' if g.get('small_group_flag') else ''}"
        for g in groups
    ]
    fig = go.Figure()
    for label, key, colour in (
        ("Selection rate", "selection_rate", C_SELECTION),
        ("TPR (recall)", "tpr", C_TPR),
        ("FPR", "fpr", C_FPR),
    ):
        values = [g.get(key) for g in groups]
        fig.add_bar(
            name=label,
            x=tick_labels,
            y=[v if v is not None else 0 for v in values],
            marker_color=colour,
            text=[_fmt(v, 3) if v is not None else "n/a" for v in values],
            textposition="outside",
            textfont=dict(size=10, color=INK_SECONDARY),
            hovertemplate="%{x}<br>" + label + " = %{y:.4f}<extra></extra>",
        )
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
    st.plotly_chart(_style(fig, height=460, y_title="Rate"), width="stretch")
    st.caption(
        "All three series are rates on the same scale, so they share one axis. "
        f"Measured at the {data.get('decision_threshold')} decision threshold."
    )

    # --- disparate impact + four-fifths context ---------------------------- #
    st.subheader("Disparate-impact ratio")
    ratios = [g.get("disparate_impact_ratio") for g in groups]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=[r if r is not None else 0 for r in ratios],
            marker_color=[
                C_THRESHOLD if (r is not None and r < 0.80) else C_SELECTION for r in ratios
            ],
            text=[_fmt(r, 3) if r is not None else "n/a" for r in ratios],
            textposition="outside",
            hovertemplate="%{x}<br>ratio = %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.80,
        line=dict(color=C_THRESHOLD, width=1.6, dash="dash"),
        annotation_text="0.80 screening threshold",
        annotation_position="top right",
    )
    fig.add_hline(y=1.0, line=dict(color=BASELINE, width=1.2))
    st.plotly_chart(
        _style(fig, height=400, y_title="Selection rate ÷ reference (1.0 = parity)"),
        width="stretch",
    )

    st.warning(
        "**The 0.80 line is a screening indicator, not a legal conclusion.** The "
        "four-fifths rule is a convention from US employment-law practice used to "
        "decide *where to look further*. It is not a statistical test, it does not "
        "establish unlawful discrimination, and nothing on this page is a legal "
        "determination about this model.",
        icon="⚖️",
    )

    # --- group table -------------------------------------------------------- #
    st.subheader("Group metrics")
    st.dataframe(groups, width="stretch", hide_index=True)

    flagged = [str(g.get("group")) for g in groups if g.get("small_group_flag")]
    if flagged:
        st.warning(
            f"**Small-group uncertainty.** {', '.join(flagged)} "
            f"{'has' if len(flagged) == 1 else 'have'} fewer than 200 test records. "
            "Their rates carry wide confidence intervals — see the "
            "`selection_rate_ci95` and `tpr_ci95` columns, which are 95% Wilson "
            "half-widths. **Do not rank these groups on their point estimates.**",
            icon="📉",
        )

    interpretation = data.get("interpretation") or {}
    col_l, col_r = st.columns(2)
    with col_l:
        render_caveats(
            interpretation.get("establishes"), "What these metrics establish", icon="✅"
        )
    with col_r:
        render_caveats(
            interpretation.get("does_not_establish"),
            "What these metrics do NOT establish",
            icon="🚫",
        )

    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Page 4 — Explainability
# --------------------------------------------------------------------------- #
def page_explainability(base_url: str) -> None:
    st.title("Explainability")
    st.caption(
        "Phase 3 permutation importance over the original human-readable features, "
        "plus exact TreeSHAP explanations for individual held-out cases."
    )

    try:
        models = fetch_models(base_url)
    except ApiError as exc:
        show_api_error(exc, "Model list")
        return

    entries = models.get("models") or []
    explained = [
        str(m.get("model_name")) for m in entries if m.get("has_explainability_audit")
    ]
    unexplained = [
        str(m.get("model_name")) for m in entries if not m.get("has_explainability_audit")
    ]

    default = "xgboost" if "xgboost" in explained else (explained[0] if explained else None)
    options = explained + unexplained
    if not options:
        st.warning("The API returned no models.", icon="⚠️")
        return

    model = st.selectbox(
        "Model",
        options,
        index=options.index(default) if default in options else 0,
        key="explain_model",
        help="Models without a Phase 3 audit are listed but have no data.",
    )

    if model in unexplained:
        st.info(
            f"**Explainability is not available for `{model}`.** The Phase 3 audit "
            f"covered {', '.join(f'`{m}`' for m in explained)} only. Rather than "
            "estimate or fabricate importances for an unaudited model, the API "
            "returns 404 and this dashboard reports it as unavailable.",
            icon="🚫",
        )
        render_provenance(base_url)
        return

    try:
        data = fetch_explainability(base_url, model)
    except ApiError as exc:
        show_api_error(exc, f"Explainability for {model}")
        render_provenance(base_url)
        return

    method = data.get("method") or {}
    with st.expander("Method", expanded=False):
        for key, text in method.items():
            st.markdown(f"**{key.capitalize()}** — {text}")

    importance = data.get("global_importance") or []
    if not importance:
        st.warning("No importance rows were returned.", icon="⚠️")
        return

    # --- global importance chart ------------------------------------------- #
    st.subheader(f"Global feature importance — {model}")
    ordered = list(reversed(importance))  # highest at top in a horizontal bar
    colour_for = {
        "protected": C_THRESHOLD,
        "likely_proxy": C_TPR,
        "other": C_SELECTION,
    }
    fig = go.Figure(
        go.Bar(
            x=[f.get("importance_mean") or 0 for f in ordered],
            y=[str(f.get("feature")) for f in ordered],
            orientation="h",
            marker_color=[colour_for.get(str(f.get("classification")), C_SELECTION)
                          for f in ordered],
            error_x=dict(
                type="data",
                array=[f.get("importance_std") or 0 for f in ordered],
                color=INK_SECONDARY,
                thickness=1,
                width=3,
            ),
            text=[_fmt(f.get("importance_mean")) for f in ordered],
            textposition="outside",
            textfont=dict(size=10, color=INK_SECONDARY),
            customdata=[str(f.get("classification")) for f in ordered],
            hovertemplate="%{y}<br>importance = %{x:.5f}<br>class = %{customdata}<extra></extra>",
        )
    )
    st.plotly_chart(
        _style(fig, height=560, x_title=f"Drop in held-out {data.get('scorer', 'roc_auc').upper()} when shuffled"),
        width="stretch",
    )
    st.caption(
        "🔴 protected attribute · 🟠 likely proxy · 🔵 other. "
        "Error bars are ±1 std over permutation repeats: where the bar is comparable "
        "to the value, the feature is not distinguishable from unimportant."
    )

    # --- proxy assessment -------------------------------------------------- #
    proxy = data.get("proxy_assessment") or {}
    st.subheader("Proxy-feature assessment")
    ranks = proxy.get("protected_attribute_ranks") or {}
    above = proxy.get("proxies_ranked_above_all_protected_attributes") or {}
    cols = st.columns(max(len(ranks), 1) + 1)
    for col, (name, rank) in zip(cols, ranks.items()):
        col.metric(f"`{name}` rank", f"{rank} / {data.get('n_features')}")
    if above:
        cols[-1].metric("Proxies ranked above both", _fmt(len(above)))

    if proxy.get("finding"):
        st.warning(f"**Finding.** {proxy['finding']}", icon="🔍")
    if proxy.get("implication"):
        st.error(f"**Governance implication.** {proxy['implication']}", icon="🚨")
    if above:
        st.caption(
            "Proxies outranking both protected attributes: "
            + ", ".join(f"`{k}` (#{v})" for k, v in sorted(above.items(), key=lambda kv: kv[1]))
        )

    # --- ranking table ----------------------------------------------------- #
    st.subheader("Feature ranking")
    st.dataframe(importance, width="stretch", hide_index=True)

    # --- local explanations ------------------------------------------------ #
    st.subheader("Local explanations (TreeSHAP)")
    local = data.get("local_explanations") or []
    if not local:
        st.info(
            f"Local case explanations are published for the primary model only. "
            f"`{model}` is a comparison model, so the API returns none.",
            icon="ℹ️",
        )
    else:
        case_ids = [str(c.get("case_id")) for c in local]
        case_id = st.selectbox("Case", case_ids, key="explain_case")
        case = next(c for c in local if str(c.get("case_id")) == case_id)

        cols = st.columns(4)
        cols[0].metric("Actual", ">50K" if case.get("actual_income") == 1 else "<=50K")
        cols[1].metric("Predicted", ">50K" if case.get("predicted_income") == 1 else "<=50K")
        cols[2].metric("P(>50K)", _fmt(case.get("predicted_probability")))
        cols[3].metric("Base log-odds", _fmt(case.get("base_log_odds")))

        factors = case.get("top_factors") or []
        if factors:
            ordered = list(reversed(factors))
            fig = go.Figure(
                go.Bar(
                    x=[f.get("shap_log_odds") or 0 for f in ordered],
                    y=[f"{f.get('feature')} = {f.get('feature_value')}" for f in ordered],
                    orientation="h",
                    marker_color=[
                        C_POSITIVE if (f.get("shap_log_odds") or 0) > 0 else C_NEGATIVE
                        for f in ordered
                    ],
                    text=[f"{f.get('shap_log_odds'):+.4f}" if f.get("shap_log_odds")
                          is not None else "n/a" for f in ordered],
                    textposition="outside",
                    hovertemplate="%{y}<br>SHAP = %{x:+.4f} log-odds<extra></extra>",
                )
            )
            fig.add_vline(x=0, line=dict(color=BASELINE, width=1.2))
            st.plotly_chart(
                _style(fig, height=340, x_title="SHAP contribution (log-odds)"),
                width="stretch",
            )
            st.caption(
                "🔵 pushes toward `>50K` · 🔴 pushes toward `<=50K`. Contributions are "
                "**log-odds, not percentage points**. Case identifiers are synthetic — "
                "dataset row indices are deliberately withheld."
            )
            st.dataframe(factors, width="stretch", hide_index=True)

    render_caveats(data.get("caveats"), "Association is not causation — API caveats", icon="🚫")
    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Page 5 — Governance Decision & Risks
# --------------------------------------------------------------------------- #
def page_governance(base_url: str) -> None:
    st.title("Governance decision & risks")

    try:
        decision = fetch_decision(base_url)
    except ApiError as exc:
        show_api_error(exc, "Governance decision")
        return

    st.caption(
        f"Subject: {decision.get('subject')} · decision date "
        f"{decision.get('decision_date')}"
    )
    render_decision_banner(decision)

    left, right = st.columns(2)
    with left:
        st.subheader("Grounds for the deployment block")
        grounds = decision.get("grounds_for_deployment_block") or []
        if grounds:
            for i, ground in enumerate(grounds, start=1):
                st.markdown(f"{i}. {ground}")
        else:
            st.info("The API returned no parsed grounds.", icon="ℹ️")
        if decision.get("blocking_risk_ids"):
            st.caption(
                "Blocking risks: "
                + ", ".join(f"`{r}`" for r in decision["blocking_risk_ids"])
            )
    with right:
        st.subheader("Conditions on research use")
        conditions = decision.get("conditions_on_research_use") or []
        if conditions:
            for condition in conditions:
                st.markdown(f"- {condition}")
        else:
            st.info("The API returned no parsed conditions.", icon="ℹ️")

    if decision.get("revisit_requirements"):
        with st.expander("What would be required to revisit the decision", expanded=False):
            st.markdown(decision["revisit_requirements"])

    # --- severity counts --------------------------------------------------- #
    st.divider()
    st.subheader("Risk severity")
    risk_profile = decision.get("risk_profile") or {}
    counts = risk_profile.get("counts_by_overall_risk") or {}
    order = ["Critical", "High", "Medium", "Low"]
    ordered = {k: counts[k] for k in order if k in counts}
    ordered.update({k: v for k, v in counts.items() if k not in order})

    if ordered:
        cols = st.columns(len(ordered) + 1)
        cols[0].metric("Total", _fmt(risk_profile.get("total_risks")))
        for col, (rating, count) in zip(cols[1:], ordered.items()):
            col.metric(rating, _fmt(count))

        severity_colour = {
            "Critical": C_THRESHOLD,
            "High": C_TPR,
            "Medium": "#eda100",
            "Low": C_FPR,
        }
        fig = go.Figure(
            go.Bar(
                x=list(ordered.keys()),
                y=list(ordered.values()),
                marker_color=[severity_colour.get(k, C_SELECTION) for k in ordered],
                text=[str(v) for v in ordered.values()],
                textposition="outside",
                hovertemplate="%{x}: %{y} risk(s)<extra></extra>",
            )
        )
        st.plotly_chart(_style(fig, height=320, y_title="Risks"), width="stretch")

    if risk_profile.get("assessment_framing"):
        st.info(f"**Assessment framing.** {risk_profile['assessment_framing']}", icon="📐")

    # --- risk register with filters ---------------------------------------- #
    st.divider()
    st.subheader("Risk register")

    col_a, col_b, col_c = st.columns(3)
    severity = col_a.selectbox(
        "Severity", ["All", *ordered.keys()], key="risk_severity",
        help="Filtered server-side by the API.",
    )
    category = col_b.text_input(
        "Category contains", key="risk_category", placeholder="e.g. Fairness"
    ).strip()
    status = col_c.text_input(
        "Status contains", key="risk_status", placeholder="e.g. blocking"
    ).strip()

    try:
        register = fetch_risks(
            base_url,
            None if severity == "All" else severity,
            category or None,
            status or None,
        )
    except ApiError as exc:
        show_api_error(exc, "Risk register")
        return

    st.caption(
        f"Showing **{register.get('count')}** of {register.get('total_in_register')} "
        f"risks · source `{register.get('source')}`"
    )
    risks = register.get("risks") or []
    if not risks:
        st.info("No risks match these filters.", icon="🔍")
    else:
        st.dataframe(risks, width="stretch", hide_index=True)
        for risk in risks:
            label = (
                f"{risk.get('risk_id')} · {risk.get('overall_risk')} · "
                f"{risk.get('category')}"
            )
            with st.expander(label, expanded=False):
                st.markdown(f"**Risk.** {risk.get('risk_statement')}")
                st.markdown(f"**Evidence.** {risk.get('evidence')}")
                st.markdown(f"**Affected groups.** {risk.get('affected_groups')}")
                cols = st.columns(3)
                cols[0].metric("Likelihood", str(risk.get("likelihood")))
                cols[1].metric("Impact", str(risk.get("impact")))
                cols[2].metric("Overall", str(risk.get("overall_risk")))
                st.markdown(f"**Recommended control.** {risk.get('recommended_control')}")
                st.markdown(f"**Residual risk.** {risk.get('residual_risk')}")
                st.caption(f"Owner: {risk.get('owner')} · Status: {risk.get('status')}")

    with st.expander("Rating scale", expanded=False):
        for key, text in (register.get("rating_scale") or {}).items():
            st.markdown(f"- **{key}** — {text}")

    # --- model card -------------------------------------------------------- #
    st.divider()
    st.subheader("Model card")
    try:
        card = fetch_model_card(base_url)
    except ApiError as exc:
        show_api_error(exc, "Model card")
        return

    st.caption(
        f"{len(card.get('sections') or [])} sections · "
        f"{_fmt(card.get('character_count'))} characters · source `{card.get('source')}`"
    )
    sections = split_markdown_sections(card.get("content") or "")
    if not sections:
        st.info("The API returned no model-card sections.", icon="ℹ️")
    else:
        for title, body in sections.items():
            with st.expander(title, expanded=False):
                st.markdown(body)

    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Page 6 — Agent Review
# --------------------------------------------------------------------------- #
SEVERITY_STYLE = {
    "critical": ("🔴", C_THRESHOLD),
    "high": ("🟠", C_TPR),
    "medium": ("🟡", "#eda100"),
    "low": ("🔵", C_SELECTION),
    "info": ("⚪", INK_SECONDARY),
}


def page_agent_review(base_url: str) -> None:
    st.title("Agent review")

    # The non-autonomy label must be impossible to miss, above everything else.
    st.info(
        "**These are deterministic governance agents, not autonomous "
        "decision-makers.** They are rule-based: they read the existing audit "
        "evidence, quote it verbatim, and classify it against fixed documented "
        "thresholds. They do not train models, recalculate metrics, exercise "
        "judgement, or make or change any governance decision. Severity labels are a "
        "triage convention, not measurements.",
        icon="🤖",
    )

    try:
        models = fetch_models(base_url)
    except ApiError as exc:
        show_api_error(exc, "Model list")
        return

    names = [str(m.get("model_name")) for m in models.get("models") or []]
    if not names:
        st.warning("The API returned no evaluated models.", icon="⚠️")
        return

    default = "xgboost" if "xgboost" in names else names[0]
    model = st.selectbox(
        "Model to review", names, index=names.index(default), key="agent_model"
    )

    # --- the agent roster -------------------------------------------------- #
    try:
        roster = fetch_agents(base_url)
        with st.expander(
            f"The {roster.get('count')} agents — what each reads and must never do",
            expanded=False,
        ):
            st.caption(roster.get("determinism", ""))
            for agent in roster.get("agents") or []:
                st.markdown(f"**`{agent.get('agent_name')}`** — {agent.get('agent_role')}")
                st.caption("Reads: " + ", ".join(f"`{r}`" for r in agent.get("reads") or []))
                for constraint in agent.get("constraints") or []:
                    st.markdown(f"  - 🚫 {constraint}")
                st.divider()
    except ApiError as exc:
        show_api_error(exc, "Agent roster")

    try:
        review = fetch_agent_review(base_url, model)
    except ApiError as exc:
        show_api_error(exc, f"Agent review for {model}")
        render_provenance(base_url)
        return

    # --- orchestrated recommendation (the preserved decision) -------------- #
    st.subheader("Orchestrated recommendation")
    preserved = review.get("preserved_decision") or {}
    render_decision_banner(
        {
            "research_use": preserved.get("research_use"),
            "real_world_deployment": preserved.get("real_world_deployment"),
            "disclaimer": review.get("disclaimer"),
        }
    )
    st.success(f"**{review.get('overall_recommendation')}**", icon="📌")
    st.caption(
        f"↳ {preserved.get('note')} Source: `{preserved.get('source')}`. "
        "The agents did not generate this recommendation and cannot alter it."
    )

    # --- severity summary -------------------------------------------------- #
    st.divider()
    st.subheader("Findings by severity")
    counts: dict[str, int] = review.get("severity_counts") or {}
    ordered = {k: counts.get(k, 0) for k in ("critical", "high", "medium", "low", "info")}

    cols = st.columns(len(ordered) + 2)
    cols[0].metric("Findings", _fmt(review.get("findings_total")))
    cols[1].metric("Highest", str(review.get("highest_severity", "")).upper())
    for col, (level, count) in zip(cols[2:], ordered.items()):
        icon, _ = SEVERITY_STYLE[level]
        col.metric(f"{icon} {level.capitalize()}", _fmt(count))

    fig = go.Figure(
        go.Bar(
            x=list(ordered.keys()),
            y=list(ordered.values()),
            marker_color=[SEVERITY_STYLE[k][1] for k in ordered],
            text=[str(v) for v in ordered.values()],
            textposition="outside",
            hovertemplate="%{x}: %{y} finding(s)<extra></extra>",
        )
    )
    st.plotly_chart(_style(fig, height=300, y_title="Findings"), width="stretch")
    st.caption(
        "Counts of findings per triage label. Severity is not a measurement and the "
        "labels are not combined into a score."
    )

    if review.get("unavailable_evidence"):
        st.warning(
            "**Evidence unavailable for this model:** "
            + "; ".join(review["unavailable_evidence"])
            + ". The agents reported this explicitly rather than estimating anything.",
            icon="🚫",
        )

    for note in review.get("consensus_notes") or []:
        st.caption(f"• {note}")

    # --- the four agent reports -------------------------------------------- #
    st.divider()
    st.subheader("Agent findings")
    reports = review.get("agents") or []
    tabs = st.tabs([f"{r.get('agent_name', '?').capitalize()}" for r in reports])

    for tab, report in zip(tabs, reports):
        with tab:
            status = str(report.get("status"))
            if status == "unavailable":
                st.warning(f"Status: **{status}** — {report.get('summary')}", icon="🚫")
            elif status == "ok":
                st.success(f"Status: **{status}**", icon="✅")
            else:
                st.error(f"Status: **{status}**", icon="⚠️")

            st.markdown(f"**Role.** {report.get('agent_role')}")
            st.markdown(f"**Summary.** {report.get('summary')}")
            st.caption(
                f"Agent type: `{report.get('agent_type')}` · evidence: "
                + ", ".join(f"`{s}`" for s in report.get("evidence_sources") or [])
            )

            for finding in report.get("findings") or []:
                level = str(finding.get("severity", "info")).lower()
                icon, _ = SEVERITY_STYLE.get(level, ("⚪", INK_SECONDARY))
                label = (
                    f"{icon} {finding.get('finding_id')} · {level.upper()} · "
                    f"{str(finding.get('finding', ''))[:90]}..."
                )
                with st.expander(label, expanded=(level in ("critical", "high"))):
                    st.markdown(f"**Finding.** {finding.get('finding')}")
                    st.markdown(
                        f"**Recommended action.** {finding.get('recommended_action')}"
                    )
                    st.caption(f"Evidence source: `{finding.get('evidence_source')}`")

                    st.markdown("**Limitations / caveats**")
                    for limitation in finding.get("limitations") or []:
                        if limitation:
                            st.markdown(f"- {limitation}")

                    with st.expander("Evidence quoted from the API", expanded=False):
                        st.caption(
                            "Values below are quoted verbatim from the evidence source. "
                            "The agent performs no arithmetic on them."
                        )
                        st.json(finding.get("evidence") or {})

            if report.get("caveats"):
                with st.expander("Caveats supplied by the source API", expanded=False):
                    for caveat in report["caveats"]:
                        if caveat:
                            st.markdown(f"- {caveat}")

    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Page 7 — Model Registry
# --------------------------------------------------------------------------- #
INTEGRITY_STYLE = {
    "verified": ("✅", "success"),
    "incomplete": ("⚠️", "warning"),
    "modified": ("🔴", "error"),
    "modified_and_incomplete": ("🔴", "error"),
}

RUN_STATUS_STYLE = {"active": "🟢", "superseded": "🟡", "archived": "⚪"}


def page_model_registry(base_url: str) -> None:
    st.title("Model registry")
    st.caption(
        "Each governance review run is recorded with SHA-256 checksums of every "
        "referenced artefact, so the evidence a conclusion rests on can be verified "
        "later."
    )
    st.info(
        "**The registry records evidence; it does not make decisions.** The governance "
        "decision shown on a run is a copy of the committed decision record.",
        icon="📒",
    )

    # --- run list ---------------------------------------------------------- #
    try:
        listing = fetch_registry_runs(base_url)
    except ApiServerError as exc:
        # A 503 here means the registry has not been created yet -- an actionable
        # setup step, so show the exact command rather than a generic error.
        if exc.status_code == 503:
            st.warning(
                "**The registry has not been created yet.** Create it from the "
                "existing evidence (this reads the artefacts and writes only to "
                "`runtime/`):",
                icon="📒",
            )
            st.code(
                "cd C:\\Users\\nreddy\\Downloads\\project_KMIT\n"
                ".\\.venv\\Scripts\\Activate.ps1\n"
                "python -m app.registry.cli register",
                language="powershell",
            )
            st.caption(f"API said: {exc.message}")
            render_provenance(base_url)
            return
        show_api_error(exc, "Registry")
        return
    except ApiError as exc:
        show_api_error(exc, "Registry")
        return

    runs = listing.get("runs") or []
    if not runs:
        st.warning(
            "The registry exists but holds no runs. Run "
            "`python -m app.registry.cli register`.",
            icon="📭",
        )
        render_provenance(base_url)
        return

    st.subheader("Audit runs")
    cols = st.columns(3)
    cols[0].metric("Registered runs", _fmt(listing.get("count")))
    cols[1].metric("Active run", listing.get("active_run_id") or "none")
    cols[2].metric("Database", str(listing.get("database")))
    st.dataframe(runs, width="stretch", hide_index=True)

    labels = [
        f"{RUN_STATUS_STYLE.get(str(r.get('status')), '•')} {r.get('run_id')} "
        f"({r.get('status')}) · {r.get('created_at')}"
        for r in runs
    ]
    choice = st.selectbox("Selected run", labels, index=0, key="registry_run")
    run_id = str(runs[labels.index(choice)].get("run_id"))

    # --- run metadata ------------------------------------------------------- #
    try:
        run = fetch_registry_run(base_url, run_id)
    except ApiError as exc:
        show_api_error(exc, f"Run {run_id}")
        return

    st.divider()
    st.subheader("Run metadata")
    cols = st.columns(4)
    cols[0].metric("Status", f"{RUN_STATUS_STYLE.get(str(run.get('status')), '')} "
                             f"{run.get('status')}")
    cols[1].metric("Artefacts", _fmt(run.get("artifact_count")))
    cols[2].metric("Refreshes", _fmt(run.get("refresh_count")))
    cols[3].metric("Schema", f"v{run.get('schema_version')}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Dataset**")
        st.markdown(f"- Name: `{run.get('dataset_name')}`")
        st.markdown(f"- Version: `{run.get('dataset_version')}`")
        st.caption(run.get("dataset_context", ""))
    with right:
        st.markdown("**Model**")
        st.markdown(f"- Name: `{run.get('model_name')}`")
        st.markdown(f"- Version: `{run.get('model_version')}`")
        st.caption(f"Run identifier: {run.get('model_run_identifier')}")

    st.markdown("**Registry identity**")
    st.markdown(f"- Run id: `{run.get('run_id')}` (content-addressed)")
    st.markdown(f"- Evidence digest: `{run.get('evidence_digest')}`")
    st.caption(
        f"Created {run.get('created_at')} · last refreshed {run.get('refreshed_at')}. "
        "The run id is derived from the evidence digest, so re-registering unchanged "
        "evidence refreshes this same run instead of creating a duplicate."
    )

    # --- recorded decision -------------------------------------------------- #
    st.divider()
    st.subheader("Governance decision (as recorded)")
    decision = run.get("governance_decision") or {}
    render_decision_banner(
        {
            "research_use": decision.get("research_use"),
            "real_world_deployment": decision.get("real_world_deployment"),
            "disclaimer": decision.get("note"),
        }
    )
    st.markdown(f"**{decision.get('headline', '')}**")
    st.caption(
        f"Recorded {decision.get('decision_date')} · source `{decision.get('source')}`"
    )
    if run.get("blocking_risk_ids"):
        st.caption(
            "Blocking risks recorded on this run: "
            + ", ".join(f"`{r}`" for r in run["blocking_risk_ids"])
        )

    # --- evidence coverage -------------------------------------------------- #
    st.divider()
    st.subheader("Evidence coverage")
    coverage = run.get("audit_coverage") or {}
    phases = ("performance", "fairness", "explainability", "governance", "agents")
    cols = st.columns(len(phases) + 1)
    for col, phase in zip(cols, phases):
        covered = bool(coverage.get(phase))
        col.metric(phase.capitalize(), "✅ yes" if covered else "❌ no")
    cols[-1].metric("Complete", "✅ yes" if coverage.get("complete") else "❌ no")

    detail_l, detail_r = st.columns(2)
    with detail_l:
        st.caption(
            "Models evaluated: "
            + ", ".join(f"`{m}`" for m in coverage.get("models_evaluated") or [])
        )
        st.caption(
            "With explainability: "
            + ", ".join(f"`{m}`" for m in coverage.get("models_with_explainability") or [])
        )
    with detail_r:
        st.caption(
            "Sensitive attributes audited: "
            + ", ".join(f"`{a}`" for a in coverage.get("sensitive_attributes") or [])
        )
        st.caption(
            "Agents: " + ", ".join(f"`{a}`" for a in coverage.get("agent_names") or [])
        )

    # --- artefact integrity ------------------------------------------------- #
    st.divider()
    st.subheader("Artefact integrity")
    st.caption(
        "Recomputes the SHA-256 of every registered artefact right now and compares "
        "it with the checksum captured at registration time."
    )
    try:
        integrity = fetch_registry_integrity(base_url, run_id)
    except ApiError as exc:
        show_api_error(exc, "Integrity check")
        render_provenance(base_url)
        return

    status = str(integrity.get("integrity_status"))
    icon, kind = INTEGRITY_STYLE.get(status, ("❔", "warning"))
    banner = getattr(st, kind)
    banner(
        f"**{icon} Integrity: {status.replace('_', ' ').upper()}** — "
        f"{integrity.get('verified_count')} verified, "
        f"{integrity.get('changed_count')} changed, "
        f"{integrity.get('missing_count')} missing "
        f"of {integrity.get('artifacts_checked')} artefacts.",
        icon=icon,
    )

    cols = st.columns(4)
    cols[0].metric("Checked", _fmt(integrity.get("artifacts_checked")))
    cols[1].metric("✅ Verified", _fmt(integrity.get("verified_count")))
    cols[2].metric("🔴 Changed", _fmt(integrity.get("changed_count")))
    cols[3].metric("⚠️ Missing", _fmt(integrity.get("missing_count")))

    counts = {
        "verified": integrity.get("verified_count") or 0,
        "changed": integrity.get("changed_count") or 0,
        "missing": integrity.get("missing_count") or 0,
    }
    fig = go.Figure(
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=[C_FPR, C_THRESHOLD, "#eda100"],
            text=[str(v) for v in counts.values()],
            textposition="outside",
            hovertemplate="%{x}: %{y} artefact(s)<extra></extra>",
        )
    )
    st.plotly_chart(_style(fig, height=300, y_title="Artefacts"), width="stretch")

    digest_match = integrity.get("registered_evidence_digest") == integrity.get(
        "current_evidence_digest"
    )
    st.markdown(
        f"- Registered digest: `{integrity.get('registered_evidence_digest')}`\n"
        f"- Current digest:    `{integrity.get('current_evidence_digest')}`\n"
        f"- Match: {'✅ yes' if digest_match else '🔴 no'}"
    )

    if integrity.get("changed_files"):
        st.error(
            "**Changed since registration:** "
            + ", ".join(f"`{p}`" for p in integrity["changed_files"]),
            icon="🔴",
        )
    if integrity.get("missing_files"):
        st.warning(
            "**Missing:** " + ", ".join(f"`{p}`" for p in integrity["missing_files"]),
            icon="⚠️",
        )

    with st.expander("Per-artefact detail", expanded=False):
        st.dataframe(integrity.get("artifacts") or [], width="stretch", hide_index=True)

    render_caveats(
        integrity.get("interpretation"),
        "What this integrity result does and does not establish",
        icon="ℹ️",
    )

    # --- timeline ----------------------------------------------------------- #
    st.divider()
    st.subheader("Timeline")
    try:
        timeline = fetch_registry_timeline(base_url, run_id)
    except ApiError as exc:
        show_api_error(exc, "Timeline")
        render_provenance(base_url)
        return

    all_events = timeline.get("events") or []
    shown = all_events[-25:]
    st.caption(
        f"{timeline.get('total_events', timeline.get('count'))} event(s) recorded"
        + (f"; showing the most recent {len(shown)}." if len(shown) < len(all_events)
           else ", oldest first.")
        + f" {timeline.get('note')}"
    )
    for event in shown:
        marker = "📄" if event.get("source") == "evidence" else "📒"
        st.markdown(
            f"{marker} **{event.get('event_time')}** · `{event.get('event_type')}` "
            f"— {event.get('detail')}"
        )

    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def main() -> None:
    page, base_url = render_sidebar()
    renderers = {
        "Overview": page_overview,
        "Model Performance": page_performance,
        "Fairness Audit": page_fairness,
        "Explainability": page_explainability,
        "Governance Decision & Risks": page_governance,
        "Agent Review": page_agent_review,
        "Model Registry": page_model_registry,
    }
    try:
        renderers[page](base_url)
    except ApiError as exc:  # safety net: never surface a raw traceback
        show_api_error(exc, page)


if __name__ == "__main__":
    main()
