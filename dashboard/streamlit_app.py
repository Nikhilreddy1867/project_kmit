"""
streamlit_app.py
================
Local Streamlit dashboard for MAAT, the Multi-Agent AI Audit and Trust Framework.

Run (with the API already running on port 8000):

    streamlit run dashboard/streamlit_app.py

Data contract
-------------
**Every value displayed here is fetched over HTTP from the governance API.** This
module opens no files, imports nothing from ``src/`` or ``app/``, and performs no
metric arithmetic -- charts and tables render the numbers exactly as the API
served them. It reads no CSV, no JSON artefact, no SQLite database, no model file
and no results directory: if a figure is on screen, an endpoint served it. Where
the interface needs prose (interpretation caveats, the four-fifths framing,
provenance, limitations), that text is also pulled from the API rather than
restated here, so the dashboard cannot drift from the audits.

The one deliberate exception is :data:`JOBLIB_WARNING`, for the reason given at
its definition.

The only client-side computation is chart layout: bar positions and axis ranges.
Notably the confusion matrix is displayed as **raw counts only** -- no row
normalisation, no derived percentages -- because deriving a rate here would be
recomputing a metric the audit already owns.

Two kinds of record
-------------------
The first seven pages read the **built-in Adult Income reference case**: committed
evidence, read-only, unaffected by anything a user submits. The last three are the
**model-intake flow** over user-submitted runs, whose evidence lives under
``runtime/``. The sidebar states which of the two is in view, because presenting
them as one audit would misrepresent both.

Writes
------
The intake pages POST to ``/api/onboarding`` and ``/api/gates`` only, and
:mod:`api_client` refuses any other write path client-side as well. Creating an
audit run, recording or revoking a waiver and re-evaluating a policy are the only
state changes reachable from this dashboard; none of them can touch the reference
case.
"""

from __future__ import annotations

import json
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
    ApiRejected,
    ApiServerError,
    ApiUnavailable,
    GovernanceApiClient,
)

#: Stated here as a literal, and it is the **one** exception to this module's rule
#: that all prose arrives over the API. The warning has to appear before the user
#: selects a file, i.e. before any request has been made -- and a security warning
#: that is only shown if a server happens to answer is not a security warning. The
#: API returns the same text in its validation response, and
#: :func:`render_security_gate` compares the two and says so if they ever diverge.
JOBLIB_WARNING = (
    "Joblib files may execute arbitrary code. Upload only models from trusted "
    "sources. This local academic prototype must not accept untrusted model files "
    "in production."
)

PAGES = [
    "Overview",
    "Model Performance",
    "Fairness Audit",
    "Explainability",
    "Governance Decision & Risks",
    "Agent Review",
    "Model Registry",
    "New Model Audit",
    "Uploaded Audit Runs",
    "Policy Gates & Conformity Bundle",
]

#: Pages 1-7 read the built-in Adult Income reference case; pages 8-10 are the
#: model-intake flow over user-submitted runs. The split is stated in the sidebar
#: because the two are separate records and must not be read as one audit.
REFERENCE_PAGES = frozenset(PAGES[:7])

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
def fetch_registry_runs(
    base_url: str, status: str | None = None, run_type: str | None = None
) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_runs(status=status, run_type=run_type)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_registry_run(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_run(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_registry_integrity(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_integrity(run_id)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_registry_timeline(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).registry_timeline(run_id)


# Uploaded-run reads use a 15s TTL. These runs change during a session -- a waiver
# is recorded, the gates are re-evaluated -- so a long cache would show a reviewer
# a verdict that has already been superseded. The write helpers below clear the
# cache outright rather than relying on the TTL to expire.
@st.cache_data(ttl=15, show_spinner=False)
def fetch_uploaded_audits(base_url: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_audits()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_uploaded_audit(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_audit(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_uploaded_performance(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_performance(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_uploaded_fairness(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_fairness(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_uploaded_explainability(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_explainability(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_uploaded_governance(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_governance(run_id)


# Integrity is a live check: a cached "verified" would be actively misleading, so
# this is the shortest TTL in the app.
@st.cache_data(ttl=5, show_spinner=False)
def fetch_uploaded_integrity(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_integrity(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_uploaded_timeline(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).uploaded_timeline(run_id)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_policies(base_url: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).policies()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_gate_evaluation(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).gate_evaluation(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_conformity_bundle(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).conformity_bundle(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_traceability(base_url: str, run_id: str) -> dict[str, Any]:
    return GovernanceApiClient(base_url).traceability(run_id)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_waivers(base_url: str, run_id: str) -> list[dict[str, Any]]:
    return GovernanceApiClient(base_url).waivers(run_id)


# --------------------------------------------------------------------------- #
# Error rendering
# --------------------------------------------------------------------------- #
def render_issues(issues: list[dict[str, Any]] | None) -> None:
    """
    Render the API's validation issues, errors first.

    Warnings are shown alongside errors rather than hidden: "ROC-AUC will not be
    available because this model has no predict_proba" is not an error, but a
    submitter who does not see it will be surprised by the audit that results.
    """
    if not issues:
        return
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") != "error"]
    for issue in errors + warnings:
        marker = "🔴" if issue.get("severity") == "error" else "⚠️"
        field = f" · `{issue['field']}`" if issue.get("field") else ""
        st.markdown(
            f"{marker} **`{issue.get('code')}`**{field} — {issue.get('message')}"
        )
        if issue.get("hint"):
            st.caption(f"↳ {issue['hint']}")


def show_api_error(exc: ApiError, context: str = "") -> None:
    """Render a typed API failure as a specific, actionable message."""
    prefix = f"**{context}** — " if context else ""
    if isinstance(exc, ApiRejected):
        # Not a malfunction: the API understood the submission and refused it. Shown
        # as something to fix, with every issue listed so one round trip is enough.
        st.error(f"{prefix}Refused. {exc.message}", icon="🚫")
        if exc.code:
            st.caption(f"Reason code: `{exc.code}`")
        render_issues(exc.issues)
    elif isinstance(exc, ApiUnavailable):
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
    st.sidebar.title("⚖️ MAAT")
    st.sidebar.caption(
        "Multi-Agent AI Audit and Trust Framework · local academic prototype"
    )

    page = st.sidebar.radio("Page", PAGES, key="page")
    if page in REFERENCE_PAGES:
        st.sidebar.caption(
            "📄 **Reference case** — the built-in Adult Income audit, read-only."
        )
    else:
        st.sidebar.caption(
            "📥 **User-submitted audits** — separate records, written under `runtime/`. "
            "They do not affect the reference case or its decision."
        )

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
        "Every figure is served by the API — from the committed audit artefacts for "
        "the reference case, and from the run's own `runtime/` artefacts for an "
        "uploaded audit. This dashboard opens no file, reads no database and "
        "recalculates nothing."
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
    #
    # Scoped to reference-case runs. The registry holds both kinds of run, but this
    # page is the reference case's record: an unfiltered list would put whichever
    # upload happens to be newest in the default selection, so the Adult Income
    # evidence a reviewer came here for would depend on who uploaded what today.
    # Uploaded audits have their own page, and the count below says how many exist
    # so their absence here is visible rather than silent.
    try:
        listing = fetch_registry_runs(base_url, run_type="reference_case")
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
    uploaded_ids = listing.get("uploaded_run_ids") or []
    cols = st.columns(4)
    cols[0].metric("Registered runs", _fmt(listing.get("count")))
    cols[1].metric("Active run", listing.get("active_run_id") or "none")
    cols[2].metric("Uploaded runs", _fmt(len(uploaded_ids)))
    cols[3].metric("Database", str(listing.get("database")))
    st.caption(
        "Reference-case runs only. The registry also holds "
        f"{len(uploaded_ids)} uploaded-model run(s); those are shown on the "
        "**Uploaded Audit Runs** page. Neither kind of run supersedes the other."
    )
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

    # Collapse CONSECUTIVE same-type events instead of windowing. The event log is
    # append-only, so repeated integrity checks pile up; a plain "last N" view
    # pushed the registration event off the screen entirely. Collapsing keeps the
    # whole history visible and short, and states the repeat count rather than
    # hiding anything.
    collapsed: list[tuple[dict[str, Any], int]] = []
    for event in all_events:
        if collapsed and collapsed[-1][0].get("event_type") == event.get("event_type"):
            previous, count = collapsed[-1]
            collapsed[-1] = (event, count + 1)  # keep the most recent occurrence
        else:
            collapsed.append((event, 1))

    st.caption(
        f"{timeline.get('total_events', len(all_events))} event(s) recorded, oldest "
        f"first; consecutive repeats are grouped. {timeline.get('note')}"
    )
    for event, count in collapsed:
        marker = "📄" if event.get("source") == "evidence" else "📒"
        repeat = f" **×{count}** (most recent shown)" if count > 1 else ""
        st.markdown(
            f"{marker} **{event.get('event_time')}** · `{event.get('event_type')}`"
            f"{repeat} — {event.get('detail')}"
        )

    render_provenance(base_url)


# --------------------------------------------------------------------------- #
# Shared blocks for user-submitted audits
# --------------------------------------------------------------------------- #
GATE_STYLE = {
    "PASS": ("✅", "success"),
    "WAIVE": ("🟡", "warning"),
    "BLOCK": ("⛔", "error"),
    "NOT_EVALUATED": ("⚪", "info"),
}

GOVERNANCE_STYLE = {
    "review_required": ("📋", "warning"),
    "insufficient_evidence": ("❔", "warning"),
    "blocked_by_policy": ("⛔", "error"),
}

FAIRNESS_STATUS_STYLE = {
    "assessed": "📊",
    "not_provided_by_user": "➖",
    "not_available": "❔",
}


def render_intake_notice() -> None:
    """The standing framing for every user-submitted audit page."""
    st.info(
        "**These are user-submitted audit runs, separate from the built-in Adult "
        "Income reference case.** Everything they produce is written under "
        "`runtime/` and is deterministic decision-support evidence for human review "
        "— not a legal compliance assessment, not proof of discrimination, not a "
        "causal claim, and not authorisation to deploy anything.",
        icon="📥",
    )


#: Fallback for the production-hardening note, used only before any request has
#: been made. Once the API has answered, its own ``production_hardening`` text is
#: shown instead -- the same API-first rule as every other piece of prose here.
PRODUCTION_HARDENING_FALLBACK = (
    "A production implementation must not deserialise user-supplied pickles in the "
    "application process. It should run model loading and inference inside an "
    "isolated sandbox (a separate container or VM, no network egress, read-only "
    "filesystem, dropped privileges, CPU/memory limits) and prefer formats that do "
    "not carry executable payloads — ONNX for the computation graph, or skops, which "
    "reconstructs scikit-learn estimators from an allow-list of types instead of "
    "executing arbitrary opcodes. This prototype does neither: it loads the uploaded "
    "model in-process, which is acceptable only because the operator is also the "
    "person supplying the file."
)


def render_security_gate(
    api_warning: str | None = None, hardening: str | None = None
) -> None:
    """
    The joblib warning, stated in full, plus what the prototype does about it.

    ``api_warning`` is the text the API returned for a submission that has already
    been made. When present it is compared with the local copy: if the two ever
    diverge, the discrepancy is shown rather than quietly resolved in favour of
    either one. ``hardening`` is the API's production-hardening note, preferred over
    :data:`PRODUCTION_HARDENING_FALLBACK` whenever it has been served.
    """
    st.error(f"**⚠️ {JOBLIB_WARNING}**", icon="⚠️")
    if api_warning and api_warning.strip() != JOBLIB_WARNING:
        st.warning(
            "**The warning text served by the API differs from the one shown above.** "
            "Treat both as authoritative until the discrepancy is explained.\n\n"
            f"API text: {api_warning}",
            icon="🧩",
        )
    with st.expander("What this prototype does about that risk", expanded=False):
        st.markdown(
            "- Only `.joblib` is accepted. `.py`, `.pkl`, `.pickle`, archives, "
            "executables and remote URLs are refused **before** anything is stored.\n"
            "- The file is never deserialised until the acknowledgement has been "
            "given and the dataset has passed structural validation, so a model "
            "that was never going to be auditable is rejected without being loaded.\n"
            "- The acknowledgement is recorded in the audit run's manifest.\n"
            "- Uploads are stored under `runtime/uploads/<upload_id>/` with a "
            "generated UUID filename. Your original filename is kept as a label "
            "only and is never used as a filesystem path. No upload can overwrite "
            "an existing file."
        )
        st.markdown("**Production hardening**")
        st.markdown(hardening or PRODUCTION_HARDENING_FALLBACK)


def uploaded_run_selector(base_url: str, key: str) -> str | None:
    """
    Shared run picker for the two uploaded-run pages.

    Returns ``None`` when there is nothing to select, having already explained what
    to do about it. The last choice is remembered in ``selected_uploaded_run`` so
    moving between the two pages keeps the same run in view.
    """
    try:
        listing = fetch_uploaded_audits(base_url)
    except ApiError as exc:
        show_api_error(exc, "Uploaded audit runs")
        return None

    runs = listing.get("runs") or []
    if not runs:
        st.warning(
            "**No user-submitted audit runs yet.** Open **New Model Audit** in the "
            "sidebar to submit a trusted local `.joblib` model and a labelled CSV.",
            icon="📭",
        )
        return None

    labels = [
        f"{GOVERNANCE_STYLE.get(str(r.get('governance_state')), ('•', ''))[0]} "
        f"{r.get('audit_run_id')} · {r.get('model_name')} v{r.get('model_version')} "
        f"· {r.get('created_at')}"
        for r in runs
    ]
    remembered = st.session_state.get("selected_uploaded_run")
    ids = [str(r.get("audit_run_id")) for r in runs]
    index = ids.index(remembered) if remembered in ids else 0
    choice = st.selectbox(
        f"Audit run ({listing.get('count')} recorded, newest first)",
        labels,
        index=index,
        key=key,
    )
    run_id = ids[labels.index(choice)]
    st.session_state["selected_uploaded_run"] = run_id
    return run_id


def render_gate_row(gate_summary: dict[str, str]) -> None:
    """The five gates as one row, in policy order."""
    order = ["DG", "TG", "VG", "RG", "OG"]
    present = [g for g in order if g in gate_summary] + [
        g for g in gate_summary if g not in order
    ]
    cols = st.columns(len(present) or 1)
    for col, gate in zip(cols, present):
        status = str(gate_summary.get(gate))
        icon, _ = GATE_STYLE.get(status, ("❔", "info"))
        col.metric(gate, f"{icon} {status}")


def render_governance_banner(governance: dict[str, Any]) -> None:
    """The run's governance state, with the API's own meaning and grounds."""
    state = str(governance.get("governance_state"))
    icon, kind = GOVERNANCE_STYLE.get(state, ("❔", "warning"))
    getattr(st, kind)(
        f"#### {icon} {state.replace('_', ' ').upper()}\n"
        f"{governance.get('state_meaning', '')}",
        icon=icon,
    )
    cols = st.columns(2)
    cols[0].metric(
        "Human review required",
        "✅ always" if governance.get("human_review_required") else "—",
    )
    cols[1].metric(
        "Deployment authorisation",
        str(governance.get("deployment_authorisation", "not_granted")).replace("_", " "),
    )
    if governance.get("state_grounds"):
        st.markdown("**Grounds for this state**")
        for ground in governance["state_grounds"]:
            st.markdown(f"- {ground}")


# --------------------------------------------------------------------------- #
# Page 8 — New Model Audit
#
# Two phases, because the dashboard must not parse the uploaded CSV itself. The
# column names, target classes and model capabilities that populate the phase-2
# controls all arrive from POST /api/onboarding/validate -- so the dropdowns are
# built from what the API read, not from anything computed here.
# --------------------------------------------------------------------------- #
def page_new_audit(base_url: str) -> None:
    st.title("New model audit")
    st.caption(
        "Submit a trusted local `.joblib` binary classifier and a labelled CSV to "
        "create a new governance audit run."
    )
    render_intake_notice()

    st.subheader("1 · Files and security acknowledgement")
    _seen = st.session_state.get("intake_validation") or {}
    render_security_gate(
        _seen.get("security_warning"), _seen.get("production_hardening")
    )

    left, right = st.columns(2)
    with left:
        model_file = st.file_uploader(
            "Model file",
            type=None,
            key="intake_model_file",
            help="A scikit-learn estimator or fitted Pipeline saved with joblib.dump.",
        )
        st.caption(
            "Only `.joblib` is accepted. The uploader is deliberately unrestricted so "
            "that the refusal is visible: anything else is rejected by the API before "
            "it is stored or opened."
        )
    with right:
        dataset_file = st.file_uploader(
            "Labelled dataset (.csv)",
            type=["csv"],
            key="intake_dataset_file",
            help="Must contain the ground-truth label column and the model's input "
            "features. Sensitive columns need not be model features.",
        )

    acknowledged = st.checkbox(
        "I have read the warning above. This model file comes from a source I trust, "
        "and I accept that loading it executes code in this process.",
        key="intake_ack",
    )

    st.subheader("2 · Target configuration")
    st.caption(
        "The target column and positive class are needed before the CSV can be read, "
        "so type them here. The remaining choices are offered as lists once the API "
        "has profiled the file."
    )
    cfg_l, cfg_r = st.columns(2)
    target_column = cfg_l.text_input(
        "Target column", key="intake_target", placeholder="e.g. income"
    ).strip()
    positive_class = cfg_r.text_input(
        "Positive class", key="intake_positive", placeholder="e.g. >50K"
    ).strip()

    validate_clicked = st.button(
        "🔍 Validate submission",
        type="primary",
        disabled=not (model_file and dataset_file and target_column and positive_class),
        help="Uploads both files and runs every intake check. No audit run is created.",
    )

    if validate_clicked:
        client = GovernanceApiClient(base_url)
        with st.spinner("Uploading and validating…"):
            try:
                validation = client.validate_upload(
                    model_name=model_file.name,
                    model_bytes=model_file.getvalue(),
                    dataset_name=dataset_file.name,
                    dataset_bytes=dataset_file.getvalue(),
                    target_column=target_column,
                    positive_class=positive_class,
                    # Fairness columns are chosen in phase 3 from the profiled
                    # column list, so this first pass asks for none.
                    sensitive_columns=[],
                    security_acknowledged=acknowledged,
                )
            except ApiError as exc:
                st.session_state.pop("intake_validation", None)
                show_api_error(exc, "Validation")
                return
        st.session_state["intake_validation"] = validation
        st.session_state.pop("intake_result", None)

    validation = st.session_state.get("intake_validation")
    if not validation:
        st.caption(
            "Nothing has been uploaded yet. Validation stores the two files under "
            "`runtime/uploads/` and creates no audit run."
        )
        return

    # --- validation report -------------------------------------------------- #
    st.divider()
    st.subheader("Validation report")
    if validation.get("valid"):
        st.success(
            f"**Ready to audit.** Upload id `{validation.get('upload_id')}`.",
            icon="✅",
        )
    else:
        st.error(
            "**Not ready.** Fix every error below and validate again. No audit run "
            "has been created.",
            icon="🚫",
        )
    render_issues(validation.get("issues"))
    st.caption(validation.get("next_step", ""))

    capabilities = validation.get("audit_capabilities") or {}
    cols = st.columns(len(capabilities) or 1)
    for col, (name, available) in zip(cols, capabilities.items()):
        col.metric(
            name.replace("_", " "), "✅ yes" if available else "❌ no"
        )
    st.caption(
        "An unavailable capability is reported honestly and stays unavailable: no "
        "substitute model is trained, no probability is synthesised, and no "
        "importance score is invented."
    )

    profile = validation.get("dataset") or {}
    caps = validation.get("model_capabilities") or {}
    compat = validation.get("feature_compatibility") or {}

    prof_tab, model_tab, compat_tab = st.tabs(
        ["Dataset profile", "Model capabilities", "Feature compatibility"]
    )
    with prof_tab:
        cols = st.columns(4)
        cols[0].metric("Rows", _fmt(profile.get("row_count")))
        cols[1].metric("Columns", _fmt(profile.get("column_count")))
        cols[2].metric("Target classes", _fmt(len(profile.get("target_classes") or [])))
        cols[3].metric(
            "Positive rows", _fmt(profile.get("positive_class_count"))
        )
        st.markdown(
            "**Class counts:** "
            + ", ".join(
                f"`{k}` = {v:,}"
                for k, v in (profile.get("target_class_counts") or {}).items()
            )
        )
        st.caption("Columns: " + ", ".join(f"`{c}`" for c in profile.get("columns") or []))
        if profile.get("duplicate_columns"):
            st.error(
                "Duplicate column names: "
                + ", ".join(f"`{c}`" for c in profile["duplicate_columns"]),
                icon="🔴",
            )
    with model_tab:
        cols = st.columns(4)
        cols[0].metric("Loaded", _fmt(caps.get("loaded")))
        cols[1].metric("Binary classifier", _fmt(caps.get("is_binary_classifier")))
        cols[2].metric("predict_proba", _fmt(caps.get("has_predict_proba")))
        cols[3].metric("Pipeline", _fmt(caps.get("is_pipeline")))
        st.markdown(
            f"- Estimator: `{caps.get('estimator_type')}` "
            f"from `{caps.get('estimator_module')}`\n"
            f"- Final estimator: `{caps.get('final_estimator')}`\n"
            f"- Pipeline steps: "
            + ", ".join(f"`{s}`" for s in caps.get("pipeline_steps") or ["—"])
            + f"\n- Classes: "
            + ", ".join(f"`{c}`" for c in caps.get("classes") or [])
            + f"\n- Permutation importance: {_fmt(caps.get('supports_permutation_importance'))}"
            f"\n- Local TreeSHAP: {_fmt(caps.get('supports_treeshap'))}"
        )
        if not caps.get("has_predict_proba"):
            st.warning(
                "No `predict_proba`. ROC-AUC will be reported as unavailable with the "
                "reason stated, the decision threshold will not be applied, and no "
                "probability will be estimated.",
                icon="⚠️",
            )
    with compat_tab:
        cols = st.columns(3)
        cols[0].metric("Checked", _fmt(compat.get("checked")))
        cols[1].metric("Compatible", _fmt(compat.get("compatible")))
        cols[2].metric("Matched features", _fmt(compat.get("matched_feature_count")))
        st.caption(f"Method: {compat.get('method')}")
        if compat.get("missing_features"):
            st.error(
                "**Missing from the CSV** (the model expects these): "
                + ", ".join(f"`{c}`" for c in compat["missing_features"]),
                icon="🔴",
            )
        if compat.get("unexpected_features"):
            st.info(
                "**In the CSV but not expected by the model:** "
                + ", ".join(f"`{c}`" for c in compat["unexpected_features"]),
                icon="ℹ️",
            )
        if compat.get("sensitive_columns_retained"):
            st.caption(
                "Retained for fairness reporting even though they are not model "
                "features: "
                + ", ".join(f"`{c}`" for c in compat["sensitive_columns_retained"])
            )

    if not validation.get("valid") or not validation.get("upload_id"):
        return

    # --- phase 3: configure and run ----------------------------------------- #
    st.divider()
    st.subheader("3 · Fairness columns, threshold and model metadata")

    columns = [c for c in profile.get("columns") or [] if c != target_column]
    classes = [str(c) for c in profile.get("target_classes") or []]

    with st.form("intake_create"):
        sensitive_columns = st.multiselect(
            "Sensitive columns for fairness reporting",
            columns,
            help="Fairness is computed only for the columns you select here. "
            "Selecting none is allowed and yields status "
            "'not_provided_by_user' — which is not a pass and not a fairness claim.",
        )
        col_a, col_b = st.columns(2)
        chosen_positive = col_a.selectbox(
            "Positive class",
            classes,
            index=classes.index(positive_class) if positive_class in classes else 0,
        )
        threshold = col_b.slider(
            "Decision threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            help="Applied to predicted probabilities. Reported as not applied for a "
            "predict-only model.",
        )

        meta_l, meta_r = st.columns(2)
        model_name = meta_l.text_input("Model name", value=Path(model_file.name).stem)
        model_version = meta_r.text_input("Model version", value="unspecified")
        model_owner = st.text_input(
            "Accountable owner",
            help="The person or team answerable for this model. Recorded in the "
            "Conformity Bundle.",
        )
        intended_use = st.text_area(
            "Intended use",
            help="What this model is for. Recorded as evidence, not evaluated.",
        )
        decision_context = st.text_area(
            "Decision context",
            help="What decisions this model would inform, and about whom.",
        )
        submitted = st.form_submit_button("🚀 Create audit run", type="primary")

    if submitted:
        missing = [
            label
            for label, value in (
                ("model name", model_name),
                ("model version", model_version),
                ("accountable owner", model_owner),
                ("intended use", intended_use),
                ("decision context", decision_context),
            )
            if not str(value).strip()
        ]
        if missing:
            # Checked here as well as server-side so the user is not made to wait
            # for an upload round trip to be told a text box is empty.
            st.error(
                "Provide: " + ", ".join(missing) + ". These are the accountability "
                "fields recorded in the Conformity Bundle, so none of them is "
                "defaulted for you.",
                icon="🚫",
            )
            return
        client = GovernanceApiClient(base_url)
        with st.spinner("Running inference, fairness, explainability and the gates…"):
            try:
                result = client.create_audit(
                    target_column=target_column,
                    positive_class=chosen_positive,
                    decision_threshold=threshold,
                    sensitive_columns=sensitive_columns,
                    security_acknowledged=acknowledged,
                    upload_id=str(validation["upload_id"]),
                    model_metadata={
                        "model_name": model_name.strip(),
                        "model_version": model_version.strip(),
                        "model_owner": model_owner.strip(),
                        "intended_use": intended_use.strip(),
                        "decision_context": decision_context.strip(),
                    },
                )
            except ApiError as exc:
                show_api_error(exc, "Audit run")
                return
        st.cache_data.clear()  # the new run must appear immediately
        st.session_state["intake_result"] = result
        st.session_state["selected_uploaded_run"] = result["audit_run_id"]

    result = st.session_state.get("intake_result")
    if not result:
        return

    st.divider()
    st.subheader("Audit run created")
    st.success(
        f"**`{result['audit_run_id']}`** — written under `{result['written_under']}`.",
        icon="✅",
    )
    cols = st.columns(4)
    cols[0].metric("Governance state", str(result.get("governance_state")))
    cols[1].metric("Fairness", str(result.get("fairness_status")))
    cols[2].metric("Explainability", str(result.get("explainability_status")))
    cols[3].metric("Artefacts", _fmt(result.get("artifact_count")))
    render_gate_row(result.get("gate_summary") or {})
    st.markdown(
        f"- Conformity Bundle: `{result.get('conformity_bundle_id')}`\n"
        f"- Registry run: `{result.get('registry_run_id') or 'not registered'}`"
    )
    render_issues(result.get("warnings"))
    st.info(result.get("next_step", ""), icon="➡️")
    st.caption(
        "Open **Uploaded Audit Runs** for the full evidence, or **Policy Gates & "
        "Conformity Bundle** for the gate decisions and traceability matrix. "
        f"{result.get('notice', '')}"
    )


# --------------------------------------------------------------------------- #
# Page 9 — Uploaded Audit Runs
# --------------------------------------------------------------------------- #
def page_uploaded_audits(base_url: str) -> None:
    st.title("Uploaded audit runs")
    st.caption(
        "Evidence for each user-submitted model, served from that run's own "
        "artefacts under `runtime/audits/`."
    )
    render_intake_notice()

    run_id = uploaded_run_selector(base_url, key="uploaded_run_pick")
    if run_id is None:
        return

    try:
        detail = fetch_uploaded_audit(base_url, run_id)
    except ApiError as exc:
        show_api_error(exc, f"Run {run_id}")
        return

    metadata = detail.get("model_metadata") or {}
    dataset = detail.get("dataset_metadata") or {}
    target = detail.get("target_configuration") or {}

    st.divider()
    cols = st.columns(4)
    cols[0].metric("Governance state", str(detail.get("governance_state")))
    cols[1].metric("Rows audited", _fmt(dataset.get("row_count")))
    cols[2].metric("Threshold", _fmt(dataset.get("decision_threshold"), 2))
    cols[3].metric("Artefacts", _fmt(len(detail.get("artifacts") or [])))

    tabs = st.tabs(
        [
            "Submission",
            "Performance",
            "Fairness",
            "Explainability",
            "Governance & risks",
            "Evidence integrity",
            "Timeline",
        ]
    )

    # --- submission --------------------------------------------------------- #
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Model**")
            st.markdown(
                f"- Name: `{metadata.get('model_name')}` "
                f"v`{metadata.get('model_version')}`\n"
                f"- Accountable owner: {metadata.get('model_owner')}\n"
                f"- SHA-256: `{detail.get('model_checksum')}`"
            )
            st.caption(f"**Intended use:** {metadata.get('intended_use', '—')}")
            st.caption(f"**Decision context:** {metadata.get('decision_context', '—')}")
        with right:
            st.markdown("**Dataset**")
            st.markdown(
                f"- Label: `{dataset.get('original_filename_label', '—')}`\n"
                f"- Rows × columns: {_fmt(dataset.get('row_count'))} × "
                f"{_fmt(dataset.get('column_count'))}\n"
                f"- SHA-256: `{detail.get('dataset_checksum')}`"
            )
            st.markdown("**Target configuration**")
            st.markdown(
                f"- Target column: `{target.get('target_column')}`\n"
                f"- Positive class: `{target.get('positive_class')}`\n"
                f"- Sensitive columns: "
                + (
                    ", ".join(f"`{c}`" for c in target.get("sensitive_columns") or [])
                    or "_none selected_"
                )
            )

        st.markdown("**Security**")
        cols = st.columns(2)
        cols[0].metric(
            "Warning acknowledged", _fmt(detail.get("security_acknowledged"))
        )
        cols[1].metric("Upload id", str(detail.get("upload_id"))[:12] + "…")
        render_security_gate(detail.get("security_warning"))

        st.markdown("**Audit coverage**")
        coverage = detail.get("audit_coverage") or {}
        cols = st.columns(len(coverage) or 1)
        for col, (name, available) in zip(cols, coverage.items()):
            col.metric(name.replace("_", " "), "✅ yes" if available else "❌ no")

        with st.expander("Model capabilities and feature compatibility", expanded=False):
            st.json(detail.get("model_capabilities") or {}, expanded=False)
            st.json(detail.get("feature_compatibility") or {}, expanded=False)
        with st.expander("Artefacts written for this run", expanded=False):
            st.dataframe(
                detail.get("artifacts") or [], width="stretch", hide_index=True
            )
        render_caveats(detail.get("limitations"), "Limitations of this audit run")

    # --- performance -------------------------------------------------------- #
    with tabs[1]:
        try:
            performance = fetch_uploaded_performance(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Performance")
        else:
            cols = st.columns(5)
            cols[0].metric("Accuracy", _fmt(performance.get("accuracy")))
            cols[1].metric("Precision", _fmt(performance.get("precision")))
            cols[2].metric("Recall", _fmt(performance.get("recall")))
            cols[3].metric("F1", _fmt(performance.get("f1")))
            cols[4].metric("ROC-AUC", _fmt(performance.get("roc_auc")))
            if performance.get("roc_auc") is None:
                st.warning(
                    f"**ROC-AUC unavailable.** "
                    f"{performance.get('roc_auc_unavailable_reason', '')}",
                    icon="⚠️",
                )
            st.caption(
                f"{_fmt(performance.get('n_samples'))} rows · positive class "
                f"`{performance.get('positive_class')}` · threshold "
                f"{_fmt(performance.get('decision_threshold'), 2)} "
                f"({'applied' if performance.get('threshold_applied') else 'not applied'})"
                f" · {performance.get('computed_from', '')}"
            )

            matrix = performance.get("confusion_matrix") or {}
            fig = go.Figure(
                go.Bar(
                    x=["TN", "FP", "FN", "TP"],
                    y=[
                        matrix.get("true_negatives", 0),
                        matrix.get("false_positives", 0),
                        matrix.get("false_negatives", 0),
                        matrix.get("true_positives", 0),
                    ],
                    marker_color=[BASELINE, C_NEGATIVE, C_NEGATIVE, C_SELECTION],
                    text=[
                        f"{matrix.get(k, 0):,}"
                        for k in (
                            "true_negatives",
                            "false_positives",
                            "false_negatives",
                            "true_positives",
                        )
                    ],
                    textposition="outside",
                    hovertemplate="%{x}: %{y:,} rows<extra></extra>",
                )
            )
            st.plotly_chart(
                _style(fig, height=320, y_title="Rows (raw counts)"), width="stretch"
            )
            st.caption(
                "Raw counts exactly as the audit computed them — no normalisation "
                "and no derived rates, which the audit already owns."
            )
            render_caveats(performance.get("caveats"), "Performance caveats")

    # --- fairness ----------------------------------------------------------- #
    with tabs[2]:
        try:
            fairness = fetch_uploaded_fairness(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Fairness")
        else:
            status = str(fairness.get("status"))
            marker = FAIRNESS_STATUS_STYLE.get(status, "❔")
            if status == "not_provided_by_user":
                st.info(
                    f"#### {marker} {status}\n{fairness.get('status_detail', '')}",
                    icon="➖",
                )
            elif status == "assessed":
                st.success(
                    f"#### {marker} Fairness assessed\n"
                    f"{fairness.get('status_detail', '')}",
                    icon="📊",
                )
            else:
                st.warning(
                    f"#### {marker} {status}\n{fairness.get('status_detail', '')}",
                    icon="❔",
                )

            st.caption(
                "Requested columns: "
                + (
                    ", ".join(
                        f"`{c}`" for c in fairness.get("sensitive_columns_requested") or []
                    )
                    or "none"
                )
                + f" · reference group rule: {fairness.get('reference_group_rule')}"
                + f" · small-group threshold: n < "
                f"{_fmt(fairness.get('small_group_threshold'))}"
            )

            for attribute in fairness.get("attributes") or []:
                st.markdown(f"#### `{attribute.get('attribute')}`")
                cols = st.columns(4)
                cols[0].metric("Groups", _fmt(attribute.get("n_groups")))
                cols[1].metric(
                    "Reference",
                    f"{attribute.get('reference_group')} "
                    f"(n={_fmt(attribute.get('reference_n'))})",
                )
                cols[2].metric(
                    "Min disparate impact ratio",
                    _fmt(attribute.get("min_disparate_impact_ratio")),
                )
                cols[3].metric(
                    "Max |equal opportunity diff|",
                    _fmt(attribute.get("max_abs_equal_opportunity_difference")),
                )
                if attribute.get("groups_failing_four_fifths"):
                    st.warning(
                        "Below the four-fifths screening ratio: "
                        + ", ".join(
                            f"`{g}`" for g in attribute["groups_failing_four_fifths"]
                        ),
                        icon="⚠️",
                    )
                if attribute.get("small_groups_present"):
                    st.caption(
                        "⚠️ Some groups are small, so their rates carry wide "
                        "uncertainty and small differences may be noise."
                    )
                if attribute.get("undefined_metric_count"):
                    st.caption(
                        f"{attribute['undefined_metric_count']} metric(s) are "
                        "undefined for this attribute and are reported as not "
                        "available rather than as zero."
                    )

                rows = [
                    g
                    for g in fairness.get("groups") or []
                    if g.get("attribute") == attribute.get("attribute")
                ]
                if rows:
                    fig = go.Figure()
                    names = [str(r.get("group")) for r in rows]
                    for label, key, colour in (
                        ("Selection rate", "selection_rate", C_SELECTION),
                        ("TPR", "true_positive_rate", C_TPR),
                        ("FPR", "false_positive_rate", C_FPR),
                    ):
                        values = [r.get(key) for r in rows]
                        fig.add_bar(
                            name=label,
                            x=names,
                            y=values,
                            marker_color=colour,
                            text=[_fmt(v, 3) for v in values],
                            textposition="outside",
                            hovertemplate="%{x} · " + label + ": %{y}<extra></extra>",
                        )
                    st.plotly_chart(
                        _style(fig, height=380, y_title="Rate"), width="stretch"
                    )
                    st.caption(
                        "A missing bar is an undefined metric (an empty denominator), "
                        "not a zero."
                    )
                    st.dataframe(rows, width="stretch", hide_index=True)

            render_caveats(fairness.get("interpretation"), "How to read this fairness "
                           "assessment", icon="ℹ️")
            st.error(f"**{fairness.get('four_fifths_notice', '')}**", icon="⚖️")

    # --- explainability ----------------------------------------------------- #
    with tabs[3]:
        try:
            explain = fetch_uploaded_explainability(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Explainability")
        else:
            status = str(explain.get("status"))
            if status == "available":
                st.success(f"#### 🔍 {status}\n{explain.get('status_detail','')}",
                           icon="🔍")
            else:
                st.warning(f"#### ❔ {status}\n{explain.get('status_detail','')}",
                           icon="❔")
            st.caption(
                f"Method: {explain.get('method') or 'n/a'} · repeats: "
                f"{_fmt(explain.get('n_repeats'))} · scorer: "
                f"{explain.get('scorer') or 'n/a'} · local method: "
                f"{explain.get('local_method') or 'not available'}"
            )

            importance = explain.get("global_importance") or []
            if importance:
                ordered = list(reversed(importance))
                fig = go.Figure(
                    go.Bar(
                        x=[i.get("importance_mean") for i in ordered],
                        y=[i.get("feature") for i in ordered],
                        orientation="h",
                        marker_color=[
                            C_THRESHOLD if i.get("is_selected_sensitive_column")
                            else C_SELECTION
                            for i in ordered
                        ],
                        error_x=dict(
                            type="data",
                            array=[i.get("importance_std") or 0 for i in ordered],
                            color=INK_SECONDARY,
                            thickness=1,
                        ),
                        text=[_fmt(i.get("importance_mean")) for i in ordered],
                        textposition="outside",
                        hovertemplate="%{y}: %{x}<extra></extra>",
                    )
                )
                st.plotly_chart(
                    _style(
                        fig,
                        height=max(320, 34 * len(ordered)),
                        x_title="Permutation importance (mean drop in score)",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Red bars are columns you selected as sensitive. Importance is "
                    "**association, not causation**: a high score means permuting the "
                    "column degrades the score, not that the column causes the outcome."
                )
                st.dataframe(importance, width="stretch", hide_index=True)

            for local in explain.get("local_explanations") or []:
                with st.expander(
                    f"Local explanation · {local.get('case_type')} "
                    f"(row {local.get('row_index')})",
                    expanded=False,
                ):
                    st.markdown(
                        f"- Predicted: `{local.get('predicted_label')}` "
                        f"(p={_fmt(local.get('predicted_probability'))})\n"
                        f"- Actual: `{local.get('actual_label')}`\n"
                        f"- Base value: {_fmt(local.get('base_value'))}"
                    )
                    st.dataframe(
                        local.get("top_factors") or [], width="stretch", hide_index=True
                    )

            if explain.get("proxy_assessment"):
                st.warning("**Possible proxy features**", icon="🔗")
                for line in explain["proxy_assessment"]:
                    st.markdown(f"- {line}")
            render_caveats(explain.get("caveats"), "Explainability caveats")

    # --- governance & risks ------------------------------------------------- #
    with tabs[4]:
        try:
            governance = fetch_uploaded_governance(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Governance")
        else:
            render_governance_banner(governance)
            counts = governance.get("severity_counts") or {}
            if counts:
                cols = st.columns(len(counts))
                for col, (severity, count) in zip(cols, counts.items()):
                    col.metric(severity.capitalize(), _fmt(count))
            if governance.get("unavailable_capabilities"):
                st.warning(
                    "**Unavailable audit capabilities:** "
                    + ", ".join(
                        f"`{c}`" for c in governance["unavailable_capabilities"]
                    )
                    + ". These are reported as unavailable, not estimated.",
                    icon="⚠️",
                )
            risks = governance.get("risks") or []
            if risks:
                st.markdown("**Risks identified**")
                for risk in risks:
                    with st.expander(
                        f"{risk.get('severity','').upper()} · "
                        f"`{risk.get('risk_id')}` — {risk.get('statement')}",
                        expanded=False,
                    ):
                        st.markdown(
                            f"- **Category:** {risk.get('category')}\n"
                            f"- **Evidence:** {risk.get('evidence')}\n"
                            f"- **Limitation:** {risk.get('limitation')}\n"
                            f"- **Recommended action:** "
                            f"{risk.get('recommended_action')}"
                        )
            st.info(governance.get("reference_case_note", ""), icon="📄")
            render_caveats(governance.get("limitations"), "Limitations")

    # --- integrity ---------------------------------------------------------- #
    with tabs[5]:
        st.caption(
            "Recomputes the SHA-256 of every baselined artefact now and compares it "
            "with the checksum recorded when the run was created."
        )
        try:
            integrity = fetch_uploaded_integrity(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Integrity")
        else:
            status = str(integrity.get("integrity_status"))
            icon, kind = INTEGRITY_STYLE.get(status, ("❔", "warning"))
            getattr(st, kind)(
                f"**{icon} Integrity: {status.replace('_', ' ').upper()}** — "
                f"{integrity.get('verified_count')} verified, "
                f"{integrity.get('changed_count')} changed, "
                f"{integrity.get('missing_count')} missing of "
                f"{integrity.get('artifacts_checked')} artefacts. "
                f"Checked {integrity.get('checked_at')}.",
                icon=icon,
            )
            cols = st.columns(4)
            cols[0].metric("Checked", _fmt(integrity.get("artifacts_checked")))
            cols[1].metric("✅ Verified", _fmt(integrity.get("verified_count")))
            cols[2].metric("🔴 Changed", _fmt(integrity.get("changed_count")))
            cols[3].metric("⚠️ Missing", _fmt(integrity.get("missing_count")))
            st.dataframe(
                integrity.get("artifacts") or [], width="stretch", hide_index=True
            )
            st.caption(f"Method: {integrity.get('method')}")
            render_caveats(
                integrity.get("interpretation"),
                "What this integrity result does and does not establish",
                icon="ℹ️",
            )

    # --- timeline ----------------------------------------------------------- #
    with tabs[6]:
        try:
            timeline = fetch_uploaded_timeline(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Timeline")
        else:
            st.caption(
                f"{timeline.get('count')} event(s), oldest first. "
                f"{timeline.get('note', '')}"
            )
            markers = {"run": "📄", "registry": "📒", "waiver": "🟡"}
            for event in timeline.get("events") or []:
                marker = markers.get(str(event.get("source")), "•")
                st.markdown(
                    f"{marker} **{event.get('event_time')}** · "
                    f"`{event.get('event_type')}` — {event.get('detail')}"
                )


# --------------------------------------------------------------------------- #
# Page 10 — Policy Gates & Conformity Bundle
# --------------------------------------------------------------------------- #
def page_policy_gates(base_url: str) -> None:
    st.title("Policy gates & Conformity Bundle")
    st.caption(
        "Governance-as-Code: a versioned policy file evaluated deterministically "
        "over one run's evidence, producing PASS / WAIVE / BLOCK / NOT_EVALUATED per "
        "gate."
    )
    render_intake_notice()

    policy_tab, run_tab = st.tabs(["Policy profile", "Run evaluation"])

    # --- policy profile ----------------------------------------------------- #
    with policy_tab:
        try:
            policies = fetch_policies(base_url)
        except ApiError as exc:
            show_api_error(exc, "Policies")
        else:
            profiles = policies.get("policies") or []
            if not profiles:
                st.warning("The API served no policy profiles.", icon="📭")
            for policy in profiles:
                st.subheader(
                    f"{policy.get('policy_name')} v{policy.get('policy_version')}"
                )
                cols = st.columns(4)
                cols[0].metric("Policy id", str(policy.get("policy_id")))
                cols[1].metric("Status", str(policy.get("policy_status")))
                cols[2].metric("Gates", _fmt(len(policy.get("gates") or [])))
                cols[3].metric("Controls", _fmt(len(policy.get("controls") or [])))
                st.caption(
                    f"Source `{policy.get('source_file')}` · SHA-256 "
                    f"`{policy.get('checksum')}` · effective from "
                    f"{policy.get('effective_from')}"
                )
                st.markdown(f"**Purpose.** {policy.get('purpose')}")

                st.markdown("**Gate sequence**")
                for gate in sorted(
                    policy.get("gates") or [], key=lambda g: g.get("order", 0)
                ):
                    never = (
                        " · **never auto-passed**" if gate.get("never_auto_pass") else ""
                    )
                    st.markdown(
                        f"{gate.get('order')}. **{gate.get('gate_code')} — "
                        f"{gate.get('gate_name')}** (owner: {gate.get('owner')})"
                        f"{never}  \n"
                        f"_{gate.get('question')}_  \n"
                        "Controls: "
                        + ", ".join(f"`{c}`" for c in gate.get("controls") or [])
                    )

                with st.expander("Controls", expanded=False):
                    st.dataframe(
                        policy.get("controls") or [], width="stretch", hide_index=True
                    )
                with st.expander("Thresholds", expanded=False):
                    st.dataframe(
                        policy.get("thresholds") or [], width="stretch", hide_index=True
                    )
                with st.expander("Decision semantics", expanded=True):
                    for name, meaning in (policy.get("statuses") or {}).items():
                        st.markdown(f"- **{name}** — {meaning}")
                    st.markdown(f"**Gate result rule.** {policy.get('gate_result_rule')}")
                    for name, meaning in (policy.get("decision_semantics") or {}).items():
                        st.markdown(f"- **{name}** — {meaning}")
                with st.expander("Never auto-passed", expanded=True):
                    st.json(policy.get("never_auto_pass") or {}, expanded=True)
                with st.expander("Waiver rules", expanded=False):
                    st.json(policy.get("waiver_rules") or {}, expanded=False)
                with st.expander("Coverage metrics", expanded=False):
                    st.json(policy.get("coverage_metrics") or {}, expanded=False)
                render_caveats(policy.get("limitations"), "Policy limitations")
            st.caption(policies.get("notice", ""))

    # --- run evaluation ----------------------------------------------------- #
    with run_tab:
        run_id = uploaded_run_selector(base_url, key="gates_run_pick")
        if run_id is None:
            return

        try:
            evaluation = fetch_gate_evaluation(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Gate evaluation")
            return

        st.subheader("Gate decisions")
        render_gate_row(evaluation.get("gate_summary") or {})
        st.caption(
            f"Policy `{evaluation.get('policy_profile_id')}` "
            f"v{evaluation.get('policy_version')} · evaluated "
            f"{evaluation.get('evaluated_at')} · deterministic: "
            f"{_fmt(evaluation.get('deterministic'))}"
        )

        for gate in sorted(
            evaluation.get("gates") or [], key=lambda g: g.get("order", 0)
        ):
            icon, kind = GATE_STYLE.get(str(gate.get("status")), ("❔", "info"))
            with st.expander(
                f"{icon} {gate.get('gate_code')} — {gate.get('gate_name')}: "
                f"{gate.get('status')}",
                expanded=str(gate.get("status")) == "BLOCK",
            ):
                st.markdown(f"_{gate.get('question')}_")
                st.markdown(
                    f"**Result.** {gate.get('reason')}  \n"
                    f"**Owner.** {gate.get('owner')}  \n"
                    "**Controls.** "
                    + ", ".join(f"`{c}`" for c in gate.get("control_ids") or [])
                )
                if gate.get("never_auto_pass"):
                    st.info(
                        "This gate is never automatically passed. It requires a human "
                        "decision recorded outside this prototype.",
                        icon="🖐️",
                    )

        if evaluation.get("blocking_controls"):
            st.error(
                "**Blocking controls:** "
                + ", ".join(f"`{c}`" for c in evaluation["blocking_controls"]),
                icon="⛔",
            )
        if evaluation.get("waivers_applied"):
            st.warning(
                "**Waivers applied in this evaluation:** "
                + ", ".join(f"`{w}`" for w in evaluation["waivers_applied"])
                + ". A WAIVE below is an accepted risk, not a satisfied control.",
                icon="🟡",
            )
        if evaluation.get("fairness_gate_notice"):
            st.warning(f"**{evaluation['fairness_gate_notice']}**", icon="⚖️")
        st.info(evaluation.get("release_gate_note", ""), icon="🖐️")

        cols = st.columns(2)
        cols[0].metric(
            "Evidence coverage", _fmt(evaluation.get("evidence_coverage_score"), 3)
        )
        cols[1].metric(
            "Control coverage", _fmt(evaluation.get("control_coverage_score"), 3)
        )
        st.caption(evaluation.get("coverage_metric_caveat", ""))

        with st.expander("Control findings", expanded=False):
            st.dataframe(
                evaluation.get("controls") or [], width="stretch", hide_index=True
            )
        render_caveats(evaluation.get("limitations"), "Evaluation limitations")

        # --- re-evaluation -------------------------------------------------- #
        st.divider()
        st.subheader("Re-evaluate")
        st.caption(
            "Re-runs the policy over this run's stored evidence. No measurement is "
            "recomputed and the model is not re-run. Identical evidence and policy "
            "must produce `changed: false` — that is the determinism check."
        )
        if st.button("♻️ Re-evaluate policy", key="gates_reevaluate"):
            try:
                outcome = GovernanceApiClient(base_url).evaluate_gates(run_id)
            except ApiError as exc:
                show_api_error(exc, "Re-evaluation")
            else:
                st.cache_data.clear()
                if outcome.get("changed"):
                    st.warning(
                        f"**The result changed.** New bundle "
                        f"`{outcome.get('conformity_bundle_id')}`, gates "
                        f"{outcome.get('gate_summary')}.",
                        icon="🔄",
                    )
                else:
                    st.success(
                        f"**Unchanged.** Same verdict and same bundle id "
                        f"`{outcome.get('conformity_bundle_id')}`.",
                        icon="✅",
                    )
                st.caption(
                    "Rewritten: "
                    + ", ".join(f"`{a}`" for a in outcome.get("artifacts_rewritten") or [])
                )

        # --- conformity bundle ---------------------------------------------- #
        st.divider()
        st.subheader("Conformity Bundle")
        try:
            bundle = fetch_conformity_bundle(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Conformity Bundle")
        else:
            cols = st.columns(3)
            cols[0].metric("Bundle id", str(bundle.get("bundle_id")))
            cols[1].metric("Bundle version", str(bundle.get("bundle_version")))
            cols[2].metric(
                "Digital signature",
                "none implemented" if bundle.get("signature") is None else "present",
            )
            st.markdown(
                f"- Model: `{bundle.get('model_name')}` v`{bundle.get('model_version')}`"
                f" · owner {bundle.get('model_owner')}\n"
                f"- Model SHA-256: `{bundle.get('model_checksum')}`\n"
                f"- Dataset: `{bundle.get('dataset_identifier')}` "
                f"({_fmt(bundle.get('dataset_row_count'))} rows)\n"
                f"- Dataset SHA-256: `{bundle.get('dataset_checksum')}`\n"
                f"- Policy: `{bundle.get('policy_profile_id')}` "
                f"v{bundle.get('policy_version')} "
                f"(SHA-256 `{bundle.get('policy_checksum')}`)\n"
                f"- Evidence digest: `{bundle.get('evidence_digest')}`\n"
                f"- Gate sequence: "
                + " → ".join(bundle.get("gate_sequence") or [])
            )
            st.caption(
                "The bundle id is content-addressed: it is derived from the evidence "
                "digest, the policy identity and the gate decisions — never from a "
                "timestamp — so re-evaluating unchanged evidence reproduces the same "
                "id, and any change to the evidence or the verdict produces a "
                "different one."
            )
            st.markdown("**Evidence**")
            st.dataframe(bundle.get("evidence") or [], width="stretch", hide_index=True)
            cols = st.columns(2)
            cols[0].metric(
                "Evidence coverage", _fmt(bundle.get("evidence_coverage_score"), 3)
            )
            cols[1].metric(
                "Control coverage", _fmt(bundle.get("control_coverage_score"), 3)
            )
            st.caption(bundle.get("coverage_metric_caveat", ""))
            st.download_button(
                "⬇️ Download conformity_bundle.json",
                data=json.dumps(bundle, indent=2, sort_keys=True),
                file_name=f"{run_id}_conformity_bundle.json",
                mime="application/json",
                help="Exactly what the API served. Every checksum in it can be "
                "verified independently against the files under runtime/.",
            )
            render_caveats(bundle.get("limitations"), "Bundle limitations")
            render_caveats(
                bundle.get("disclaimers"), "What this bundle is not", icon="⛔"
            )

        # --- traceability --------------------------------------------------- #
        st.divider()
        st.subheader("Traceability matrix")
        st.caption(
            "One row per control: the policy requirement, the artefact evidencing it, "
            "the endpoint that serves it, and the expected against actual checksum."
        )
        try:
            matrix = fetch_traceability(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Traceability")
        else:
            st.dataframe(matrix.get("rows") or [], width="stretch", hide_index=True)
            if matrix.get("unresolved_evidence"):
                st.error(
                    "**Unresolved evidence** (missing or changed since creation):\n"
                    + "\n".join(f"- {u}" for u in matrix["unresolved_evidence"]),
                    icon="🔴",
                )
            else:
                st.success(
                    "Every control's evidence path resolves and every checksum "
                    "matches the value recorded at audit time.",
                    icon="✅",
                )

        # --- waivers -------------------------------------------------------- #
        st.divider()
        st.subheader("Waivers")
        st.warning(
            "A waiver is an **explicit, time-bounded human decision** to accept a "
            "risk. The platform never creates or approves one, nothing here is "
            "defaulted for you, and **a waiver can never satisfy or override the "
            "Release Gate**. Status is recomputed against the clock on every read, so "
            "an expired waiver simply stops applying.",
            icon="🟡",
        )
        try:
            waivers = fetch_waivers(base_url, run_id)
        except ApiError as exc:
            show_api_error(exc, "Waivers")
            waivers = []

        if waivers:
            st.dataframe(waivers, width="stretch", hide_index=True)
            active = [w for w in waivers if w.get("status") == "active"]
            if active:
                labels = [
                    f"{w.get('waiver_id')} · {w.get('control_id')} · expires "
                    f"{w.get('expires_at')}"
                    for w in active
                ]
                target = st.selectbox("Active waiver to revoke", labels, key="revoke_pick")
                if st.button("🚫 Revoke selected waiver", key="revoke_go"):
                    waiver_id = str(active[labels.index(target)].get("waiver_id"))
                    try:
                        outcome = GovernanceApiClient(base_url).revoke_waiver(
                            run_id, waiver_id
                        )
                    except ApiError as exc:
                        show_api_error(exc, "Revocation")
                    else:
                        st.cache_data.clear()
                        st.success(
                            f"`{outcome.get('waiver_id')}` is now "
                            f"**{outcome.get('status')}**. The row is retained rather "
                            "than deleted — a waiver that once applied is part of this "
                            "run's history. Re-evaluate to apply the change.",
                            icon="✅",
                        )
        else:
            st.caption(
                "No waiver has been recorded for this run. None is created "
                "automatically, and none exists for the built-in Adult Income "
                "reference case."
            )

        controls = [
            c.get("control_id")
            for c in evaluation.get("controls") or []
            if c.get("waiver_eligible")
        ]
        with st.expander("Record a waiver", expanded=False):
            if not controls:
                st.caption("No control in this evaluation is waiver-eligible.")
            else:
                with st.form("waiver_form"):
                    control_id = st.selectbox("Control", controls)
                    scope = st.text_input(
                        "Scope",
                        help="Exactly what is being accepted, and for what use.",
                    )
                    owner = st.text_input(
                        "Accountable owner",
                        help="The person accepting this risk by name or role.",
                    )
                    expires_at = st.text_input(
                        "Expires at (ISO-8601)",
                        placeholder="2027-01-01T00:00:00+00:00",
                        help="Must be in the future. A waiver with no expiry is not "
                        "accepted.",
                    )
                    rationale = st.text_area("Rationale")
                    compensating = st.text_area(
                        "Compensating controls (one per line)",
                        help="At least one is required.",
                    )
                    record = st.form_submit_button("🟡 Record waiver")
                if record:
                    payload = {
                        "control_id": control_id,
                        "scope": scope.strip(),
                        "owner": owner.strip(),
                        "expires_at": expires_at.strip(),
                        "rationale": rationale.strip(),
                        "compensating_controls": [
                            line.strip()
                            for line in compensating.splitlines()
                            if line.strip()
                        ],
                    }
                    try:
                        created = GovernanceApiClient(base_url).create_waiver(
                            run_id, payload
                        )
                    except ApiError as exc:
                        show_api_error(exc, "Waiver")
                    else:
                        st.cache_data.clear()
                        st.success(
                            f"Recorded `{created.get('waiver_id')}` against "
                            f"`{created.get('control_id')}`, status "
                            f"**{created.get('status')}**, expiring "
                            f"{created.get('expires_at')}. "
                            "Re-evaluate the policy to apply it.",
                            icon="🟡",
                        )
                        st.caption(created.get("notice", ""))


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
        "New Model Audit": page_new_audit,
        "Uploaded Audit Runs": page_uploaded_audits,
        "Policy Gates & Conformity Bundle": page_policy_gates,
    }
    try:
        renderers[page](base_url)
    except ApiError as exc:  # safety net: never surface a raw traceback
        show_api_error(exc, page)


if __name__ == "__main__":
    main()
