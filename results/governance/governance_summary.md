# Governance Summary and Decision Record

**Subject:** `xgboost_pipeline` — Adult / Census Income classification baseline
**Assessment:** Phase 4 governance risk assessment (AI Governance Platform)
**Date:** 2026-08-10
**Inputs reviewed:** Phase 1 metrics (`results/model_metrics.csv`), Phase 2 fairness
audit (`results/fairness/`), Phase 3 explainability audit
(`results/explainability/`), model card (`results/governance/model_card.md`)
**Companion documents:** `model_card.md`, `governance_risk_register.csv` (12 risks)

No model was retrained and no existing artefact was modified to produce this
assessment.

---

## 1. Decision recommendation

> ### ✅ CONDITIONALLY APPROVED FOR RESEARCH AND EDUCATIONAL USE ONLY
> ### ⛔ BLOCKED FROM REAL-WORLD DEPLOYMENT

The model is fit for its actual purpose — serving as a well-documented,
audited baseline for teaching and governance-methodology development. It is not
fit, and cannot be made fit by tuning alone, for any decision about a real person.

### 1.1 Why research use is approved

1. **Methodologically sound for its stated purpose.** Data-leakage prevention is
   enforced structurally: the split precedes all fitting, every learned
   transformation lives inside the `Pipeline` and is fitted on the training fold
   only, and `pipeline.fit(X_train, y_train)` is the single `fit` call in the
   codebase. Phase 3 independently re-scored the saved artefact and reproduced the
   Phase 1 metrics exactly.
2. **Reproducible end to end.** Deterministic `random_state=42` throughout, an
   immutable raw data snapshot in `data/raw/`, and three commands that regenerate
   every artefact.
3. **Performance is credible and consistent with published results** for this
   dataset (ROC-AUC 0.9299, F1 0.7239), which is itself evidence that no leakage is
   inflating the numbers.
4. **The known risks are documented rather than hidden.** Fairness, explainability
   and limitation caveats travel with the artefacts, and the audits state what they
   do *not* establish as explicitly as what they do.
5. **It has genuine pedagogical value precisely because it is imperfect.** The
   model demonstrates a governance lesson that is hard to teach abstractly: a model
   can show near-zero importance for `sex` and `race` while producing a 0.315
   selection-rate ratio by sex. An audit that had run explainability alone would
   have concluded the protected attributes were barely used, and missed the
   disparity entirely.

### 1.2 Why real-world deployment is blocked

Five of twelve risks are rated **Critical** and seven are **blocking**. The four
independent grounds for blocking, any one of which would be sufficient:

1. **The target is obsolete (R08).** The model encodes a 1994 US wage distribution
   and a 1994 nominal \$50,000 threshold — roughly \$105,000–\$110,000 in 2026
   terms. Applied to any present-day population it would be systematically
   miscalibrated. This is certain, not speculative, and **no amount of retuning
   fixes it** — it requires different data.
2. **Substantial measured output disparities by sex and race (R01, R02).**
   Selection-rate ratios of 0.315 for women and 0.402–0.471 for three of four
   non-reference race groups sit far below the 0.80 four-fifths screening threshold
   conventionally applied in US employment-law analysis. In a consequential setting
   this would represent serious adverse-impact exposure requiring legal review.
3. **Proxy structure makes naive mitigation ineffective (R04).** `sex` and `race`
   rank 10th and 11th of 14 in importance, yet disparities are large, because
   correlated features carry the same information — `relationship` is sex-coded by
   construction (`Husband`/`Wife`) and outranks `sex` itself. Dropping the
   protected attributes would remove the ability to *measure* disparity without
   removing it.
4. **Prerequisites for responsible deployment do not exist (R05–R12).** No cost
   model for the two error types, no justified decision threshold, no calibration
   analysis, no intersectional audit, no drift monitoring, no human-review or
   appeals pathway, no privacy review, and no named accountable owner.

### 1.3 Conditions attached to the research approval

Research use is approved **only** while all of the following hold. Breach of any
condition voids the approval.

- Outputs are used solely for teaching, methodology development, or governance
  demonstration — never to inform a decision about any person.
- Every published metric is accompanied by its disaggregated fairness table. No
  aggregate accuracy figure travels alone.
- The model card's non-intended-use list (§3) accompanies any redistribution of the
  artefact.
- No claim is made or repeated that this model measures ability, productivity or
  merit, or that its outputs are findings about any demographic group.
- Local explanation outputs continue to omit dataset row indices, and no attempt is
  made to re-identify individual records.
- Any change to features, threshold, data or model triggers a re-run of Phases 2–4
  before results are reused.

### 1.4 What would be required to revisit the deployment decision

A future model could be reconsidered — this one cannot. Reconsideration requires,
at minimum: retraining on contemporary jurisdiction-appropriate data with a
documented present-day target; a selected and signed-off fairness criterion; a
threshold derived from an explicit error-cost model; per-group calibration and
bootstrap intervals; an intersectional audit; a privacy/DPIA and legal review; a
defined human-review and appeals pathway; and live monitoring with trigger
thresholds. These are set out in `model_card.md` §11.

---

## 2. Risk profile

**Rating scale.** Likelihood and impact are each rated Low / Medium / High, and
`overall_risk` follows a fixed matrix — High×High = Critical; High×Medium or
Medium×High = High; Medium×Medium, Low×High or High×Low = Medium; otherwise Low.

**Assessment framing.** Impact is rated **as if the model were used for a
consequential decision about individuals**. That is a hypothetical, stated
explicitly so the ratings are not mistaken for observed operational harm — there is
no deployment and therefore no realised harm to date. Ratings are qualitative
expert judgement against documented evidence, not calibrated probabilities.

| overall_risk | count | risk ids |
|---|---:|---|
| **Critical** | 5 | R01, R02, R03, R04, R08 |
| **High** | 6 | R05, R06, R07, R09, R11, R12 |
| **Medium** | 1 | R10 |
| Low | 0 | — |

| status | count |
|---|---:|
| Open — blocking for deployment | 7 |
| Open — control implemented (research scope) | 3 |
| Open — accepted (research only) | 1 |
| Open — not yet assessed | 1 |

**Owners are nominal role placeholders** (Governance Lead, ML Lead, Data Steward,
Fairness Reviewer, Privacy Officer, Deployment Owner). This is an academic project
with no accountable organisational owner — itself a governance gap, and the reason
the Deployment Owner is recorded as unassigned.

---

## 3. Key evidence at a glance

| Finding | Value | Source |
|---|---|---|
| Selected model performance | ROC-AUC 0.9299 · F1 0.7239 · accuracy 0.8782 | Phase 1 |
| Majority-class accuracy floor | 0.7607 | Phase 1 |
| Error asymmetry | 778 false negatives vs 412 false positives (1.9×) | Phase 1 |
| True high earners missed | 33.3% overall; 42.0% of women; 42.1% of Black records | Phases 1–2 |
| Disparate impact — sex | **0.315** (Female vs Male) | Phase 2 |
| Disparate impact — race | 0.428 Black · 0.402 Amer-Indian-Eskimo · 0.471 Other · 1.069 Asian-Pac-Islander | Phase 2 |
| Race groups below four-fifths | 3 of 4 non-reference groups | Phase 2 |
| Equal-opportunity difference | −10.3 pp (sex) · −12.8 pp (race, worst) | Phase 2 |
| Smallest groups | Amer-Indian-Eskimo n=104 (9 positives) · Other n=69 (11 positives) | Phase 2 |
| Protected-attribute importance | `sex` 10/14 (0.0018) · `race` 11/14 (0.0011) | Phase 3 |
| Top feature | `marital-status` (0.0680) — a likely proxy | Phase 3 |
| Sex-coded proxy | `relationship` (`Husband`/`Wife`) outranks `sex` | Phase 3 |
| Label base rates | 30.4% (Male) vs 11.2% (Female) actual `>50K` | Phase 2 |

---

## 4. Cross-phase governance findings

1. **Low protected-attribute importance is not evidence of fairness.** This is the
   single most transferable finding. `sex` at rank 10/14 alongside a 0.315
   selection-rate ratio shows that the two measurements answer different questions.
   Fairness must be tested on **outcomes, disaggregated by group** — never inferred
   from a feature list or an importance ranking.
2. **Explainability and fairness auditing are complementary, not substitutable.**
   Phase 3 alone would have produced a false reassurance; Phase 2 alone would have
   found the disparity without the proxy mechanism that makes it intelligible. Both
   are required.
3. **Disparity is jointly attributable to model and label, and the two cannot be
   separated here.** Group base rates differ in the data itself (30.4% vs 11.2%).
   No method used in any phase can distinguish "the model is unfair" from "the 1994
   labour market recorded in the labels was unequal."
4. **Fairness metrics conflict mathematically.** Women are selected far less often
   *and* receive far fewer false positives, while precision is near-equal. Forcing
   selection-rate parity would raise their false-positive rate. There is no
   configuration that is fair on every metric, so the criterion must be chosen and
   documented, not optimised for silently.
5. **Small groups are where governance is weakest.** The two groups with the worst
   disparate impact are also the two too small to measure reliably. Uncertainty
   should be reported rather than resolved by picking the point estimate.
6. **No finding of discrimination or causation is made anywhere in this
   assessment.** Phase 2 measured outcome differences; Phase 3 measured model
   reliance and per-case attributions. Establishing discrimination would require a
   deployment context, a legal standard and a jurisdiction; establishing causation
   would require a causal model. This assessment has none of those, claims neither,
   and its recommendation rests on documented measurements and absent prerequisites
   rather than on any legal conclusion.

---

## 5. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Preparer (Phases 1–4) | project author | Recommend: research only; block deployment | 2026-08-10 |
| Governance Lead | *unassigned* | *pending* | — |
| Accountable Owner | *unassigned* | *pending* | — |

The absence of an accountable owner is recorded as an open governance gap. In an
organisational setting this document would not be considered complete without one.

**Review trigger:** re-assess on any change to data, features, threshold, model or
intended use, and on any proposal to move beyond research use.

---

*Prepared from the Phase 1–3 artefacts in this repository. All source models,
predictions, fairness and explainability outputs were read only and left
unmodified.*
