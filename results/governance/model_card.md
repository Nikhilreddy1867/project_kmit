# Model Card — Adult / Census Income Classification Baseline

**Model name:** `xgboost_pipeline` (Phase-1 baseline, AI Governance Platform)
**Version:** Phase 1 baseline, unversioned research artefact
**Date of this card:** 2026-08-10
**Card owner:** project author (academic coursework / internal platform prototype)
**Status:** research artefact — **not approved for real-world deployment** (see
`governance_summary.md` for the decision record)

This card follows the structure of *Model Cards for Model Reporting*
(Mitchell et al., 2019). Every quantitative claim is traceable to a file in this
repository; sources are named inline. No model was retrained to produce this card.

---

## 1. Model details

| Item | Value |
|---|---|
| Task | Binary classification: does a person's annual income exceed \$50,000? |
| Selected model | XGBoost (`XGBClassifier`), gradient-boosted trees |
| Artefact | `models/xgboost_pipeline.joblib` (full scikit-learn `Pipeline`) |
| Pipeline contents | `ColumnTransformer` (median impute + one-hot with `handle_unknown="ignore"`) → `XGBClassifier` |
| Key hyperparameters | `n_estimators=400`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `tree_method="hist"`, `random_state=42` |
| Input | 14 original features (6 numeric, 8 categorical), expanded to 105 encoded columns |
| Output | `predict_proba` → P(income > \$50K); class assigned at a **0.5** threshold |
| Training data | 39,073 rows (80% stratified split) |
| Evaluation data | 9,769 held-out rows (20%), never seen during fitting |
| Reproducibility | `python src/train.py` — deterministic (`random_state=42` throughout) |
| Licence / provenance | UCI Machine Learning Repository, dataset id 2 (public research dataset) |

The saved artefact contains **preprocessing and model together**. Inference cannot
accidentally apply different preprocessing than training — there is no train/serve
skew by construction.

---

## 2. Intended purpose

**Primary intended use — academic and educational only.**

This model exists to predict, from historical 1994 US Census records, whether an
individual's recorded annual income exceeds \$50,000. It was built as Phase 1 of a
teaching/prototype AI Governance Platform, and its purpose is to serve as a
**subject of governance analysis**: a realistic, imperfect baseline against which
fairness auditing (Phase 2), explainability auditing (Phase 3) and this risk
assessment (Phase 4) can be demonstrated end to end.

Legitimate uses:

- Coursework, teaching and methodological demonstration.
- Benchmarking of ML pipeline hygiene (leakage prevention, reproducibility).
- A worked example of a fairness and explainability audit, including the
  demonstration that a model can show low protected-attribute importance while
  producing substantial group disparities.
- Internal methodology development for governance tooling.

The intended "users" are the analyst and reviewers of this repository. There is no
end user, no affected data subject, and no decision that this model informs.

---

## 3. Non-intended uses

The following uses are **out of scope and unsupported**. This list is prohibitive,
not illustrative.

**Never use this model for any decision about a real person**, including:

- **Employment decisions** — hiring, screening, shortlisting, promotion,
  compensation setting, salary benchmarking, or workforce planning.
- **Credit and financial decisions** — lending, credit limits, underwriting,
  insurance pricing, affordability or risk scoring.
- **Housing, tenancy or property decisions.**
- **Government, benefits or eligibility determinations**, including means-testing
  or fraud triage.
- **Marketing or pricing segmentation** that targets or excludes individuals by
  inferred income.
- **Any inference about a specific living individual's actual income**, past,
  present or future.
- **Any use as evidence about groups** — e.g. citing model outputs as findings
  about the earnings of a demographic group. The model reproduces its training
  labels; it is not a measurement instrument.

**Specific technical reasons these are unsupported**

1. The model predicts a **1994** income threshold. \$50,000 in 1994 is roughly
   \$105,000–\$110,000 in 2026 terms; the threshold, the wage distribution and the
   occupational structure have all moved. Predictions are not interpretable in
   present-day money.
2. Phase 2 measured selection-rate ratios of **0.315 for women** and **0.402 for
   `Amer-Indian-Eskimo`** relative to reference groups — far below the 0.80
   four-fifths screening threshold conventionally used in US employment-law
   analysis. Deploying this model in a consequential setting would carry serious
   adverse-impact exposure.
3. The model was never validated for any operational purpose, has no calibration
   analysis, no drift monitoring, no human-review pathway and no appeals process.
4. Two features (`sex`, `race`) are protected attributes used directly as model
   inputs. This is deliberate and appropriate for *auditing*, and inappropriate for
   most real decision systems.

---

## 4. Deployment limitations

There is **no deployment context**, and that is itself a limitation on every claim
in this card. Specifically absent:

- No defined decision, decision-maker, or affected population.
- No service-level, latency, availability or scale requirements.
- No human-in-the-loop design, override, contestation or appeals mechanism.
- No monitoring, alerting, logging or incident-response plan in operation.
- No Data Protection Impact Assessment, legal review or stakeholder consultation.
- No cost model for the two error types (a false negative and a false positive are
  treated as equally undesirable by the 0.5 threshold, which is almost never true
  in a real application).

Because harm is a function of deployment, the risk ratings in
`governance_risk_register.csv` are assessed **as if** this model were used for a
consequential decision about individuals. That is a hypothetical framing, stated
explicitly so the ratings are not mistaken for observed operational harm.

---

## 5. Data: provenance and historical context

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset id 2 (`Adult` / Census Income), retrieved via `ucimlrepo` |
| Raw snapshot | `data/raw/adult_raw.csv` — 48,842 rows × 15 columns, saved verbatim as an audit trail |
| Metadata snapshot | `data/raw/adult_metadata.txt` |
| Origin | Extracted by Barry Becker from the **1994 US Census** database |
| Target | `income`, encoded `>50K` = 1, `<=50K` = 0 |
| Class balance | 23.93% positive (`>50K`); 76.07% negative |
| Missing data | 6,465 cells after `?` → null conversion: `occupation` 2,809, `workclass` 2,799, `native-country` 857 |

**Label normalisation note.** The published file mixes `<=50K` with `<=50K.` — the
trailing period originates in the original `adult.test` split. Both spellings were
collapsed during cleaning; without this the target would silently have had four
levels instead of two.

### 5.1 1994 historical context — why this matters for every downstream claim

The target variable is not a neutral measurement of ability, productivity or merit.
It records **who was paid more than \$50,000 in the United States in 1994**, and is
therefore an artefact of that labour market, including its inequalities:

- Observed base rates differ sharply by group in the data itself: **30.4% of men**
  versus **11.2% of women** in the test set have recorded income above \$50K; by
  race, **25.4% (White)** versus **12.8% (Black)**, **8.7%
  (`Amer-Indian-Eskimo`)** and **15.9% (`Other`)**.
- Those gaps reflect the historical distribution of occupations, hours, promotion,
  pay-setting practices, education access, and discrimination in that period. The
  dataset records the *outcome*; it does not record the *causes*.
- A model that learns this target faithfully will reproduce those gaps. **High
  accuracy therefore means fidelity to a historically unequal world, not
  fairness in it.** Improving the model cannot correct the label.
- Categories are of their time and coarse: `race` has five fixed values, and `sex`
  is recorded as a binary, which erases non-binary people entirely. Neither is
  self-identified in any modern sense.
- `fnlwgt` is a census **sampling weight**, not a personal attribute. It was
  retained as a feature to avoid a silent methodological change, but it has no
  individual-level meaning and should not be interpreted as one.

**Consequence for interpretation:** every disparity reported below is jointly
attributable to the model *and* to the historical labels it was trained on. No
method used in Phases 1–3 can separate those two sources.

---

## 6. Model comparison and selected baseline

Three models were trained on identical folds with identical preprocessing logic,
no hyperparameter tuning, and no class re-weighting or resampling — so the
comparison is like-for-like. Source: `results/model_metrics.csv`.

| model | accuracy | precision | recall | F1 | ROC-AUC | fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| **xgboost (selected)** | **0.8782** | **0.7911** | **0.6672** | **0.7239** | **0.9299** | 7.2 |
| random_forest | 0.8684 | 0.7834 | 0.6219 | 0.6934 | 0.9186 | 3.6 |
| logistic_regression | 0.8507 | 0.7314 | 0.5941 | 0.6557 | 0.9042 | 0.8 |

**Selection rationale.** XGBoost leads on every metric, and — per Phase 2 — was
also the *least* disparate of the three on most fairness measures (lowest
equalised-odds difference: 0.103 by sex, 0.128 by race). No accuracy-versus-fairness
trade-off had to be made in this particular ranking. That is a convenient property
of this dataset and model set, **not a general rule**, and it should not be cited as
evidence that accuracy and fairness align in general.

Logistic Regression is retained as an interpretable comparison model; its signed
coefficients are used in the Phase 3 explainability audit.

### 6.1 Data-leakage controls (why these numbers are trustworthy)

The reported scores are believable because the split precedes all fitting:

1. The train/test split is performed on cleaned but **untransformed** data.
2. Every learned transformation (imputation medians, one-hot vocabulary, scaler
   statistics) lives inside the `Pipeline` and is fitted on the training fold only.
3. `pipeline.fit(X_train, y_train)` is the only `fit` call in the codebase; the test
   set is touched exclusively through `predict` / `predict_proba`.
4. Phase 3 independently re-scored the saved artefact and reproduced the Phase 1
   metrics exactly, confirming the artefact and the reported numbers agree.

---

## 7. Performance metrics

Held-out test set, 9,769 rows, positive class `>50K`, threshold 0.5.

| Metric | Value | Reading |
|---|---:|---|
| Accuracy | 0.8782 | Against a 0.7607 majority-class floor — a "predict everyone `<=50K`" model scores 0.7607, so accuracy alone is close to uninformative here |
| Precision | 0.7911 | Of those flagged `>50K`, ~79% truly are |
| Recall / TPR | 0.6672 | Of true high earners, ~67% are found |
| F1 | 0.7239 | Preferred headline metric on this imbalanced target |
| ROC-AUC | 0.9299 | Threshold-free ranking quality; the only metric here unaffected by the 0.5 choice |

**Confusion matrix:** TN 7,019 · FP 412 · FN 778 · TP 1,560.

### 7.1 Error patterns

- **Errors are strongly asymmetric.** 778 false negatives against 412 false
  positives. The model is roughly **1.9× more likely to miss a true high earner
  than to wrongly flag a low earner**. This is the expected consequence of a 0.5
  threshold on a 24%-positive target, not a bug.
- **One in three true high earners is missed** (778 of 2,338 actual positives,
  33.3%). Any application where a false negative denies someone something would
  concentrate that harm on real high earners.
- **The miss rate is unevenly distributed** (Phase 2, computed from group TPRs):

  | group | true `>50K` in test | missed (FN) | miss rate |
  |---|---:|---:|---:|
  | Male | 1,971 | 624 | 31.7% |
  | **Female** | 367 | 154 | **42.0%** |
  | White | 2,119 | 691 | 32.6% |
  | Black | 121 | 51 | 42.1% |
  | Asian-Pac-Islander | 78 | 27 | 34.6% |
  | Amer-Indian-Eskimo | 9 | 4 | 44.4% |
  | Other | 11 | 5 | 45.5% |

  The last two rows rest on **9 and 11 actual positives respectively** and must not
  be ranked (see §9.3 and §10).
- **False positives are rarer in the disadvantaged groups.** FPR is 0.020 for women
  versus 0.078 for men. The model is *less* likely to wrongly promote someone into
  `>50K` in those groups — the arithmetic flip side of selecting them less often.
- **Precision is nearly equal across sex** (0.792 male vs 0.783 female). So
  predictive parity approximately holds while demographic parity fails badly — a
  concrete instance of the metric conflict in §9.4.

---

## 8. Fairness findings

Full detail: `results/fairness/fairness_report.md`,
`results/fairness/fairness_metrics_by_group.csv`. Reference groups are the largest
by sample count (`Male`, `White`) — conventional for disparate-impact analysis, and
**not** an assertion that the majority group's treatment is correct.

### 8.1 By sex (XGBoost)

| group | n | actual `>50K` rate | selection rate | TPR | FPR | precision | DI ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Male (ref) | 6,480 | 0.3042 | 0.2623 | 0.6834 | 0.0783 | 0.7924 | 1.000 |
| Female | 3,289 | 0.1116 | 0.0827 | 0.5804 | 0.0202 | 0.7831 | **0.315** |

- Demographic parity difference: **−18.0 pp**; disparate impact ratio **0.315**.
- Equal opportunity (TPR) difference: **−10.3 pp**.
- Equalised odds: TPR −10.3 pp, FPR −5.8 pp; equalised-odds difference **0.103**.
- The pattern is consistent across all three models (DI ratio 0.299 / 0.307 /
  0.315), so it is a property of the data-and-task, not of one algorithm.

### 8.2 By race (XGBoost)

| group | n | actual `>50K` rate | selection rate | TPR | FPR | precision | DI ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| White (ref) | 8,348 | 0.2538 | 0.2155 | 0.6739 | 0.0596 | 0.7938 | 1.000 |
| Black | 944 | 0.1282 | 0.0922 | 0.5785 | 0.0207 | 0.8046 | **0.428** |
| Asian-Pac-Islander | 304 | 0.2566 | 0.2303 | 0.6538 | 0.0841 | 0.7286 | 1.069 |
| Amer-Indian-Eskimo ⚠ | 104 | 0.0865 | 0.0865 | 0.5556 | 0.0421 | 0.5556 | **0.402** |
| Other ⚠ | 69 | 0.1594 | 0.1014 | 0.5455 | 0.0172 | 0.8571 | **0.471** |

- **Three of four non-reference race groups fall below the 0.80 four-fifths
  threshold.** `Asian-Pac-Islander` is the exception and slightly *exceeds* parity
  (1.069) — the pattern is not a simple majority/minority ordering.
- Worst equal-opportunity difference: **−12.8 pp** (`Other`).
- ⚠ flags groups with fewer than 200 test records.

### 8.3 Interpretation discipline

**What is established:** the model's *outputs* differ substantially by sex and by
race on this test set, in the direction and magnitude tabulated above. These are
direct measurements and are not in dispute.

**What is not established:**

- **Not discrimination.** These metrics describe *what* differs, never *why*. This
  card makes **no claim that the model is unlawfully or legally discriminatory**.
  Such a determination requires a deployment context, a legal standard, a
  jurisdiction, and a causal analysis — none of which exist here.
- **Not causation.** No fairness metric in Phase 2 identifies a causal effect of
  sex or race on the prediction, and none is claimed.
- **Not attributable to the model alone.** Base rates differ in the labels
  themselves (30.4% vs 11.2% by sex). A model that merely reproduces that gap will
  fail demographic parity while being "right" in a narrow accuracy sense. Whether
  reproducing a historical gap is acceptable is a policy question.
- **The four-fifths rule is a screening trigger, not a verdict.** Failing it
  conventionally prompts scrutiny; it is not a statistical test and establishes no
  legal conclusion.

---

## 9. Explainability findings and proxy-feature concerns

Full detail: `results/explainability/explainability_report.md`. Permutation
importance was computed on the **raw input columns through the whole pipeline**
(so importance attaches to human-readable features, not one-hot fragments), scored
on the held-out test set over 10 repeats. Local attributions use XGBoost's exact
TreeSHAP; additivity was verified numerically to < 1e-4.

### 9.1 Global importance (drop in test ROC-AUC when shuffled)

| # | feature | ΔROC-AUC | ± std | class |
|---:|---|---:|---:|---|
| 1 | `marital-status` | 0.0680 | 0.0019 | 🟡 likely proxy |
| 2 | `capital-gain` | 0.0643 | 0.0027 | financial |
| 3 | `age` | 0.0451 | 0.0028 | demographic |
| 4 | `education-num` | 0.0249 | 0.0017 | 🟡 likely proxy |
| 5 | `capital-loss` | 0.0167 | 0.0013 | financial |
| 6 | `hours-per-week` | 0.0128 | 0.0007 | 🟡 likely proxy |
| 7 | `occupation` | 0.0125 | 0.0011 | 🟡 likely proxy |
| 8 | `relationship` | 0.0057 | 0.0010 | 🟡 likely proxy (sex-coded) |
| 9 | `workclass` | 0.0026 | 0.0004 | — |
| 10 | **`sex`** | **0.0018** | 0.0003 | 🔴 protected |
| 11 | **`race`** | **0.0011** | 0.0003 | 🔴 protected |
| 12 | `fnlwgt` | 0.0008 | 0.0007 | sampling weight |
| 13 | `native-country` | 0.0006 | 0.0003 | 🟡 likely proxy |
| 14 | `education` | −0.0002 | 0.0002 | 🟡 diluted duplicate |

### 9.2 The central proxy finding

`sex` ranks **10/14** and `race` ranks **11/14** — near the bottom. Shuffling either
barely changes performance. A naive reading would be "the model hardly uses them,
so it is fair." **Phase 2 measured the opposite**: a 0.315 selection-rate ratio by
sex and three race groups below the four-fifths threshold.

These coexist without contradiction:

1. **Redundant encoding.** `marital-status` (#1), `hours-per-week` (#6),
   `occupation` (#7) and `relationship` (#8) all outrank both protected
   attributes, and jointly carry much of their information. Permutation importance
   measures the *incremental* contribution of a column given the others, so it
   correctly reports a small number while the information remains fully in use.
2. **`relationship` is sex-coded by construction** — its categories include
   `Husband` and `Wife`. It carries `sex` explicitly, under another name, and
   outranks `sex` itself.
3. **`education` ranks last with a slightly negative score**, which is not evidence
   that education is irrelevant: it is a duplicate of `education-num` (#4), so the
   model recovers the signal from the twin column. This is correlated-feature
   dilution caught in the act, and a worked demonstration that a low importance
   score must never be read as "unused".

**Therefore: removing `sex` and `race` would not be a mitigation.** It would leave
the proxies — and very likely the disparities — while destroying the ability to
measure them. "Fairness through unawareness" fails here for reasons visible in the
table above.

### 9.3 Association, not causation

Every number in Phase 3 is **associational**. Permutation importance and SHAP
describe how a *fitted function* responds to its inputs on a *particular dataset*;
neither is a causal estimand. "`marital-status` is the most important feature" means
the model's accuracy depends on that column — **not** that marriage raises income,
and not that changing someone's marital status would change their earnings. A SHAP
value of +0.87 is a statement about the model's arithmetic, not about the world.

A local example makes the point concretely: in the audited false-negative case,
`relationship = Wife` pushed the prediction **+0.58 toward `>50K`**, not against it;
the downward push came from `age`, `fnlwgt` and `capital-gain`. Sex-coded features
do not act in one uniform direction, and single-case attributions cannot be
generalised.

---

## 10. Limitations

Consolidated from Phases 1–3. Each is a real constraint on what this model and its
audits can support.

1. **Historical labels.** The target encodes 1994 US wage outcomes, including that
   era's inequality. The model can be an accurate predictor of an unequal world;
   accuracy is fidelity to the label, not fairness. No mitigation applied to the
   model can correct the label. (§5.1)
2. **Small-group uncertainty.** `Amer-Indian-Eskimo` (n = 104) and `Other`
   (n = 69) are flagged. Their TPRs rest on **9 and 11 actual positives**, with 95%
   Wilson intervals of **±28.9 pp** and **±26.5 pp**. Their apparent ordering is
   noise. **Do not rank these groups on point estimates**; conclusions about them
   need bootstrap intervals or more data.
3. **Single split, single seed.** All results come from one 9,769-row test set with
   `random_state=42`. Nothing is averaged over repeated splits, and no confidence
   intervals were computed on the fairness *differences* themselves. Adjacent
   importance ranks have overlapping error bars.
4. **Threshold dependence.** Every selection rate, TPR, FPR, precision, recall, F1
   and every fairness gap above is measured at the default **0.5** cut-off, which
   was chosen for nothing in particular. These quantities — and the size of the
   disparities — would change under a different threshold. Only ROC-AUC is
   threshold-free.
5. **Correlation versus causation.** No phase establishes a causal relationship,
   between features and income or between group membership and model output. The
   audits measure association and model reliance only. (§8.3, §9.3)
6. **No deployment context.** No decision, population, human-review path, appeal
   route, or cost model for errors exists. Risk ratings are therefore hypothetical
   (§4).
7. **No intersectional analysis.** `sex` and `race` were audited one at a time. A
   Black-female subgroup gap could exceed either marginal gap and would be
   invisible here. Cell sizes shrink quickly, which is why it was not attempted —
   not an argument that it does not matter.
8. **No calibration analysis.** Per-group reliability of `predicted_probability`
   was not assessed; calibration is the third leg of the fairness impossibility
   result and remains unexamined.
9. **Fairness metrics are mutually incompatible.** Where base rates differ between
   groups — they do here — no classifier can simultaneously equalise demographic
   parity, equalised odds and calibration except in degenerate cases (Kleinberg et
   al. 2016; Chouldechova 2017). The audit shows the trade-off directly: the groups
   selected least often also receive the fewest false positives. **There is no
   configuration that is fair on every metric**, so a criterion must be chosen and
   documented rather than optimised for silently.
10. **Explainability caveats.** Permutation importance is diluted by correlated
    features (so scores are lower bounds) and scores the model partly off-manifold,
    since shuffling creates impossible records such as `relationship = Wife` with
    `sex = Male`. TreeSHAP is exact but Shapley attribution is one choice among
    several; log-odds contributions are not percentage points. No interaction
    values were computed.
11. **Coarse and dated categories.** Five fixed `race` values; binary `sex`;
    `native-country` dominated by one level. Non-binary people are absent from the
    data entirely.
12. **Modelling choices left deliberately untouched** and therefore unexamined:
    `fnlwgt` and `education-num` retained, duplicate rows not removed, no class
    re-weighting or resampling, no hyperparameter tuning, no cross-validation.

---

## 11. Monitoring recommendations (hypothetical — if this were ever deployed)

Presented as what *would be required*, not as controls that exist. Nothing in this
list is currently implemented. **None of it would make this particular model
deployable** — the 1994 labels alone rule that out (§3). It is specified so the
governance pattern is reusable for a model that legitimately could be.

**Pre-deployment gates (all blocking)**

1. Re-train on **contemporary, jurisdiction-appropriate** data with a documented,
   present-day target definition. Do not port a 1994 threshold.
2. Choose, justify and **document a single fairness criterion** before mitigation
   (they conflict — §10.9), with sign-off from an accountable owner.
3. Sweep the decision threshold and publish the **fairness/accuracy frontier**
   rather than inheriting 0.5; set the threshold from the real cost of each error
   type.
4. Add **per-group calibration curves** and bootstrap confidence intervals on every
   fairness gap, with a minimum-sample rule so small groups are not reported as
   point estimates.
5. Complete an **intersectional** audit, a privacy/DPIA review, and a legal review
   in the deployment jurisdiction.
6. Define the **human-review, contestation and appeals** pathway, and the
   remediation owed to a person harmed by an error.

**Ongoing monitoring**

| What | Metric | Cadence | Trigger |
|---|---|---|---|
| Input drift | PSI / KL per feature vs training reference | Weekly | PSI > 0.2 on any top-5 feature |
| Prediction drift | Overall selection rate vs baseline | Daily | ±20% relative change |
| Performance | ROC-AUC, F1, precision, recall on labelled outcomes | Monthly, as labels arrive | AUC drop > 0.02 |
| **Fairness, disaggregated** | selection rate, TPR, FPR, precision per group; DI ratio; equal-opportunity and equalised-odds differences | Monthly | DI ratio < 0.80, or any gap worsening > 5 pp |
| Small groups | n per group per window | Monthly | n < 200 → suppress point estimate, widen window |
| Calibration | per-group reliability, Brier score | Quarterly | material per-group divergence |
| Explainability stability | permutation-importance rank churn | Quarterly | protected attribute or new proxy entering top 5 |
| Label integrity | label-source audit, label delay, feedback-loop check | Quarterly | any evidence outputs influence future labels |
| Error-cost review | FN/FP counts and their real consequences | Quarterly | asymmetry diverges from the assumed cost model |

**Governance requirements**

- Model registry with versioned artefacts, data snapshot hashes and the exact split
  seed (this project's `data/raw/` snapshot and `random_state=42` are the pattern).
- Immutable audit trail: every prediction logged with model version, input version,
  probability and threshold.
- Scheduled **re-audit of Phases 2–4 on live data**, not just on the original test
  set, plus re-audit on any retrain, feature change or threshold change.
- A named accountable owner, a documented kill-switch, and a rollback plan.
- Incident process for a fairness-threshold breach, with a defined
  suspend-versus-continue decision right.

---

## 12. Related documents

| Document | Contents |
|---|---|
| `results/model_metrics.csv` | Phase 1 model comparison |
| `results/fairness/fairness_report.md` | Phase 2 fairness audit |
| `results/explainability/explainability_report.md` | Phase 3 explainability audit |
| `results/governance/governance_risk_register.csv` | Phase 4 risk register (12 risks) |
| `results/governance/governance_summary.md` | Phase 4 executive summary and **decision record** |
| `README.md` | Setup and reproduction instructions |

**Reproduction:** `python src/train.py` → `python src/fairness_audit.py` →
`python src/explainability_audit.py`. Deterministic throughout
(`random_state=42`).

---

*This card documents observed measurements and their limits. It deliberately makes
no legal determination about discrimination and no causal claim about any feature or
group.*
