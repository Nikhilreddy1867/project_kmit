# Phase 2 — Fairness Audit: Adult Income Models

Group fairness audit of the 3 Phase-1 models (`logistic_regression`, `random_forest`, `xgboost`) on the held-out test set of **9,769 records**, disaggregated by **sex** and **race**.

This audit is **read-only**: it consumes the prediction CSVs written in Phase 1 and does not retrain, reload or modify any model. Every number below is reproducible from `results/fairness/fairness_metrics_by_group.csv`.

## 1. Method

- **Positive class** = `>50K` (encoded 1). A "selection" means the model predicted a person into the high-income class.
- **Reference groups** are the largest by sample count, per spec: **Male** for sex (n = 6,480) and **White** for race (n = 8,348). Choosing the majority group as the yardstick is conventional for disparate-impact analysis and gives the most precisely estimated reference. It is **not** an assertion that the majority group's treatment is correct or desirable.
- **Decision threshold** = 0.5, inherited unchanged from Phase 1.
- `recall` and `TPR` are the same quantity by definition; both appear because the schema asks for both. Likewise the **disparate impact ratio** and the **demographic parity ratio** are one formula (group selection rate ÷ reference selection rate) under two vocabularies — one measurement, reported under both names.
- 95% **Wilson score intervals** accompany selection rate and TPR so that a gap can be compared against its own sampling uncertainty.

## 2. Observed group metrics

### 2.1 By `sex`

**logistic_regression**

| group | n | actual +rate | selection rate | TPR | FPR | precision | F1 | DP diff | DI ratio | EqOpp diff | EqOdds FPR diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Male** (ref) | 6,480 | 0.304 | 0.254 | 0.613 | 0.098 | 0.733 | 0.667 | +0.0pp | 1.000 | +0.0pp | +0.0pp |
| Female | 3,289 | 0.112 | 0.076 | 0.493 | 0.024 | 0.724 | 0.587 | -17.8pp | 0.299 | -12.0pp | -7.4pp |

**random_forest**

| group | n | actual +rate | selection rate | TPR | FPR | precision | F1 | DP diff | DI ratio | EqOpp diff | EqOdds FPR diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Male** (ref) | 6,480 | 0.304 | 0.248 | 0.639 | 0.077 | 0.784 | 0.704 | +0.0pp | 1.000 | +0.0pp | +0.0pp |
| Female | 3,289 | 0.112 | 0.076 | 0.531 | 0.019 | 0.780 | 0.632 | -17.2pp | 0.307 | -10.7pp | -5.8pp |

**xgboost**

| group | n | actual +rate | selection rate | TPR | FPR | precision | F1 | DP diff | DI ratio | EqOpp diff | EqOdds FPR diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Male** (ref) | 6,480 | 0.304 | 0.262 | 0.683 | 0.078 | 0.792 | 0.734 | +0.0pp | 1.000 | +0.0pp | +0.0pp |
| Female | 3,289 | 0.112 | 0.083 | 0.580 | 0.020 | 0.783 | 0.667 | -18.0pp | 0.315 | -10.3pp | -5.8pp |

⚠ = group smaller than 200 test records; see limitations.

### 2.2 By `race`

**logistic_regression**

| group | n | actual +rate | selection rate | TPR | FPR | precision | F1 | DP diff | DI ratio | EqOpp diff | EqOdds FPR diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **White** (ref) | 8,348 | 0.254 | 0.209 | 0.605 | 0.074 | 0.736 | 0.664 | +0.0pp | 1.000 | +0.0pp | +0.0pp |
| Black | 944 | 0.128 | 0.069 | 0.397 | 0.021 | 0.738 | 0.516 | -14.0pp | 0.330 | -20.8pp | -5.3pp |
| Asian-Pac-Islander | 304 | 0.257 | 0.266 | 0.654 | 0.133 | 0.630 | 0.642 | +5.8pp | 1.278 | +4.9pp | +5.9pp |
| Amer-Indian-Eskimo ⚠ | 104 | 0.087 | 0.067 | 0.556 | 0.021 | 0.714 | 0.625 | -14.1pp | 0.323 | -4.9pp | -5.3pp |
| Other ⚠ | 69 | 0.159 | 0.072 | 0.364 | 0.017 | 0.800 | 0.500 | -13.6pp | 0.347 | -24.1pp | -5.7pp |

**random_forest**

| group | n | actual +rate | selection rate | TPR | FPR | precision | F1 | DP diff | DI ratio | EqOpp diff | EqOdds FPR diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **White** (ref) | 8,348 | 0.254 | 0.202 | 0.626 | 0.058 | 0.787 | 0.697 | +0.0pp | 1.000 | +0.0pp | +0.0pp |
| Black | 944 | 0.128 | 0.083 | 0.521 | 0.018 | 0.808 | 0.633 | -11.9pp | 0.409 | -10.6pp | -4.0pp |
| Asian-Pac-Islander | 304 | 0.257 | 0.253 | 0.692 | 0.102 | 0.701 | 0.697 | +5.1pp | 1.253 | +6.6pp | +4.4pp |
| Amer-Indian-Eskimo ⚠ | 104 | 0.087 | 0.077 | 0.556 | 0.032 | 0.625 | 0.588 | -12.5pp | 0.381 | -7.1pp | -2.6pp |
| Other ⚠ | 69 | 0.159 | 0.087 | 0.455 | 0.017 | 0.833 | 0.588 | -11.5pp | 0.430 | -17.2pp | -4.1pp |

**xgboost**

| group | n | actual +rate | selection rate | TPR | FPR | precision | F1 | DP diff | DI ratio | EqOpp diff | EqOdds FPR diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **White** (ref) | 8,348 | 0.254 | 0.216 | 0.674 | 0.060 | 0.794 | 0.729 | +0.0pp | 1.000 | +0.0pp | +0.0pp |
| Black | 944 | 0.128 | 0.092 | 0.579 | 0.021 | 0.805 | 0.673 | -12.3pp | 0.428 | -9.5pp | -3.9pp |
| Asian-Pac-Islander | 304 | 0.257 | 0.230 | 0.654 | 0.084 | 0.729 | 0.689 | +1.5pp | 1.069 | -2.0pp | +2.5pp |
| Amer-Indian-Eskimo ⚠ | 104 | 0.087 | 0.087 | 0.556 | 0.042 | 0.556 | 0.556 | -12.9pp | 0.402 | -11.8pp | -1.7pp |
| Other ⚠ | 69 | 0.159 | 0.101 | 0.545 | 0.017 | 0.857 | 0.667 | -11.4pp | 0.471 | -12.8pp | -4.2pp |

⚠ = group smaller than 200 test records; see limitations.

## 3. Headline gaps

| model | attribute | ref | worst DP diff (group) | min DI ratio (group) | worst EqOpp diff (group) | worst EqOdds FPR diff | EqOdds diff | # failing 4/5 |
|---|---|---|---|---|---|---|---:|---:|
| logistic_regression | sex | Male | -17.8pp (Female) | 0.299 (Female) | -12.0pp (Female) | -7.4pp (Female) | 0.120 | 1 |
| logistic_regression | race | White | -14.1pp (Amer-Indian-Eskimo) | 0.323 (Amer-Indian-Eskimo) | -24.1pp (Other) | +5.9pp (Asian-Pac-Islander) | 0.241 | 3 |
| random_forest | sex | Male | -17.2pp (Female) | 0.307 (Female) | -10.7pp (Female) | -5.8pp (Female) | 0.107 | 1 |
| random_forest | race | White | -12.5pp (Amer-Indian-Eskimo) | 0.381 (Amer-Indian-Eskimo) | -17.2pp (Other) | +4.4pp (Asian-Pac-Islander) | 0.172 | 3 |
| xgboost | sex | Male | -18.0pp (Female) | 0.315 (Female) | -10.3pp (Female) | -5.8pp (Female) | 0.103 | 1 |
| xgboost | race | White | -12.9pp (Amer-Indian-Eskimo) | 0.402 (Amer-Indian-Eskimo) | -12.8pp (Other) | -4.2pp (Other) | 0.128 | 3 |

## 4. Observed disparities vs. conclusions about discrimination

### 4.1 What the numbers *do* establish

These are direct measurements of model output on this test set, and they are not in dispute:

1. **Every model selects women at a far lower rate than men.** The disparate impact ratio for `Female` ranges 0.299–0.315 across the three models — all well below the 0.80 four-fifths screening threshold.
2. **Every model finds true high-earning women less often than true high-earning men** (negative equal-opportunity difference in all 3 model/sex rows).
3. **Race-based selection-rate gaps are present but smaller**, and are not uniform across groups: the worst race DI ratio per model is 0.323–0.402.
4. **False-positive rates are consistently lower for the disadvantaged groups.** The models are *less* likely to wrongly promote someone into `>50K` in those groups — the flip side of selecting them less often. A single-metric audit would have missed this.

### 4.2 What the numbers do *not* establish

**None of the above is, by itself, proof of discrimination.** Specifically:

- **A disparity is not a mechanism.** These metrics say *what* differs, never *why*. Distinguishing "the model treats similar people differently" from "the groups differ in the recorded features" requires counterfactual or causal analysis that is out of scope here.
- **The base rates genuinely differ in the data.** In this test set the actual `>50K` rate is 0.304 for men and 0.112 for women. A model that merely reproduces that gap will fail demographic parity while being, in a narrow accuracy sense, "right". Whether reproducing a historical gap is acceptable is a policy question, not a statistical one.
- **The label is not ground truth about merit.** `income > 50K` in 1994 census data records the outcome of a labour market that was itself unequal. Learning it faithfully means inheriting that inequality — the model can be an accurate predictor of a biased world.
- **Legal thresholds are screening tools, not verdicts.** Failing the four-fifths rule triggers scrutiny under US employment-law convention; it is not a statistical test, and it does not establish unlawful discrimination.
- **This dataset is not a hiring or lending system.** Adult/Census Income is a benchmark. Real-world harm depends on deployment context, which does not exist here.

## 5. Limitations

### 5.1 Small group sizes

Group metrics are only as stable as the group is large. Flagged groups (n < 200):

| attribute | group | n | 95% CI half-width on selection rate |
|---|---|---:|---:|
| race | Amer-Indian-Eskimo | 104 | ±6.8pp |
| race | Other | 69 | ±9.0pp |

For these groups a difference of several percentage points is not distinguishable from sampling noise, and TPR is worse still because it is computed on the handful of actual positives only. **Do not rank these groups by their point estimates.** Any conclusion about them needs bootstrap intervals or a larger sample.

### 5.2 Historical bias in census data

The data is the 1994 US Census extract. The label encodes real historical wage inequality between demographic groups, so the target itself is value-laden. Correcting the model cannot correct the label, and reported "accuracy" measures fidelity to a biased world, not fairness in it. Nothing here transfers unexamined to the present day.

### 5.3 Proxy variables

Removing `sex` and `race` from the features would **not** make the model blind to them. `relationship` contains the sex-coded categories `Husband` and `Wife`; `marital-status`, `occupation`, `hours-per-week`, `education` and `native-country` all correlate with the protected attributes and jointly reconstruct them. Fairness-through-unawareness therefore removes the ability to *measure* disparity without removing the disparity — which is exactly why Phase 1 retained both attributes.

### 5.4 The 0.5 decision threshold

Every disparity above is measured at the default 0.5 cut-off, which was never chosen for fairness — or for anything else. Selection rates, TPR and FPR are all threshold-dependent, so these gaps would change, and could partly close, under a different or per-group threshold. The ranking-quality metric (ROC-AUC, Phase 1) is threshold-free and would not move. Treat threshold choice as an open governance decision, and note that per-group thresholds are themselves legally and ethically contested.

### 5.5 Fairness metrics conflict — mathematically, not incidentally

The metrics in this report cannot all be satisfied at once. When base rates differ between groups (they do here), it is a **proven impossibility** that a classifier equalises demographic parity, equalised odds and calibration simultaneously except in degenerate cases (Kleinberg et al. 2016; Chouldechova 2017). This audit is visible evidence of the trade-off: the same groups that are selected less often also receive fewer false positives. Forcing selection rates to parity would raise their false-positive rates. **There is no configuration that is fair on every metric**, so the platform must state which definition it is committing to, and why, rather than optimising a single number.

### 5.6 Further scope limits

- **No intersectional analysis.** `sex` and `race` are audited one at a time; a Black-female subgroup gap could be larger than either marginal gap and would be invisible here. Cell sizes shrink fast, which is the reason it was not attempted rather than an argument that it does not matter.
- **Point estimates only** for the headline gaps; no significance testing or bootstrap CIs on the *differences* themselves.
- **`race` uses the dataset's own five fixed categories**, which are coarse, of their time, and not self-identified in any modern sense. `sex` is recorded as a binary, which erases non-binary people entirely — an artefact of the 1994 instrument, and a limitation of the audit as much as of the data.
- **Single split.** All numbers come from one 9,769-row test set with `random_state=42`; they are not averaged over repeated splits.
- **No calibration analysis** (per-group reliability of `predicted_probability`), which is the third leg of the impossibility result and a natural next step.

## 6. Recommended next steps

1. **Choose and document a fairness criterion** before mitigating anything — the metrics conflict, so this is a policy decision, not an optimisation.
2. **Add per-group calibration curves** for `predicted_probability`.
3. **Bootstrap the gaps** to get confidence intervals on the differences, which is what the small groups actually need.
4. **Sweep the decision threshold** and plot the fairness/accuracy frontier instead of accepting 0.5.
5. **Then** evaluate mitigations (reweighing, threshold optimisation, constrained training), re-auditing after each — and re-check that a mitigation does not simply move the harm to another metric or group.

---

Generated by `src/fairness_audit.py`. Phase 1 models and outputs unmodified.
