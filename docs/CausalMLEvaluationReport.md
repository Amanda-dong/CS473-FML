# Causal ML Evaluation Report

Updated: 2026-04-29

## Purpose

This report documents the first end-to-end test of the causal machine learning evaluation framework added to the repository. The goal of this work was to move beyond predictive evaluation and test whether a model-guided intervention appears to cause improved outcomes over time.

The implementation supports:

- uplift curve generation
- Qini coefficient calculation
- average treatment effect (ATE) estimation
- uplift at top decile
- policy risk and policy value evaluation
- temporal backtesting with expanding or rolling windows
- covariate balance checks using standardized mean difference (SMD)
- optional sensitivity analysis
- optional MLflow logging

## Code Added

The following files were added or updated:

- [src/validation/causal.py](/Users/Catherine/Desktop/CS473-FML/src/validation/causal.py)
  - core causal evaluation logic
- [src/validation/run_causal_evaluation.py](/Users/Catherine/Desktop/CS473-FML/src/validation/run_causal_evaluation.py)
  - command-line runner for causal backtesting
- [src/validation/__init__.py](/Users/Catherine/Desktop/CS473-FML/src/validation/__init__.py)
  - exports for the new validation helpers
- [tests/test_causal_validation.py](/Users/Catherine/Desktop/CS473-FML/tests/test_causal_validation.py)
  - targeted automated tests for the new framework

## Data Used

The first causal test used the existing project dataset:

- [data/processed/feature_matrix.parquet](/Users/Catherine/Desktop/CS473-FML/data/processed/feature_matrix.parquet)

This file already contained:

- `zone_id`
- `time_key`
- outcome-like target signal in `target`
- zone-year features such as license activity, rent pressure, and valuation features

Because the project data does not yet include a true logged model intervention variable, a derived evaluation dataset was created:

- [data/processed/causal_eval_frame.parquet](/Users/Catherine/Desktop/CS473-FML/data/processed/causal_eval_frame.parquet)

## Derived Causal Setup

For this first run, the causal frame was defined as follows:

- Outcome:
  - `outcome = target`
- Treatment:
  - `treatment = 1 if license_velocity > 0 else 0`
- Features:
  - `lag_license_velocity`
  - `lag_net_opens`
  - `lag_net_closes`
  - `lag_target`
  - `rent_pressure`
  - `mean_assessed_value`

This setup was chosen to reduce leakage by using prior-year zone signals wherever possible.

## Data Preparation Steps

The preparation pipeline did the following:

1. Loaded `feature_matrix.parquet`.
2. Sorted records by `zone_id` and `time_key`.
3. Filtered to rows with non-null `target`.
4. Restricted the tested period to years 2001 through 2024.
5. Created lagged features within each `zone_id`:
   - `lag_license_velocity`
   - `lag_net_opens`
   - `lag_net_closes`
   - `lag_target`
6. Created the binary treatment variable from current-year `license_velocity`.
7. Filled missing slow-moving structural covariates using medians:
   - `rent_pressure`
   - `mean_assessed_value`
8. Dropped rows without the required lagged history.

Resulting frame:

- 609 rows
- 24 years
- treatment counts:
  - treated: 441
  - control: 168

## Backtesting Configuration

The first evaluation run used:

- dataset: `data/processed/causal_eval_frame.parquet`
- time column: `time_key`
- treatment column: `treatment`
- outcome column: `outcome`
- window type: `expanding`
- minimum training periods: `5`
- test size: `1`

This produced:

- 19 temporal splits

Each split trained on all prior years and tested on the next year only.

## What the Evaluation Does

### 1. Temporal Backtesting

For each split:

- train on historical years only
- test on the next future year
- estimate uplift and causal metrics
- generate plots and reports

This is the main defense against time leakage.

### 2. Uplift Modeling

The current implementation uses a T-learner:

- one model estimates outcome under treatment
- one model estimates outcome under control
- predicted uplift is the difference between those two estimates

### 3. Propensity Estimation

If no propensity score is provided, the framework estimates treatment propensity from the feature set and uses those scores in inverse propensity weighted evaluation.

### 4. Uplift Curve

Rows are sorted by predicted uplift from highest to lowest. The evaluator then measures cumulative incremental gain as we move down the ranked list.

Interpretation:

- if the curve rises above the random baseline, the ranking is useful
- if it does not, the model is not prioritizing the right cases

### 5. Qini Coefficient

The Qini coefficient summarizes how far the uplift curve stays above the random baseline.

Interpretation:

- positive Qini is good
- higher positive Qini is better
- negative Qini means worse than random

### 6. Average Treatment Effect (ATE)

The evaluator computes both:

- a naive observed difference
- an inverse propensity weighted ATE estimate

Interpretation:

- positive ATE suggests treatment is associated with better outcomes
- small p-values suggest the estimated effect is statistically distinguishable from zero

### 7. Uplift at Top Decile

This measures observed uplift among the top 10 percent of rows ranked by predicted uplift.

Interpretation:

- useful for evaluating targeting quality
- strong top-decile uplift suggests the ranking is useful even if average effects are modest

### 8. Policy Risk

The evaluator compares a model policy against a baseline policy.

In the current run, the model policy is:

- treat if predicted uplift is positive

Interpretation:

- lower policy risk is better
- negative policy risk means the model policy outperformed the baseline

### 9. Covariate Balance

The evaluator computes standardized mean differences across covariates in train data.

Interpretation:

- smaller absolute SMD values are better
- values under 0.1 are preferred
- large SMD values indicate treatment and control are not very comparable

## Test Results

The new causal test suite passed:

- `venv/bin/pytest tests/test_causal_validation.py -q`
- result: `8 passed`

The broader validation suite still has an unrelated environment issue tied to loading an existing serialized XGBoost model in another test path. That issue was not introduced by this work.

## Run Output

The first full causal run completed successfully and wrote outputs here:

- [data/processed/causal_uplift_demo](CS473-FML/data/processed/causal_uplift_demo)

Key files:

- [backtesting_report.html](CS473-FML/data/processed/causal_uplift_demo/backtesting_report.html)
- [time_series_performance.csv](CS473-FML/data/processed/causal_uplift_demo/time_series_performance.csv)
- [final_recommendation_summary.json](Desktop/CS473-FML/data/processed/causal_uplift_demo/final_recommendation_summary.json)
- [run_19/uplift_curve.png](CS473-FML/data/processed/causal_uplift_demo/run_19/uplift_curve.png)
- [run_19/qini_curve.png](CS473-FML/data/processed/causal_uplift_demo/run_19/qini_curve.png)
- [run_19/backtesting_report.html](CS473-FML/data/processed/causal_uplift_demo/run_19/backtesting_report.html)

## Summary Metrics

Aggregate results across 19 splits:

- mean Qini coefficient: `0.302656`
- median Qini coefficient: `0.256850`
- min Qini coefficient: `-0.045125`
- max Qini coefficient: `0.722057`

- mean ATE: `0.228298`
- median ATE: `0.262696`
- min ATE: `-0.040354`
- max ATE: `0.525406`

- mean uplift at top decile: `0.290849`
- median uplift at top decile: `0.295000`
- min uplift at top decile: `0.193030`
- max uplift at top decile: `0.369568`

- mean validation performance score: `0.578947`
- median validation performance score: `0.500000`

- mean max absolute SMD: `0.487858`
- median max absolute SMD: `0.479466`
- min max absolute SMD: `0.417770`
- max max absolute SMD: `0.583775`

## What These Results Mean

### Positive findings

- The framework ran successfully on project data without needing synthetic placeholders.
- Most splits showed positive Qini.
- 18 of 19 splits had positive uplift relative to the random baseline.
- The latest split also had positive Qini.
- Mean ATE was positive.
- Uplift at the top decile was consistently positive.
- The model’s policy evaluation generally favored the uplift-based policy over the baseline.

This suggests there may be a real ranking signal in the data that identifies where treatment-like conditions are associated with better outcomes.

### Important limitations

- Treatment was a proxy, not a true logged intervention.
- Treated and control groups were still substantially imbalanced.
- Balance thresholds were not met.
- Because of that imbalance, the results should not be interpreted as strong causal proof.

The largest issue is covariate balance:

- preferred threshold: `abs(SMD) < 0.1`
- observed range in this run: roughly `0.42` to `0.58`

That is too high for a strong causal claim.

## Production Readiness Decision

Final recommendation from the framework:

- `production_ready: false`
- `stable_uplift_splits: 18`
- `latest_split_qini: 0.1144923418450971`
- recommendation: `Do not promote to Production`

Why the run did not pass:

- uplift was mostly stable, which is encouraging
- but covariate imbalance remained too large
- therefore the success criteria were not fully satisfied

## Feature Importance Snapshot

Latest split feature importances:

- `lag_net_opens`: `0.339660`
- `lag_net_closes`: `0.166978`
- `lag_target`: `0.133296`
- `lag_license_velocity`: `0.130363`
- `rent_pressure`: `0.115005`
- `mean_assessed_value`: `0.114698`

Interpretation:

- prior opening activity was the strongest driver
- prior closing activity and prior target level also mattered
- structural zone context contributed, but less strongly

## Interpretation for the Project

This first run is best understood as a framework validation and a diagnostic observational backtest.

It demonstrates that:

- the repository now has a working causal evaluation pipeline
- the project’s time-indexed feature matrix can support uplift-style testing
- temporal evaluation and artifact generation are working
- the current proxy treatment yields encouraging but not decision-grade results

It does not yet demonstrate that:

- a real deployed intervention causes better outcomes
- the system is ready for production model promotion
- causal claims are robust enough for high-confidence business decisions

## Next Steps

### Highest priority

1. Replace the proxy treatment with a better intervention definition.
   - Best option: a true model-driven action flag from logged decisions
   - Good fallback: a domain-grounded intervention event that clearly precedes outcome measurement

2. Improve balance before interpreting effects causally.
   - trim extreme propensity regions
   - add matching or stratification
   - test alternative control cohorts

3. Tighten feature timing.
   - ensure every feature used at decision time truly precedes treatment
   - remove any same-period variables that may partially encode the outcome

### Medium priority

4. Run alternative treatment definitions.
   - `net_opens > 0`
   - `license_velocity > 0`
   - thresholded market change events
   - intervention definitions tied to actual recommendation or exposure logic

5. Compare expanding and rolling windows.
   - expanding windows test long-horizon learning
   - rolling windows test recent stability under drift

6. Add richer diagnostics to the report.
   - per-feature balance tables
   - sensitivity-analysis plots
   - recent-period degradation checks

### Product and workflow

7. Wire report outputs into the Streamlit UI.
   - allow browsing uplift plots
   - show per-split performance table
   - expose final recommendation summary

8. Add MLflow to the project environment if experiment tracking is desired immediately.
   - the framework already supports optional MLflow logging
   - current run succeeded without requiring MLflow

## Recommended Immediate Follow-Up

The most useful next iteration is:

1. define a stronger treatment variable
2. rerun the causal backtest
3. compare balance and Qini against this baseline run

That will tell us whether the current optimism comes from a meaningful intervention signal or from observational imbalance.

## Bottom Line

The causal ML evaluation framework is now implemented, tested, and successfully run on real project data.

The first backtest is encouraging because uplift metrics are usually positive over time. However, the current treatment proxy and covariate imbalance mean the result should be treated as an exploratory observational finding, not as production-ready causal evidence.

The framework is ready for the next iteration. The main task now is improving treatment definition and balance so that future runs can support stronger causal conclusions.
