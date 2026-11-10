# Independent OIBC Phase 2 forecasting

## Status and limits
Implementation draft, NOT Phase 2 completion. No Python tests or browser tests have been executed by this chat. No actual competition files have been read. No real Phase 1 output is assumed. Official OIBC documentation was not located/supplied; unrelated search results are not treated as rules. The CLI intentionally refuses real-data execution. Source schema, provider units, exact issuance delivery/latency, coordinate permissions, peak-hour endpoint convention and official row ordering need local verification first.

Phase 1 is unchanged. This branch derives from phase1-independent-grouping. No M3 or other forecasting implementation is imported. No GitHub Actions, external weather, package installation, GPU or submission is configured.

## Architecture
`src/core.py`: canonical data validation, grouping adapter, provider conversion/as-of join, features, train-only preprocessing, climatology, Ridge, histogram gradient boosting, aggregation, metrics, penalty formula, expanding folds, forecast validator.
`src/backtest.py`: common plant predictions for every fixture grouping, fixed chronological development validation and separate historical holdout, batched inference, model checkpoints and reports.
`src/synthetic.py`: explicitly synthetic plants/weather/labels/compositions. These compositions are fixtures, NOT alleged Phase 1 results or duplicated grouping optimization.
`src/visualizer.py`: offline canvas dashboard, model/candidate/group/split/date selectors, PNG and HTML export.
`tests/test_contracts.py`: synthetic tests.
`config/`: explicit canonical schema and synthetic defaults.

## Features and information availability
Only configured forecast fields, capacity, cyclical hour/year features. No coordinates, label lags, future observations, site label encodings, or pretrained models. Weather joins use station and target timestamp, choosing latest issuance with BOTH issued_at and available_at <= decision_at. Forecasts unavailable by decision are missing, never backfilled from future revisions. Decisions are an explicit input, not inferred from source target dates. Synthetic fixtures assume previous-day 17:00 decisions and 10:00 issue with one-hour latency; this is not an assertion about official rules. Both 10:00 and 17:00 rounds can be represented by canonical rows, but official round-selection policy remains unverified.

Canonical ts denotes interval END. All timestamps must have timezone and exact hourly boundary. Naive timestamps and numeric epochs are rejected. Generation labels need actual availability timestamps >= interval end. No labels on/after label_cutoff are accepted. Provider rules state source column, canonical unit and affine conversion. Synthetic WB Kelvin-to-Celsius is ONLY a fixture convention. Median/scaling parameters fit on each training fold only; all-missing columns use zero plus missing indicator. Fixed feature order is checked. Missing labels are not imputed; validation excludes incomplete whole target hours for fair complete-portfolio comparison.

## Models
Climatology: mean training generation (or normalized generation) by site/hour with training-only hour/global fallback.
Ridge: standardize/impute on training; solve (X'X + alpha I) beta = X'y with unpenalized intercept. No hyperparameter tuning using holdout.
Stronger candidate: scikit-learn HistGradientBoostingRegressor, from scratch, fixed seed, early_stopping=False to avoid random internal holdout. Requires an already-installed library and fails explicitly if unavailable. CPU only; thread limits set before numerical imports.
Capacity normalization can be enabled/disabled for controlled later comparisons; no claim of superiority.

## Validation
Expanding origin folds require train targets < origin AND labels available <= origin. Validation targets are at/after origin and their decisions at/after origin; this deliberately purges boundary hours whose forecasts would require a model not fitted yet. Fold validation ranges cannot overlap. Holdout follows development validation. Same trained plant forecasts are aggregated for every composition; group capacity is checked and CSV inputs are read-only with hashes available via load_grouping. Direct group-level forecasting is not implemented; its future adapter can aggregate labels first but requires separate per-composition fitting, never relabeling a plant-model result as a direct model.
Metrics: MAE, RMSE, capacity-normalized MAE, hourly/monthly and diagnostic day/night CSVs, plant residuals, group performance. Day/night labels are configured clock-hour diagnostics, not physical sunrise calculations. penalty() requires exact matching label/prediction keys and complete SMP; peak convention is mandatory. Test-only hand calculation is 0.1*(100*3*2+50*1*3)=75. No prices are synthesized by the runner, and no official score is produced.

## Windows CMD commands (one line each)
From the existing repository, first check uncommitted work. Do not force checkout or reset.

```bat
cd /d C:\Users\yhy42\modeling
git status
git fetch origin
git switch --track origin/phase2-independent-forecasting
cd OIBC_NEW_FORECASTING
py -c "import numpy,pandas,sklearn; from zoneinfo import ZoneInfo; print(numpy.__version__,pandas.__version__,sklearn.__version__,ZoneInfo('Asia/Seoul'))"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
py -m unittest discover -s tests -p "test_*.py" -v
py run_forecasting.py --synthetic --config config\forecasting_config.json --output outputs\synthetic_run_001
start "" outputs\synthetic_run_001\forecast_diagnostics.html
```
If the branch already exists locally: `git switch phase2-independent-forecasting`. If checkout reports local changes, stop and preserve them; do not discard. No package install command is supplied. A missing dependency needs explicit installation authorization. Use a NEW output directory for each run.

## Outputs and checkpoints
Local outputs: plant_forecasts_*.csv, plant_errors_*.csv, group_forecasts_*.csv, error_statistics_*.csv, model_comparison.csv, grouping_forecast_comparison.csv, backtest_report.json, forecast_diagnostics.html, validation_report.json, model_metadata.json, environment.json, forecasting_report.md, models/*.pkl, forecast.csv. Exactly one holdout-fitted checkpoint per model. Pickle files are local trusted artifacts; no untrusted checkpoint loader exists. A failure can leave a partial run directory; retain it for diagnosis and rerun into a new directory, not over it.
Annual forecast.csv is a full-grid SYNTHETIC TRAINING-CLIMATOLOGY FORMAT EXAMPLE: 8760 hours per group, 17520 rows, finite nonnegative values, +09:00 ISO text. It is not future observed labels, an official competition forecast, or a scored result. Real plant-level annual inference remains locked. Real source adapters and source units must first be verified against official documentation. There is intentionally NO runnable real-data command yet; fabricating one would falsely imply readiness.

## Offline acceptance
Disconnect network. Open generated HTML, change model/group/candidate/split, filter dates and export PNG/HTML. Reopen exported HTML and verify charts. Source tests detect external dependencies but do not execute a browser. The dashboard displays aggregated error series/hour/month; plant-level error DISTRIBUTION CHART remains pending (CSV residuals are retained). No browser verification is claimed.

## Security
.gitignore excludes actual data, models, outputs and reports, but cannot remove already tracked files. Review git status and git diff --cached before commits. Keep all competition inputs outside the repo. Do not upload actual IDs, metadata, derived features, forecasts, errors, logs or models. Public/private repository visibility does not change this rule.

## Phase 3 handoff (not implemented)
Retain locally: source/composition hashes, IDs/order, immutable composition CSVs, label-availability and station/issue-time join policy, preprocessing and feature order, fixed config/environment, per-fold origin/coverage, per-plant predictions and residuals, group forecasts/truth, capacity totals, hourly/monthly/daytime statistics, holdout designation, checkpoints and model metadata. No settlement optimization or grouping changes are included.

## Remaining completion gates
Execute and fix synthetic tests, execute full synthetic integration, verify offline browser behavior, add plant-error distribution chart, verify actual Phase 1 outputs locally, obtain official documentation and approve canonical source adapters/real execution policy. Until then this is a development implementation, not a complete validated Phase 2 system.
