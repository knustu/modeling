# Independent Phase 4 synthetic-only prototype

## Status and boundaries
This is a partial development implementation, NOT a completed validated final system. Tests and experiments have NOT been run by this chat. No official OIBC documentation or official validator has been supplied. No actual competition inputs are read by the CLI. No public score reconstruction, M3 reuse, external weather, package installation, upstream modifications or competition submission. All candidate labels remain UNVERIFIED. Upstream projects are inherited read-only on the branch; this commit adds only OIBC_NEW_SETTLEMENT.

## Integration inspection
Repository root confirms OIBC_NEW_GROUPING, OIBC_NEW_FORECASTING, OIBC_NEW_ERROR_ANALYSIS. Existing source interfaces were reviewed in this conversation, not the user's private output bytes. Phase 1 composition=site_id,group; Phase 2 plant residual=site_id,ts,forecast_kwh,energy_kwh with file-context model/fold; Phase 3 requires decision/training/OOS provenance absent from those CSVs alone. Phase 3 local 36-test success is user-reported; full Phase 3 pipeline/browser status is not assumed. Locally patched Phase 2 core.py and Phase 3 UTF-8 test changes are not overwritten.
contracts.input_manifest checks explicit local paths and supplied SHA256 values without copying files; composition validates original IDs/order and capacities. Canonical array partitions include model IDs, site ordering, forecast round, training-through, decision times and OOS assertion. File-based Phase 1–3 adapter orchestration remains pending; synthetic runner does not claim genuine end-to-end artifact integration. Direct-group lineage is rejected for reassignment; it cannot fabricate plant forecasts.

## Components
- objective.py: R, W, S; group signed sums before absolute value; exact SMP join and configurable peaks.
- grouping_optimizer.py: greedy best-improvement relocation, sampled pairwise swaps, mixed multi-start and simulated annealing. Proposed changes update only affected error sums, check capacities before acceptance and retain the best feasible assignment even after annealing uphill moves. No optimality certificate. Costs roughly O(starts*iterations*(N+swap_trials)*(T+N)); memory O(TN).
- corrections.py: bounded additive weighted-median correction, prespecified multiplicative alpha grid, hour cells and fixed issuance-weather bins. Additive weighted median solves unconstrained weighted L1 offset; nonnegative clipping can change boundary optima, so these are validated candidates, not exact constrained correction optimizers. Sparse cells fall back to global development offset. No neural residuals or training of base models.
- ensemble.py: nonnegative simplex combinations of two compatible plant models; equal, single and small grid weights. No time-conditional fitting; no extra learned regularization, small search space only.
- experiment.py: fixed development residuals for grouping/correction/ensemble; separate validation for bounded A-F/joint candidate comparisons; optional frozen synthetic holdout assessment only after selection. Default holdout report contains no score. Corrections are refitted separately by grouping/model, always development only.
- robustness.py: paired full-calendar-block bootstrap, 1/7-day sensitivity, total-delta CI on covered blocks only, block counts and insufficient-block status. Approximate independence, no multiplicity correction; does not establish real generalization.
- submission.py: synthetic zero-valued format fixture and independent custom checks for 253 assignments, capacities, 17520 finite nonnegative rows, exact 8760-hour KST range. Generated package is explicitly NON_SUBMITTABLE_SYNTHETIC. No real forecast inference or official-validator invocation.
- visualization.py: offline prototype, actual/corrected forecast, cumulative penalty difference, convergence, metrics and assignment tables, configuration/group/date selectors and PNG export.

## Leakage and objective conventions
Only development partitions can fit corrections/weights. Chronological order, training before issuance, interval-start availability, model/site/round/lineage compatibility and complete matrices are required. Validation/holdout decisions follow prior-stage target end. Forecast-round label 10:00/17:00 is supported but official decision schedule/latency mapping must still be verified; labels alone are not proof. Synthetic weights use synthetic known prices; prices are retrospective objective weights, never operational weather/correction inputs. Negative SMP optimization is rejected because nonnegative convex-loss assumptions need review. Under stated invariant-revenue formula minimizing W maximizes S; decreases in W and percentage changes in S are NOT conflated.
Actual horizon and source unit/version audits require verified real manifests before real use. Claims in manifests cannot independently prove training lineage. Protected holdout data is structurally validated but not scored by default; repeated releases can still contaminate judgment and are not controlled by a persistent single-use registry.

## Candidate/ablation policy
A reference/none; B optimized grouping/none; C reference/additive; D optimized/additive; E optimized/development convex ensemble/none; F optimized/ensemble/additive. All use identical validation labels/prices/hours. Full small correction/model matrix is retained. Promotion prototype requires relative W margin, no worse maximum daily W and every configured block CI available with upper bound below zero; otherwise keep baseline. Simplicity breaks selection priorities. All categories remain unverified in synthetic mode, even if this rule favors one fixture candidate.

## Windows CMD: separate worktree preserves upstream local edits
Run each command on its own line. Do not force checkout/reset or delete existing folders.

```bat
cd /d C:\Users\yhy42\modeling
git status
git fetch origin
git worktree add -b phase4-local ..\modeling_phase4 origin/phase4-independent-settlement
cd /d C:\Users\yhy42\modeling_phase4\OIBC_NEW_SETTLEMENT
py -c "import numpy,pandas; from zoneinfo import ZoneInfo; print(numpy.__version__,pandas.__version__,ZoneInfo('Asia/Seoul'))"
py -m unittest discover -s tests -p "test_*.py" -v
py run_optimization.py --synthetic --config config\optimization_config.json --output outputs\synthetic\run_001
py run_validation.py --synthetic --directory outputs\synthetic\run_001\synthetic_submission
start "" outputs\synthetic\run_001\optimization_dashboard.html
```
Stop on any test failure. If folder/branch exists, use existing worktree rather than overwrite. A pipeline failure may leave partial outputs; use a NEW output path after repair. No automatic package installation. Real-data commands deliberately unavailable: official schema/provenance/rule review and separate authorization are required before enabling them.

## Output inventory
Grouping optimization CSV/logs; correction results; ensemble/joint comparison; bootstrap CSV; robustness JSON; A-F ablation CSV; unreleased holdout JSON; candidate manifest; selected grouping/model/correction; experiment manifest/history; integration validation; dashboard; final report and instructions. Synthetic submission subdirectory contains composition.csv, forecast.csv, submission_validation.json and non-submittable marker. All actual data/results/models stay local and ignored by Git. Review staged files: ignore patterns do not protect already tracked data.

## Acceptance and unfinished requirements
Implemented acceptance checks are prototype canonical-fixture gates, NOT certification of official rules or complete production integration. Tests are supplied, not executed here. Required remaining work:
1. Execute tests/integration and offline browser acceptance locally, fix all reported errors.
2. Full adversarial small exhaustive-optimum fixture, vanished-holdout gains and monthly ensemble overfit scenarios; current tests do not cover every requested case.
3. Multiple nested rolling validation folds, seasonal/round robustness, per-month/peak/season score tables, assignment stability and optimizer setting sensitivity. Synthetic fixture currently has winter partitions and one round.
4. Weather correction preselected bins are simplistic; physical upper bounds/night constraints need authoritative rules. No arbitrary upper clipping is claimed physical.
5. starts config is reserved; multistart currently uses every supplied composition with greedy/annealing, not additional generated assignments. Logs capture accepted moves only; rejected-proposal audit counts remain pending.
6. Full source-file adapters, fold manifests, independent diagnostic acceptance, immutable input hash propagation and source-version gate orchestration. No duplicated upstream algorithms.
7. Complete dashboard: monthly/peak/uncertainty/capacity charts, generic metadata navigation, model/correction independent selectors, report/HTML export. Current prototype uses candidate IDs/tables and three charts only.
8. Persistent holdout release registry, final freeze/checkpoint reload/inference, build_submission CLI for genuine validated forecasts, optional authorized official validator. Zero annual format fixture is not that final pipeline.
9. Config validation for all numerical bounds, structured failure journal/recovery and large-data memory benchmarks.
Do not call Phase 4 complete or real-data ready until these gates are closed. Software-development progress is not settlement evidence.
