# ML Pipeline: How often will a car-insurance policy have a claim? (freMTPL2)

Started 2026-10-07. Data: French motor third-party liability policies, `freMTPL2freq` from the CASdatasets collection (github.com/dutangc/CASdatasets), 677,991 policies, read-only. Tools: scikit-learn plus XGBoost 3.4.1. Budget $0, CPU only.

## Checklist
- [x] 1. Data inspection - done 2026-10-07: 677,991 policies x 12 columns; 0 missing; 3.68% have a claim; 0.0738 claims per policy-year; 1,224 policies with impossible exposure above 1 year (max 2.01); 1,116 cars older than 30 years; 401 drivers older than 90; up to 16 claims on one policy; 28,768 rows identical on every rating column (different policies, kept); no dates
- [x] 2. EDA - done 2026-10-07: 5 figures; claims per policy-year are 0.21 for drivers under 22 vs about 0.06 for 60+; 0.05 for the best bonus-malus (past record) vs 0.44 for the worst; very short policies look riskiest (0.17 vs 0.04), a known quirk because policies often end after a claim
- [x] 3. Define the prediction problem - done 2026-10-07, contract below, awaiting Gate A approval
- Gate A: approved 2026-10-07
- [x] 4. Data cleaning - done 2026-10-07: 0 missing; capped (not deleted) exposure at 1 year (1,224 rows), claims at 4 (8), vehicle age at 20 (8,321), driver age at 90 (401), bonus-malus at 150 (209); all 677,991 policies kept
- [x] 5. Data engineering - done 2026-10-07: one row per policy (IDpol unique); 11 features known at pricing time; ClaimNb, Exposure, IDpol asserted not to be features
- [x] 6. Train / validation / test split - done 2026-10-07: guard.split random stratified (no dates); 406,795 train / 135,598 validation / 135,598 final exam; claim rate 0.0736 / 0.0738 / 0.0738 per policy-year; 15,838 / 5,290 / 5,277 claims; 0 shared policy ids
- [x] 7. Feature engineering - done 2026-10-07: log density, young driver (<25) flag, high bonus-malus (>=100) flag; all row-wise
- [x] 8. Preprocessing - done 2026-10-07: scale 7 numeric columns, one-hot 4 categorical columns (largest: 22 regions) = 48 columns, fit on train only
- Gate B: approved 2026-10-07
- [x] 9. Baseline model - done 2026-10-07: one average rate (0.0736 claims per policy-year) gives validation deviance 0.4802; lift in riskiest 20% = 0.20 (random)
- [x] 10. Model training - done 2026-10-07: Poisson regression (with smooth age curves), scikit-learn gradient boosting, XGBoost 3.4.1 (459 trees kept by early stopping on a random 15% slice of train)
- [x] 11. Hyperparameter tuning - done 2026-10-07: 11 settings, 3-fold CV on a 150,000-policy slice of train only; CV deviance 0.4471-0.4520, boosted models close together (cv_results.csv)
- [x] 12. Model evaluation - done 2026-10-07: validation deviance 0.4802 baseline / 0.4573 Poisson regression / 0.4513 gradient boosting / 0.4506 XGBoost; XGBoost 6.2% better than the average (95% range 5.6-6.7), regression 4.8% (4.2-5.3); riskiest 20% hold 38.5% of claims (XGBoost) vs 34.2% (regression); pre-declared rule selects XGBoost depth 5 (gap to gradient boosting 0.0007 > 0.0005)
- Gate C: approved 2026-10-07
- [x] 13. Error analysis - done 2026-10-07: calibration is good for age, past record, vehicle age, brand and most regions (actual/predicted 0.9-1.1); the clear miss is policy length: actual claims are 2.0x predicted for policies under 5 weeks (exposure <= 0.1), 1.5x for 5-13 weeks, and 0.87x for near-full-year policies, because claim counts do not grow in proportion to time insured (policies end after claims); small regions (Corse 0.56, Limousin 1.40) are noisy
- [x] 14. Final test - done 2026-10-07: ONE pass over 135,598 locked policies; selected XGBoost deviance 0.4479 (validation 0.4506), 6.5% better than the average rate (95% range 5.9-7.0); gradient boosting 6.4% (5.9-6.9), Poisson regression 4.9% (4.4-5.3); riskiest 20% hold 38.9% of claims; final_results.json
- [x] 15. Deployment - skipped 2026-10-07: user did not ask for deployment
- [x] 16. Monitoring + retraining plan - done 2026-10-07: see Monitoring plan below
- Gate D: report delivered 2026-10-07, awaiting the user's sign-off

## Problem contract (step 3, awaiting Gate A approval)
- Target: number of claims on a policy (`ClaimNb`), modeled as a claim RATE per policy-year, the way insurers price: expected claims = rate x exposure
- Prediction unit: one policy, scored when it is priced or renewed
- Allowed features: VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, Region
- `Exposure` (share of the year insured) is NOT a feature, it is the multiplier (offset): short policies have fewer chances to claim. Using it as a feature would leak, because many short policies are short BECAUSE a claim ended them
- EXCLUDED: `IDpol` (identifier), `ClaimNb` (the answer)
- Cleaning rules proposed: cap exposure at 1 year; cap claims per policy at 4; cap vehicle age at 20, driver age at 90, bonus-malus at 150 (extreme values are kept but limited so a few odd rows do not drive the model)
- Objective: rank policies by expected claim frequency so pricing can charge more for risky ones
- Metric (proposed): Poisson deviance (the standard for claim counts; lower is better) as the main score, reported as % better than the average-rate baseline; plus lift: share of all claims that fall in the riskiest 20% of policies; every number with a bootstrap range
- Baselines: (a) one average rate for everyone (0.0738 claims per policy-year); (b) Poisson regression (a GLM, the industry workhorse)
- Candidates: Poisson GLM; scikit-learn gradient boosting with Poisson loss; XGBoost 3.4.1 with Poisson loss and log-exposure as base margin
- Split plan: RANDOM 60/20/20, stratified on whether a policy has a claim, because the file has no dates and no repeated policies (guard.split without time_col); about 136,000 policies and about 5,000 claims in the final exam
- Risks: claims are rare (3.7%) and noisy, so gains over the average-rate baseline will be modest; the file carries no calendar date, so we cannot test a future period; rows that look identical are kept because they are different policies in the same rating cell
- Not pushed anywhere: all files are local; nothing is committed or pushed without the user's say-so

## Step notes
- Figure 01_missingness.png: No column has missing values.
- Figure 02_target_balance.png: any_claim=0 is 96.3% of 677,991 rows, so a majority-class dummy scores 96.3%; the minority share is 3.7%.
- Figure 02_distributions.png: Histograms of 8 numeric features; 5 are strongly skewed (ClaimNb, VehPower, VehAge, BonusMalus, Density), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with any_claim: ClaimNb (0.96); anything above 0.95 is a leakage suspect.
- Figure 02_frequency_by_group.png: Claims per policy-year for young drivers, for poor past records (bonus-malus), and for short policies. Young drivers and bad records claim far more, and very short policies look oddly risky, which is a data quirk to handle, not a signal to trust.
- Note on Gate A: approved by the user in chat 2026-10-07 ('approve go ahead'), after the contract was presented; Gates B-C run next with the standing plan to report once at the end.
- Figure 04_outliers.png: Three columns with impossible or extreme values; the red line is where each is capped. Only a small number of rows sit beyond the line, so capping keeps every policy while stopping odd rows from steering the model.
- Figure 06_split.png: The three parts of the data. Each has almost the same claim rate, so scores on one are fair to compare with the others; the final exam is locked until the very end.
- Figure 08_log_density.png: Population density is extremely skewed: most policies sit in small towns and a few in dense cities. Taking the log (then scaling with training rows only) spreads it out so the models can use it.
- Note on Gate B: recorded under the user's 'approve go ahead' after Gate A, with the plan (stated to the user) to run Gates B-C and report once at the end; every decision is listed in the final report.

## Model rationale (written before any training, Gate B)
Model rationale:
- traits: count target (claims) with exposure, 677,991 rows (large), 3.7% of policies have a claim (imbalanced, but a rate model, not a yes/no one), no datetime, no groups
- baseline: one average claim rate for everyone + Poisson regression (GLM)
- candidates: scikit-learn gradient boosting with Poisson loss; XGBoost 3.4.1 with Poisson loss and log-exposure base margin
- ruled out: neural nets (tabular, modest signal); plain accuracy or yes/no classification (hides frequency differences); using Exposure as a feature (leaks: policies end after claims); dropping the identical-looking rows (different policies in the same rating cell)
- metric: Poisson deviance primary (% better than average-rate baseline); lift in the riskiest 20% of policies secondary; bootstrap ranges
- tuning: 3-fold CV on a 150,000-policy slice of TRAIN only; validation used once per model to compare

## Pre-declared rules (written 2026-10-07 BEFORE any modeling)
- Selection: the model with the lowest validation Poisson deviance wins; if the gap in mean deviance between two models is below 0.0005, the simpler one wins (average rate < Poisson regression < gradient boosting < XGBoost)
- Final exam: ONE pass over the locked 135,598 policies scores every pre-declared predictor (average rate, Poisson regression, gradient boosting, XGBoost, all fit on train only). The claim is about the pre-selected model only; the others are shown for comparison. That is one touch, so no override is needed
- Exposure is the multiplier for every model (sample weight for scikit-learn, log base margin for XGBoost); it is never an input feature
- Tuning uses 3-fold CV on a random 150,000-policy slice of train (seed 0); full train is used for the final fits
- Figure 10_model_comparison.png: Each model scored on the validation policies. The left bars show how much better than the flat average rate each model predicts claim counts; the right bars show how well it ranks risky policies, where the dashed line is what a coin flip would do.
- Figure 12_evaluation.png: Policies sorted into ten equal groups from safest to riskiest. Points on the dashed line mean the prediction matches reality; all models rank the groups correctly, and the gap between curves shows where one is better calibrated.
- Note on Gate C: recorded under the plan stated to the user. Artifacts: Figure 10_model_comparison.png, Figure 12_evaluation.png, runs.csv, cv_results.csv, selected.json. Selection followed the rule fixed before modeling.
- Figure 13_error_analysis.png: How far the chosen model is off inside groups of policies: a value of 1 means predicted claims match actual claims. Points far from the dashed line are the groups where pricing from this model would be too cheap (above 1) or too dear (below 1).
- Step 14 final test: poisson_deviance=0.4479 (touch 1, 2026-10-07)
- Figure 17_pricing_view.png: Policies the model scored as safest actually claim far less than average and the riskiest far more, on policies it never saw. A flat price charges all five groups the same, so the safe groups pay for the risky ones.

## Monitoring plan (step 16)
- Each quarter, compare the mix of new policies (driver age, bonus-malus, density, region) with training; a large shift means the model is seeing a different book
- Each year, compare actual claims with predicted claims by risk group (the five bands in Figure 17) once claims have developed; retrain if any band is off by more than 10%, or deviance is more than 1% worse than at launch
- Retrain yearly on the newest policies, same split and rules; keep the final exam locked until the end each time
- Known weakness to fix first: short policies (under about 13 weeks) are under-predicted by up to 2x; test an exposure-dependent adjustment (for example a power of exposure) or model policy length separately
- Limits: one French portfolio from one period, no calendar dates (cannot test a future year), claim counts only (not claim size), so this is a frequency model, not a price
