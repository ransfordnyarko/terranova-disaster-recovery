# Terranova Disaster Cost Forecasting

A machine learning system for forecasting federal disaster declaration costs using FEMA public data. Built with a confidence-aware prediction architecture that distinguishes between typical disasters the model handles reliably and catastrophic outliers that require human review.

---

## Project Overview

**Goal:** Predict the total cost of a federal disaster declaration at the time of declaration, using only information available before costs are incurred.

**Target variable:** `total_disaster_cost` — the sum of Public Assistance obligations, Individual Assistance grants, and Hazard Mitigation funding per disaster declaration.

**Key design principle:** Honest uncertainty quantification. The system does not attempt to predict catastrophic outliers ($1B+) confidently. Instead it flags them and routes to human review.

---

## Data Sources

All data sourced from FEMA's OpenFEMA API.

| Dataset               | Raw rows | Description                                 |
| --------------------- | -------- | ------------------------------------------- |
| Disaster Declarations | ~60,000  | One row per designated area per declaration |
| Public Assistance     | ~811,609 | One row per project worksheet               |
| Disaster Summaries    | 3,940    | Pre-rolled cost totals per disaster         |

### Why so many rows?

FEMA data is intentionally granular. Disaster Declarations fan out by designated area — one COVID declaration in Maine had 438 rows, one per county and township. Public Assistance fans out by project worksheet — one disaster can have thousands of individual funded projects.

---

## Data Pipeline

### 1. Deduplication Strategy

Before dropping duplicates, duplicates were used to fill null values in `incidentEndDate` by taking the first non-null value within each duplicate group.

**Key:** `disasterNumber + declarationDate + incidentType + declarationType`

Audit confirmed `incidentType` never varies within a disaster (all 5,179 groups had exactly 1 unique value), validating the key choice.

### 2. Public Assistance Aggregation

PA data collapsed from 811K rows to disaster level:

```
pa_agg = groupby(disasterNumber).agg(
    pa_total_obligated    = sum(totalObligated)
    pa_applicant_count    = nunique(applicationId)
    pa_project_count      = count(projectAmount)
    pa_large_project_count= sum(projectSize == 'Large')
    has_debris/emergency/roads/water/buildings/utilities/parks
)
```

### 3. Dataset Joins

```
Declarations (5,179)
    ↓ inner join on disasterNumber
Summaries (3,940)     ← universe limiter
    ↓ left join
PA aggregated (1,765) ← not all disasters receive PA
    = Final: 3,940 rows
```

Pre-1980 declarations were excluded by the inner join — FEMA summaries didn't exist for that era. Only 1 pre-summary disaster had PA records, making the exclusion safe.

FM (Fire Management) declarations were retained in the dataset with an `is_fire_management` flag rather than dropped — they represent a structurally different cost mechanism and the model needs to learn the distinction.

---

## Feature Engineering

### Leakage Prevention

Features available only **after** a disaster were excluded from training. This was the most critical design decision.

| Feature                  | Available at declaration?   | Decision                |
| ------------------------ | --------------------------- | ----------------------- |
| `designatedArea_count`   | ✅ Yes                      | Keep                    |
| `ihProgramDeclared`      | ✅ Yes                      | Keep                    |
| `project_count`          | ❌ No — post-disaster       | Drop                    |
| `large_project_count`    | ❌ No — post-disaster       | Drop                    |
| `totalObligatedAmountPa` | ❌ No — component of target | Drop                    |
| `incident_duration_days` | ❌ No — end date unknown    | Replace with historical |

### Features Used

**Disaster characteristics (known at declaration):**

| Feature                     | Description                              |
| --------------------------- | ---------------------------------------- |
| `fyDeclared`                | Fiscal year of declaration               |
| `designatedArea_count`      | Number of counties/areas designated      |
| `ihProgramDeclared`         | Individual & Household Program activated |
| `iaProgramDeclared`         | Individual Assistance activated          |
| `paProgramDeclared`         | Public Assistance activated              |
| `hmProgramDeclared`         | Hazard Mitigation activated              |
| `programs_activated`        | Count of activated programs (0-4)        |
| `is_low_program_activation` | 1 if ≤1 programs activated               |
| `is_emergency_declaration`  | 1 if declarationType == EM               |
| `is_fire_management`        | 1 if declarationType == FM               |
| `is_biological_em`          | 1 if Biological + EM (COVID-type)        |
| `is_ongoing`                | 1 if incidentEndDate is null             |

**Geographic features:**

| Feature             | Description                           |
| ------------------- | ------------------------------------- |
| `area_bin_score`    | Designated areas bucketed 1-5         |
| `area_bin_score_dr` | Area score zeroed for EM declarations |
| `area_x_dr`         | Raw area count × DR flag interaction  |
| `state_median_cost` | Target-encoded state median cost      |

**Historical cost signals (engineered from past disasters):**

All historical features computed by `incidentType + declarationType` grouping with `incidentType`-only fallback for sparse combinations (<5 records).

| Feature                          | Description                                       |
| -------------------------------- | ------------------------------------------------- |
| `historical_median_cost`         | Median total cost for this type combo             |
| `historical_project_count`       | Median PA project count                           |
| `historical_large_project_count` | Median large project count                        |
| `historical_median_ihp`          | Median Individual & Household Program spend       |
| `historical_median_ha`           | Median Housing Assistance spend                   |
| `historical_median_ona`          | Median Other Needs Assistance spend               |
| `historical_median_pa`           | Median Public Assistance spend                    |
| `historical_median_hmgp`         | Median Hazard Mitigation spend                    |
| `historical_median_catab`        | Median Category A+B (emergency/debris) spend      |
| `historical_median_catc2g`       | Median Category C-G (infrastructure) spend        |
| `historical_duration_days`       | Median duration with 95th percentile cap per type |

**Temporal features (extracted from dates):**

| Feature               | Description                                  |
| --------------------- | -------------------------------------------- |
| `declaration_month`   | Month of declaration                         |
| `declaration_quarter` | Quarter of declaration                       |
| `days_to_declaration` | Days from incident start to FEMA declaration |

**Confidence scoring features:**

| Feature                      | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `catastrophe_risk_score`     | Composite risk score (0-10+)                    |
| `is_biological_dr`           | Biological + DR = COVID-scale risk              |
| `is_high_exposure_hurricane` | Hurricane/Tropical Storm in high-exposure state |
| `confidence_tier`            | HIGH / MEDIUM / LOW                             |

### Key Feature Engineering Decisions

**`designatedArea_count` is conditioned on declarationType:**
Audit revealed 100+ area declarations are predominantly COVID (Biological + EM) where FEMA designated entire states geographically but PA spending was near zero. Raw area count is misleading for EM declarations — `area_bin_score_dr` zeros the signal for EM.

**`incident_duration_days` replaced by `historical_duration_days`:**
The end date is not known at declaration time. Historical median duration by incident type (with COVID-outlier cap at 95th percentile) is used instead.

**EM declarations uniformly cheap:**
Every cheap combo in the bottom-15 analysis was `+ EM`. Emergency declarations are preparedness-focused, not recovery-focused, making `is_emergency_declaration` a strong negative cost predictor.

**Programs activated is the strongest validated signal:**
Median cost by programs activated: 0-1 programs ≈ $0, 2 programs = $4.5M, 3 programs = $42M, 4 programs = $30M.

---

## Target Variable

```python
total_disaster_cost = (
    totalAmountIhpApproved      # Individual assistance grants
  + totalObligatedAmountPa      # Public assistance infrastructure
  + totalObligatedAmountHmgp    # Hazard mitigation grants
)
```

Null cost components filled with $0 — absence of a program means $0 spend, not missing data.

**Log transformation:** `log_total_cost = log1p(total_disaster_cost)` applied before training to handle severe right skew. Metrics reported in both log scale (R²) and dollar scale (RMSE, MAE) for interpretability.

---

## Model Training

### Train/Test Split

**Time-based split** — disasters sorted by `fyDeclared`, 80/20 split. Future years always in test set. Random splitting would allow the model to train on 2020 disasters and test on 1990 ones, which is unrealistic.

```
Train: 3,152 disasters | 1970 - 2020
Test:    788 disasters | 2020 - 2026
```

### Models Evaluated

| Model             | CV R²     | Test R²   | Test MAE |
| ----------------- | --------- | --------- | -------- |
| Linear Regression | 0.231     | 0.205     | $365M    |
| Lasso             | 0.303     | 0.054     | $110M    |
| Ridge             | 0.494     | 0.166     | $767B\*  |
| Gradient Boosting | 0.656     | 0.503     | $104M    |
| XGBoost (GPU)     | 0.644     | 0.470     | $3.6B\*  |
| LightGBM (GPU)    | 0.662     | 0.503     | $126M    |
| **Random Forest** | **0.653** | **0.565** | **$96M** |

\*Ridge and XGBoost dollar-scale metrics inflated by extreme outlier sensitivity.

**Random Forest selected** — best generalization gap (CV vs Test R²), most stable across folds, lowest dollar-scale MAE on typical disasters.

### Hyperparameter Tuning

Optuna TPE sampler, 50 trials, 5-fold CV on training set only. Test set never touched during tuning.

GPU models (XGBoost, LightGBM) preprocessed once upfront before tuning — avoids 250 redundant `ColumnTransformer` fits (50 trials × 5 folds). Manual CV loop used instead of `cross_val_score` to prevent joblib/GPU semaphore conflicts on Colab.

---

## Confidence Scoring System

### Why confidence tiers?

The model fundamentally cannot predict catastrophic events ($1B+) accurately. These events — COVID relief, major hurricane landfalls, unprecedented inland flooding — are structurally unlike training data. Attempting to predict them produces errors in the billions.

### How it works

```python
catastrophe_risk_score = (
    is_biological_dr          × 5  # COVID-scale risk
  + is_high_exposure_hurricane × 3  # Major hurricane in vulnerable state
  + is_pr_earthquake           × 2  # Infrastructure vulnerability
  + (programs_activated == 4)  × 2  # All programs = major event
  + (area_bin_score_dr >= 4)   × 1  # Large geographic scope
)
```

| Tier   | Score | Count | % of data | Median actual cost |
| ------ | ----- | ----- | --------- | ------------------ |
| HIGH   | 0-1   | 3,629 | 92%       | $1.2M              |
| MEDIUM | 2-3   | 176   | 4.5%      | $24M               |
| LOW    | 4+    | 135   | 3.4%      | $325M              |

### Validation results

- **15 of 16 catastrophic test events ($1B+) correctly flagged** as MEDIUM or LOW
- The 1 miss: NC Tropical Storm (Helene 2024) — novel inland flooding pattern with no historical precedent
- **HIGH confidence MAE: $19M** (median $916K) vs overall MAE of $95M
- 80% reduction in error when isolating HIGH confidence predictions

### The one miss — a known limitation

Helene (NC 2024) was classified HIGH confidence because historically, tropical storms in NC were cheap. Climate change is shifting storm behavior in ways that create new catastrophic patterns outside the training distribution. This is documented as a known limitation and motivates annual retraining.

---

## Production Architecture

```
Disaster declared
        │
        ▼
   Feature extraction
   (all known at declaration time)
        │
        ▼
   Catastrophe risk scorer
        │
   ┌────┴────────┐
   │             │
HIGH (92%)   MEDIUM/LOW (8%)
   │             │
   ▼             ▼
RF Model     Historical analog
$19M MAE     matching + human
             expert review
```

### Historical analog matching

For MEDIUM/LOW confidence predictions, the system finds the most similar historical disaster by incident type, state, programs activated, and geographic scope — providing a reference cost rather than a confident prediction.

---

## Project Structure

```
terranova-disaster-recovery/
├── data/
│   ├── raw/                    # Original FEMA CSV downloads
│   └── engineered/
│       └── disaster_costs.csv  # Final model-ready dataset
├── models/
│   ├── random_forest_pipeline.pkl
│   └── random_forest_metadata.pkl
├── plots/
│   ├── random_forest_actual_vs_predicted.png
│   ├── random_forest_residuals.png
│   ├── random_forest_error_distribution.png
│   ├── random_forest_feature_importance.png
│   └── model_comparison.png
├── src/
│   └── models/
│       └── disaster_cost_training.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
optuna>=3.4.0
mlflow>=2.9.0
python-dotenv>=1.0.0
```

---

## Known Limitations

| Limitation                              | Impact                                                                | Mitigation                            |
| --------------------------------------- | --------------------------------------------------------------------- | ------------------------------------- |
| Catastrophic events unpredictable       | 16/788 test disasters have $1B+ errors                                | Confidence tier flagging              |
| Post-2020 catastrophes underrepresented | Most LOW confidence failures occur 2020–2026 — outside training range | Annual retraining as data accumulates |
| Novel disaster patterns (Helene)        | New climate-driven patterns outside training distribution             | Annual retraining                     |
| Pre-1980 data excluded                  | Historical context limited                                            | Summaries dataset coverage gap        |
| No external data                        | Population, GDP, infrastructure value not included                    | Future enhancement                    |
| FM declarations cheapness               | Fire Management costs flow through different mechanism                | `is_fire_management` flag             |

### The post-2020 data gap — a structural limitation

The most significant limitation of this model is temporal. The time-based train/test split places all 2020–2026 disasters in the test set. This period contains the most expensive events in FEMA history:

- **COVID-19 (2020)** — 8 Biological DR declarations across NY, TX, FL, LA, NC, MI, VA at $1B–$18B each. Federal relief at unprecedented scale through mechanisms the model has never seen.
- **Hurricane Ida (2021)** — $4.9B in Louisiana alone.
- **Hurricane Ian (2022)** — $4.2B in Florida, $2.5B in Puerto Rico.
- **Hurricane Helene (2024)** — $2.6B in North Carolina via inland flooding with no historical precedent.
- **Hurricane Milton (2025)** — $2.4B in Florida.

These are not random failures — they are structurally concentrated in a period the model was never trained on. As FEMA finalizes obligations for 2020–2026 disasters and more training data accumulates, the model's catastrophic event performance is expected to improve significantly with annual retraining.

**The implication:** The LOW confidence tier is not just flagging uncertain disaster types — it is largely flagging a known temporal blind spot. Approximately 80% of LOW confidence predictions involve post-2020 events where the model has no training analog at the required cost scale.

---

## Future Enhancements

- **External data integration** — Census population, BEA regional GDP, NOAA storm intensity, USGS soil moisture for affected counties
- **Separate catastrophe model** — train on $500M+ events only once sufficient data accumulates
- **Annual retraining pipeline** — automated ingestion of new FEMA data and model refresh
- **Uncertainty quantification** — quantile regression or conformal prediction intervals alongside point estimates
- **Geographic enrichment** — county-level infrastructure replacement value from FHWA

---

## Authors

Ransford Nyarko — M.S. Artificial Intelligence, University of Michigan-Dearborn
