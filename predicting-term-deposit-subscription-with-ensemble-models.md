# Portfolio Project Data — Predicting Term Deposit Subscription with Ensemble Models

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/predicting_term_deposit_subscription/`
- **Required Local Images:** `banner.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "predicting-term-deposit-subscription-with-ensemble-models",
  "code": "PRO-22 // ML-05",
  "title": "Predicting Term Deposit Subscription with Ensemble Models",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Predicting-Term-Deposit-Subscription-with-Ensemble-Models",
  "license": "MIT License",
  "banner": "/images/projects/predicting_term_deposit_subscription/banner.webp",
  "summary": "Leakage-free ensemble ML pipeline predicting retail bank term deposit subscriptions with temporal validation and interactive Streamlit lead scoring.",
  "projectDescription": "End-to-end supervised ensemble classification pipeline predicting bank term deposit subscriptions across 45,211 telemarketing records. Excludes post-call duration to prevent leakage, uses chronological validation with Extra Trees, and serves real-time scoring via Streamlit.",
  "architectureSummary": "Chronological 80/20 Split -> PreContactFeatureEngineer -> ColumnTransformer -> 5-Fold TimeSeriesSplit Calibration -> Extra Trees Classifier -> Streamlit Lead Scoring",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Joblib",
    "Streamlit",
    "Pillow"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Ensemble ML Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "Retail Banking Telemarketing",
    "architecturePipeline": "Scikit-Learn ColumnTransformer & ExtraTrees",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Pre-Contact Lead Prioritization Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Retail Banking Telemarketing",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Zero Leakage Scoring",
      "Temporal Holdout Validation",
      "Interactive Streamlit App"
    ]
  },
  "mlProblemContext": "Retail banks rely on outbound telemarketing to grow term deposits, but only 11.70% of 45,211 calls convert. Standard models retain call duration, causing severe post-call leakage, while random splits hide macroeconomic concept drift across the 2008 to 2010 campaign timeline.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Bank Marketing",
    "source": "UCI Machine Learning Repository (ID: 222; DOI: 10.24432/C5K306)",
    "modality": "Tabular",
    "scale": "45,211 Contact Records (36,168 Train / 9,043 Holdout Test)",
    "inputShape": "45,211 x 15 Pre-Contact Features",
    "dataFormat": "CSV",
    "description": "Chronologically ordered Portuguese retail bank direct telemarketing records spanning 30 monthly blocks from May 2008 to November 2010 with zero duplicate rows, partitioned strictly 80/20 by row index.",
    "provenanceNote": "Published by Moro, Rita, and Cortez (2014) via the UCI Machine Learning Repository; stored in dataset/original/Bank_Marketing.csv with metadata.json."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Pre-Contact Feature Space",
    "spaceNote": "Strictly excludes post-call duration; adds engineered previously_contacted binary indicator.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "y (Term Deposit Subscription)",
      "type": "Binary Class",
      "distribution": "Overall: 88.30% no (39,922) vs. 11.70% yes (5,289); Train: 6.73% yes; Holdout Test: 31.58% yes"
    },
    "inputSignals": [
      {
        "name": "poutcome",
        "type": "Categorical",
        "importanceOrRole": "Rank 1 Driver (0.0801 ROC-AUC drop)",
        "description": "Outcome of the prior marketing campaign (success, failure, other, or Unknown)."
      },
      {
        "name": "month",
        "type": "Categorical",
        "importanceOrRole": "Rank 2 Driver (0.0207 ROC-AUC drop)",
        "description": "Scheduled calendar month of campaign outreach capturing strong seasonal responsiveness."
      },
      {
        "name": "housing",
        "type": "Categorical",
        "importanceOrRole": "Rank 3 Driver (0.0170 ROC-AUC drop)",
        "description": "Binary indicator of whether the client holds an active housing mortgage loan."
      },
      {
        "name": "marital",
        "type": "Categorical",
        "importanceOrRole": "Rank 4 Driver (0.0107 ROC-AUC drop)",
        "description": "Client marital status (married, single, or divorced) reflecting household liquidity allocation."
      },
      {
        "name": "contact",
        "type": "Categorical",
        "importanceOrRole": "Rank 5 Driver (0.0093 ROC-AUC drop)",
        "description": "Outreach communication channel (cellular, telephone, or Unknown)."
      },
      {
        "name": "previously_contacted",
        "type": "Numerical",
        "importanceOrRole": "Engineered Binary Flag",
        "description": "Derived indicator set to 1 when pdays != -1 (18.26% of records), capturing prior campaign touchpoints."
      }
    ],
    "dataCharacteristicsSummary": "The dataset exhibits a 7.5:1 class skew alongside strong temporal drift, where conversion rises from 6.73% in early batches to 31.58% in late re-contact campaigns. Categorical missingness concentrates in poutcome (81.75%) and contact (28.80%), while numeric predictors have zero missing values."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Binary Classification with Imbalance-Aware Threshold Calibration",
    "methodologyDescription": "Five supervised classifiers—Dummy, Logistic Regression, Random Forest, HistGradientBoosting, and Extra Trees—are trained within identical leakage-free pipelines. Models are evaluated on an 80/20 chronological holdout using 5-fold TimeSeriesSplit for out-of-fold F1 threshold calibration.",
    "justification": "Extra Trees with balanced class weights achieves the highest holdout PR-AUC (0.4907) and strong specificity without relying on post-call duration or optimistic random-split leakage."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Target Leakage Exclusion",
      "description": "PreContactFeatureEngineer drops the post-call duration column before any transformation so only pre-dialing attributes enter the model.",
      "badge": "Anti-Leakage Guard"
    },
    {
      "step": "02",
      "title": "Prior Contact Flag Derivation",
      "description": "Converts the sentinel pdays == -1 condition (81.74% of records, 100% consistent with previous == 0) into the binary previously_contacted feature.",
      "badge": "Feature Engineering"
    },
    {
      "step": "03",
      "title": "Explicit Unknown Categorical Imputation",
      "description": "Replaces missing values in poutcome, contact, education, and job with the explicit 'Unknown' category string instead of discarding 85%+ of records.",
      "badge": "Categorical Cleaning"
    },
    {
      "step": "04",
      "title": "ColumnTransformer Encoding & Scaling",
      "description": "Applies median SimpleImputer (plus StandardScaler for linear models) to 7 numeric columns and OneHotEncoder(handle_unknown='ignore') to 9 categorical columns.",
      "badge": "Pipeline Transform"
    }
  ],
  "mlDataPreparationNote": "All imputers, encoders, and scalers are wrapped inside Scikit-Learn Pipeline and ColumnTransformer objects fitted strictly on training partitions to eliminate cross-fold and temporal leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "ExtraTreesClassifier (Extremely Randomized Trees Ensemble)",
    "architectureSummary": "An ensemble of 200 randomized decision trees constrained to max_depth=12 with class_weight='balanced', wrapped behind PreContactFeatureEngineer and ColumnTransformer inside a unified Scikit-Learn Pipeline.",
    "configurationBadge": "200 Trees | Depth 12",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "200",
        "description": "Number of randomized decision trees averaged in the bagging ensemble."
      },
      {
        "name": "max_depth",
        "value": "12",
        "description": "Maximum tree depth limiting variance and overfitting on one-hot encoded categorical levels."
      },
      {
        "name": "min_samples_split",
        "value": "10",
        "description": "Minimum samples required to split an internal node for regularized partitioning."
      },
      {
        "name": "min_samples_leaf",
        "value": "4",
        "description": "Minimum samples required at leaf nodes to stabilize probability estimates."
      },
      {
        "name": "class_weight",
        "value": "balanced",
        "description": "Inversely weights minority subscriber instances during split scoring to counter the 7.5:1 class skew."
      },
      {
        "name": "decision_threshold",
        "value": "0.49",
        "description": "Optimal probability cutoff selected via 5-fold TimeSeriesSplit F1 sweep across 0.10 to 0.90."
      }
    ],
    "objectiveCriterion": "Gini Impurity with Balanced Class Weighting",
    "optimizationSolver": "Randomized Feature Split Selection & Out-of-Fold Threshold Sweep",
    "trainingProcess": "Training runs on the first 36,168 chronological records with deterministic random_state=42 and balanced class weighting. A 5-fold TimeSeriesSplit loop sweeps 81 decision thresholds from 0.10 to 0.90 to maximize validation F1 before refitting on the full training partition."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "80/20 Chronological Temporal Holdout & 5-Fold TimeSeriesSplit CV",
    "description": "Models are tuned on the first 36,168 rows via 5-fold TimeSeriesSplit and evaluated out-of-sample on the final 9,043 rows (31.58% positive prevalence), alongside post-contact leakage and random-split control benchmarks.",
    "metrics": [
      {
        "label": "HOLDOUT PR-AUC",
        "value": "0.4907",
        "benchmark": "Baseline 0.3158"
      },
      {
        "label": "HOLDOUT ROC-AUC",
        "value": "0.6817",
        "benchmark": "Baseline 0.5000"
      },
      {
        "label": "RECALL (TH=0.49)",
        "value": "69.08%",
        "benchmark": "Target >= 0.65"
      },
      {
        "label": "F1-SCORE (TH=0.49)",
        "value": "0.5303",
        "benchmark": "Target > 0.50"
      },
      {
        "label": "BALANCED ACCURACY",
        "value": "63.43%",
        "benchmark": "Baseline 0.5000"
      },
      {
        "label": "PRECISION (TH=0.49)",
        "value": "43.03%",
        "benchmark": "Baseline 0.3158"
      }
    ]
  },
  "mlPerformanceVisualizations": [],
  "mlVisualizationTelemetryNote": "5-fold TimeSeriesSplit CV on the early 6.73% prevalence cohort yielded 0.0924 mean PR-AUC, while the late 31.58% prevalence temporal holdout reached 0.4907 PR-AUC; static PNG evaluation plots are stored in figures/.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Permutation importance on the temporal holdout identifies prior campaign outcome (0.0801 ROC-AUC drop), contact month (0.0207), and housing loan status (0.0170) as the primary pre-contact conversion drivers, while excluded duration registers zero importance inside the pipeline.",
    "keyDrivers": [
      {
        "factor": "poutcome",
        "impact": "+0.0801 mean ROC-AUC decrease when permuted across 5 repeats on holdout data"
      },
      {
        "factor": "month",
        "impact": "+0.0207 mean ROC-AUC decrease reflecting seasonal campaign conversion peaks"
      },
      {
        "factor": "housing",
        "impact": "+0.0170 mean ROC-AUC decrease separating mortgage-free liquidity profiles"
      },
      {
        "factor": "marital & contact",
        "impact": "+0.0107 and +0.0093 mean ROC-AUC decrease across household and channel segments"
      }
    ],
    "limitations": [
      {
        "title": "Macroeconomic Concept Drift",
        "description": "Subscription prevalence shifts from 6.73% in the training slice to 31.58% in the holdout slice due to increased re-contact targeting, requiring ongoing drift monitoring."
      },
      {
        "title": "High Channel Missingness",
        "description": "Over 28.80% of records lack a recorded communication channel (contact = 'Unknown'), limiting pre-call channel granularity."
      },
      {
        "title": "Absence of Macro Interest Indicators",
        "description": "This dataset variant omits external macroeconomic variables such as benchmark Euribor rates or consumer confidence indices."
      }
    ],
    "futureImprovements": [
      {
        "title": "Macroeconomic Feature Enrichment",
        "target": "Integrate central bank interest rates and CPI indicators to capture macro regime shifts"
      },
      {
        "title": "Rolling Window Retraining & Calibration",
        "target": "Schedule quarterly probability recalibration and automated PSI drift alerts in production"
      }
    ],
    "discoveryNote": "Permutation importance quantifies predictive association within historical outreach workflows rather than causal intervention effects."
  },
  "mlModelArtifacts": [
    {
      "name": "best_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "metrics_summary.json",
      "format": "JSON"
    },
    {
      "name": "permutation_importance.json",
      "format": "JSON"
    },
    {
      "name": "plot_data.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Binary Classification",
    "modelArchitecture": "Extra Trees Ensemble",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Chronological Temporal Holdout"
  }
}
```
