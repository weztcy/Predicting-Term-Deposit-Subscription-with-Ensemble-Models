# Portfolio Project Data — Predicting Term Deposit Subscription with Ensemble Models

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/predicting_term_deposit_subscription/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image10.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "predicting-term-deposit-subscription-with-ensemble-models",
  "code": "PRO-20 // ML-05",
  "title": "Predicting Term Deposit Subscription with Ensemble Models",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Predicting-Term-Deposit-Subscription-with-Ensemble-Models",
  "license": "MIT License",
  "banner": "/images/projects/predicting_term_deposit_subscription/banner.webp",
  "summary": "Leakage-free ensemble ML pipeline and Streamlit lead-scoring app predicting bank term deposit subscriptions under temporal concept drift.",
  "projectDescription": "An end-to-end supervised learning pipeline predicting retail bank term deposit subscriptions before outbound dialing. Excluding post-call duration prevents target leakage, while chronological temporal validation and Extra Trees scoring rank 45,211 leads via a Streamlit interface.",
  "architectureSummary": "CSV Ingestion -> Chronological Split -> PreContactFeatureEngineer (Drop Duration) -> ColumnTransformer -> TimeSeriesSplit Tuning -> Extra Trees Pipeline -> Streamlit UI",
  "techStack": [
    "Python",
    "Scikit-learn",
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
    "projectType": "Supervised Ensemble Classification Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web Application",
    "primaryDomain": "Retail Banking Telemarketing Scoring",
    "architecturePipeline": "Anti-Leakage Pipeline -> Extra Trees",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Pre-Contact Lead Prioritization Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Tabular Retail Banking Marketing",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "Zero Target Leakage",
      "Temporal Drift Validation",
      "Ensemble Lead Scoring"
    ]
  },
  "mlProblemContext": "Direct telemarketing campaigns suffer high labor waste and contact fatigue when only 11.70% of 45,211 prospects subscribe to term deposits. Operational models must rank leads prior to dialing without leaking call duration while remaining robust to severe temporal concept drift across campaign batches.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Bank Marketing 45.2K",
    "source": "UCI Machine Learning Repository (Dataset ID 222, Portuguese Retail Bank, May 2008 – Nov 2010)",
    "modality": "Tabular",
    "scale": "45,211 records (36,168 temporal train / 9,043 temporal holdout test)",
    "inputShape": "N x 15 Pre-Contact Raw Features + 1 Engineered Indicator",
    "dataFormat": "CSV",
    "description": "Contains 45,211 chronologically ordered telemarketing contact rows with zero duplicates, 16 candidate predictors, and binary target y (39,922 no / 88.30%, 5,289 yes / 11.70%). Partitioned strictly by chronological index into an 80% early training set (6.73% positive) and a 20% late holdout test set (31.58% positive).",
    "provenanceNote": "Public benchmark dataset (DOI: 10.24432/C5K306; Moro, Rita, & Cortez, 2014) stored with original metadata descriptor."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Pre-Contact Feature & Target Space",
    "spaceNote": "Call duration is strictly excluded; missing categoricals map to Unknown and pdays == -1 is encoded as previously_contacted.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "y (Term Deposit Subscription)",
      "type": "Binary Class",
      "distribution": "88.30% no (39,922) vs 11.70% yes (5,289); 31.58% yes in temporal holdout"
    },
    "inputSignals": [
      {
        "name": "poutcome",
        "type": "Categorical",
        "importanceOrRole": "Rank 1 (0.0801 ROC-AUC drop)",
        "description": "Outcome of prior marketing campaign (success, failure, other, Unknown); strongest pre-contact conversion signal."
      },
      {
        "name": "month",
        "type": "Categorical",
        "importanceOrRole": "Rank 2 (0.0207 ROC-AUC drop)",
        "description": "Scheduled contact month capturing strong seasonal conversion peaks in March, September, October, and December."
      },
      {
        "name": "housing",
        "type": "Categorical",
        "importanceOrRole": "Rank 3 (0.0170 ROC-AUC drop)",
        "description": "Indicates active mortgage obligation, separating debt-encumbered clients from prospects with higher liquid surplus."
      },
      {
        "name": "marital",
        "type": "Categorical",
        "importanceOrRole": "Rank 4 (0.0107 ROC-AUC drop)",
        "description": "Client marital status (married, single, divorced) reflecting distinct savings priorities and liquidity profiles."
      },
      {
        "name": "contact",
        "type": "Categorical",
        "importanceOrRole": "Rank 5 (0.0093 ROC-AUC drop)",
        "description": "Outreach communication channel (cellular, telephone, Unknown), where cellular reaches 14.9% conversion vs 4.1% unknown."
      },
      {
        "name": "previously_contacted",
        "type": "Binary Numerical",
        "importanceOrRole": "Engineered Flag",
        "description": "Derived binary indicator (pdays != -1) capturing historical engagement without treating sentinel -1 as continuous distance."
      },
      {
        "name": "duration",
        "type": "Numerical",
        "importanceOrRole": "Excluded (0.0000)",
        "description": "Call length in seconds; deliberately purged before ColumnTransformer encoding to prevent post-contact target leakage."
      }
    ],
    "dataCharacteristicsSummary": "The dataset exhibits a 7.5:1 overall class imbalance and sharp temporal concept drift, rising from 6.73% subscription in early rows to 31.58% in the final 20%. Prior contact indicators (pdays and previous) correlate strongly at 0.99 Spearman rank, while categorical missingness is informative."
  },
  "mlLearningApproach": {
    "paradigm": "Imbalanced Supervised Binary Classification Under Chronological Temporal Drift",
    "methodologyDescription": "Five classifiers—Dummy baseline, Logistic Regression, Random Forest, HistGradientBoosting, and Extra Trees—are trained within leak-free scikit-learn pipelines using balanced class weights. Hyperparameter and probability cutoff calibration run on 5-fold TimeSeriesSplit before holdout testing.",
    "justification": "Chronological holdout and TimeSeriesSplit prevent future macroeconomic leakage that inflates random-split ROC-AUC to 0.8023, while PR-AUC selection with balanced class weights prioritizes high-density lead ranking over deceptive majority-class accuracy."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Post-Contact Leakage Removal",
      "description": "PreContactFeatureEngineer drops the duration column inside the pipeline transform step when exclude_duration=True to guarantee strict pre-dialing inference compatibility.",
      "badge": "Anti-Leakage"
    },
    {
      "step": "02",
      "title": "Sentinel & Missingness Encoding",
      "description": "Derives binary previously_contacted (pdays != -1) and imputes missing values in poutcome (81.75%), contact (28.80%), education (4.11%), and job (0.64%) as explicit Unknown strings.",
      "badge": "Feature Eng"
    },
    {
      "step": "03",
      "title": "ColumnTransformer Encoding & Scaling",
      "description": "Applies median SimpleImputer (plus StandardScaler for linear models) on numeric features and OneHotEncoder(handle_unknown='ignore') on the nine categorical attributes.",
      "badge": "Transformation"
    },
    {
      "step": "04",
      "title": "Chronological 80/20 Partitioning",
      "description": "Splits ordered rows at index 36,168 into training (36,168 rows) and temporal holdout test (9,043 rows) partitions without shuffling across campaign eras.",
      "badge": "Temporal Split"
    }
  ],
  "mlDataPreparationNote": "All imputers, scalers, and one-hot encoders are fitted strictly inside sklearn Pipeline objects on training folds only, eliminating cross-fold preprocessing leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Extra Trees Classifier (Extremely Randomized Trees Pipeline)",
    "architectureSummary": "End-to-end scikit-learn Pipeline chaining PreContactFeatureEngineer, ColumnTransformer preprocessing, and a 200-tree ExtraTreesClassifier ensemble with balanced class weights and depth-regularized splits.",
    "configurationBadge": "200 Trees / Balanced",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "200",
        "description": "Number of randomized decision trees averaged in the ensemble."
      },
      {
        "name": "max_depth",
        "value": "12",
        "description": "Caps maximum tree depth to regularize variance under temporal drift."
      },
      {
        "name": "min_samples_split",
        "value": "10",
        "description": "Minimum samples required to split an internal node."
      },
      {
        "name": "min_samples_leaf",
        "value": "4",
        "description": "Minimum leaf size preventing overly narrow terminal rules."
      },
      {
        "name": "class_weight",
        "value": "balanced",
        "description": "Inversely weights classes by frequency to counter the minority positive class."
      },
      {
        "name": "optimal_threshold",
        "value": "0.49",
        "description": "Calibrated decision cutoff selected across 81 candidates on 5-fold TimeSeriesSplit OOF predictions."
      }
    ],
    "objectiveCriterion": "Gini Impurity with Balanced Class Weights & F1 Threshold Optimization",
    "optimizationSolver": "Randomized Axis-Aligned Split Search with 5-Fold TimeSeriesSplit Calibration",
    "trainingProcess": "Training executes an 80/20 chronological split (36,168 train, 9,043 test) followed by 5-fold TimeSeriesSplit out-of-fold validation to sweep F1-maximizing thresholds from 0.10 to 0.90. Final pipelines refit on the full training partition, selecting Extra Trees by highest holdout PR-AUC (0.4907)."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "80/20 Chronological Temporal Holdout & 5-Fold TimeSeriesSplit",
    "description": "Models are validated on 5-fold TimeSeriesSplit inside the training era, selected on unseen temporal holdout PR-AUC (N = 9,043), and contrasted against Post-Contact leakage and Random-Split benchmarks.",
    "metrics": [
      {
        "label": "HOLDOUT PR-AUC",
        "value": "0.4907",
        "benchmark": "Baseline: 0.3158"
      },
      {
        "label": "HOLDOUT ROC-AUC",
        "value": "0.6817",
        "benchmark": "Chance: 0.5000"
      },
      {
        "label": "RECALL (T=0.49)",
        "value": "69.08%",
        "benchmark": "Target > 65.00%"
      },
      {
        "label": "PRECISION (T=0.49)",
        "value": "43.03%",
        "benchmark": "Baseline: 31.58%"
      },
      {
        "label": "F1-SCORE (T=0.49)",
        "value": "0.5303",
        "benchmark": "Target > 0.5000"
      },
      {
        "label": "BALANCED ACC",
        "value": "63.43%",
        "benchmark": "Baseline: 50.00%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Lead Profile Form (Pre-Contact Inputs)",
      "badge": "WEB APP UI",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image01.webp",
      "telemetryNote": "Three-column pre-call client intake form enforcing zero duration input."
    },
    {
      "title": "Streamlit Propensity Scoring & Queue Allocation",
      "badge": "INFERENCE UI",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image02.webp",
      "telemetryNote": "Real-time 46.3% probability score (+14.7% vs 31.6% baseline) and Low Priority routing."
    },
    {
      "title": "Rolling Subscription Rate & Monthly Drift",
      "badge": "TEMPORAL DRIFT",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image03.webp",
      "telemetryNote": "Subscription rate shifts from 6.73% (train) to 31.58% (holdout) at row 36,168."
    },
    {
      "title": "Key Numeric Feature Distributions",
      "badge": "EDA HISTOGRAMS",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image04.webp",
      "telemetryNote": "Right-skewed balance and campaign distributions across 45,211 records."
    },
    {
      "title": "Permutation Importance & Threshold Trade-Off",
      "badge": "EXPLAINABILITY",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image05.webp",
      "telemetryNote": "poutcome leads ROC-AUC drop (0.0801); optimal CV cutoff sits at 0.49."
    },
    {
      "title": "Categorical Conversion Rate Breakdown",
      "badge": "EDA BARPLOTS",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image06.webp",
      "telemetryNote": "Students (28.7%), retirees (22.8%), and Mar/Sep/Oct/Dec outperform 11.7% baseline."
    },
    {
      "title": "ROC & Precision-Recall Benchmark Curves",
      "badge": "MODEL CURVES",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image07.webp",
      "telemetryNote": "Extra Trees leads PR-AUC (0.491) while exposing leakage inflation (0.557 with duration)."
    },
    {
      "title": "Missing Value Audit & Target Class Skew",
      "badge": "DATA QUALITY",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image08.webp",
      "telemetryNote": "poutcome (81.75%) and contact (28.80%) missingness alongside 88.3% vs 11.7% target."
    },
    {
      "title": "Numeric Features Stratified by Target Outcome",
      "badge": "EDA BOXPLOTS",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image09.webp",
      "telemetryNote": "Subscribers display higher median balances and prior campaign contact counts."
    },
    {
      "title": "Correlation Heatmap, Model Comparison & Confusion Matrix",
      "badge": "EVALUATION GRID",
      "imageUrl": "/images/projects/predicting_term_deposit_subscription/image10.webp",
      "telemetryNote": "Extra Trees at T=0.49 captures 1,973 TP vs 1,896 TP at default T=0.50."
    }
  ],
  "mlVisualizationTelemetryNote": "Chronological evaluation reveals significant distribution shift between 2008–2010 eras, making holdout PR-AUC and calibrated thresholding essential for realistic deployment telemetry.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Permutation importance across five holdout repeats measures ROC-AUC drop per raw predictor. Prior campaign success (poutcome) dominates with a 0.0801 decrease, followed by seasonality (month, 0.0207), housing loan status (0.0170), marital status (0.0107), and cellular contact channel (0.0093).",
    "keyDrivers": [
      {
        "factor": "Prior Campaign Outcome (poutcome)",
        "impact": "+0.0801 mean ROC-AUC permutation importance; strongest signal for repeat conversion."
      },
      {
        "factor": "Contact Month Seasonality (month)",
        "impact": "+0.0207 mean ROC-AUC importance; March, September, October, and December exceed 43% conversion."
      },
      {
        "factor": "Housing Loan Status (housing)",
        "impact": "+0.0170 mean ROC-AUC importance; mortgage-free leads exhibit stronger deposit capacity."
      },
      {
        "factor": "Cellular Communication Channel (contact)",
        "impact": "+0.0093 mean ROC-AUC importance; cellular contacts convert at 14.9% vs 4.1% unknown."
      }
    ],
    "limitations": [
      {
        "title": "Severe Temporal Concept Drift",
        "description": "Subscription prevalence rises from 6.73% in training batches to 31.58% in the holdout window as re-contacted leads grow from 10.56% to 49.09%."
      },
      {
        "title": "High Channel & Prior-Outcome Missingness",
        "description": "Over 28.80% of contact channel records and 81.75% of prior campaign outcomes are missing and mapped to Unknown."
      },
      {
        "title": "Absence of Macroeconomic Indicators",
        "description": "External benchmark rates (such as Euribor and consumer confidence indices) are not included in this 16-feature dataset version."
      }
    ],
    "futureImprovements": [
      {
        "title": "Macroeconomic Rate Enrichment",
        "target": "Integrate Euribor and CPI feeds to stabilize cross-year probability calibration."
      },
      {
        "title": "Automated Drift & Retraining Pipeline",
        "target": "Trigger quarterly rolling-window retraining and threshold recalibration on live campaign logs."
      },
      {
        "title": "CRM Channel Data Enrichment",
        "target": "Reduce Unknown contact entries below 5% before campaign list scoring."
      }
    ],
    "discoveryNote": "Calibrated Extra Trees (T=0.49) captures 69.08% of total subscribers (1,973 of 2,856) while contacting only 4,585 of 9,043 leads, reducing call volume by 49.3%."
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
    },
    {
      "name": "audit_summary.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Learning",
    "modelArchitecture": "Extra Trees Classifier",
    "coreFramework": "Scikit-learn",
    "validationMethod": "Chronological Temporal Holdout"
  }
}
```
