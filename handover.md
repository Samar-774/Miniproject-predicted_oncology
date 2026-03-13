Technical Handover: Precision Oncology Pipeline Results
1. Model 1: Tumor Histology Classification
Goal: Categorize tumors into three classes (Astrocytoma, Glioblastoma, Meningioma) using a Random Forest classifier.

Performance Metrics
Global Accuracy: 99%.

Class-Specific Metrics:

Astrocytoma: 1.00 Precision, 0.98 Recall.

Glioblastoma: 0.98 Precision, 1.00 Recall.

Meningioma: 1.00 Precision, 1.00 Recall.

Clinical Interpretability (SHAP):

Primary Diagnostic Drivers: Patient Age and encoded Tumor Stage were identified as the highest-impact features for histology differentiation.

Visual Evidence: Reference confusion_matrix_histology.png for the class-wise error distribution.

2. Model 2: Survival Prognostication
Goal: Predict patient survival rates using an XGBoost regressor integrated with Model 1 probabilities (Stacked Architecture).

Predictive Accuracy
Coefficient of Determination (R^2): 0.813, indicating the model explains approximately 81% of the 
variance in survival data.

Error Metrics: RMSE of 5.25% and MAE of 2.25%.

Ablation Study (The "Stacked" Advantage)
This section proves the technical contribution of our 2-stage pipeline:

Baseline RMSE (Clinical Only): 6.07%.

Stacked RMSE (Integrated with Model 1): 5.25%.

Net Improvement: 0.81 percentage points reduction in error by incorporating histology probabilities.

3. Visual Assets & Mapping
Please place these figures in the Results section of the manuscript:


Figure Name                                  purpose
confusion_matrix_histology.png            Shows Model 1's classification precision

shap_beeswarm_survival.png                Displays global feature importance for survival

shap_waterfall_high_risk.png              Case study explaining a high-risk prognosis

shap_waterfall_low_risk.png               Case study explaining a low-risk prognosis.

