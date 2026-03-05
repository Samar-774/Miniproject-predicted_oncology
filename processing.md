The data transformation process converts the raw brain_tumor_dataset.csv into a fully numerical feature matrix suitable for tree-based ensemble learning models such as Random Forest and XGBoost. All categorical clinical variables were encoded numerically, and selected continuous variables were normalized to ensure numerical stability and consistent contribution during model training and explainability analysis.

////////////////////////////////////////////////////////////////////////////////////////////////////

Feature Removal

Dropped Column: Patient_ID

Reasoning:
Patient_ID is a high-cardinality unique identifier with no clinical or predictive relevance. Retaining this column could allow the model to memorize individual records rather than learn generalizable clinical patterns, increasing the risk of overfitting. The column was therefore removed prior to training.

////////////////////////////////////////////////////////////////////////////////////////////////////

Symptom Feature Engineering

Raw Symptom Columns:

Symptom_1

Symptom_2

Symptom_3

Transformation Applied:
A multi-label binary indicator mapping strategy was implemented. A Boolean OR operation was applied across the three symptom columns. If a patient exhibited a symptom in any of the three fields, the corresponding binary indicator was set to 1, otherwise 0.

New Engineered Features:

Has_Headache

Has_Nausea

Has_Seizures

Has_Vision_Issues

This approach preserves symptom presence while removing positional dependency from the original columns.

////////////////////////////////////////////////////////////////////////////////////////////////////

Numerical Normalization (Z-Score)

Normalization Method:
Z-score normalization (StandardScaler) was applied to selected continuous clinical variables.

Normalized Features:

Age

Tumor_Size

Tumor_Growth_Rate

After normalization:

Mean ≈ 0

Standard Deviation ≈ 1

This ensures that continuous variables operate on comparable scales, supporting stable model training and reliable SHAP-based feature attribution.

////////////////////////////////////////////////////////////////////////////////////////////////////

Target Variable Mapping
Histology Encoding (Histology_Encoded)

The tumor histology label was encoded as a multi-class numerical variable:

0 → Astrocytoma

1 → Glioblastoma

2 → Medulloblastoma

3 → Meningioma

This encoded target is used exclusively by Model 1 (Histology Classifier).

Survival Target (Survival_Rate)

Continuous numerical target

Range: 0–100 (%)

No transformation applied

The survival rate is preserved in its original scale to maintain direct clinical interpretability and is used exclusively by Model 2 (Survival Prognosticator).

////////////////////////////////////////////////////////////////////////////////////////////////////

Ordinal and Categorical Feature Encoding
Ordinal Encoding

Stage Encoding (Stage_Encoded):

1 → Stage I

2 → Stage II

3 → Stage III

4 → Stage IV

Ordinal encoding preserves the inherent progression of tumor severity.

Binary / Categorical Encoding

The following clinical attributes were encoded numerically:

Gender_Encoded → 0: Female, 1: Male

Tumor_Type_Encoded → 0: Benign, 1: Malignant

Location_Encoded → 0: Frontal, 1: Occipital, 2: Parietal, 3: Temporal

MRI_Result_Encoded → 0: Negative, 1: Positive

Family_History_Encoded → 0: No, 1: Yes

Radiation_Treatment_Encoded → 0: No, 1: Yes

Surgery_Performed_Encoded → 0: No, 1: Yes

Chemotherapy_Encoded → 0: No, 1: Yes

Follow_Up_Required_Encoded → 0: No, 1: Yes

All encoded features are numeric and compatible with tree-based learning models.

////////////////////////////////////////////////////////////////////////////////////////////////////

Final Dataset Characteristics

Fully numerical feature space

No identifier leakage

No missing values

Clear separation of targets for multi-stage modeling

The processed dataset is optimized for:

Model 1: Histology Classification (Random Forest)

Model 2: Survival Rate Prediction (XGBoost)

Explainability: SHAP-based global and patient-level clinical interpretation