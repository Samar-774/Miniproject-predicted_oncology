import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')

def run_precision_oncology_model1():
    print("=" * 80)
    print("PRECISION ONCOLOGY ENGINE – MODEL 1")
    print("Histology Classification + Explainability + Metrics")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load Preprocessed Data
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv("processed_data.csv")
    except FileNotFoundError:
        print("❌ processed_data.csv not found.")
        return

    class_target = "Histology_Encoded"
    X = df.drop(columns=[class_target, "Survival_Rate"], errors="ignore")
    y = df[class_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Train Model
    # ------------------------------------------------------------------
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    print("✓ Histology Classification Model Trained")
    
    # ==================================================================
    # PERSON A'S WORK: IEEE METRICS FOR TABLE I
    # ==================================================================
    y_pred = rf_classifier.predict(X_test)
    tumor_labels = ["Astrocytoma", "Glioblastoma", "Meningioma"] # FIXED: 3 Classes Only

    print("\n" + "=" * 80)
    print("MODEL 1 EVALUATION (FOR IEEE MANUSCRIPT)")
    print("=" * 80)
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=tumor_labels))

    # ==================================================================
    # PERSON B'S WORK: CONFUSION MATRIX VISUALIZATION
    # ==================================================================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=tumor_labels, yticklabels=tumor_labels)
    plt.xlabel("Predicted Tumor Type")
    plt.ylabel("Actual Tumor Type")
    plt.title("Confusion Matrix – Tumor Histology Classification")
    plt.tight_layout()
    plt.savefig("confusion_matrix_histology.png", dpi=300)
    plt.close()
    print("✓ Confusion matrix saved as confusion_matrix_histology.png")
    
    # ------------------------------------------------------------------
    # 3. SHAP Explainability (Terminal Outputs)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PATIENT-LEVEL DIAGNOSTIC EXPLANATIONS")
    print("=" * 80)

    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test)
    sample_patients = X_test.iloc[:3]

    for i in range(len(sample_patients)):
        patient_data = sample_patients.iloc[i:i + 1]
        patient_id = sample_patients.index[i]
        predicted_class = int(rf_classifier.predict(patient_data)[0])

        patient_shap = shap_values[predicted_class][i]
        top_feature_idx = np.argmax(np.abs(patient_shap))
        top_feature = X.columns[top_feature_idx]
        direction = "Positive" if patient_shap[top_feature_idx] > 0 else "Negative"

        print(f"PATIENT ID: {patient_id}")
        print(f"  - Predicted Histology Class: {tumor_labels[predicted_class]}")
        print(f"  - Primary Diagnostic Driver: {top_feature} ({direction} Influence)")
        print("-" * 40)

    # ==================================================================
    # PERSON A'S WORK: STACKED PROBABILITY HAND-OFF
    # ==================================================================
    probabilities = rf_classifier.predict_proba(X)
    prob_df = pd.DataFrame(probabilities, columns=['Prob_Astro', 'Prob_Glio', 'Prob_Meningioma'])
    prob_df.to_csv("predicted_probabilities_for_model2.csv", index=False)

    print("\n✓ 3-Class Probabilities saved for Model 2")
    print("=" * 80)

if __name__ == "__main__":
    run_precision_oncology_model1()