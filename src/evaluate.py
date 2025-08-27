import joblib, os, yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()
    X_train, X_val, y_train, y_val = joblib.load(
        os.path.join(config["data"]["plk_path"], "dataset.pkl")
    )

    models = ["logreg", "rf", "lgb", "xgb"]
    model_titles = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "lgb": "LightGBM",
        "xgb": "XGBoost"
    }

    y_probas = {}
    aucs = {}
    y_preds = {}

    # === 1) ROC Curve comparative ===
    plt.figure(figsize=(8, 6))

    for name in models:
        model = joblib.load(os.path.join(config["data"]["plk_path"], f"{name}_model.pkl"))
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        y_preds[name] = y_pred
        y_probas[name] = y_proba
        aucs[name] = roc_auc_score(y_val, y_proba)

        # Metrics textuels
        print(f"\n####### Results for {name.upper()}")
        print("F1-score:", f1_score(y_val, y_pred))
        print("AUC:", aucs[name])
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
        print("Classification Report:\n", classification_report(y_val, y_pred))

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{model_titles[name]} (AUC = {aucs[name]:.4f})')

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparative ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    output_dir = os.path.join("results/figures/")
    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"\nROC curve saved to {roc_path}")

    # === 2) Matrices de confusion ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors=['Blues','Greens','Oranges','Reds']

    for ax, name, cmap in zip(axes.ravel(), models, colors):
        cm = confusion_matrix(y_val, y_preds[name])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax)
        ax.set_title(model_titles[name])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrices saved to {cm_path}")

if __name__ == "__main__":
    evaluate()
