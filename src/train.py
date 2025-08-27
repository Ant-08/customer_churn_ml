import joblib, yaml, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import numpy as np

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Config chargée :", config)
    return config

def train_models():



    config = load_config()
    X_train, X_val, y_train, y_val = joblib.load(os.path.join(config["data"]["plk_path"], "dataset.pkl"))

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_w = n_neg / n_pos

    models = {}

    # Logistic Regression
    models["logreg"] = LogisticRegression(C=config["models"]["logistic_regression"]["C"], max_iter=5000,
        random_state=42, class_weight="balanced")
    models["logreg"].fit(X_train, y_train)

    # Random Forest
    models["rf"] = RandomForestClassifier(
        n_estimators=config["models"]["random_forest"]["n_estimators"],
        max_depth=config["models"]["random_forest"]["max_depth"],
        min_samples_split=config["models"]["random_forest"]["min_samples_split"],
        random_state=42, class_weight="balanced"
    )
    models["rf"].fit(X_train, y_train)

    # LightGBM
    models["lgb"] = lgb.LGBMClassifier(
        objective='binary',
        learning_rate=config["models"]["lightgbm"]["learning_rate"],
        n_estimators=config["models"]["lightgbm"]["n_estimators"],
        num_leaves=config["models"]["lightgbm"]["num_leaves"],
        random_state=42, scale_pos_weight=scale_pos_w 
    )
    models["lgb"].fit(X_train, y_train)

    # XGBoost
    models["xgb"] = xgb.XGBClassifier(
        learning_rate=config["models"]["xgboost"]["learning_rate"],
        max_depth=config["models"]["xgboost"]["max_depth"],
        n_estimators=config["models"]["xgboost"]["n_estimators"],
        subsample=config["models"]["xgboost"]["subsample"],
        use_label_encoder=False, scale_pos_weight=scale_pos_w,
        eval_metric="logloss", random_state=42,
    )
    models["xgb"].fit(X_train, y_train)

    # Sauvegarde
    for name, model in models.items():
        joblib.dump(model, os.path.join(config["data"]["plk_path"], f"{name}_model.pkl"))

    print(" Training over, models save")

if __name__ == "__main__":
    train_models()
