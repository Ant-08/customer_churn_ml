import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Créer le dossier figures s'il n'existe pas
os.makedirs("results/figures", exist_ok=True)

# Charger les modèles
models = {
    "LogReg": joblib.load("data/plk/logreg_model.pkl"),
    "RandomForest": joblib.load("data/plk/rf_model.pkl"),
    "LightGBM": joblib.load("data/plk/lgb_model.pkl"),
    "XGBoost": joblib.load("data/plk/xgb_model.pkl")
}

# Charger scaler et encoders
scaler = joblib.load("data/plk/scaler.pkl")
encoders = joblib.load("data/plk/encoders.pkl")

# Charger les nouveaux clients
new_clients = pd.read_csv("data/raw/new_clients.csv")

# Appliquer les encoders sauvegardés aux colonnes catégorielles
for col, le in encoders.items():
    if col in new_clients.columns:
        # Mapper les valeurs inconnues sur -1
        new_clients[col] = new_clients[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Identifier les colonnes numériques connues par le scaler
num_cols_scaled = scaler.feature_names_in_
new_clients[num_cols_scaled] = scaler.transform(new_clients[num_cols_scaled])

# Charger X_train pour récupérer l’ordre exact des colonnes
X_train, _, _, _ = joblib.load("data/plk/dataset.pkl")

# Réorganiser X_new exactement comme X_train
X_new = new_clients[X_train.columns]

# Prédictions pour chaque modèle
results = {}
for name, model in models.items():
    results[name] = model.predict_proba(X_new)[:,1]

# Transformer en DataFrame pour le plotting
proba_df = pd.DataFrame(results, index=new_clients["customerID"])
print(proba_df)

# Plot
ax = proba_df.plot(kind='bar', figsize=(10,6))
plt.title("Churn Probability for New Clients by Model")
plt.ylabel("Predicted Probability of Churn")
plt.xlabel("Customer ID")
plt.ylim(0,1)
plt.legend(title="Model")
plt.grid(axis='y')
plt.tight_layout()

# Enregistrer le graphique
plt.savefig("results/figures/new_clients_churn.png")
plt.close()

print("Graph saved to results/figures/new_clients_churn.png")
