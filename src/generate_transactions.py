import pandas as pd
import numpy as np
import os

# Créer dossier si nécessaire
os.makedirs("data/raw", exist_ok=True)

# Définir les 3 nouveaux clients
new_clients = pd.DataFrame([
    {
        "customerID": "0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5
    },
    {
        "customerID": "0002",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 56.25,
        "TotalCharges": 282.5
    },
    {
        "customerID": "0003",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 24,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 99.65,
        "TotalCharges": 2391.6
    }
])

# Générer les données marketing
np.random.seed(42)
marketing = pd.DataFrame({
    "customerID": new_clients["customerID"],
    "emails_opened": np.random.poisson(3, len(new_clients)),
    "campaigns_received": np.random.randint(1, 10, len(new_clients)),
    "promo_used": np.random.binomial(1, 0.3, len(new_clients))
})

# Générer les données transactions
transactions = pd.DataFrame({
    "customerID": new_clients["customerID"],
    "monthly_usage_minutes": np.random.normal(300, 50, len(new_clients)),
    "avg_monthly_bill": np.random.normal(70, 15, len(new_clients)),
    "payment_delays": np.random.binomial(1, 0.1, len(new_clients))
})

# Générer les données support
support = pd.DataFrame({
    "customerID": new_clients["customerID"],
    "num_tickets": np.random.poisson(0.5, len(new_clients)),
    "avg_response_time": np.random.uniform(1, 10, len(new_clients)),
    "last_ticket_days_ago": np.random.randint(0, 100, len(new_clients))
})

# Sauvegarder tous les CSV
new_clients.to_csv("data/raw/new_clients.csv", index=False)
marketing.to_csv("data/raw/marketing_new.csv", index=False)
transactions.to_csv("data/raw/transactions_new.csv", index=False)
support.to_csv("data/raw/support_new.csv", index=False)

print("CSV files for 3 new clients saved in data/raw/")
