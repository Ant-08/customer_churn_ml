import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

def load_and_merge():
    df = pd.read_csv("data/raw/telco_churn.csv")
    support = pd.read_csv("data/raw/support.csv")
    transactions = pd.read_csv("data/raw/transactions.csv")
    marketing = pd.read_csv("data/raw/marketing.csv")

    df_merged = df.merge(support, on="customerID", how="left")
    df_merged = df_merged.merge(transactions, on="customerID", how="left")
    df_merged = df_merged.merge(marketing, on="customerID", how="left")
    
    return df_merged

def preprocess(df):
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns.drop('customerID')
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Fill NaN
    df = df.fillna(0)

    # Scale numeric
    num_cols = df.select_dtypes(include='float64').columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, encoders, scaler

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    df = load_and_merge()
    df_clean, encoders, scaler = preprocess(df)

    df_clean.to_csv("data/processed/df_clean.csv", index=False)

    # Split train/val
    X = df_clean.drop(["Churn", "customerID"], axis=1)
    y = df_clean["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Sauvegardes
    joblib.dump((X_train, X_val, y_train, y_val), "data/plk/dataset.pkl")
    joblib.dump(encoders, "data/plk/encoders.pkl")
    joblib.dump(scaler, "data/plk/scaler.pkl")

    print(" Preprocessing done. Datasets + encoders + scaler saved in data/processed/")
