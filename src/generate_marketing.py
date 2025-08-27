import pandas as pd
import numpy as np

def generate_marketing_data(df, seed=42):
    np.random.seed(seed)
    marketing = pd.DataFrame({
        "customerID": df["customerID"],
        "emails_opened": np.random.poisson(3, len(df)),
        "campaigns_received": np.random.randint(1, 10, len(df)),
        "promo_used": np.random.binomial(1, 0.3, len(df))
    })
    return marketing

if __name__ == "__main__":
    telco = pd.read_csv("data/raw/telco_churn.csv")
    marketing = generate_marketing_data(telco)
    marketing.to_csv("data/raw/marketing.csv", index=False)
