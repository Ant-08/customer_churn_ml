import pandas as pd
import numpy as np

def generate_support_data(df, seed=42):
    np.random.seed(seed)
    support = pd.DataFrame({
        "customerID": df["customerID"],
        "num_tickets": np.random.poisson(0.5, len(df)),
        "avg_response_time": np.random.uniform(1, 10, len(df)),
        "last_ticket_days_ago": np.random.randint(0, 100, len(df))
    })
    return support

if __name__ == "__main__":
    telco = pd.read_csv("data/raw/telco_churn.csv")
    support = generate_support_data(telco)
    support.to_csv("data/raw/support.csv", index=False)
