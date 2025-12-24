import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Encode categorical
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # Scaling
    scaler = StandardScaler()
    features = df.drop("Churn", axis=1)
    target = df["Churn"]

    features_scaled = scaler.fit_transform(features)

    df_clean = pd.DataFrame(features_scaled, columns=features.columns)
    df_clean["Churn"] = target.values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print("Preprocessing selesai. File disimpan di:", output_path)


if __name__ == "__main__":
    preprocess(
        "telco_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "preprocessing/data_clean/telco_clean.csv"
    )