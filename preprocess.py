import pandas as pd

def preprocess(filename="rainfall_data.csv"):
    df = pd.read_csv(filename, parse_dates=["date"], index_col="date")
    monthly = df.resample("M").sum()
    monthly.to_csv("rainfall_monthly.csv")
    return monthly

if __name__ == "__main__":
    monthly = preprocess()
    print("✅ Monthly data saved as rainfall_monthly.csv")
    print(monthly.head())