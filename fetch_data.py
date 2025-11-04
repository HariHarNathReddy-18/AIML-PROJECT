import requests
import pandas as pd

def fetch_rainfall(lat, lon, start="2000-01-01", end="2020-12-31"):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,T2M,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    parameters = data["properties"]["parameter"]
    df = pd.DataFrame({
        "date": list(parameters["PRECTOTCORR"].keys()),
        "rainfall": list(parameters["PRECTOTCORR"].values()),
        "temperature": list(parameters["T2M"].values()),
        "humidity": list(parameters["RH2M"].values())
    })

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_rainfall(lat=17.3850, lon=78.4867, start="2000-01-01", end="2020-12-31")
    df.to_csv("rainfall_data.csv")
    print("✅ Data saved as rainfall_data.csv")
    print(df.head())
