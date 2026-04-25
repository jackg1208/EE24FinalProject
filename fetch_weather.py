import requests
import pandas as pd
from datetime import date

# NYC coordinates (Central Park)
LAT = 40.7829
LON = -73.9654

START_DATE = "2012-07-01"
END_DATE = date.today().isoformat()

print(f"Fetching NYC weather from {START_DATE} to {END_DATE}...")

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m",
        "precipitation",
        "weathercode",
        "windspeed_10m",
    ],
    "temperature_unit": "fahrenheit",
    "windspeed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": "America/New_York",
}

response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
response.raise_for_status()
data = response.json()

df = pd.DataFrame(data["hourly"])
df.rename(columns={
    "time":              "DATETIME",
    "temperature_2m":    "TEMP",
    "precipitation":     "PRECIPITATION",
    "weathercode":       "WEATHER CODE",
    "windspeed_10m":     "WIND SPEED",
}, inplace=True)

df["DATETIME"] = pd.to_datetime(df["DATETIME"])
df["DATE"] = df["DATETIME"].dt.date
df["HOUR"] = df["DATETIME"].dt.hour

# https://open-meteo.com/en/docs#weathervariables
WMO_DESCRIPTIONS = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

df["WEATHER DESCRIPTION"] = df["WEATHER CODE"].map(WMO_DESCRIPTIONS).fillna("Unknown")

def precip_type(row):
    code = row["WEATHER CODE"]
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    elif code in (51, 53, 55, 61, 63, 65, 80, 81, 82):
        return "rain"
    elif code in (95, 96, 99):
        return "rain"  
    else:
        return None

df["PRECIPITATION TYPE"] = df.apply(precip_type, axis=1)

df.to_csv("weather.csv", index=False)
print(f"Saved weather.csv — {len(df)} days of data.")
print(df.head())
