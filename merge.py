import pandas as pd
import numpy as np

print("Loading collisions.csv...")
collisions = pd.read_csv("collisions.csv", low_memory=False)

print("Loading weather.csv...")
weather = pd.read_csv("weather.csv")

collisions["CRASH DATE"] = pd.to_datetime(collisions["CRASH DATE"], errors="coerce")
collisions["CRASH TIME"] = pd.to_datetime(collisions["CRASH TIME"], format="%H:%M", errors="coerce")
collisions.dropna(subset=["CRASH DATE"], inplace=True)

weather["DATE"] = pd.to_datetime(weather["DATE"]).dt.date
weather["PRECIPITATION TYPE"] = weather["PRECIPITATION TYPE"].fillna("none")


for col_prefix, new_col in [
    ("CONTRIBUTING FACTOR VEHICLE", "CONTRIBUTING FACTOR VEHICLES"),
    ("VEHICLE TYPE CODE",           "VEHICLE TYPES"),
]:
    cols = [f"{col_prefix} {i}" for i in range(1, 6) if f"{col_prefix} {i}" in collisions.columns]
    collisions[new_col] = (
        collisions[cols]
        .apply(lambda row: ", ".join(v for v in row if pd.notna(v) and v != ""), axis=1)
    )

def build_street_info(row):
    names, types = [], []
    for col, label in [
        ("ON STREET NAME",    "ON STREET"),
        ("CROSS STREET NAME", "CROSS STREET"),
        ("OFF STREET NAME",   "OFF STREET"),
    ]:
        if col in row and pd.notna(row[col]) and row[col] != "":
            names.append(row[col])
            types.append(label)
    return pd.Series({
        "STREET NAME": ", ".join(names),
        "STREET TYPE": ", ".join(types),
    })

street_info = collisions.apply(build_street_info, axis=1)
collisions["STREET NAME"] = street_info["STREET NAME"]
collisions["STREET TYPE"] = street_info["STREET TYPE"]

numeric_cols = [
    "NUMBER OF PERSONS INJURED",    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF CYCLIST INJURED",    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF PERSONS KILLED",     "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST KILLED",     "NUMBER OF MOTORIST KILLED",
]
for col in numeric_cols:
    if col in collisions.columns:
        collisions[col] = pd.to_numeric(collisions[col], errors="coerce").fillna(0).astype(int)

if all(c in collisions.columns for c in numeric_cols[:4]):
    collisions["NUMBER OF INJURIES"] = (
        collisions["NUMBER OF PERSONS INJURED"] +
        collisions["NUMBER OF PEDESTRIANS INJURED"] +
        collisions["NUMBER OF CYCLIST INJURED"] +
        collisions["NUMBER OF MOTORIST INJURED"]
    )
if all(c in collisions.columns for c in numeric_cols[4:]):
    collisions["NUMBER OF DEATHS"] = (
        collisions["NUMBER OF PERSONS KILLED"] +
        collisions["NUMBER OF PEDESTRIANS KILLED"] +
        collisions["NUMBER OF CYCLIST KILLED"] +
        collisions["NUMBER OF MOTORIST KILLED"]
    )

collisions["DATE"] = collisions["CRASH DATE"].dt.date
collisions["HOUR"] = collisions["CRASH TIME"].dt.hour

collisions.dropna(subset=["HOUR"], inplace=True)
collisions["HOUR"] = collisions["HOUR"].astype(int)

print("Merging collisions with hourly weather...")
merged_raw = pd.merge(collisions, weather, on=["DATE", "HOUR"], how="left")

print("Aggregating to (date, hour) collision counts...")
merged = (
    merged_raw
    .groupby(["DATE", "HOUR"])
    .agg(
        COLLISION_COUNT=("COLLISION_ID",            "count"),
        TOTAL_INJURIES=("NUMBER OF INJURIES",       "sum"),
        TOTAL_DEATHS=("NUMBER OF DEATHS",           "sum"),
        TEMP=("TEMP",                               "first"),
        PRECIPITATION=("PRECIPITATION",             "first"),
        WEATHER_CODE=("WEATHER CODE",               "first"),
        WIND_SPEED=("WIND SPEED",                   "first"),
        WEATHER_DESCRIPTION=("WEATHER DESCRIPTION", "first"),
        PRECIPITATION_TYPE=("PRECIPITATION TYPE",   "first"),
    )
    .reset_index()
)

merged["DATE"]        = pd.to_datetime(merged["DATE"])
merged["DAY_OF_WEEK"] = merged["DATE"].dt.dayofweek
merged["DAY_NAME"]    = merged["DATE"].dt.day_name()
merged["MONTH"]       = merged["DATE"].dt.month
merged["YEAR"]        = merged["DATE"].dt.year
merged["IS_WEEKEND"]  = merged["DAY_OF_WEEK"].isin([5, 6]).astype(int)

merged["DATE"] = merged["DATE"].astype(str)
merged["PRECIPITATION_TYPE"] = merged["PRECIPITATION_TYPE"].fillna("none")
merged.replace("", np.nan, inplace=True)

merged.to_csv("merged.csv", index=False)
print(f"Saved merged.csv — {len(merged)} rows.")
print(merged.head())
print("\nColumns:", list(merged.columns))
print("\nMissing values:\n", merged.isnull().sum()[merged.isnull().sum() > 0])
