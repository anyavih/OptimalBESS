from datetime import date
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

NORTH = {"Actual": ["EAST", "NORTH_C"], "Forecast": ["East", "NorthCentral"]}
SOUTH = {"Actual": ["SOUTHERN", "SOUTH_C"], "Forecast": ["Southern", "SouthCentral"]}
WEST = {"Actual": ["WEST", "FAR_WEST", "NORTH"], "Forecast": ["West", "FarWest", "North"]}
HOUSTON = {"Actual": ["COAST"], "Forecast": ["Coast"]}
ZONES = {"NORTH": NORTH, "SOUTH": SOUTH, "WEST": WEST, "HOUSTON": HOUSTON}

def date_format(df, name):
    df[name] = pd.to_datetime(df[name], format="%m/%d/%Y")
    df["Date"] = df[name].dt.strftime("%m-%d-%Y")
    df["Hour Ending"] = df["HourEnding"].str.replace(":00", "", regex=False).astype(int)

def get_actual(year, ercot, endpoint="/np6-345-cd/act_sys_load_by_wzn"):
    start = date(year, 1, 2)
    end = date(year + 1, 1, 2)
    df = ercot.get_historical_data(endpoint, start_date=start, end_date=end)
    date_format(df, "OperDay")
    for zone in ZONES:
        df[zone] = df[ZONES[zone]["Actual"]].sum(axis=1)
    actual = df.melt(id_vars=["Date", "Hour Ending", "DSTFlag"],
                 value_vars=["NORTH", "SOUTH", "WEST", "HOUSTON"],
                 var_name="Load Zone",
                 value_name="Real Time")
    actual = actual[["Date", "Hour Ending", "Load Zone", "DSTFlag", "Real Time"]]
    return actual.sort_values(by=['Date', "Hour Ending"]).reset_index(drop=True)
    

def get_forecast(year, ercot, endpoint="/np3-565-cd/lf_by_model_weather_zone"):
    start = date(year - 1, 12, 31)
    end = date(year + 1, 1, 1)

    forecast = pd.DataFrame(columns=["Date", "Hour Ending", "Load Zone", "Day Ahead", "DSTFlag"])
    df = ercot.get_historical_data(endpoint, start_date=start, end_date=end)
    date_format(df, "DeliveryDate")

    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')

    for d in dates:
        date_filter = df[(df["DeliveryDate"] == d) & (df["Model"] == "E")].tail(24*24*2)
        for t in range(1, 25):
            try:
                time_filter = date_filter[date_filter["Hour Ending"] == t].iloc[t-1]
                for zone in ZONES:
                    load = time_filter[ZONES[zone]["Forecast"]].sum()
                    row = {"Date": d, "Hour Ending": t, "Load Zone": zone, "Day Ahead": load, "DSTFlag": "N"}
                    forecast = pd.concat([forecast, pd.DataFrame([row])], ignore_index=True)
            except IndexError:
                continue
    
    dst = df[(df["DSTFlag"] == "Y") & (df["Model"] == "E")].tail(2)
    d = dst["DeliveryDate"].iloc[0]
    t = dst["Hour Ending"].iloc[0]
    for zone in ZONES:
        load = dst[ZONES[zone]["Forecast"]].iloc[0].sum()
        row = {"Date": d, "Hour Ending": t, "Load Zone": zone, "Day Ahead": load, "DSTFlag": "Y"}
        forecast = pd.concat([forecast, pd.DataFrame([row])], ignore_index=True)
    forecast["Date"] = forecast["Date"].dt.strftime("%m-%d-%Y")
    return forecast
            

ercot = ErcotAPI()
'''
year = 2024

actual = get_actual(year, ercot)
forecast = get_forecast(year, ercot)
combined = pd.merge(actual, forecast, on=['Date', 'Hour Ending', 'Load Zone', 'DSTFlag'], how="outer")

print(actual.shape)
print(forecast.shape)
print(combined.shape)
combined.to_csv(f'load_{year}.csv', index=False)


years = [2022, 2023, 2024]
dfs = []
for year in years:
    dfs.append(pd.read_csv(f'load_{year}.csv'))
df = pd.concat(dfs, ignore_index=True)
df.to_csv('load.csv', index=False)
'''