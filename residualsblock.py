import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
import logging
import os
from collections import Counter
from typing import Dict, Any, Tuple, List
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Constants
PRICE_MIN = -300.0 
PRICE_MAX = 5000.0 
MC_SURPLUS_SAMPLES = 100 
REWARD_SCALE = 100.0 
LOAD_SCALE_FACTOR = 0.25

PRICE_ZONE_MAP = {'LZ_HOUSTON': 'HOUSTON', 'LZ_NORTH': 'NORTH','LZ_SOUTH': 'SOUTH', 'LZ_WEST': 'WEST'}
ZONES = ['HOUSTON', 'NORTH', 'SOUTH', 'WEST']

class Battery:
    def __init__(self):
        self.capacity_mwh = 400.0
        self.max_power_mw = 100.0
        self.charge_eff = 0.9487
        self.discharge_eff = 0.9487
        self.c_deg = 89.14
        self.dt = 1.0 

#5 possible actions for the battery (pretty simple, next steps would be to add more/make continuous)
ACTIONS = {0: 'HOLD', 1: 'CHARGE_MAX', 2: 'CHARGE_HALF', 3: 'DISCHARGE_MAX', 4: 'DISCHARGE_HALF'}
ACTION_MW = {0: 0.0, 1: -100.0, 2: -50.0, 3: 100.0, 4: 50.0}
N_ACTIONS = len(ACTIONS)

#Block Bootstrap Sampler
#Common method used for sampling from time series data 
#Sample blocks with replacements to maintain autocorrelation from time
class BlockBootstrapSampler:
    def __init__(self, price_csv_path, load_csv_path, renewable_csv_path, zone='HOUSTON'):
        self.zone = zone
        self.price_df = None; self.load_df = None; self.ren_df = None
        self.residuals_df = pd.DataFrame()
        #Load data 
        if os.path.exists(price_csv_path): self.price_df = pd.read_csv(price_csv_path)
        if os.path.exists(load_csv_path): self.load_df = pd.read_csv(load_csv_path)
        if os.path.exists(renewable_csv_path): self.ren_df = pd.read_csv(renewable_csv_path)
        if self.price_df is not None and self.load_df is not None and self.ren_df is not None:
            try: self.residuals_df = self._build_residual_history()
            except Exception: pass

    #Convert date column to datetime object 
    def _parse_dt(self, df, cols):
        for c in cols:
            if c in df.columns: return pd.to_datetime(df[c], errors='coerce')
        return None

    def _build_residual_history(self):
        #PRICE DATA PROCESSING 
        #Get the price data frame 
        p = self.price_df.copy()
        #Get the data for the specified zone 
        settle_pt = {v: k for k, v in PRICE_ZONE_MAP.items()}.get(self.zone)
        p = p[p['Settlement Point'] == settle_pt].copy()
        p_dt = self._parse_dt(p, ['Delivery Date', 'Date'])
        p['datetime'] = p_dt + pd.to_timedelta(p.get('Hour Ending', 0).astype(int) - 1, unit='h')
        #Calculate error if error column does not exist 
        if 'Err' not in p.columns and 'RTM Price' in p.columns: p['price_err'] = p['RTM Price'] - p['DAM Price']
        elif 'Err' in p.columns: p['price_err'] = p['Err']
        else: p['price_err'] = 0.0
        #Discards all columns except for error (and time)
        p = p[['datetime', 'price_err']].dropna().set_index('datetime')
        #Keep only unique timestamps 
        p = p[~p.index.duplicated(keep='last')]
        
        #LOAD DATA PROCESSING 
        #Same process as above 
        l = self.load_df.copy()
        l = l[l['Load Zone'] == self.zone].copy()
        l_dt = self._parse_dt(l, ['Date'])
        l['datetime'] = l_dt + pd.to_timedelta(l.get('Hour Ending', 0).astype(int) - 1, unit='h')
        #Scale the load error relative to the load size 
        l['load_err'] = (l['Real Time'] - l['Day Ahead']) * LOAD_SCALE_FACTOR
        l = l[['datetime', 'load_err']].dropna().set_index('datetime')
        l = l[~l.index.duplicated(keep='last')]
        
        #RENEWABLE DATA PROCESSING
        r = self.ren_df.copy()
        r_dt = self._parse_dt(r, ['Date'])
        r['datetime'] = r_dt + pd.to_timedelta(r.get('Hour Ending', 0).astype(int) - 1, unit='h')
        rt_col, da_col = f'{self.zone}_RT_Actual', f'{self.zone}_DA_Forecast'
        if rt_col in r.columns: r['ren_err'] = r[rt_col] - r[da_col]
        else: r['ren_err'] = 0.0
        r = r[['datetime', 'ren_err']].dropna().set_index('datetime')
        r = r[~r.index.duplicated(keep='last')]
        
        #Join dataframes
        return p.join(l, how='inner').join(r, how='inner').sort_index().reset_index()

    def sample_block(self, block_size: int) -> pd.DataFrame:
        n = len(self.residuals_df)
        if n <= block_size: return self.residuals_df 
        #Pick start index of block as random integer 
        start_idx = np.random.randint(0, n - block_size)
        #Return block of residuals (price, demand, and renewables)
        return self.residuals_df.iloc[start_idx : start_idx + block_size].copy()

    def sample_joint(self, zone, hour):
        if self.residuals_df.empty: return 0.0, 0.0, 0.0
        #Find every historical example of the specific hour 
        subset = self.residuals_df[self.residuals_df['datetime'].dt.hour == (hour - 1)]
        if subset.empty: return 0.0, 0.0, 0.0
        #Sample one row from the hour subset 
        row = subset.sample(1).iloc[0]
        return row['price_err'], row['load_err'], row['ren_err']

#Dataset Generation 
def create_historical_forecast_df(price_csv, load_csv, renewable_csv, zone='HOUSTON'):
    logging.info(f"Creating forecast db: {zone}")
    #Get the price, load, and renewable data
    p = pd.read_csv(price_csv); l = pd.read_csv(load_csv); r = pd.read_csv(renewable_csv)
    
    #Get the data for the specified zone
    p_pt = {v: k for k, v in PRICE_ZONE_MAP.items()}.get(zone)
    p = p[p['Settlement Point'] == p_pt][['Delivery Date', 'Hour Ending', 'DAM Price']]
    p['datetime'] = pd.to_datetime(p['Delivery Date']) + pd.to_timedelta(p['Hour Ending'].astype(int) - 1, unit='h')
    p = p.set_index('datetime').rename(columns={'DAM Price': 'dam_price'})[['dam_price']]
    #Keep only the price column 
    p = p[~p.index.duplicated(keep='last')]
    
    #Same thing with the load data 
    l = l[l['Load Zone'] == zone][['Date', 'Hour Ending', 'Day Ahead']]
    l['datetime'] = pd.to_datetime(l['Date']) + pd.to_timedelta(l['Hour Ending'].astype(int) - 1, unit='h')
    # Scale the Day Ahead Load
    l['dam_load'] = l['Day Ahead'] * LOAD_SCALE_FACTOR
    l = l.set_index('datetime').rename(columns={'Day Ahead': 'dam_load_original'})[['dam_load']]
    l = l[~l.index.duplicated(keep='last')]
    
    #Same thing with the renewables genereation data 
    r_col = f'{zone}_DA_Forecast'
    r = r[['Date', 'Hour Ending', r_col]]
    r['datetime'] = pd.to_datetime(r['Date']) + pd.to_timedelta(r['Hour Ending'].astype(int) - 1, unit='h')
    r = r.set_index('datetime').rename(columns={r_col: 'dam_renewable'})[['dam_renewable']]
    r = r[~r.index.duplicated(keep='last')]
    
    #Join all for a final DA dataset 
    final = p.join(l, how='inner').join(r, how='inner').dropna().sort_index()
    
    for i in range(1, 9):
        final[f'dam_price_lead_{i}'] = final['dam_price'].shift(-i)
    
    final = final.fillna(0.0)
    return final[~final.index.duplicated(keep='first')]
