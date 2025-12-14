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
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Constants
PRICE_MIN = -300.0
PRICE_MAX = 5000.0
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
        self.c_deg = 28.3
        self.dt = 1.0

ACTIONS = {
    0: 'HOLD',
    1: 'CHARGE_MAX',    
    2: 'CHARGE_75',    
    3: 'CHARGE_50',    
    4: 'CHARGE_25',     
    5: 'DISCHARGE_20', 
    6: 'DISCHARGE_40',  
    7: 'DISCHARGE_60',  
    8: 'DISCHARGE_80', 
    9: 'DISCHARGE_MAX'}

#Find indices for heuristic actions
try:
    CHARGE_MAX_IDX = [k for k, v in ACTIONS.items() if v == 'CHARGE_MAX'][0]
    DISCHARGE_MAX_IDX = [k for k, v in ACTIONS.items() if v == 'DISCHARGE_MAX'][0]
except IndexError:
    #Fallback defaults if names change
    CHARGE_MAX_IDX = 1
    DISCHARGE_MAX_IDX = 9

ACTION_MW = {
    0: 0.0,
    1: -100.0,
    2: -75.0,
    3: -50.0,
    4: -25.0,
    5: 20.0,
    6: 40.0,
    7: 60.0,
    8: 80.0,
    9: 100.0}

N_ACTIONS = len(ACTIONS)

#Block Bootstrap Sampler
class BlockBootstrapSampler:
    def __init__(self, price_csv_path, load_csv_path, renewable_csv_path, zone='WEST'):
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
        p = self.price_df.copy()
        settle_pt = {v: k for k, v in PRICE_ZONE_MAP.items()}.get(self.zone)
        p = p[p['Settlement Point'] == settle_pt].copy()
        p_dt = self._parse_dt(p, ['Delivery Date', 'Date'])
        p['datetime'] = p_dt + pd.to_timedelta(p.get('Hour Ending', 0).astype(int) - 1, unit='h')
        if 'Err' not in p.columns and 'RTM Price' in p.columns: p['price_err'] = p['RTM Price'] - p['DAM Price']
        elif 'Err' in p.columns: p['price_err'] = p['Err']
        else: p['price_err'] = 0.0
        
        #Recent Volatility (24h Rolling Std Dev) 
        p['volatility'] = p['price_err'].rolling(window=24, min_periods=1).std().fillna(0.0)
    
        p = p[['datetime', 'price_err', 'volatility']].dropna().set_index('datetime')
        p = p[~p.index.duplicated(keep='last')]
        #LOAD DATA PROCESSING
        l = self.load_df.copy()
        l = l[l['Load Zone'] == self.zone].copy()
        l_dt = self._parse_dt(l, ['Date'])
        l['datetime'] = l_dt + pd.to_timedelta(l.get('Hour Ending', 0).astype(int) - 1, unit='h')
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
        return p.join(l, how='inner').join(r, how='inner').sort_index().reset_index()

    def sample_block(self, block_size: int) -> pd.DataFrame:
        n = len(self.residuals_df)
        if n <= block_size: return self.residuals_df
        start_idx = np.random.randint(0, n - block_size)
        return self.residuals_df.iloc[start_idx : start_idx + block_size].copy()

    def sample_joint(self, zone, hour):
        if self.residuals_df.empty: return 0.0, 0.0, 0.0
        subset = self.residuals_df[self.residuals_df['datetime'].dt.hour == hour]
        if subset.empty: return 0.0, 0.0, 0.0
        row = subset.sample(1).iloc[0]
        return row['price_err'], row['load_err'], row['ren_err']

#Dataset Generation
def create_historical_forecast_df(price_csv, load_csv, renewable_csv, zone='WEST'):
    logging.info(f"Creating forecast db: {zone}")
    p = pd.read_csv(price_csv); l = pd.read_csv(load_csv); r = pd.read_csv(renewable_csv)
    p_pt = {v: k for k, v in PRICE_ZONE_MAP.items()}.get(zone)
    p = p[p['Settlement Point'] == p_pt][['Delivery Date', 'Hour Ending', 'DAM Price']]
    p['datetime'] = pd.to_datetime(p['Delivery Date']) + pd.to_timedelta(p['Hour Ending'].astype(int) - 1, unit='h')
    p = p.set_index('datetime').rename(columns={'DAM Price': 'dam_price'})[['dam_price']]
    p = p[~p.index.duplicated(keep='last')]
    l = l[l['Load Zone'] == zone][['Date', 'Hour Ending', 'Day Ahead']]
    l['datetime'] = pd.to_datetime(l['Date']) + pd.to_timedelta(l['Hour Ending'].astype(int) - 1, unit='h')
    l['dam_load'] = l['Day Ahead'] * LOAD_SCALE_FACTOR
    l = l.set_index('datetime').rename(columns={'Day Ahead': 'dam_load_original'})[['dam_load']]
    l = l[~l.index.duplicated(keep='last')]
    r_col = f'{zone}_DA_Forecast'
    r = r[['Date', 'Hour Ending', r_col]]
    r['datetime'] = pd.to_datetime(r['Date']) + pd.to_timedelta(r['Hour Ending'].astype(int) - 1, unit='h')
    r = r.set_index('datetime').rename(columns={r_col: 'dam_renewable'})[['dam_renewable']]
    r = r[~r.index.duplicated(keep='last')]
    final = p.join(l, how='inner').join(r, how='inner').dropna().sort_index()
    for i in range(1, 9):
        final[f'dam_price_lead_{i}'] = final['dam_price'].shift(-i)
    final = final.fillna(0.0)
    return final[~final.index.duplicated(keep='first')]

def get_s(s_soc, f, t_hour, vol=0.0):
    base = [s_soc, f['dam_load']-f['dam_renewable'], np.sin(2*np.pi*t_hour/24), np.cos(2*np.pi*t_hour/24), f['dam_price'], vol]
    future = [f.get(f'dam_price_lead_{i}', 0.0) for i in range(1, 9)]
    return np.array(base + future)

def generate_batch_dataset(n_transitions, forecast_df, sampler, battery, block_len=24):
    logging.info(f"Generating {n_transitions} transitions...")
    work_df = forecast_df
    if 'datetime' in forecast_df.columns: work_df = forecast_df.set_index('datetime')
    fc_lookup = work_df.to_dict('index')
    states, actions, rewards, next_states = [], [], [], []
    collected = 0
    while collected < n_transitions:
        res_block = sampler.sample_block(block_len)
        if res_block.empty: continue
        
        #Start SoC at 150.0 instead of random
        soc = 150.0
        
        for i in range(len(res_block) - 1):
            ts, ts_next = res_block.iloc[i]['datetime'], res_block.iloc[i+1]['datetime']
            if ts not in fc_lookup or ts_next not in fc_lookup: break
            fc, fc_next = fc_lookup[ts], fc_lookup[ts_next]
            vol = res_block.iloc[i].get('volatility', 0.0)
            
            s = get_s(soc, fc, ts.hour, vol)
            net_load = s[1]
            has_surplus = (net_load < -0.001)
            valid_indices = []
            for idx, mw in ACTION_MW.items():
                if mw == 0: valid_indices.append(idx)
                elif mw < 0 and has_surplus and (soc < battery.capacity_mwh): valid_indices.append(idx)
                elif mw > 0 and (soc > 0): valid_indices.append(idx)
            a = np.random.choice(valid_indices)
            row = res_block.iloc[i]
            rtm_price = np.clip(fc['dam_price'] + row['price_err'], PRICE_MIN, PRICE_MAX)
            surplus = max(0, (fc['dam_renewable'] + row['ren_err']) - (fc['dam_load'] + row['load_err']))
            mw = ACTION_MW[a]
            r_val, d_soc = 0.0, 0.0
            
            #Charge Logic
            if mw < 0:
                feasible = min(abs(mw), battery.max_power_mw, surplus, (battery.capacity_mwh - soc))
                feasible /= battery.charge_eff
                #Silent failure penalty logic
                if feasible < 0.001 and abs(mw) > 0:
                    r_val = -1000.0
                else:
                    r_val = 0.0 
                    d_soc = feasible * battery.dt
            #Discharge Logic
            elif mw > 0:
                feasible = min(mw, battery.max_power_mw, soc)
                r_val = (rtm_price - battery.c_deg) * feasible * battery.discharge_eff * battery.dt
                d_soc = -feasible * battery.dt

            soc_new = np.clip(soc + d_soc, 0, battery.capacity_mwh)
            states.append(s); actions.append(a); rewards.append(r_val / REWARD_SCALE)
            vol_next = res_block.iloc[i+1].get('volatility', 0.0)
            next_states.append(get_s(soc_new, fc_next, ts_next.hour, vol_next))
            soc = soc_new
            collected += 1
            if collected >= n_transitions: break
            
    return {'states': np.array(states), 'actions': np.array(actions), 'rewards': np.array(rewards), 'next_states': np.array(next_states)}

#FQI (Multi-Model)
class FittedQIteration:
    def __init__(self, n_actions, gamma=0.90, n_iterations=50, sampler=None, zone='WEST'):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.sampler = sampler
        self.zone = zone
        self.models = {}
        self.base_regressor = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=500,
        learning_rate=0.02,
        max_depth=20,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42)
        self.is_fitted = False

    def fit(self, batch):
        S, A, R, S_next = batch['states'], batch['actions'], batch['rewards'], batch['next_states']
        for a in range(self.n_actions):
            self.models[a] = clone(self.base_regressor)
        target_q = R.copy()
        for it in range(self.n_iterations):
            if it % 10 == 0: logging.info(f"FQI Iteration {it+1}/{self.n_iterations}")
            if it > 0:
                q_next_all = np.zeros((len(S_next), self.n_actions))
                for a in range(self.n_actions):
                    try: q_next_all[:, a] = self.models[a].predict(S_next)
                    except NotFittedError: pass
                next_soc = S_next[:, 0]
                next_net_load = S_next[:, 1]
                has_surplus_next = (next_net_load < -0.001)
                has_capacity_next = (next_soc < 400.0)
                can_charge_mask = has_surplus_next & has_capacity_next
                can_discharge_mask = (next_soc > 0.0)
                for a, mw in ACTION_MW.items():
                    if mw < 0: 
                        q_next_all[~can_charge_mask, a] = -np.inf
                    elif mw > 0: 
                        q_next_all[~can_discharge_mask, a] = -np.inf
                max_q_next = np.max(q_next_all, axis=1)
                max_q_next[max_q_next == -np.inf] = 0.0
                target_q = R + self.gamma * max_q_next
            for a in range(self.n_actions):
                mask = (A == a)
                if np.sum(mask) > 0:
                    self.models[a].fit(S[mask], target_q[mask])
        self.is_fitted = True

    def save_model(self, filename):
        if not self.is_fitted:
            logging.warning("Warning: You are saving a model that has not been fitted yet.")
        joblib.dump(self.models, filename)
        logging.info(f"Models successfully saved to {filename}")

    def get_q_values(self, state_arr):
        q_vals = np.zeros((state_arr.shape[0], self.n_actions))
        for a in range(self.n_actions):
            try:
                q_vals[:, a] = self.models[a].predict(state_arr)
            except NotFittedError:
                q_vals[:, a] = -1e9
        return q_vals
    def get_valid_actions(self, state, battery, debug=False):
        state = np.asarray(state).reshape(-1)
        soc = float(state[0])
        net_load_forecast = float(state[1])
        has_surplus = (net_load_forecast < -0.001)
        can_charge = (soc < battery.capacity_mwh) and has_surplus
        can_discharge = (soc > 0)
        valid = []
        for a, mw in ACTION_MW.items():
            if mw == 0: valid.append(a)
            elif mw < 0 and can_charge: valid.append(a)
            elif mw > 0 and can_discharge: valid.append(a)
        return valid

def run_sanity_checks(agent, battery):
    print("\n" + "="*60)
    print("RUNNING 4 CRITICAL UNIT TESTS (SANITY CHECK)")
    print("="*60)
    def print_decision(name, s):
        q_vals = agent.get_q_values(s)[0]
        valid_indices = agent.get_valid_actions(s, battery)
        if not valid_indices:
            best_a = 0 
            note = "(Forced HOLD - Constraints)"
        else:
            valid_q = {a: q_vals[a] for a in valid_indices}
            best_a = max(valid_q, key=valid_q.get)
            note = ""
        print(f"TEST: {name}")
        print(f" Raw Q-Values: {np.round(q_vals, 1)}")
        print(f" Valid Actions: {[ACTIONS[v] for v in valid_indices]}")
        print(f" Best Allowed Action: {ACTIONS[best_a]} {note}")
        print("-" * 40)

    s1 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0] + [500.0]*8).reshape(1, -1)
    print_decision("1. Price=0, Future=500 (Empty Battery)", s1)

    s2 = np.array([400.0, 50.0, -0.9, -0.2, 5000.0, 0.0] + [10.0]*8).reshape(1, -1)
    print_decision("2. Price=5000, Future=10 (Full Battery)", s2)

    s3 = np.array([200.0, 20.0, 0.0, 1.0, 40.0, 0.0] + [40.0]*8).reshape(1, -1)
    print_decision("3. Price=40 vs Cost=51.2 (Stagnant)", s3)

    s4 = np.array([0.0, 0.0, 0.0, 1.0, 20.0, 0.0] + [20.0]*3 + [5000.0]*5).reshape(1, -1)
    print_decision("4. Price=20, Spike in 4 hrs (Empty Battery)", s4)
    print("="*60 + "\n")

def run_backtest(agent, test_df, sampler, battery, zone='WEST'):
    logging.info("--- RUNNING CHRONOLOGICAL BACKTEST ---")
    soc = 0.0
    naive_soc = 0.0
    total_profit = 0.0
    naive_profit = 0.0
    total_discharge_mwh = 0.0
    history = []
    action_counts = Counter()
    test_df = test_df.sort_index()
    res_lookup = sampler.residuals_df.set_index('datetime').to_dict('index')
    
    for i, (ts, row) in enumerate(test_df.iterrows()):
        if ts in res_lookup:
            rec = res_lookup[ts]
            p_err, l_err, r_err = rec['price_err'], rec['load_err'], rec['ren_err']
            vol = rec.get('volatility', 0.0)
        else:
            print("residuals not found for that hour")
            p_err, l_err, r_err = 0.0, 0.0, 0.0
            vol = 0.0
        
        rtm_price = np.clip(row['dam_price'] + p_err, PRICE_MIN, PRICE_MAX)
        surplus = max(0, (row['dam_renewable'] + r_err) - (row['dam_load'] + l_err))
        s = get_s(soc, row, ts.hour, vol)
        valid = agent.get_valid_actions(s, battery)
        
        if not valid:
            action_idx = 0
        else:
            qs = agent.get_q_values(s.reshape(1, -1))[0]
            valid_qs = {a: qs[a] for a in valid}
            action_idx = max(valid_qs, key=valid_qs.get)
        
        if i < 5:
            logging.info(f"Step {i} ({ts}): Price={row['dam_price']:.2f}, SoC={soc:.1f}")
            logging.info(f" Chosen: {ACTIONS[action_idx]}")
        action_counts[ACTIONS[action_idx]] += 1
        
        mw = ACTION_MW[action_idx]
        r_val, d_soc, energy = 0.0, 0.0, 0.0
        
        if mw < 0:
            feasible = min(abs(mw), battery.max_power_mw, surplus, (battery.capacity_mwh - soc))
            feasible /= battery.charge_eff
            r_val = 0.0 
            d_soc = feasible * battery.dt
            
        elif mw > 0:
            feasible = min(mw, battery.max_power_mw, soc)
            #Energy leaving the battery (SoC reduction)
            d_soc = -feasible * battery.dt
            #Energy entering grid = feasible * discharge_eff
            #Reward = Price * Grid_Energy
            r_val = (rtm_price - battery.c_deg) * feasible * battery.discharge_eff * battery.dt
            #For tracking usage, we count battery-side energy discharged
            energy = feasible * battery.dt

        total_discharge_mwh += energy
        total_profit += r_val
        soc = np.clip(soc + d_soc, 0, battery.capacity_mwh)
        
        naive_action = 0
        if row['dam_price'] < 25.0 and (naive_soc < battery.capacity_mwh): naive_action = CHARGE_MAX_IDX
        elif row['dam_price'] > 100 and (naive_soc > 0): naive_action = DISCHARGE_MAX_IDX 
        naive_d_soc = 0.0
        if naive_action == CHARGE_MAX_IDX:
            naive_feasible = min(abs(ACTION_MW[CHARGE_MAX_IDX]), battery.max_power_mw, surplus, (battery.capacity_mwh - naive_soc))
            naive_feasible /= battery.charge_eff
            naive_d_soc = naive_feasible * battery.dt 
        elif naive_action == DISCHARGE_MAX_IDX:
            naive_feasible_d = min(ACTION_MW[DISCHARGE_MAX_IDX], battery.max_power_mw, naive_soc)
            naive_profit += (rtm_price - battery.c_deg) * naive_feasible_d * battery.discharge_eff * battery.dt
            naive_d_soc = -naive_feasible_d * battery.dt

        naive_soc = np.clip(naive_soc + naive_d_soc, 0, battery.capacity_mwh)

        history.append({'timestamp': ts,
        'da_price': row['dam_price'],
        'rtm_price': rtm_price,
        'soc': soc,
        'action': ACTIONS[action_idx],
        'profit_step': r_val,
        'cumulative_profit': total_profit})
    results_df = pd.DataFrame(history)
    cycles = total_discharge_mwh / battery.capacity_mwh
    logging.info(f"Backtest Complete. Total AI Profit: ${total_profit:,.2f}")
    logging.info(f"Naive Benchmark Profit (Approx): ${naive_profit:,.2f}")
    print("\nACTION DISTRIBUTION:")
    for act, count in action_counts.items():
        print(f" {act}: {count}")
    print("\nBATTERY USAGE:")
    print(f" Total Discharged: {total_discharge_mwh:,.2f} MWh")
    print(f" Equivalent Cycles: {cycles:.2f}")
    return results_df, total_profit

if __name__ == "__main__":
    np.random.seed(123)
    PATH_PRICE = 'price_df.csv'
    PATH_LOAD = 'load.csv'
    PATH_RENEWABLE = 'zone_renewable_data.csv'
    logging.info("--- INITIALIZING ---")
    TARGET_ZONE = 'WEST'
    sampler = BlockBootstrapSampler(PATH_PRICE, PATH_LOAD, PATH_RENEWABLE, zone='WEST')
    battery = Battery()

    try:
        full_df = create_historical_forecast_df(PATH_PRICE, PATH_LOAD, PATH_RENEWABLE, zone='WEST')
        print("\n" + "="*40)
        print("PRICE DATA SUMMARY STATS")
        print("="*40)
        print(full_df['dam_price'].describe())
        print("="*40 + "\n")
    except Exception as e:
        logging.error(f"Could not create forecast database: {e}")
        raise

    surplus_mask = full_df['dam_renewable'] > full_df['dam_load']
    total_hours = len(full_df)
    surplus_hours = surplus_mask.sum()
    percent = (surplus_hours / total_hours) * 100
    print("\n" + "="*40)
    print(f"DIAGNOSTIC: Surplus Check (Ren > Load)")
    print(f"Total Hours: {total_hours}")
    print(f"Surplus Hours: {surplus_hours}")
    print(f"Percent Allowed: {percent:.4f}%")
    print("="*40 + "\n")

    last_ts = full_df.index.max()
    cutoff_date = last_ts - relativedelta(months=3)
    logging.info(f"Data Range: {full_df.index.min()} to {last_ts}")
    logging.info(f"Splitting data at {cutoff_date} (Last 3 months for testing)")
    train_df = full_df[full_df.index <= cutoff_date]
    test_df = full_df[full_df.index > cutoff_date]
    logging.info(f"Train Size: {len(train_df)} hours")
    logging.info(f"Test Size: {len(test_df)} hours")

    logging.info(f"Set REWARD_SCALE to: {REWARD_SCALE:.2f}")
    #generate data
    batch = generate_batch_dataset(n_transitions=300000, forecast_df=train_df,
    sampler=sampler, battery=battery, block_len=24)
    fqi = FittedQIteration(n_actions=N_ACTIONS, gamma=0.90,
    n_iterations=100, sampler=sampler, zone='WEST')
    fqi.fit(batch)
    #Save the model 
    model_filename = f"fqi_model_{TARGET_ZONE}.joblib"
    fqi.save_model(model_filename)

    run_sanity_checks(fqi, battery)

    results, final_profit = run_backtest(fqi, test_df, sampler, battery, zone='WEST')
    print("\n" + "="*60)
    print(f"FINAL RESULTS (Last 3 Months)")
    print("="*60)
    print(f"Total Net Profit: ${final_profit:,.2f}")
    print(f"Final SoC: {results.iloc[-1]['soc']:.2f} MWh")
    print("\nSample of first 5 decisions:")
    print(results[['timestamp', 'da_price', 'soc', 'action', 'profit_step']].head().to_string())
