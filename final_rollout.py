import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from residualsblock import BlockBootstrapSampler, Battery, create_historical_forecast_df
import time
from dateutil.relativedelta import relativedelta
import logging

# Set random seed for reproducibility
np.random.seed(123)

# ---- PARAMETERS ----

class Battery:
    def __init__(self):
        self.capacity_mwh = 400.0
        self.max_power_mw = 100.0
        self.charge_eff = 0.9487
        self.discharge_eff = 0.9487
        self.c_deg = 28.3
        self.dt = 1.0

battery = Battery()

H = 24  # lookahead horizon

ACTIONS = {0: 'HOLD', 
           1: 'CHARGE_MAX', 
           2: 'CHARGE_75', 
           3: 'CHARGE_50',
           4: 'CHARGE_25',  
           5: 'DISCHARGE_20',
           6: 'DISCHARGE_40',
           7: 'DISCHARGE_60',
           8: 'DISCHARGE_80',
           9: 'DISCHARGE_MAX'}
ACTION_MW = {0: 0.0, 
             1: 100.0, 
             2: 75.0,
             3: 50.0,
             4: 25.0,
             5: -20.0,
             6: -40.0,
             7: -60.0,
             8: -80.0,
             9: -100.0}
N_ACTIONS = len(ACTIONS)
ACTION_COUNTS_LOOKAHEAD = {a: 0 for a in ACTIONS.keys()}
ACTION_COUNTS_NAIVE = {a: 0 for a in ACTIONS.keys()}

PRICE_MIN = -300.0
PRICE_MAX = 5000.0
E_min = 0
E_max = battery.capacity_mwh
n_iters = 100
gamma = 0.9
zone_dict = {
    'LZ_HOUSTON': 0,
    'LZ_NORTH':   1,
    'LZ_SOUTH':   2,
    'LZ_WEST':    3,
}

# ---- UPDATE FUNCTIONS ----

def soc_update(E, feasible_action, battery_instance=battery):
    if feasible_action >= 0:  # charging
        E_new = E + (battery_instance.dt * feasible_action)
        return min(battery_instance.capacity_mwh, E_new)
    else:  # discharging
        E_new = E + (battery_instance.dt * feasible_action)
        return max(E_min, E_new)

def objective(desired_mw, rtm_P, rtm_surplus, s, battery_instance=battery):
    current_soc = s['E_t']
    if desired_mw >= 0:
        # Charging
        available_capacity = battery_instance.capacity_mwh - current_soc
        #max_charge_energy = available_capacity / battery_instance.charge_eff
        feasible_mw = min(desired_mw, battery_instance.max_power_mw, rtm_surplus,available_capacity)
        feasible_mw /= battery_instance.charge_eff
        reward = 0
    else:
        # Discharging
        #max_discharge_energy = current_soc * battery_instance.discharge_eff
        feasible_mw = -min(abs(desired_mw), battery_instance.max_power_mw, current_soc)
        reward = (rtm_P - battery_instance.c_deg) * abs(feasible_mw) * battery_instance.discharge_eff
    return reward, feasible_mw

# ---- HEURISTIC POLICY ----

def choose_action_from_future(current_P, future_prices):
    P_min = np.min(future_prices)
    P_max = np.max(future_prices)
    z = (current_P - P_min) / (P_max - P_min + 1e-5)
    if z < 0.1: return ACTION_MW[1]
    elif z < 0.2: return ACTION_MW[2]
    elif z < 0.3: return ACTION_MW[3]
    elif z < 0.4: return ACTION_MW[4]
    elif z < 0.5: return ACTION_MW[0]
    elif z < 0.6: return ACTION_MW[5]
    elif z < 0.7: return ACTION_MW[6]
    elif z < 0.8: return ACTION_MW[7]
    elif z < 0.9: return ACTION_MW[8]
    else: return ACTION_MW[9]

# ---- ROLLOUT POLICY ----

def rollout_policy(a0, state_t, P_t, R_t, D_t, forecasted_scenarios, n_iters, H, gamma, battery_instance=battery):
    """
    Estimate Q(state_t, a0) via rollout over forecasted scenarios.
    a0: initial action to test
    """
    total_reward = 0
    K = 10  # lookahead window for greedy action selection

    for omega in range(n_iters):
        s = state_t.copy()
        r_sum = 0

        # Step 0: apply a0 with feasible MW considering SOC and surplus
        surplus = max(0.0, R_t - D_t)
        reward, feasible_a = objective(a0, P_t, surplus, s, battery_instance)
        s['E_t'] = soc_update(s['E_t'], feasible_a, battery_instance)
        r_sum += reward

        # Greedy simulation over horizon
        for h in range(H):
            current_P = forecasted_scenarios[h, omega, 0]
            current_R = forecasted_scenarios[h, omega, 1]
            current_D = forecasted_scenarios[h, omega, 2]

            surplus = max(0.0, current_R - current_D)

            # choose next action greedily using future prices
            future_end = min(h + K, H)
            future_prices = forecasted_scenarios[h:future_end, omega, 0]
            a_current = choose_action_from_future(current_P, future_prices)

            # apply feasible action with surplus constraint
            reward, feasible_a = objective(a_current, current_P, surplus, s, battery_instance)
            s['E_t'] = soc_update(s['E_t'], feasible_a, battery_instance)

            r_sum += (gamma ** h) * reward

        total_reward += r_sum

    return total_reward / n_iters

# ---- DATA LOADING AND SCENARIO GENERATION ----

def generate_scenarios(current_date, H, zone, sampler, forecast_df, n_iters=100):
    scenarios = np.zeros((H, n_iters, 3))
    current_datetime = pd.to_datetime(current_date)
    for h in range(H):
        future_time = current_datetime + timedelta(hours=h+1)
        if future_time in forecast_df.index:
            base_price = forecast_df.loc[future_time, 'dam_price']
            base_load = forecast_df.loc[future_time, 'dam_load'] 
            base_renewable = forecast_df.loc[future_time, 'dam_renewable']
        else:
            base_price = 50.0
            base_load = 500.0
            base_renewable = 500.0
        for omega in range(n_iters):
            price_err, load_err, ren_err = sampler.sample_joint(zone, future_time.hour + 1)
            scenarios[h, omega, 0] = base_price + price_err
            scenarios[h, omega, 1] = base_renewable + ren_err
            scenarios[h, omega, 2] = base_load + load_err
    return scenarios

# ---- RUNNING SIMULATION ----

def run_lookahead_policy(test_df, sampler, battery, zone='HOUSTON'):
    lookahead_E_t = 150
    naive_E_t = 150
    lookahead_results = []
    naive_results = []
    test_df = test_df.sort_index()
    res_lookup = sampler.residuals_df.set_index('datetime').to_dict('index')
    
    for i, (ts, row) in enumerate(test_df.iterrows()):
        # print every day
        if i % 24 == 0:
            print(f"Processing timestamp: {ts}")
        if ts in res_lookup:
            rec = res_lookup[ts]
            p_err, l_err, r_err = rec['price_err'], rec['load_err'], rec['ren_err']
        else:
            print("residuals not found for that hour")
            p_err, l_err, r_err = 0.0, 0.0, 0.0

        rtm_price = np.clip(row['dam_price'] + p_err, PRICE_MIN, PRICE_MAX)
        rtm_load = row['dam_load'] + l_err
        rtm_renewable = row['dam_renewable'] + r_err
        rtm_surplus = max(0, rtm_renewable - rtm_load)
        
        lookahead_state_t = {'E_t': lookahead_E_t, 'P_t': row['dam_price'] , 'R_t': row['dam_renewable'], 'D_t': row['dam_load']}
        forecasted_scenarios = generate_scenarios(ts, H, zone, sampler, test_df, n_iters)

        Q_est = {}
        for a_t in ACTION_MW.values():
            Q_est[a_t] = rollout_policy(a_t, lookahead_state_t, row['dam_price'], row['dam_renewable'], row['dam_load'], forecasted_scenarios, n_iters, H, gamma)
        
        a_star = max(Q_est, key=Q_est.get)
        ACTION_COUNTS_LOOKAHEAD[[k for k,v in ACTION_MW.items() if v == a_star][0]] += 1

        lookahead_reward, feasible_action = objective(a_star, rtm_price, rtm_surplus, lookahead_state_t, battery)
        lookahead_E_t = soc_update(lookahead_E_t, feasible_action)

        lookahead_results.append({
            'datetime': ts,
            'SOC': lookahead_E_t,
            'action': a_star,
            'price': row['dam_price'],
            'renewable': row['dam_renewable'],
            'demand': row['dam_load'],
            'reward': lookahead_reward
        })

        naive_state_t = {'E_t': naive_E_t, 'P_t': row['dam_price'], 'R_t': row['dam_renewable'], 'D_t': row['dam_load']}
        if row['dam_price'] <= 25 and (naive_E_t < battery.capacity_mwh):
            naive_action = 1  # charge max
        elif row['dam_price'] >= 100 and (naive_E_t > 0):  
            naive_action = 9  # discharge max
        else:
            naive_action = 0  # hold    
        feasible_naive_mw = ACTION_MW[naive_action]
        ACTION_COUNTS_NAIVE[naive_action] += 1
        naive_reward, feasible_naive_action = objective(feasible_naive_mw, rtm_price, rtm_surplus, naive_state_t, battery)
        naive_E_t = soc_update(naive_E_t, feasible_naive_action)

        naive_results.append({
            'datetime': ts,
            'SOC': naive_E_t,
            'action': feasible_naive_mw,
            'price': row['dam_price'],
            'renewable': row['dam_renewable'],
            'demand': row['dam_load'],
            'reward': naive_reward
        })

    lookahead_results_df = pd.DataFrame(lookahead_results)
    print(f"Simulation complete! Total reward: {lookahead_results_df['reward'].sum():.2f}")

    naive_results_df = pd.DataFrame(naive_results)
    print(f"Naive policy total reward: {naive_results_df['reward'].sum():.2f}")


    return lookahead_results_df, naive_results_df

# ---- MAIN ----

if __name__ == "__main__":

    np.random.seed(123)
    PATH_PRICE = 'price_data/price_df.csv'
    PATH_LOAD = 'load.csv'
    PATH_RENEWABLE = 'Renewables/zone_renewable_data.csv'
    TARGET_ZONE = 'HOUSTON'
    sampler = BlockBootstrapSampler(PATH_PRICE, PATH_LOAD, PATH_RENEWABLE, zone='HOUSTON')
    battery = Battery()

    try:
        full_df = create_historical_forecast_df(PATH_PRICE, PATH_LOAD, PATH_RENEWABLE, zone='HOUSTON')
        print("\n" + "="*40)
        print("PRICE DATA SUMMARY STATS")
        print("="*40)
        print(full_df['dam_price'].describe())
        print("="*40 + "\n")
    except Exception as e:
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
    
    start = time.time()
    lookahead_results_df, naive_results_df = run_lookahead_policy(test_df, sampler, battery, zone='HOUSTON')
    end = time.time()
    print(f"Total simulation time: {end - start:.2f} seconds")

    print("\nSimulation Results Summary:")
    print(f"Total hours simulated: {len(lookahead_results_df)}")
    print(f"Total reward: ${lookahead_results_df['reward'].sum():.2f}")
    print(f"Average SOC: {lookahead_results_df['SOC'].mean():.1f} MWh")
    print(f"Final SOC: {lookahead_results_df['SOC'].iloc[-1]:.1f} MWh")

    # Save results
    lookahead_results_df.to_csv('lookahead_results.csv', index=False)
    print("Results saved to 'lookahead_results.csv'")

    naive_results_df.to_csv('naive_results.csv', index=False)
    print("Naive baseline results saved to 'naive_results.csv'")

    # print action counts
    print("\nAction Counts (Lookahead Policy):")
    for action, count in ACTION_COUNTS_LOOKAHEAD.items():
        print(f"{ACTIONS[action]}: {count}")

    print("\nAction Counts (Naive Policy):")
    for action, count in ACTION_COUNTS_NAIVE.items():
        print(f"{ACTIONS[action]}: {count}")


