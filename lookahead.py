import pandas as pd
import numpy as np

# ---- PARAMETERS ----

H = 24 # lookahead horizon
delta_t = 1 # timestep
action_space = np.array([-50, -40, -30, -20, -10, -5, 0, 
                         5, 10, 20, 30, 40, 50]) # if we charge, this assumes we have positive action. if we discharge, this assumes we have negative action. right now assuming discrete action space
n_c = 0.9487 # charge efficiency 
n_d = 0.9487 # discharge efficiency
c_deg = 89.15 # degradation cost reflecting parameter wear
c_pen = 5 # penalty for not meeting demand
E_min = 0 # min energy storage capacity
E_max = 400 # max energy storage capacity
n_iters = 100 # number of simulations to run
gamma = 0.9 # discount factor
P_max = 100 # not sure how to utilize this
OM_cost = 3877700 # fixed annual operating cost


# ---- UPDATE FUNCTIONS ----

def soc_update(E, a):
    """
    Update SOC based on current SOC and action taken.
        - If charging, we add charge to the battery by n_c
    """
    if a >= 0: # charging
        E_new = E + (delta_t * a * n_c)
        return np.min(E_max, E_new)
    else: # discharging
        E_new = E + (delta_t * a / n_d)
        return np.max(E_min, E_new)


def action_space_update(R, D, SOC, E_max, E_min):
    """
    Clips action space based on restraint defined by R - D and current SOC
    """
    s = max(0, R - D)
    allowed_actions = []
    for a in action_space:
        if a > s: # can't charge more than surplus
            continue
        if SOC + a > E_max: # can't overcharge
            continue
        if SOC + a < E_min: # can't overdischarge
            continue

        allowed_actions.append(a)

    return allowed_actions


def objective(a, P, R, D):
    """
    Compute objective function
    
    """
    # power balance equation: R + g = D + a => g = R - D - a
    g = R - D - a # supply - demand
    revenue = P * g * delta_t

    # degradation: assuming a 10MW charge for 1 hr and a 10MW charge for 1 hr both wear battery equally
    degradation = c_deg * abs(a) * delta_t

    # unmet demand penalty
    # supply = R - a
    # unmet_demand = max(0, D - supply) # demand - supply
    # penalty = c_pen * unmet_demand * delta_t

    return revenue - degradation


# ---- GREEDY POLICY ----

def greedy_policy(a_t, state_t, forecasted_scenarios, n_iters, H, gamma):
    """
    Calculates reward for current state and action for forecasted scenarios given greedy heuristic policy.
        - charge when price below some threshold
        - discharge when price above some threshold
        - otherwise do nothing
    """
    total_reward = 0

    for omega in range(n_iters):
        s = state_t.copy()
        a_current = a_t
        r_sum = 0

        all_prices = forecasted_scenarios[:, omega, 0]
        price_low = np.percentile(all_prices, 30)
        price_high = np.percentile(all_prices, 70)
        
        for h in range(H):
            P = forecasted_scenarios[h, omega, 0]
            R = forecasted_scenarios[h, omega, 1]
            D = forecasted_scenarios[h, omega, 2]

            actions = action_space_update(R, D, s['E_t'], E_max, E_min)

            # current action
            reward = objective(a_current, P, R, D)
            s['E_t'] = soc_update(s['E_t'], a_current)
            r_sum += (gamma ** h) * reward

            if P < price_low:
                # if price is lower than some threshold, we want to charge now (buy energy) since prices are low
                a_next = max(actions)
            
            elif P > price_high:
                a_next = min(actions)
            
            else:
                a_next = 0

            a_current = a_next
        
        total_reward += r_sum
    
    return total_reward / n_iters

# ---- TBD: Monte Carlo Tree Search Policy ----




# ---- RUNNING SIMULATION ----

E_t = 150 # some initial SOC
for t in time_steps: # iterate over 3 years?
    
    # current state
    P_t, R_t, D_t = current_market(t)
    state_t = (E_t, P_t, R_t, D_t)
    
    # feasible actions
    actions = action_space_update(R_t, D_t, state_t['E_t'], E_max, E_min)

    # generate forecasts for price, renewable generation, and demand
    forecasted_scenarios = generate_scenarios(H, t) # n_iters x H x 3

    Q_est = np.zeros(len(actions))
    for a_t in actions:
        Q_est[a_t] = greedy_policy(a_t, state_t, forecasted_scenarios, n_iters, H, gamma)

    a_star = max(Q_est, key = Q_est.get)

