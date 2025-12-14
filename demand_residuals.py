import numpy as np
import pandas as pd
from scipy import stats

ZONES = ['HOUSTON', 'NORTH', 'SOUTH', 'WEST']

class DemandResiduals:
    def __init__(self, csv_path = 'load.csv'):
        # One-time setup: load CSV and fit KDEs
        demand = pd.read_csv(csv_path)
        demand['Residual'] = demand['Real Time'] - demand['Day Ahead']
        res_by_hour = {hour: (group['Load Zone'].values, group['Residual'].values) for hour, group in demand.groupby('Hour Ending')}

        res_kdes = {}
        for hour, vals in res_by_hour.items():
            zone, res = vals
            zone_kdes = {}
            for z in ZONES:
                kde = stats.gaussian_kde(res[zone == z])
                zone_kdes[z] = kde
            res_kdes[hour] = zone_kdes
        self.kdes = res_kdes


    def sample(self, t, h):
        samples = np.zeros((4, h))
        for i in range(4):
            for j in range(h):
                samples[i, j] = self.kdes[t+j+1][ZONES[i]].resample(1)
        return samples