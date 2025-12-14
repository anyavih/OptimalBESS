import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

# function that takes in current hour, H (how many samples we want) and 
# gives us residuals for hours t+1 to t+H

zone_dict = {'LZ_HOUSTON': 0,
             'LZ_NORTH': 1,
             'LZ_SOUTH': 2,
             'LZ_WEST': 3}

def generate_price_residuals(current_hour, H):
    price_df = pd.read_csv('price_data/price_df.csv')
    residuals = np.zeros((4, H))

    target_hours = [((h - 1) % 24) + 1 for h in range(current_hour + 1,
                                                  current_hour + H + 1)]
    print(target_hours)
    df = price_df[price_df["Hour Ending"].isin(target_hours)]

    for zone, zone_idx in zone_dict.items():
        zone_df = df[df["Settlement Point"] == zone]

        for i, t in enumerate(target_hours):
            err_values = zone_df[zone_df["Hour Ending"] == t]["Err"].values
            data = err_values.reshape(-1,1)
            kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
            kde.fit(data)

            residuals[zone_idx, i] = kde.sample(1)[0,0]
    
    return residuals

if __name__ == "__main__":
    residuals = generate_price_residuals(22, H=5)
    print(residuals)