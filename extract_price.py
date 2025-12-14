import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
# Houston, North, South, West

def extract_DAM_RTM(year):
    settlement_points = ['LZ_NORTH', 'LZ_WEST', 'LZ_HOUSTON', 'LZ_SOUTH']

    # DAM
    DAM_path = f'price_data/DAM_{year}.xlsx'
    DAM_sheets = pd.read_excel(DAM_path, sheet_name=None)
    DAM = pd.concat(DAM_sheets.values(), ignore_index=True)
    DAM = DAM[DAM['Settlement Point'].isin(settlement_points)]
    DAM['Hour Ending'] = DAM['Hour Ending'].str.split(':').str[0].astype(int)
    DAM = DAM[['Delivery Date', 'Hour Ending', 'Settlement Point', 'Settlement Point Price']]
    DAM = DAM.rename(columns={'Settlement Point Price': 'DAM Price'})

    # RTM
    RTM_path = f'price_data/RTM_{year}.xlsx'
    RTM_sheets = pd.read_excel(RTM_path, sheet_name=None)
    RTM = pd.concat(RTM_sheets.values(), ignore_index=True)
    RTM = RTM[(RTM['Settlement Point Name'].isin(settlement_points)) & 
                        (RTM['Settlement Point Type'] == 'LZ') & 
                        (RTM['Delivery Interval'] == 1)]
    RTM = RTM[['Delivery Date', 'Delivery Hour', 'Settlement Point Name', 'Settlement Point Price']]
    RTM = RTM.rename(columns={'Delivery Hour': 'Hour Ending', 
                                        'Settlement Point Name': 'Settlement Point',
                                        'Settlement Point Price': 'RTM Price'})

    # merge
    DAM['Delivery Date'] = pd.to_datetime(DAM['Delivery Date'])
    RTM['Delivery Date'] = pd.to_datetime(RTM['Delivery Date'])
    DAM['Hour Ending'] = DAM['Hour Ending'].astype(int)
    RTM['Hour Ending'] = RTM['Hour Ending'].astype(int)

    #DAM = DAM.drop_duplicates(subset=['Delivery Date', 'Hour Ending', 'Settlement Point'])
    #RTM = RTM.drop_duplicates(subset=['Delivery Date', 'Hour Ending', 'Settlement Point'])

    DAM = DAM.sort_values(by=['Delivery Date', 'Hour Ending', 'Settlement Point']).reset_index(drop=True)
    RTM = RTM.sort_values(by=['Delivery Date', 'Hour Ending', 'Settlement Point']).reset_index(drop=True)

    assert len(DAM) == len(RTM), "DAM and RTM row counts do not match!"
    RTM['DAM Price'] = DAM['DAM Price'].values


    # price = pd.merge(RTM, DAM, 
    #                     on=['Delivery Date', 'Hour Ending', 'Settlement Point'],
    #                     how='inner')
    price = RTM.copy()
    
    return price


if __name__ == "__main__":
    # year = '2022'
    # price = extract_DAM_RTM(year)
    # print(price.head())
    # print(price.shape)
    # price.to_csv(f'price_data/price_{year}.csv', index=False)

    # price_2022 = pd.read_csv('price_data/price_2022.csv')
    # price_2023 = pd.read_csv('price_data/price_2023.csv')
    # price_2024 = pd.read_csv('price_data/price_2024.csv')

    # price_df = pd.concat([price_2022, price_2023, price_2024], ignore_index=True)
    # price_df['Err'] = price_df['RTM Price'] - price_df['DAM Price']
    # price_df.to_csv('price_data/price_df.csv', index=False)

    price_df = pd.read_csv('price_data/price_df.csv')
    sample_data = price_df[(price_df['Hour Ending'] == 1) &
                           (price_df['Settlement Point'] == 'LZ_HOUSTON')]
    data = sample_data['Err']
    plt.figure(figsize=(8, 6)) # Set the figure size for better visualization

    # Use sns.histplot for a combined histogram and KDE plot
    # kde=True overlays the Kernel Density Estimate
    # stat='density' normalizes the histogram so the area sums to 1, matching the KDE
    sns.histplot(data, kde=True, stat='density', bins=50, color='skyblue', edgecolor='black', linewidth=0.8)

    plt.title('Histogram and Density Plot of Sample Data')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(axis='y', alpha=0.75) # Add a grid for readability
    plt.show()

    print(price_df.shape)





    
