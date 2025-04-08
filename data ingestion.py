import pandas as pd
import os

raw_data_path = "C:/Users/lsmith3/OneDrive - MBTA/Documents/Personal Projects/AB Testing/WA_Marketing-Campaign.csv"
processed_data_path = "C:/Users/lsmith3/OneDrive - MBTA/Documents/Personal Projects/AB Testing/cleaned_WA_Marketing-Campaign.csv"
def load_and_clean_data(input_path = raw_data_path, output_path = processed_data_path):
    print("Loading raw data...")
    df = pd.read_csv(input_path)
    print(df.head())
    
    print("Cleaning column names")
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    
    print("Converting week to integer")
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
    
    print("Checking fo rmissing values")
    missing= df.isnull().sum()
    print("Missing values:\n", missing)
    
    print("Dropping rows with any missing values")
    df.dropna(inplace=True)
    
    print("Creating useful features...")
    #Each promotion is unique to a storeID
    df['TotalSales_4Weeks'] = df.groupby('LocationID')['SalesInThousands'].transform('sum')
    df['AvgWeeklySales'] = df['TotalSales_4Weeks']/4
    
    print("Encoding Marketsize ordinally...")
    df['marketSizeEncoded'] = df['MarketSize'].map({'Small': 1, 'Medium': 2, 'Large': 3})
    
    print("sorting data")
    df= df.sort_values(['LocationID', 'week'])
    
    print("Calculating weekly sales growth rate per store/promotion")
    threshold = df['TotalSales_4Weeks'].median()
    df['HighPerformer'] = (df['TotalSales_4Weeks'] > threshold).astype(int)
    
    print("Saving cleaned data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Cleaned data saved to {}".format(output_path))
    

if __name__ == "__main__":
    load_and_clean_data()