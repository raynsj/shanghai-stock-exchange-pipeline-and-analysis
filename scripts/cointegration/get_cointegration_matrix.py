import pandas as pd
import yfinance as yf
import numpy as np 
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import datetime

from pymongo import MongoClient
import os
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set.")

client = MongoClient(MONGO_URI)
db = client["stocks_db"]
collection = db["daily_prices"]



end_date = datetime.datetime.today()
target_start_date = end_date - datetime.timedelta(days=365 * 3)  # 3 years ago

print(f"Using a 3-year lookback window: {target_start_date.date()} to {end_date.date()}")

pipeline = [
    {
        "$group": {
            "_id": "$ticker",
            "min_date": {"$min": "$date"}
        }
    },
    {
        "$match": {
            "min_date": {"$lte": target_start_date}
        }
    }
]

try:
    # Execute the pipeline and convert the result to a list
    results = list(collection.aggregate(pipeline))
    
    # Extract just the ticker names from the aggregation results
    tickers_since_start_date = [doc['_id'] for doc in results]

    if tickers_since_start_date:
        print(f"\nFound {len(tickers_since_start_date)} tickers with data since {target_start_date.date()}.")
        print("Here are the first 50:")
        print(tickers_since_start_date[:50])
    else:
        print("No tickers found with data starting on or before the specified date.")

except Exception as e:
    print(f"An error occurred during the database query: {e}")





# PEARSON CORRELATION MATRIX
if 'tickers_since_start_date' in locals() and tickers_since_start_date:
    query = {
        "ticker": {"$in": tickers_since_start_date},
        "date": {"$gte": target_start_date}
    }
    projection = {"_id": 0, "date": 1, "ticker": 1, "close": 1}
    all_data = pd.DataFrame(list(collection.find(query, projection)))

    if not all_data.empty:
        df = all_data.pivot(index='date', columns='ticker', values='close')
        df.dropna(inplace=True)
        print(f"Price matrix created with {len(df)} common trading days.")
        print("Sample of the price matrix (df):")
        print(df.head())

        print(f"\nGenerating unique pairs from {len(df.columns)} tickers...")
        corr_matrix = df.corr()

        print("Correlation matrix calculation complete.")

        # 2. Generate unique pairs from the tickers that are in the matrix
        print(f"\nGenerating unique pairs from {len(corr_matrix.columns)} tickers...")
        ticker_pairs = combinations(corr_matrix.columns, 2)
        pairs_df = pd.DataFrame(list(ticker_pairs), columns=["ticker1", "ticker2"])
        print(f"Generated {len(pairs_df)} pairs.")

        # 3. Look up the correlation for each pair from the pre-computed matrix.
        # This is much faster than calculating for each pair individually.
        print("\nMapping correlation values to pairs...")
        pairs_df['correlation'] = pairs_df.apply(lambda row: corr_matrix.loc[row['ticker1'], row['ticker2']], axis=1)
        
        print("\nTop 10 most correlated pairs:")
        print(pairs_df.sort_values(by='correlation', ascending=False).head(10))
    else:
        print("No data available for the specified tickers and date range.")
else:
    print("No tickers found with data since start_date. Please check your database or the date filter.")


pairs_df = pairs_df[pairs_df['correlation'] > 0.95]


def do_regression(ticker1, ticker2):
    X = df[ticker1].values
    y = df[ticker2].values

    X = sm.add_constant(X)  # Add a constant term for the intercept
    model = sm.OLS(y, X).fit()
    alpha,beta = model.params

    residuals = y - (alpha + beta * X[:,1])

    if np.isnan(residuals).any():
        return np.nan

    adf_results = adfuller(residuals)

    return adf_results[0], adf_results[1], beta, alpha


pairs_df[['adf_stat', 'p_value', 'beta', 'alpha']] = pairs_df.apply(lambda row: pd.Series(do_regression(row['ticker1'], row['ticker2'])), axis=1)
pairs_df = pairs_df[pairs_df.p_value < 0.01].sort_values(by='adf_stat')


if not pairs_df.empty:
    # 1. Select the top pair (the one with the most negative ADF statistic)
    top_pair = pairs_df.iloc[0]
    ticker1 = top_pair['ticker1']
    ticker2 = top_pair['ticker2']
    beta = top_pair['beta']
    alpha = top_pair['alpha']

    # 2. Calculate the spread using the hedge ratio (beta)
    # The spread is the price of stock 2 minus the hedged price of stock 1
    spread = df[ticker2] - beta * df[ticker1]
    
    # 3. Set a modern plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    # 4. Plot the spread over time
    spread.plot(ax=ax, color='royalblue', label='Spread')
    
    # 5. Plot the mean and standard deviation bands for context
    # The mean of the spread is 'alpha' from our regression
    mean_line = ax.axhline(alpha, color='black', linestyle='--', label=f'Long-Term Mean (alpha)')
    
    # Calculate standard deviation of the spread
    spread_std = spread.std()
    
    # Plot +/- 2 standard deviation bands
    upper_band = ax.axhline(alpha + 2 * spread_std, color='red', linestyle=':', label='+2 Std Dev')
    lower_band = ax.axhline(alpha - 2 * spread_std, color='green', linestyle=':', label='-2 Std Dev')

    # Add shaded region for +/- 1 standard deviation for a nicer look
    ax.fill_between(spread.index, alpha + spread_std, alpha - spread_std, color='gray', alpha=0.2)

    # 6. Make the chart look nice
    ax.set_title(f'Cointegration Spread: {ticker2} vs. {ticker1}', fontsize=18, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Spread Value', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a text box with key stats
    stats_text = (f"Beta (Hedge Ratio): {beta:.2f}\n"
                  f"ADF Stat: {top_pair['adf_stat']:.2f}\n"
                  f"P-Value: {top_pair['p_value']:.4f}")
    ax.text(0.8, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=1.0', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

else:
    print("No cointegrated pairs were found to plot.")
