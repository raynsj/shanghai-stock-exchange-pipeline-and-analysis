from pymongo import MongoClient, UpdateOne
from pymongo.errors import ServerSelectionTimeoutError
import yfinance as yf
import pandas as pd
import os
import datetime
from dotenv import load_dotenv


def fetch_hist_data_to_mongodb(tickers_list, start_date, end_date, interval):
    """
    Fetch historical stock data from Yahoo Finance and store it in MongoDB.
    
    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    """

    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise Exception("MONGO_URI environment variable is not set.")

    client = MongoClient(MONGO_URI)
    db = client["stocks_db"]
    collection = db["daily_prices"]

    try:
        print(client.admin.command({"ping": 1}))
    except ServerSelectionTimeoutError as e:
        print("MongoDB not reachable:", e)
        raise

    # Ensure efficient upserts
    collection.create_index([("ticker", 1), ("date", 1)], unique=True)

    print("Downloading historical data from yfinance (may take a while)...")
    # Download the data
    df = yf.download(tickers_list, start=start_date, end=end_date, interval=interval, progress=True)

    if df.empty:
        print("No data returned from yfinance.")
    else: 
        df_stacked = df.stack()

        df_stacked = df_stacked.reset_index()

        df_stacked.rename(columns={
            'Ticker': 'ticker',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # PREPARE DATA FOR MONGOGDB
        ops = []
        for record in df_stacked.to_dict("records"):
            ops.append(
                UpdateOne(
                    {"ticker": record["ticker"], "date": record["date"]},
                    {"$set": record},
                    upsert=True
                )
            )

        # BULK WRITE TO MONGODB
        if ops:
            res = collection.bulk_write(ops, ordered=False)
            print(f"Upserted: {res.upserted_count}, Modified: {res.modified_count}")
        else:
            print("No operations to perform on MongoDB.")