import datetime
from web_scrape_sse_stock_ticker.fetch_sse_ticker import fetch_sse_tickers
from load_data.fetch_hist_data_to_mongodb import fetch_hist_data_to_mongodb

def run_data_pipeline():
    """
    Orchestrates the process of fetching tickers and loading their data.
    """
    print("--- Starting Data Pipeline ---")

    ###########################
    # CONFIGS -----------------
    ###########################
    NUM_TICKERS_TO_FETCH = 500
    START_DATE = "2010-01-01"
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    INTERVAL = "1d"  # Daily data

    # 1. FETCH TICKER SYMBOLS FROM SHANGHAI STOCK EXCHANGE
    print(f"\n Step 1: Fetching top {NUM_TICKERS_TO_FETCH} tickers from SSE...")
    sse_tickers = fetch_sse_tickers(NUM_TICKERS_TO_FETCH)

    if not isinstance(sse_tickers, list) or not sse_tickers:
        print(f"Failed to fetch tickers. Error: {sse_tickers}")
        return
    
    print(f"Fetched {len(sse_tickers)} tickers: {sse_tickers}")

    # 2. PREPARE TICKERS FOR YFINANCE
    tickers_for_yfinance = [f"{ticker}.SS" for ticker in sse_tickers]
    print(f"Prepared tickers for yfinance: {tickers_for_yfinance}")

    # 3. FETCH HISTORICAL DATA AND STORE IN MONGODB 
    print("\n Step 2: Fetching historical data and storing in MongoDB...")
    fetch_hist_data_to_mongodb(
        tickers_list=tickers_for_yfinance,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL
    )

    print("--- Data Pipeline Completed ---")

if __name__ == "__main__":
    run_data_pipeline()