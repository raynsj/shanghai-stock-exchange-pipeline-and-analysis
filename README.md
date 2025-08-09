# Stock Data Pipeline

A simple data pipeline that scrapes Shanghai Stock Exchange (SSE) tickers, fetches historical price data, and stores it in MongoDB. A basic REST API is included to serve the data.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment:**
    Create a `.env` file in the root directory and add your MongoDB connection string:
    ```
    MONGO_URI="mongodb://127.0.0.1:27017/?directConnection=true"
    ```

## How to Run

1.  **Run the Data Pipeline:**
    This script fetches all the data and loads it into the database.
    ```bash
    python scripts/run_data_pipeline.py
    ```

2.  **Start the API Server:**
    ```bash
    python scripts/api/api.py
    ```

## API Usage

-   **Endpoint:** `GET /api/timeseries`
-   **Example:**
    ```bash
    curl "http://127.0.0.1:5000/api/timeseries?ticker=601288.SS&start_date=2020-01-01&end_date=2020-01-10&fields=high,close,volume"
    ```