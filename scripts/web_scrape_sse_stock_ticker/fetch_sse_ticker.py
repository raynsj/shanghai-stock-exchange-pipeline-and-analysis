import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import json

# FETCHNG TICKERS FROM SHANGHAI STOCK EXCHANGE ACCORDING TO MARKET CAPITALIZATION
def fetch_sse_tickers(num_of_tickers):
    url = "https://stockanalysis.com/list/shanghai-stock-exchange/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses

        # Parsing the HTML Content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all table cells 'td') with the class 'sym' which contains the stock tickers
        ticker_cells = soup.find_all('td', class_='sym')

        # Extracting the tickers
        tickers_list = []
        for cell in ticker_cells:
            if cell.a:
                tickers_list.append(cell.a.text.strip())

        tickers_to_return = tickers_list[:num_of_tickers] 
        return tickers_to_return
    except requests.exceptions.RequestException as e:
        error_message = f"An error occurred while fetching the data: {e}"
        return error_message


    # if first_500_tickers:
    #     df = pd.DataFrame(first_500_tickers, columns=['ticker'])
    #     print(f"Successfully scraped {len(df)} tickers.")
    #     print(df[:5])
    # else:
    #     print("No tickers found or an error occurred during scraping.")