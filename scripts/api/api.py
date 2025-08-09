from flask import Flask, request, jsonify
from pymongo import MongoClient
import datetime
import os
import json
from dotenv import load_dotenv 

load_dotenv()


# Custom JSON encoder to handle MongoDB datetime objects
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


# Initialize Flask App and custom encoder
app = Flask(__name__)
app.json_encoder = JSONEncoder


MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MONGO_URI environment variable is not set.")
    
client = MongoClient(MONGO_URI)
db = client["stocks_db"]
collection = db["daily_prices"]



@app.route('/api/timeseries', methods=['GET'])
def get_timeseries_data():
    """
    Exposes stock data via a GET request.
    Query Parameters:
        - ticker: The stock symbol (e.g., "AAPL"). Required.
        - start_date: The start of the date range (e.g., "2025-01-01"). Required.
        - end_date: The end of the date range (e.g., "2025-01-31"). Required.
        - fields: Comma-separated list of fields to return (e.g., "high,low,close"). Optional.
    """
    # 1. Get query parameters
    ticker = request.args.get('ticker')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    fields = request.args.get('fields')

    # 2. Validate required parameters
    if not all([ticker, start_date, end_date]):
        return jsonify({"error": "Missing required parameters: ticker, start_date, end_date"}), 400

    # 3. Build MongoDB query
    try:
        query = {
            "ticker": ticker.upper(),
            "date": {
                "$gte": datetime.datetime.fromisoformat(start_date),
                "$lte": datetime.datetime.fromisoformat(end_date)
            }
        }
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400


    # 4. Build projection to select fields
    projection = {"_id": 0} # exclude MongoDB's default _id field
    if fields:
        for field in fields.split(','):
            projection[field.strip()] = 1
        projection["date"] = 1  # Always include date field

    # 5. Execute Query MongoDB and return results
    try:
        cursor = collection.find(query, projection).sort("date", 1)
        results = list(cursor)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 6. Run Flask app
    app.run(debug=True, port=5000)

# curl "http://127.0.0.1:5000/api/timeseries?ticker=AAPL&start_date=2025-01-02&end_date=2025-01-10&fields=high,close,volume"