from flask import Flask, jsonify
from flask_cors import CORS
import random
from datetime import datetime, timedelta

# Initialize Flask App
app = Flask(__name__)
# Enable CORS to allow requests from your React frontend
CORS(app)

@app.route('/api/data', methods=['GET'])
def get_data():
    # Simulate some data fetching
    data = {
        "id": random.randint(1, 100),
        "value": random.random()
    }
    return jsonify(data)

# Running on http://127.0.0.1:5000
app.run(debug=True, port=5000)