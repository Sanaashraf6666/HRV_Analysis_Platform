from flask import Flask, jsonify
from flask_cors import CORS
import random
from vercel_wsgi import handle_wsgi_app

app = Flask(__name__)
CORS(app)

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        "id": random.randint(1, 100),
        "value": random.random()
    }
    return jsonify(data)

# Vercel uses this handler to invoke the app
handler = handle_wsgi_app(app)