import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, jsonify, request
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS
import logging
import requests

# Set Flask logging to a higher level to see more debug info
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

last_analysis_results = {}

@app.route('/')
def index():
    return render_template('index.html')

def generate_mock_rr_intervals(num_points=500):
    """
    Generates a mock time series of RR intervals for demonstration.
    """
    base_rr = np.random.uniform(700, 950)
    rr_intervals = np.random.normal(base_rr, 50, num_points)
    sin_wave = np.random.uniform(10, 30) * np.sin(np.linspace(0, np.random.uniform(1, 3) * np.pi, num_points))
    rr_intervals += sin_wave
    shift_start = np.random.randint(100, num_points - 100)
    shift_end = shift_start + np.random.randint(5, 15)
    rr_intervals[shift_start:shift_end] += np.random.uniform(30, 80)
    rr_intervals = np.clip(rr_intervals, 400, 1500)
    return rr_intervals

def generate_plot_base64(rr_intervals, sd1, sd2):
    """
    Generates HRV plots and returns them as a base64 encoded string.
    """
    fs = 1000 / np.mean(rr_intervals)
    freqs, pxx = signal.welch(x=rr_intervals, fs=fs, nperseg=256)
    
    img = io.BytesIO()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.plot(rr_intervals, color='b')
    ax1.set_title('Time-Domain Tachogram')
    ax1.set_xlabel('Beat Number')
    ax1.set_ylabel('RR Interval (ms)')
    ax1.grid(True)
    
    ax2.plot(freqs, pxx, color='g')
    ax2.set_title('Frequency-Domain Power Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power ($ms^2/Hz$)')
    ax2.set_xlim(0, 0.5)
    ax2.grid(True)
    
    rr_x = rr_intervals[:-1]
    rr_y = rr_intervals[1:]
    ax3.scatter(rr_x, rr_y, alpha=0.5, color='r')
    ax3.set_title('Poincaré Plot')
    ax3.set_xlabel('$RR_n (ms)$')
    ax3.set_ylabel('$RR_{n+1} (ms)$')
    ax3.text(1.05, 0.5, f'SD1: {sd1:.2f} ms\nSD2: {sd2:.2f} ms',
             transform=ax3.transAxes, ha='left', va='center')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return plot_base64

def calculate_hrv_parameters(rr_intervals):
    """
    Calculates various HRV parameters from RR-interval data.
    """
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    
    fs = 1000.0 / mean_rr
    freqs, pxx = signal.welch(x=rr_intervals, fs=fs, nperseg=256)
    
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    # Using np.trapezoid instead of np.trapz to address the DeprecationWarning
    vlf_power = np.trapezoid(pxx[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])], freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.trapezoid(pxx[(freqs >= lf_band[0]) & (freqs < lf_band[1])], freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapezoid(pxx[(freqs >= hf_band[0]) & (freqs < hf_band[1])], freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    sd1 = np.std(np.diff(rr_intervals) / np.sqrt(2))
    sd2 = np.std((rr_intervals[:-1] + rr_intervals[1:]) / np.sqrt(2))
    
    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'lf_hf_ratio': lf_hf_ratio,
        'sd1': sd1,
        'sd2': sd2
    }


@app.route('/analyze_hrv', methods=['POST'])
def analyze_hrv_with_results_storage():
    global last_analysis_results
    logging.info("Starting HRV data analysis...")

    request_data = request.get_json(silent=True)
    if not request_data or 'method' not in request_data:
        return jsonify({'error': 'Invalid request body. Method not specified.'}), 400

    data = request_data.get('data')
    method = request_data.get('method')
    
    rr_intervals = None
    if method == 'csv' or method == 'manual':
        if not data:
            return jsonify({'error': 'No data provided for the selected method.'}), 400
        try:
            rr_intervals_str = data.split(',')
            rr_intervals_list = [float(rr.strip()) for rr in rr_intervals_str if rr.strip()]
        except ValueError:
            return jsonify({'error': 'Invalid data format. Please provide comma-separated numbers only.'}), 400
        
        if len(rr_intervals_list) < 200:
            return jsonify({'error': 'Insufficient data. Please provide at least 200 RR-intervals for proper analysis.'}), 400
        
        rr_intervals = np.array(rr_intervals_list)
    elif method in ['device', 'mock']:
        logging.info(f"Method '{method}' selected. Generating mock data.")
        rr_intervals = generate_mock_rr_intervals()
    else:
        return jsonify({'error': 'Invalid analysis method specified.'}), 400

    try:
        rr_intervals = np.array(rr_intervals)
        
        analysis_results = calculate_hrv_parameters(rr_intervals)
        last_analysis_results = analysis_results
        
        sd1 = analysis_results['sd1']
        sd2 = analysis_results['sd2']
        
        plot_base64 = generate_plot_base64(rr_intervals, sd1, sd2)

        logging.info("Analysis complete. Sending results to frontend.")
        return jsonify({
            'results': analysis_results,
            'plot': plot_base64
        })
    except Exception as e:
        # We will now print the actual error to the console for better debugging.
        logging.error(f"An error occurred during analysis: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during analysis. Please try again.'}), 500


@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    global last_analysis_results
    logging.info("Fetching feedback...")
    
    if not last_analysis_results:
        logging.warning("No analysis results found. Requesting analysis first.")
        return jsonify({'feedback': 'Please analyze the data first.'})

    rmssd = last_analysis_results.get('rmssd', 0)
    sdnn = last_analysis_results.get('sdnn', 0)
    lf_hf_ratio = last_analysis_results.get('lf_hf_ratio', 0)
    sd1 = last_analysis_results.get('sd1', 0)
    
    # Get API key from environment variables.
    api_key = os.environ.get('API_KEY', '') # Add a default value to prevent NoneType error
    if not api_key:
        logging.error("API_KEY environment variable is not set.")
        return jsonify({'error': 'Feedback service is not available. API key is missing.'}), 503
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    system_prompt = "Act as a world-class heart rate variability (HRV) analyst. Provide a single-paragraph summary of the key findings from the provided HRV parameters. Conclude with a list of 3 specific and actionable recommendations for improvement, with each recommendation on a new line. Format the entire response as a list. dont use * or ** .list parameter values as list"

    user_query = f"Provide an HRV analysis based on the following parameters: RMSSD: {rmssd:.2f} ms, SDNN: {sdnn:.2f} ms, LF/HF Ratio: {lf_hf_ratio:.2f}, SD1: {sd1:.2f} ms."
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Failed to generate feedback.')
        
        logging.info("Feedback generated by LLM and sent to frontend.")
        return jsonify({'feedback': generated_text})
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return jsonify({'error': 'An error occurred while connecting to the feedback service.'}), 500
    except (KeyError, IndexError) as e:
        logging.error(f"Failed to parse API response: {e}")
        return jsonify({'error': 'An error occurred while parsing the feedback response.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
