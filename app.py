# app.py (updated version)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from model import TC_Model
from data import RealTimeData
import logging
import base64
from cyclone_tracker import RealTimeCycloneTracker  # Import the tracker class
# Change this line in app.py
from cyclone_tracker import RealTimeCycloneTracker


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize model and data clients
model = TC_Model(weights_path='ri_model_weights.weights.h5')
data_client = RealTimeData()
cyclone_tracker = RealTimeCycloneTracker()  # Initialize the cyclone tracker

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_risk_gauge(probability):
    """Generate a horizontal bar gauge for RI probability"""
    fig, ax = plt.subplots(figsize=(6, 1))
    color = '#e57373' if probability > 0.7 else '#ffb300' if probability > 0.4 else '#4a90e2'
    ax.barh(['Risk'], [probability], color=color)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.set_facecolor('#3b4a6b')
    fig.patch.set_facecolor('#3b4a6b')
    ax.tick_params(colors='#ffffff')
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    plt.close(fig)
    return buffer

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/prediction')
def prediction_page():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/realtime')
def realtime_page():
    try:
        # Get active cyclones with marked images
        cyclone_data = cyclone_tracker.get_active_cyclones()
        
        if cyclone_data['success'] and cyclone_data['cyclones']:
            # Use the first cyclone's image as initial image
            first_cyclone = cyclone_data['cyclones'][0]
            initial_image = first_cyclone['image']
            timestamp = cyclone_data['timestamp']
            source = first_cyclone['image_url']
        else:
            # Get regular satellite image if no cyclones
            satellite_data = data_client.get_latest_satellite()
            if satellite_data['success']:
                img_buffer = io.BytesIO(satellite_data['compressed_image'])
                img_buffer.seek(0)
                initial_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                timestamp = satellite_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')
                source = satellite_data['source']
            else:
                initial_image = None
                timestamp = None
                source = None
                
    except Exception as e:
        logger.error(f"Error fetching initial data: {e}")
        initial_image = None
        timestamp = None
        source = None
        
    return render_template('realtime.html', 
                         initial_image=initial_image, 
                         timestamp=timestamp, 
                         source=source)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and predict RI probability"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            img = Image.open(file).convert('RGB')
            img = img.resize((256, 256))
            img_array = np.array(img) / 255.0
            
            # Predict RI probability
            ri_prob = model.predict(img_array)
            
            # Generate risk gauge
            gauge_buffer = generate_risk_gauge(ri_prob)
            
            # Save uploaded image temporarily
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='WEBP', quality=90)
            img_buffer.seek(0)
            
            return jsonify({
                'success': True,
                'ri_probability': f'{ri_prob*100:.1f}',
                'risk_level': 'High' if ri_prob > 0.7 else 'Moderate' if ri_prob > 0.4 else 'Low',
                'image': f'data:image/webp;base64,{base64.b64encode(img_buffer.getvalue()).decode("utf-8")}',
                'gauge': f'data:image/png;base64,{base64.b64encode(gauge_buffer.getvalue()).decode("utf-8")}'
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/realtime', methods=['POST'])
def get_realtime_data():
    """Fetch real-time cyclone data with marked images and predict RI"""
    try:
        # Get active cyclones with marked images
        cyclone_data = cyclone_tracker.get_active_cyclones()
        
        if not cyclone_data['success']:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch cyclone data',
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }), 500
            
        if not cyclone_data['cyclones']:
            # No active cyclones - return basic satellite image
            satellite_data = data_client.get_latest_satellite()
            if not satellite_data['success']:
                return jsonify({
                    'success': False,
                    'error': 'No active cyclones and failed to fetch satellite image',
                    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                }), 500
                
            return jsonify({
                'success': True,
                'cyclones': [],
                'message': 'No active tropical cyclones detected',
                'image': f'data:image/webp;base64,{base64.b64encode(satellite_data["compressed_image"]).decode("utf-8")}',
                'timestamp': satellite_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC'),
                'source': satellite_data['source']
            })
        
        # Process each cyclone
        processed_cyclones = []
        for cyclone in cyclone_data['cyclones']:
            # Get the marked image array for prediction
            img_array = cyclone['image_array']
            
            # Predict RI probability with location context
            ri_prob = model.predict_with_location(
                img_array,
                cyclone['lat'],
                cyclone['lon']
            )
            
            processed_cyclones.append({
                'name': cyclone['name'],
                'lat': cyclone['lat'],
                'lon': cyclone['lon'],
                'wind_speed': cyclone['wind_speed'],
                'pressure': cyclone['pressure'],
                'movement': cyclone['movement'],
                'image': cyclone['image'],  # Marked image in base64
                'image_url': cyclone['image_url'],
                'ri_probability': f'{ri_prob*100:.1f}',
                'risk_level': 'High' if ri_prob > 0.7 else 'Moderate' if ri_prob > 0.4 else 'Low'
            })
        
        # Generate a combined risk gauge (average of all cyclones)
        avg_prob = sum(float(c['ri_probability']) for c in processed_cyclones) / len(processed_cyclones) / 100
        gauge_buffer = generate_risk_gauge(avg_prob)
        
        return jsonify({
            'success': True,
            'cyclones': processed_cyclones,
            'gauge': f'data:image/png;base64,{base64.b64encode(gauge_buffer.getvalue()).decode("utf-8")}',
            'timestamp': cyclone_data['timestamp'],
            'message': f'Tracking {len(processed_cyclones)} active cyclones'
        })
        
    except Exception as e:
        logger.error(f"Error in realtime data: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)