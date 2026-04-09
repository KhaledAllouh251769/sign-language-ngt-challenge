"""
NGT Sign Language - Flask Backend
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import sys
import os
import base64
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set working directory to project root so data/reference is found
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from classification import LetterClassifier

app = Flask(__name__)
classifier = LetterClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    landmarks = data.get('landmarks')
    if not landmarks or len(landmarks) != 21:
        return jsonify({'letter': None, 'confidence': 0})
    letter, confidence = classifier.classify_letter(landmarks, use_smoothing=False)
    return jsonify({'letter': letter, 'confidence': round(float(confidence), 3)})

@app.route('/save_sample', methods=['POST'])
def save_sample():
    import json
    data        = request.get_json()
    letter      = data.get('letter', '').upper()
    person_name = data.get('person_name', '').strip()
    frames      = data.get('frames', [])

    if not letter or not person_name or not frames:
        return jsonify({'success': False, 'error': 'Missing data'})
    if len(frames) < 5:
        return jsonify({'success': False, 'error': 'Not enough frames'})

    data_dir = 'data/reference'
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f'{letter}_{person_name}.json')

    existing = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing = json.load(f)

    averaged = np.mean(frames, axis=0).tolist()
    existing.append(averaged)

    with open(filepath, 'w') as f:
        json.dump(existing, f)

    classifier.load_reference_data()
    return jsonify({'success': True, 'total_samples': len(existing)})

if __name__ == '__main__':
    print("="*50)
    print("NGT Sign Language App")
    print("Open http://localhost:5000 in your browser")
    print("="*50)
    app.run(debug=False, host='0.0.0.0', port=5000)