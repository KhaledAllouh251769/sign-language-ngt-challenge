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
    """
    Receives landmarks from the browser (21 x [x,y,z] points)
    and returns the classified letter + confidence.
    """
    data = request.get_json()
    landmarks = data.get('landmarks')
    
    if not landmarks or len(landmarks) != 21:
        return jsonify({'letter': None, 'confidence': 0})

    letter, confidence = classifier.classify_letter(landmarks, use_smoothing=False)
    return jsonify({'letter': letter, 'confidence': round(float(confidence), 3)})

if __name__ == '__main__':
    print("="*50)
    print("NGT Sign Language App")
    print("Open http://localhost:5000 in your browser")
    print("="*50)
    app.run(debug=False, host='0.0.0.0', port=5000)