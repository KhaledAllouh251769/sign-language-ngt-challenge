"""
Letter classification
Author: Person 2
"""

import numpy as np
import json
import os

class LetterClassifier:
    def __init__(self):
        self.reference_data = {}
        self.load_reference_data()
    
    def load_reference_data(self):
        data_dir = 'data/reference'
        if not os.path.exists(data_dir):
            return
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                letter = filename[0]
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    self.reference_data[letter] = json.load(f)
        print(f'Loaded {len(self.reference_data)} letters')
    
    def classify_letter(self, landmarks):
        if not landmarks or not self.reference_data:
            return None, 0.0
        return None, 0.0

if __name__ == '__main__':
    print('Classification module')
