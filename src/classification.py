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
        self.reference_data_dynamic = {}
        self.dynamic_letters = ['H', 'J','U','X', 'Z']  # Define dynamic letters
        self.load_reference_data()
        self.load_dynamic_data()
    
    def load_reference_data(self):
        """Load static letter data (excludes H, J, X, Z)"""
        data_dir = 'data/reference'
        if not os.path.exists(data_dir):
            return
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                letter = filename[0]
                
                # SKIP dynamic letters in static data
                if letter in self.dynamic_letters:
                    print(f'  Skipping {letter} from static data (dynamic letter)')
                    continue
                
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    # Create list for this letter if doesn't exist
                    if letter not in self.reference_data:
                        self.reference_data[letter] = []
                    # Add samples from this file
                    samples = json.load(f)
                    self.reference_data[letter].extend(samples)
        
        print(f'Loaded {len(self.reference_data)} static letters')
        for letter, samples in self.reference_data.items():
            print(f'  {letter}: {len(samples)} samples')
    
    def load_dynamic_data(self):
        """Load dynamic letter data (ONLY H, J, X, Z)"""
        data_dir = 'data/reference_dynamic'
        if not os.path.exists(data_dir):
            print('No dynamic data found')
            return
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                letter = filename[0]
                
                # ONLY load if it's a dynamic letter
                if letter not in self.dynamic_letters:
                    print(f'  Skipping {letter} from dynamic folder (not a dynamic letter)')
                    continue
                
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Create list for this letter if doesn't exist
                    if letter not in self.reference_data_dynamic:
                        self.reference_data_dynamic[letter] = []
                    # Add sequences from this file
                    if 'sequences' in data:
                        self.reference_data_dynamic[letter].extend(data['sequences'])
        
        if self.reference_data_dynamic:
            print(f'Loaded {len(self.reference_data_dynamic)} dynamic letters')
            for letter, sequences in self.reference_data_dynamic.items():
                print(f'  {letter}: {len(sequences)} sequences')
    
    def classify_letter(self, landmarks):
        """Classify static letter from single landmark frame (excludes H, J, X, Z)"""
        if not landmarks or not self.reference_data:
            return None, 0.0
        
        current = np.array(landmarks).flatten()
        best_letter = None
        min_distance = float('inf')
        
        # Compare to all static reference samples (H, J, X, Z already excluded)
        for letter, samples in self.reference_data.items():
            for sample in samples:
                sample_array = np.array(sample).flatten()
                distance = np.linalg.norm(current - sample_array)
                
                if distance < min_distance:
                    min_distance = distance
                    best_letter = letter
        
        # Convert distance to confidence
        confidence = 1.0 / (1.0 + min_distance)
        return best_letter, confidence
    
    def classify_dynamic_letter(self, landmark_sequence):
        """
        Classify dynamic letter from sequence of landmarks (ONLY H, J, X, Z)
        landmark_sequence: list of 3 landmark frames [start, middle, end]
        """
        if not landmark_sequence or len(landmark_sequence) < 3:
            return None, 0.0
        
        if not self.reference_data_dynamic:
            return None, 0.0
        
        best_letter = None
        min_total_distance = float('inf')
        
        # Compare to all dynamic reference sequences (ONLY H, J, U,X, Z)
        for letter, sequences in self.reference_data_dynamic.items():
            for ref_sequence in sequences:
                total_distance = 0
                
                # Compare each position in the sequence
                for i in range(min(3, len(landmark_sequence), len(ref_sequence))):
                    current_pos = np.array(landmark_sequence[i]).flatten()
                    ref_pos = np.array(ref_sequence[i]).flatten()
                    distance = np.linalg.norm(current_pos - ref_pos)
                    total_distance += distance
                
                # Average distance across all positions
                avg_distance = total_distance / 3
                
                if avg_distance < min_total_distance:
                    min_total_distance = avg_distance
                    best_letter = letter
        
        # Convert distance to confidence
        confidence = 1.0 / (1.0 + min_total_distance)
        return best_letter, confidence

if __name__ == '__main__':
    classifier = LetterClassifier()
    print('\n' + '='*50)
    print('Classification module ready')
    print('='*50)
    print(f'Static letters ({len(classifier.reference_data)}): {sorted(classifier.reference_data.keys())}')
    print(f'Dynamic letters ({len(classifier.reference_data_dynamic)}): {sorted(classifier.reference_data_dynamic.keys())}')
    print('='*50)