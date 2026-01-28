"""
Letter classification
Author: Person 2
"""

import numpy as np
import json
import os
from collections import Counter

class LetterClassifier:
    def __init__(self):
        self.reference_data = {}
        self.prediction_buffer = []
        self.buffer_size = 5
        self.load_reference_data()
    
    def load_reference_data(self):
        """Load ALL letters (static and dynamic averaged)"""
        data_dir = 'data/reference'
        if not os.path.exists(data_dir):
            print('No reference data found')
            return
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                letter = filename[0].upper()
                filepath = os.path.join(data_dir, filename)
                
                with open(filepath, 'r') as f:
                    # Create list for this letter if doesn't exist
                    if letter not in self.reference_data:
                        self.reference_data[letter] = []
                    
                    # Load samples
                    samples = json.load(f)
                    
                    # Handle both formats (list of samples or dict with sequences)
                    if isinstance(samples, dict) and 'sequences' in samples:
                        # Old dynamic format - convert sequences to averages
                        for sequence in samples['sequences']:
                            averaged = np.mean(sequence, axis=0).tolist()
                            self.reference_data[letter].append(averaged)
                    else:
                        # Normal format - just extend
                        self.reference_data[letter].extend(samples)
        
        print(f'Loaded {len(self.reference_data)} letters')
        for letter in sorted(self.reference_data.keys()):
            samples = self.reference_data[letter]
            print(f'  {letter}: {len(samples)} samples')
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to remove position and scale variations"""
        landmarks = np.array(landmarks)
        
        # Center around wrist (landmark 0)
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # Calculate hand size (distance from wrist to middle finger tip)
        middle_finger_tip = centered[12]
        hand_size = np.linalg.norm(middle_finger_tip)
        
        # Normalize by hand size
        if hand_size > 0:
            normalized = centered / hand_size
        else:
            normalized = centered
        
        return normalized.flatten()
    
    def classify_letter(self, landmarks, use_smoothing=True):
        """
        Classify ANY letter (static or dynamic averaged)
        
        Args:
            landmarks: List of 21 hand landmarks
            use_smoothing: Whether to use prediction smoothing
            
        Returns:
            letter: Predicted letter (A-Z) or None
            confidence: Confidence score (0.0-1.0)
        """
        if not landmarks or not self.reference_data:
            return None, 0.0
        
        # Normalize current landmarks
        current = self.normalize_landmarks(landmarks)
        
        # Calculate average distance to each letter's samples
        letter_distances = {}
        
        for letter, samples in self.reference_data.items():
            distances = []
            for sample in samples:
                # Normalize reference sample
                sample_normalized = self.normalize_landmarks(sample)
                
                # Calculate distance
                distance = np.linalg.norm(current - sample_normalized)
                distances.append(distance)
            
            # Use average of 3 closest samples (more stable)
            distances.sort()
            top_3_avg = np.mean(distances[:min(3, len(distances))])
            letter_distances[letter] = top_3_avg
        
        # Find best match
        best_letter = min(letter_distances, key=letter_distances.get)
        min_distance = letter_distances[best_letter]
        
        # Convert distance to confidence
        confidence = 1.0 / (1.0 + min_distance * 0.5)
        
        # Return None if confidence too low
        if confidence < 0.3:
            return None, 0.0
        
        # Apply smoothing to reduce flickering
        if use_smoothing and best_letter:
            self.prediction_buffer.append(best_letter)
            if len(self.prediction_buffer) > self.buffer_size:
                self.prediction_buffer.pop(0)
            
            # Return most common prediction in buffer (need at least 3 predictions)
            if len(self.prediction_buffer) >= 3:
                most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
                return most_common, confidence
        
        return best_letter, confidence

if __name__ == '__main__':
    classifier = LetterClassifier()
    print('\n' + '='*50)
    print('Classification module ready')
    print('='*50)
    print(f'Total letters loaded: {len(classifier.reference_data)}')
    print(f'Letters: {sorted(classifier.reference_data.keys())}')
    print('='*50)