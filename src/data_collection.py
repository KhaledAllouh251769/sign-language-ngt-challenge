"""
Data collection tool
Author: Person 2
"""

import cv2
import json
import os
from detection import HandDetector

class DataCollector:
    def __init__(self):
        self.detector = HandDetector()
        self.data_dir = 'data/reference'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_letter(self, letter):
        cap = cv2.VideoCapture(0)
        samples = []
        print(f'Recording {letter} - Press SPACE to capture, Q to finish')
        while len(samples) < 5:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, annotated_frame = self.detector.find_hands(frame)
            text = f'{letter}: {len(samples)}/5'
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Data Collection', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and landmarks:
                samples.append(landmarks)
                print(f'Captured {len(samples)}/5')
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if samples:
            person_name = input("Enter your name: ")
            filepath = os.path.join(self.data_dir, f'{letter}_{person_name}.json')
            with open(filepath, 'w') as f:
                json.dump(samples, f)
            print(f'Saved {len(samples)} samples')
        return len(samples)

if __name__ == '__main__':
    collector = DataCollector()
    for letter in ['A', 'B', 'C', 'D', 'E','F','G','I','J','K','L','M','N','O','P','Q','R','S','T','V','W','Y']:
        input(f'Ready to record {letter}? Press Enter')
        collector.collect_letter(letter)

    print('Done!')
