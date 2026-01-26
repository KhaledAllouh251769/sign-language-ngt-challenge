"""
Dynamic letter data collection (H, J, X, Z)
Author: Team
"""

import cv2
import json
import os
from detection import HandDetector

class DynamicDataCollector:
    def __init__(self):
        self.person_name = input("Enter your name: ")
        self.detector = HandDetector()
        self.data_dir = 'data/reference_dynamic'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_dynamic_letter(self, letter):
        cap = cv2.VideoCapture(0)
        sequences = []
        
        print(f'\nRecording DYNAMIC letter: {letter}')
        print('For each sequence, press SPACE at: START -> MIDDLE -> END')
        print('Need 5 complete sequences (15 total captures)')
        print('Press Q to finish')
        
        current_sequence = []
        sequence_count = 0
        
        while sequence_count < 5:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, annotated_frame = self.detector.find_hands(frame)
            
            position_in_sequence = len(current_sequence)
            position_names = ['START', 'MIDDLE', 'END']
            current_pos = position_names[position_in_sequence] if position_in_sequence < 3 else 'COMPLETE'
            
            text1 = f'{letter} - Sequence {sequence_count + 1}/5'
            text2 = f'Position: {current_pos} ({position_in_sequence}/3)'
            
            cv2.putText(annotated_frame, text1, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, text2, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(annotated_frame, 'SPACE=Capture Q=Finish', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Dynamic Letter Recording', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and landmarks:
                current_sequence.append(landmarks)
                print(f'  Captured {current_pos}')
                
                if len(current_sequence) == 3:
                    sequences.append(current_sequence)
                    sequence_count += 1
                    print(f'  ✅ Sequence {sequence_count}/5 complete!')
                    current_sequence = []
                    
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if sequences:
            data = {
                'letter': letter,
                'type': 'dynamic',
                'person': self.person_name,
                'sequences': sequences
            }
            filepath = os.path.join(self.data_dir, f'{letter}_{self.person_name}.json')
            with open(filepath, 'w') as f:
                json.dump(data, f)
            print(f'✅ Saved {len(sequences)} motion sequences for {letter}\n')
        
        return len(sequences)

if __name__ == '__main__':
    collector = DynamicDataCollector()
    
    dynamic_letters = ['H', 'J', 'U','X', 'Z']
    
    print("=" * 60)
    print("DYNAMIC LETTER RECORDING - Motion Capture")
    print("=" * 60)
    print("\nMotion Instructions:")
    print("  H: Move hand LEFT -> CENTER -> RIGHT (side to side)")
    print("  J: Start TOP -> curve DOWN -> hook LEFT (J-shape)")
    print("  X: Finger STRAIGHT -> HALF-BENT -> HOOKED (hook motion)")
    print("  Z: Draw Z: TOP-RIGHT -> DIAGONAL -> BOTTOM-LEFT")
    print("\nFor each letter:")
    print("  - Press SPACE at START position")
    print("  - Press SPACE at MIDDLE position")
    print("  - Press SPACE at END position")
    print("  - Repeat 5 times")
    print("=" * 60)
    
    for letter in dynamic_letters:
        input(f'\nReady to record {letter}? Press Enter...')
        collector.collect_dynamic_letter(letter)
    
    print('\n' + "=" * 60)
    print('✅ All dynamic letters recorded!')
    print(f'Files saved in: data/reference_dynamic/')
    print("=" * 60)