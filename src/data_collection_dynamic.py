"""
Dynamic letter collection - Averaging approach
"""

import cv2
import json
import os
import numpy as np
import time
from detection import HandDetector

class DynamicDataCollector:
    def __init__(self):
        self.person_name = input("Enter your name: ")
        self.detector = HandDetector()
        self.data_dir = 'data/reference'  # SAME folder as static!
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_dynamic_letter(self, letter):
        cap = cv2.VideoCapture(0)
        all_samples = []
        
        print(f'\nRecording letter: {letter} (DYNAMIC)')
        print('For each sample:')
        print('  1. Press SPACE to start')
        print('  2. Perform the motion continuously for 2 seconds')
        print('  3. System will average all frames')
        print('Need 5 samples total')
        
        for sample_num in range(5):
            print(f'\n--- Sample {sample_num + 1}/5 ---')
            input('Press Enter when ready...')
            
            frames_collected = []
            recording = False
            start_time = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks, annotated_frame = self.detector.find_hands(frame)
                
                if not recording:
                    cv2.putText(annotated_frame, f'{letter} - Sample {sample_num + 1}/5',
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, 'Press SPACE to start 2-sec recording',
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    elapsed = time.time() - start_time
                    remaining = 2.0 - elapsed
                    
                    if remaining > 0:
                        cv2.putText(annotated_frame, f'RECORDING... {remaining:.1f}s',
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(annotated_frame, f'Frames: {len(frames_collected)}',
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(annotated_frame, 'Keep moving!',
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        if landmarks:
                            frames_collected.append(landmarks)
                    else:
                        break
                
                cv2.imshow('Dynamic Letter Recording', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and not recording:
                    recording = True
                    start_time = time.time()
                    print('  Recording started! Perform the motion...')
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return 0
            
            # Average all collected frames
            if len(frames_collected) > 0:
                averaged = np.mean(frames_collected, axis=0).tolist()
                all_samples.append(averaged)
                print(f'  ✓ Averaged {len(frames_collected)} frames into 1 sample')
            else:
                print('  ⚠️ No frames captured, retrying...')
                sample_num -= 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save averaged samples (SAME format as static letters!)
        if all_samples:
            filepath = os.path.join(self.data_dir, f'{letter}_{self.person_name}.json')
            with open(filepath, 'w') as f:
                json.dump(all_samples, f)
            print(f'✅ Saved {len(all_samples)} averaged samples for {letter}')
        
        return len(all_samples)

if __name__ == '__main__':
    collector = DynamicDataCollector()
    
    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    
    print("="*60)
    print("DYNAMIC LETTER RECORDING")
    print("="*60)
    print("\nHow it works:")
    print("  - Press SPACE to start 2-second recording")
    print("  - Perform the letter motion continuously")
    print("  - System captures many frames and averages them")
    print("  - Repeat 5 times per letter")
    print("\nMotions:")
    print("  H: Move hand left-center-right smoothly")
    print("  J: Draw J-shape downward")
    print("  U: Draw U-shape with two fingers")
    print("  X: Hook and unhook finger")
    print("  Z: Draw Z-shape in air")
    print("="*60)
    
    for letter in dynamic_letters:
        input(f'\nReady to record {letter}? Press Enter...')
        collector.collect_dynamic_letter(letter)
    
    print('\n✅ All dynamic letters recorded!')