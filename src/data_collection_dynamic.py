"""
Dynamic letter collection - Averaging approach
Camera closes after each sample
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
        self.data_dir = 'data/reference'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_one_sample(self, letter, sample_num):
        """Collect one sample (opens and closes camera)"""
        detector = HandDetector()
        cap = cv2.VideoCapture(0)
        
        # Warm up camera
        for _ in range(5):
            cap.read()
        
        frames_collected = []
        recording = False
        start_time = None
        
        print(f'  Sample {sample_num}/5 - Press SPACE to start recording')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, annotated_frame = detector.find_hands(frame)
            
            if not recording:
                cv2.putText(annotated_frame, f'{letter} - Sample {sample_num}/5',
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
                    # Recording complete
                    break
            
            cv2.imshow('Dynamic Letter Recording', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording:
                recording = True
                start_time = time.time()
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        # Close camera immediately
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process close event
        
        # Average frames
        if len(frames_collected) > 0:
            averaged = np.mean(frames_collected, axis=0).tolist()
            print(f'  ✓ Averaged {len(frames_collected)} frames')
            return averaged
        else:
            print(f'  ⚠️ No frames captured')
            return None
    
    def collect_dynamic_letter(self, letter):
        """Collect 5 samples for a dynamic letter"""
        all_samples = []
        
        print(f'\n{"="*50}')
        print(f'Recording letter: {letter}')
        print(f'{"="*50}')
        print('Instructions:')
        print('  1. Get ready with the sign')
        print('  2. Press SPACE in the camera window')
        print('  3. Perform motion smoothly for 2 seconds')
        print('  4. Camera closes automatically')
        print('  5. Repeat 5 times')
        print(f'{"="*50}\n')
        
        for sample_num in range(1, 6):
            input(f'Ready for sample {sample_num}/5? Press Enter...')
            
            sample = self.collect_one_sample(letter, sample_num)
            
            if sample is not None:
                all_samples.append(sample)
            else:
                print('  Retrying this sample...')
                sample_num -= 1  # Don't increment, retry
            
            # Small pause between samples
            time.sleep(0.3)
        
        # Save all samples
        if all_samples:
            filepath = os.path.join(self.data_dir, f'{letter}_{self.person_name}.json')
            with open(filepath, 'w') as f:
                json.dump(all_samples, f)
            print(f'\n✅ Saved {len(all_samples)} samples for {letter}')
        
        return len(all_samples)

if __name__ == '__main__':
    collector = DynamicDataCollector()
    
    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    
    print("\n" + "="*60)
    print("DYNAMIC LETTER RECORDING - Averaging Approach")
    print("="*60)
    print("\nMotions:")
    print("  H: Move hand left-center-right smoothly")
    print("  J: Draw J-shape downward")
    print("  U: Draw U-shape with two fingers")
    print("  X: Hook and unhook finger")
    print("  Z: Draw Z-shape in air")
    print("="*60)
    
    for letter in dynamic_letters:
        collector.collect_dynamic_letter(letter)
        print('\n' + '-'*60)
    
    print('\n' + '='*60)
    print('✅ All dynamic letters recorded!')
    print('='*60)