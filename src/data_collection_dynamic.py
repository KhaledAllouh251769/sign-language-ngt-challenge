"""
Hand detection using MediaPipe
Author: Person 1
"""

import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def find_hands(self, frame, mirror_for_left_hand=False):
        """
        Detect hands and return landmarks
        
        Args:
            frame: Input camera frame
            mirror_for_left_hand: If True, mirrors landmarks for left hand
        
        Returns:
            landmarks: List of 21 hand landmarks or None
            annotated_frame: Frame with hand drawn on it
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        landmarks = None
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            # Mirror for left hand if requested
            if mirror_for_left_hand:
                for i in range(len(landmarks)):
                    landmarks[i][0] = 1.0 - landmarks[i][0]
        
        return landmarks, annotated_frame

def test_detector():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    print('Press Q to quit')
    print('Press M to toggle mirroring (for left hand)')
    
    mirror_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, annotated_frame = detector.find_hands(frame, mirror_for_left_hand=mirror_mode)
        
        if landmarks:
            cv2.putText(annotated_frame, 'Hand Detected!', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show mirror status
        mirror_text = "Mirror: ON (Left Hand)" if mirror_mode else "Mirror: OFF (Right Hand)"
        cv2.putText(annotated_frame, mirror_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Hand Detection', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirror_mode = not mirror_mode
            print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_detector()