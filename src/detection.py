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
    
    def find_hands(self, frame):
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
        return landmarks, annotated_frame

def test_detector():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    print('Press Q to quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks, annotated_frame = detector.find_hands(frame)
        if landmarks:
            cv2.putText(annotated_frame, 'Hand Detected!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_detector()
