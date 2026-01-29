"""
Streamlit UI - Complete Integration
Author: Person 3
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import json
import os
import time
sys.path.append('src')
from detection import HandDetector
from classification import LetterClassifier

def main():
    st.set_page_config(page_title='NGT Recognition', page_icon='🤟')
    st.title('🤟 Dutch Sign Language Recognition')
    
    # Sidebar settings
    st.sidebar.header('Settings')
    mode = st.sidebar.radio('Mode', ['Learn', 'Practice', 'Record'])
    mirror_mode = st.sidebar.checkbox('Left-handed (Mirror Mode)', value=False)
    
    if mode == 'Learn':
        show_learn_mode()
    elif mode == 'Practice':
        show_practice_mode(mirror_mode)
    else:
        show_record_mode()

def show_learn_mode():
    st.header('📚 Learn Mode')
    st.write('Select a letter to see how to sign it')
    
    letter = st.selectbox('Choose a letter:', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    
    st.write(f'### Letter: {letter}')
    
    # Show letter type
    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    if letter in dynamic_letters:
        st.warning(f'⚡ {letter} is a DYNAMIC letter - involves movement')
    else:
        st.success(f'✋ {letter} is a STATIC letter - hold the position')
    
    st.write('---')
    
    # Show YOUR video for this letter
    video_path = f'data/videos/{letter}.mov'
    
    if os.path.exists(video_path):
        st.write(f'### 🎥 How to Sign {letter}:')
        
        # Read and display .mov video file
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        video_file.close()
        
    else:
        st.warning(f'⚠️ Video for {letter} not found')
        st.info(f'Expected location: {video_path}')
        
        # Fallback to YouTube
        st.write('### 📹 General Tutorial:')
        st.video('https://youtu.be/C3n_B5UGBKs')

def show_practice_mode(mirror_mode):
    st.header('🎯 Practice Mode')
    st.write('Show signs to the camera and get real-time feedback!')
    
    run = st.checkbox('▶️ Start Camera', value=False)
    
    if run:
        detector = HandDetector()
        classifier = LetterClassifier()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write('### 📹 Camera Feed')
            frame_placeholder = st.empty()
        
        with col2:
            st.write('### 🎯 Detection Results')
            result_placeholder = st.empty()
            confidence_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera!")
            return
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            landmarks, annotated_frame = detector.find_hands(frame, mirror_for_left_hand=mirror_mode)
            
            if landmarks:
                letter, confidence = classifier.classify_letter(landmarks)
                
                if letter and confidence > 0.3:
                    if confidence > 0.7:
                        result_placeholder.success(f"# ✅ {letter}")
                    elif confidence > 0.5:
                        result_placeholder.warning(f"# ⚠️ {letter}")
                    else:
                        result_placeholder.info(f"# 🤔 {letter}?")
                    
                    confidence_placeholder.progress(float(confidence))
                    confidence_placeholder.write(f"Confidence: {confidence*100:.0f}%")
                else:
                    result_placeholder.info("👋 Show a clear sign")
            else:
                result_placeholder.warning("🖐️ No hand detected")
            
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, width=640)
            cv2.waitKey(1)
        
        cap.release()
    else:
        st.info('👆 Check the box above to start the camera')
        
        st.write('### 📖 Instructions:')
        st.write('1. Check "Start Camera" box')
        st.write('2. Make a sign from the NGT alphabet')
        st.write('3. See real-time detection results')
        st.write('4. Uncheck box to stop')
        
        if mirror_mode:
            st.success('✅ Mirror mode is ON - use your LEFT hand')
        else:
            st.info('ℹ️ Mirror mode is OFF - use your RIGHT hand')

def show_record_mode():
    st.header('💾 Record Training Data')
    st.write('Add your own training samples to improve the system!')
    
    # Initialize detector
    if 'recording_detector' not in st.session_state:
        st.session_state.recording_detector = HandDetector()
    
    detector = st.session_state.recording_detector
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        person_name = st.text_input('Your name:', placeholder='Enter your name')
    
    with col2:
        letter_to_record = st.selectbox('Letter to record:', 
                                        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    
    # Check if dynamic
    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    is_dynamic = letter_to_record in dynamic_letters
    
    if is_dynamic:
        st.warning(f'⚡ {letter_to_record} is DYNAMIC - perform the motion')
    else:
        st.info(f'✋ {letter_to_record} is STATIC - hold steady')
    
    st.write('---')
    
    if not person_name:
        st.warning('⚠️ Please enter your name first')
        return
    
    # Check existing samples
    data_dir = 'data/reference'
    os.makedirs(data_dir, exist_ok=True)
    filename = f'{letter_to_record}_{person_name}.json'
    filepath = os.path.join(data_dir, filename)
    
    existing_samples = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing_samples = json.load(f)
    
    st.metric('Samples for this letter', len(existing_samples))
    
    # Recording process
    st.write(f'### Ready to Record: **{letter_to_record}**')
    
    if st.button('🎬 Record New Sample', type='primary'):
        # Create placeholders
        camera_placeholder = st.empty()
        countdown_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera!")
            return
        
        # Warm up camera
        for _ in range(5):
            cap.read()
        
        # Show preview with countdown
        countdown_placeholder.warning('📹 Camera opened - Get ready!')
        
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                landmarks, annotated_frame = detector.find_hands(frame)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Add countdown text
                cv2.putText(frame_rgb, f'Starting in {countdown}...', (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                
                camera_placeholder.image(frame_rgb, width=640)
            time.sleep(0.8)
        
        countdown_placeholder.empty()
        status_placeholder.warning('🔴 RECORDING NOW - Make the sign!')
        
        # Record frames
        collected_frames = []
        max_frames = 30 if not is_dynamic else 60
        
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, annotated_frame = detector.find_hands(frame)
            
            if landmarks:
                collected_frames.append(landmarks)
            
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Add recording text
            cv2.putText(frame_rgb, '🔴 RECORDING', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame_rgb, f'{len(collected_frames)} frames captured', (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            camera_placeholder.image(frame_rgb, width=640)
            
            progress = (i + 1) / max_frames
            progress_placeholder.progress(progress)
            
            cv2.waitKey(1)
        
        # CLOSE CAMERA IMMEDIATELY
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.2)
        
        # Process frames
        camera_placeholder.empty()
        progress_placeholder.empty()
        
        if len(collected_frames) >= 10:
            if is_dynamic:
                new_sample = np.mean(collected_frames, axis=0).tolist()
                status_placeholder.success(f'✅ Averaged {len(collected_frames)} frames!')
            else:
                median_idx = len(collected_frames) // 2
                new_sample = collected_frames[median_idx]
                status_placeholder.success(f'✅ Captured from {len(collected_frames)} frames!')
            
            # Save
            existing_samples.append(new_sample)
            with open(filepath, 'w') as f:
                json.dump(existing_samples, f)
            
            st.success(f'💾 SAVED! Total samples: **{len(existing_samples)}**')
            
            if len(existing_samples) < 5:
                st.info(f'💡 Record {5 - len(existing_samples)} more samples')
            elif len(existing_samples) == 5:
                st.balloons()
                st.success('🎉 Perfect! 5 samples complete!')
            else:
                st.success(f'🌟 Excellent! {len(existing_samples)} samples!')
            
        else:
            status_placeholder.error(f'❌ Only {len(collected_frames)} frames captured')
            st.error('Hand not visible enough - try again!')
    
    # Instructions
    st.write('---')
    st.write('### 📝 Instructions:')
    st.write('1. Enter your name and select letter')
    st.write('2. Click "Record New Sample"')
    st.write('3. Camera opens with 3-second countdown')
    st.write('4. Make the sign clearly')
    st.write('5. Recording lasts 1-2 seconds')
    st.write('6. Camera closes automatically')
    st.write('7. Sample is saved')
    st.write('8. Click again to add more samples')

if __name__ == '__main__':
    main()