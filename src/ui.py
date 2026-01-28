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
    
    # Show reference videos/instructions
    st.info('📹 Reference videos coming soon')
    
    # Show which type of letter
    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    if letter in dynamic_letters:
        st.warning(f'⚡ {letter} is a DYNAMIC letter - involves movement')
    else:
        st.success(f'✋ {letter} is a STATIC letter - hold the position')

def show_practice_mode(mirror_mode):
    st.header('🎯 Practice Mode')
    st.write('Show signs to the camera and get real-time feedback!')
    
    # Start/Stop control
    run = st.checkbox('▶️ Start Camera', value=False)
    
    if run:
        # Initialize detector and classifier
        detector = HandDetector()
        classifier = LetterClassifier()
        
        # Create placeholders
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write('### 📹 Camera Feed')
            frame_placeholder = st.empty()
        
        with col2:
            st.write('### 🎯 Detection Results')
            result_placeholder = st.empty()
            confidence_placeholder = st.empty()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera! Check if camera is connected.")
            return
        
        # Process video
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Detect hand
            landmarks, annotated_frame = detector.find_hands(frame, mirror_for_left_hand=mirror_mode)
            
            # Classify if hand detected
            if landmarks:
                letter, confidence = classifier.classify_letter(landmarks)
                
                if letter and confidence > 0.3:
                    # Show detected letter
                    if confidence > 0.7:
                        result_placeholder.success(f"# ✅ {letter}")
                    elif confidence > 0.5:
                        result_placeholder.warning(f"# ⚠️ {letter}")
                    else:
                        result_placeholder.info(f"# 🤔 {letter}?")
                    
                    # Show confidence bar
                    confidence_placeholder.progress(float(confidence))
                    confidence_placeholder.write(f"Confidence: {confidence*100:.0f}%")
                else:
                    result_placeholder.info("👋 Show a clear sign")
            else:
                result_placeholder.warning("🖐️ No hand detected")
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, width=640)
            
            # Small delay
            cv2.waitKey(1)
        
        cap.release()
    else:
        st.info('👆 Check the box above to start the camera')
        
        # Show instructions
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
    
    # Initialize detector in session state
    if 'recording_detector' not in st.session_state:
        st.session_state.recording_detector = HandDetector()
    
    detector = st.session_state.recording_detector
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        person_name = st.text_input('Your name:', value='', placeholder='Enter your name')
    
    with col2:
        letter_to_record = st.selectbox('Letter to record:', 
                                        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    
    # Check if dynamic letter
    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    is_dynamic = letter_to_record in dynamic_letters
    
    if is_dynamic:
        st.warning(f'⚡ {letter_to_record} is a DYNAMIC letter - perform the motion when recording')
    else:
        st.info(f'✋ {letter_to_record} is a STATIC letter - hold the position steady')
    
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
    
    st.info(f'📊 Current samples for **{letter_to_record}_{person_name}**: {len(existing_samples)}')
    
    # Recording button
    if st.button('🔴 Record Sample', type='primary'):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera!")
            return
        
        # Placeholders
        camera_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        status_placeholder.warning('📹 Recording in progress...')
        
        collected_frames = []
        max_frames = 30 if not is_dynamic else 60
        
        # Warm up camera
        for _ in range(5):
            cap.read()
        
        # Collect frames
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, annotated_frame = detector.find_hands(frame)
            
            if landmarks:
                collected_frames.append(landmarks)
            
            # Show live feed
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Add text overlay
            cv2.putText(frame_rgb, f'Recording {letter_to_record}...', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame_rgb, f'Frame {i+1}/{max_frames}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            camera_placeholder.image(frame_rgb, width=640)
            
            # Update progress
            progress = (i + 1) / max_frames
            progress_placeholder.progress(progress)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Process frames
        if len(collected_frames) >= 10:
            if is_dynamic:
                # Average for dynamic letters
                new_sample = np.mean(collected_frames, axis=0).tolist()
                status_placeholder.success(f'✅ Averaged {len(collected_frames)} frames!')
            else:
                # Use median for static letters
                median_idx = len(collected_frames) // 2
                new_sample = collected_frames[median_idx]
                status_placeholder.success(f'✅ Captured sample from {len(collected_frames)} frames!')
            
            # Add to existing
            existing_samples.append(new_sample)
            
            # Save
            with open(filepath, 'w') as f:
                json.dump(existing_samples, f)
            
            st.success(f'💾 Saved! Total samples: **{len(existing_samples)}**')
            
            # Suggestions
            if len(existing_samples) < 5:
                st.info(f'💡 Tip: Record {5 - len(existing_samples)} more samples for better accuracy')
            elif len(existing_samples) == 5:
                st.balloons()
                st.success('🎉 Perfect! 5 samples recorded!')
            else:
                st.success(f'🌟 Excellent! {len(existing_samples)} samples recorded!')
            
            # Clear camera placeholder
            camera_placeholder.empty()
            progress_placeholder.empty()
            
        else:
            status_placeholder.error(f'❌ Only captured {len(collected_frames)} frames with hand visible')
            st.error('Please make sure your hand is clearly visible in the camera')
    
    # Show all recorded data
    st.write('---')
    if st.button('📊 View All Training Data'):
        try:
            classifier = LetterClassifier()
            
            st.write('### All Training Data:')
            
            col1, col2, col3 = st.columns(3)
            letters = sorted(classifier.reference_data.keys())
            
            for i, letter in enumerate(letters):
                samples = len(classifier.reference_data[letter])
                
                with [col1, col2, col3][i % 3]:
                    if samples >= 15:
                        st.success(f'{letter}: {samples} ✅')
                    elif samples >= 10:
                        st.warning(f'{letter}: {samples} ⚠️')
                    else:
                        st.error(f'{letter}: {samples} ❌')
            
            total = sum(len(s) for s in classifier.reference_data.values())
            st.metric('Total Samples', total)
            
        except Exception as e:
            st.error(f'Error: {e}')

if __name__ == '__main__':
    main()