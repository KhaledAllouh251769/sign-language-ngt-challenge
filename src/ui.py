"""
Streamlit UI - Complete Integration
Author: Person 3
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
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
            
            # Small delay to prevent freezing
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
    st.write('Help improve the system by adding more training samples!')
    
    st.warning('⚠️ Feature coming soon - use data_collection.py script for now')
    
    st.write('### How to add training data:')
    st.code('''
# In Anaconda Prompt:
conda activate signlang
python src/data_collection.py
    ''', language='bash')
    
    # Show current data stats
    if st.button('📊 Show Current Training Data'):
        try:
            classifier = LetterClassifier()
            
            st.write('### Current Training Data:')
            
            col1, col2, col3 = st.columns(3)
            letters = sorted(classifier.reference_data.keys())
            
            for i, letter in enumerate(letters):
                samples = len(classifier.reference_data[letter])
                
                with [col1, col2, col3][i % 3]:
                    if samples >= 15:
                        st.success(f'{letter}: {samples} samples ✅')
                    elif samples >= 10:
                        st.warning(f'{letter}: {samples} samples ⚠️')
                    else:
                        st.error(f'{letter}: {samples} samples ❌')
            
            total_samples = sum(len(samples) for samples in classifier.reference_data.values())
            st.info(f'**Total samples: {total_samples}**')
            
        except Exception as e:
            st.error(f'Error loading data: {e}')

if __name__ == '__main__':
    main()