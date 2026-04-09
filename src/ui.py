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

# ── Theme definitions ─────────────────────────────────────────────────────────
THEMES = {
    'Default': {
        'primary':       '#534AB7',
        'secondary':     '#7F77DD',
        'bg':            '#EEEDFE',
        'border':        '#7F77DD',
        'text':          '#534AB7',
        'label_bg':      '#EEEDFE',
        'label_text':    '#3C3489',
        'bar_color':     (83, 74, 183),
        'cv_text':       (83, 74, 183),
        'cv_bar':        (83, 74, 183),
    },
    'High Contrast': {
        'primary':       '#FFFFFF',
        'secondary':     '#FFFF00',
        'bg':            '#000000',
        'border':        '#FFFFFF',
        'text':          '#FFFFFF',
        'label_bg':      '#000000',
        'label_text':    '#FFFF00',
        'bar_color':     (255, 255, 0),
        'cv_text':       (255, 255, 0),
        'cv_bar':        (255, 255, 0),
    },
    'Deuteranopia': {
        'primary':       '#004488',
        'secondary':     '#0077BB',
        'bg':            '#DDEEFF',
        'border':        '#0077BB',
        'text':          '#004488',
        'label_bg':      '#DDEEFF',
        'label_text':    '#003366',
        'bar_color':     (0, 119, 187),
        'cv_text':       (0, 68, 136),
        'cv_bar':        (0, 119, 187),
    },
    'Protanopia': {
        'primary':       '#7A5000',
        'secondary':     '#F5A500',
        'bg':            '#FFF3CD',
        'border':        '#F5A500',
        'text':          '#7A5000',
        'label_bg':      '#FFF3CD',
        'label_text':    '#5C3A00',
        'bar_color':     (245, 165, 0),
        'cv_text':       (122, 80, 0),
        'cv_bar':        (245, 165, 0),
    },
}

def get_theme():
    return st.session_state.get('theme', THEMES['Default'])

def get_font():
    return st.session_state.get('font_size', 16)

def styled_header(text, font_size=None):
    """Render a header that respects the current font size setting."""
    t  = get_theme()
    fs = font_size or (get_font() + 8)
    st.markdown(
        f'<h2 style="color:{t["primary"]};font-size:{fs}px;font-weight:600;margin-bottom:0.5rem">{text}</h2>',
        unsafe_allow_html=True,
    )

def styled_text(text, font_size=None, color=None):
    """Render body text that respects font size and theme."""
    t  = get_theme()
    fs = font_size or get_font()
    c  = color or t['primary']
    st.markdown(
        f'<p style="color:{c};font-size:{fs}px;margin:4px 0">{text}</p>',
        unsafe_allow_html=True,
    )

def styled_box(content, font_size=None):
    """Themed box — used for the sentence display."""
    t  = get_theme()
    fs = font_size or (get_font() + 16)
    st.markdown(
        f"""
        <div style="
            background:{t['bg']};
            border:3px solid {t['border']};
            border-radius:10px;
            padding:16px 20px;
            min-height:64px;
            font-size:{fs}px;
            font-weight:600;
            letter-spacing:6px;
            color:{t['text']};
            font-family:monospace;
            word-break:break-all;
        ">{content}</div>
        """,
        unsafe_allow_html=True,
    )

def styled_label(text, font_size=None):
    """Themed pill label."""
    t  = get_theme()
    fs = font_size or (get_font() - 2)
    st.markdown(
        f"""
        <span style="
            background:{t['label_bg']};
            color:{t['label_text']};
            border:2px solid {t['border']};
            border-radius:6px;
            padding:4px 10px;
            font-size:{fs}px;
            font-weight:600;
        ">{text}</span>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title='NGT Recognition', page_icon='🤟')

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header('Settings')
    mode        = st.sidebar.radio('Mode', ['Learn', 'Practice', 'Record', 'Sentence'])
    mirror_mode = st.sidebar.checkbox('Left-handed (Mirror Mode)', value=False)

    st.sidebar.markdown('---')
    st.sidebar.header('♿ Accessibility')

    theme_name = st.sidebar.selectbox(
        'Color theme',
        list(THEMES.keys()),
        help='Choose a color-blind friendly or high-contrast theme',
    )
    font_size = st.sidebar.slider(
        'Font size',
        min_value=12, max_value=28, value=16, step=2,
        help='Makes text bigger or smaller throughout the app',
    )

    # Store in session state so helper functions can read them
    st.session_state['theme']     = THEMES[theme_name]
    st.session_state['font_size'] = font_size

    # Show a live preview of the theme in the sidebar
    t = THEMES[theme_name]
    st.sidebar.markdown(
        f"""
        <div style="
            background:{t['bg']};
            border:2px solid {t['border']};
            border-radius:8px;
            padding:8px 12px;
            margin-top:8px;
        ">
            <span style="color:{t['text']};font-size:{font_size}px;font-weight:600">
                Preview: Aa Bb Cc
            </span><br>
            <span style="color:{t['label_text']};font-size:{max(font_size-4,10)}px">
                Theme: {theme_name}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page title — themed + sized ───────────────────────────────────────────
    st.markdown(
        f'<h1 style="color:{t["primary"]};font-size:{font_size+14}px">🤟 Dutch Sign Language Recognition</h1>',
        unsafe_allow_html=True,
    )

    # ── Route ─────────────────────────────────────────────────────────────────
    if mode == 'Learn':
        show_learn_mode()
    elif mode == 'Practice':
        show_practice_mode(mirror_mode)
    elif mode == 'Sentence':
        show_sentence_mode(mirror_mode)
    else:
        show_record_mode()


# ─────────────────────────────────────────────────────────────────────────────
# LEARN MODE
# ─────────────────────────────────────────────────────────────────────────────
def show_learn_mode():
    fs = get_font()
    styled_header('📚 Learn Mode')
    styled_text('Select a letter to see how to sign it', font_size=fs)

    letter = st.selectbox('Choose a letter:', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    styled_label(f'Letter: {letter}', font_size=fs + 4)
    st.write('')

    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    if letter in dynamic_letters:
        st.warning(f'⚡ {letter} is a DYNAMIC letter - involves movement')
    else:
        st.success(f'✋ {letter} is a STATIC letter - hold the position')

    st.write('---')

    video_path = f'data/videos/{letter}.mov'
    if os.path.exists(video_path):
        styled_text(f'🎥 How to Sign {letter}:', font_size=fs + 2)
        video_file  = open(video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        video_file.close()
    else:
        st.warning(f'⚠️ Video for {letter} not found')
        st.info(f'Expected location: {video_path}')
        styled_text('📹 General Tutorial:', font_size=fs + 2)
        st.video('https://youtu.be/C3n_B5UGBKs')


# ─────────────────────────────────────────────────────────────────────────────
# PRACTICE MODE
# ─────────────────────────────────────────────────────────────────────────────
def show_practice_mode(mirror_mode):
    fs = get_font()
    styled_header('🎯 Practice Mode')
    styled_text('Show signs to the camera and get real-time feedback!', font_size=fs)

    run = st.checkbox('▶️ Start Camera', value=False)

    if run:
        detector   = HandDetector()
        classifier = LetterClassifier()

        col1, col2 = st.columns([2, 1])
        with col1:
            styled_text('📹 Camera Feed', font_size=fs + 2)
            frame_placeholder = st.empty()
        with col2:
            styled_text('🎯 Detection', font_size=fs + 2)
            result_placeholder     = st.empty()
            confidence_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('❌ Cannot open camera!')
            return

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error('Failed to read from camera')
                break

            landmarks, annotated_frame = detector.find_hands(frame, mirror_for_left_hand=mirror_mode)

            if landmarks:
                letter, confidence = classifier.classify_letter(landmarks)
                if letter and confidence > 0.3:
                    if confidence > 0.7:
                        result_placeholder.success(f'# ✅ {letter}')
                    elif confidence > 0.5:
                        result_placeholder.warning(f'# ⚠️ {letter}')
                    else:
                        result_placeholder.info(f'# 🤔 {letter}?')
                    confidence_placeholder.progress(float(confidence))
                    confidence_placeholder.write(f'Confidence: {confidence*100:.0f}%')
                else:
                    result_placeholder.info('👋 Show a clear sign')
            else:
                result_placeholder.warning('🖐️ No hand detected')

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, width=640)
            cv2.waitKey(1)

        cap.release()
    else:
        st.info('👆 Check the box above to start the camera')
        for step in [
            '1. Check "Start Camera" box',
            '2. Make a sign from the NGT alphabet',
            '3. See real-time detection results',
            '4. Uncheck box to stop',
        ]:
            styled_text(step, font_size=fs)
        if mirror_mode:
            st.success('✅ Mirror mode is ON - use your LEFT hand')
        else:
            st.info('ℹ️ Mirror mode is OFF - use your RIGHT hand')


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE MODE — keyboard guided
# ─────────────────────────────────────────────────────────────────────────────
def show_sentence_mode(mirror_mode):
    t  = get_theme()
    fs = get_font()

    styled_header('✍️ Sentence Builder')
    styled_text('Click a letter on the keyboard, then sign it to add it to your sentence.', font_size=fs)

    if 'sentence' not in st.session_state:
        st.session_state.sentence = []
    if 'target_letter' not in st.session_state:
        st.session_state.target_letter = None
    if 'last_added_letter' not in st.session_state:
        st.session_state.last_added_letter = ''

    sentence_str = ''.join(st.session_state.sentence)

    st.write('')
    styled_text('📝 Your sentence:', font_size=fs + 2)
    display_content = (
        sentence_str if sentence_str
        else f'<span style="color:#AAA;font-size:{fs}px;font-weight:400;letter-spacing:0">Start signing...</span>'
    )
    styled_box(display_content, font_size=fs + 16)
    st.write('')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('⎵  Space', use_container_width=True):
            st.session_state.sentence.append(' ')
            st.session_state.target_letter = None
            st.rerun()
    with col2:
        if st.button('⌫  Delete', use_container_width=True):
            if st.session_state.sentence:
                st.session_state.sentence.pop()
            st.rerun()
    with col3:
        if st.button('🗑️  Clear', use_container_width=True):
            st.session_state.sentence = []
            st.session_state.target_letter = None
            st.rerun()
    with col4:
        if st.button('📋  Copy', use_container_width=True):
            st.code(sentence_str if sentence_str else '(empty)')

    st.write('---')

    styled_text('Step 1 — Click the letter you want to sign:', font_size=fs + 2)
    st.write('')

    ROWS = [list('QWERTYUIOP'), list('ASDFGHJKL'), list('ZXCVBNM')]
    for row in ROWS:
        cols = st.columns(len(row))
        for col, letter in zip(cols, row):
            with col:
                btn_label = f'**{letter}**' if letter == st.session_state.target_letter else letter
                if st.button(btn_label, key=f'key_{letter}', use_container_width=True):
                    st.session_state.target_letter = letter
                    st.rerun()

    st.write('---')

    target = st.session_state.target_letter

    if not target:
        st.info('👆 Click a letter above to start.')
        return

    styled_text('Step 2 — Sign this letter:', font_size=fs + 2)
    st.markdown(
        f'<div style="text-align:center;font-size:{fs+64}px;font-weight:700;color:{t["primary"]};line-height:1.1;margin:0.5rem 0">{target}</div>',
        unsafe_allow_html=True,
    )

    dynamic_letters = ['H', 'J', 'U', 'X', 'Z']
    if target in dynamic_letters:
        st.warning(f'⚡ {target} is dynamic — perform the motion during recording')

    # Camera opens automatically when a letter is selected
    run = True

    detector   = HandDetector()
    classifier = LetterClassifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error('❌ Cannot open camera!')
        return

    cam_col, info_col = st.columns([3, 1])
    with cam_col:
        frame_placeholder = st.empty()
    with info_col:
        styled_text('Detecting:', font_size=fs)
        live_letter_placeholder = st.empty()
        styled_text('Confidence:', font_size=fs)
        live_conf_placeholder   = st.empty()
        styled_text('Hold progress:', font_size=fs)
        progress_placeholder    = st.empty()
        styled_text('Last added:', font_size=fs)
        last_added_placeholder  = st.empty()

    STREAK_NEEDED  = 20
    correct_streak = [0]

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated_frame = detector.find_hands(frame, mirror_for_left_hand=mirror_mode)

        cv2.putText(annotated_frame, f'Sign: {target}',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, t['cv_text'], 3)

        added_this_frame = False

        if landmarks:
            letter, confidence = classifier.classify_letter(landmarks, use_smoothing=False)

            if letter:
                is_correct = letter == target
                if confidence > 0.7:
                    conf_color = '#22a822' if is_correct else '#E24B4A'
                elif confidence > 0.5:
                    conf_color = '#F5A500'
                else:
                    conf_color = '#E24B4A'

                live_letter_placeholder.markdown(
                    f'<p style="font-size:{fs+28}px;font-weight:700;color:{conf_color};margin:0">{letter}</p>',
                    unsafe_allow_html=True,
                )
                live_conf_placeholder.markdown(
                    f'<p style="font-size:{fs+4}px;font-weight:600;color:{conf_color};margin:0">{confidence*100:.0f}%</p>',
                    unsafe_allow_html=True,
                )
                cv2.putText(annotated_frame, f'{letter}  {confidence*100:.0f}%',
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, t['cv_text'], 2)

            if letter == target and confidence > 0.5:
                correct_streak[0] += 1
            else:
                correct_streak[0] = max(0, correct_streak[0] - 2)

            hold_fraction = min(correct_streak[0] / STREAK_NEEDED, 1.0)
            progress_placeholder.progress(hold_fraction)

            bar_w = int(hold_fraction * frame.shape[1])
            cv2.rectangle(annotated_frame,
                          (0, frame.shape[0] - 14),
                          (bar_w, frame.shape[0]),
                          t['cv_bar'], -1)

            if correct_streak[0] >= STREAK_NEEDED:
                st.session_state.sentence.append(target)
                st.session_state.last_added_letter = target
                st.session_state.target_letter     = None
                added_this_frame = True
        else:
            correct_streak[0] = 0
            live_letter_placeholder.markdown(
                f'<p style="font-size:{fs}px;color:#AAA;margin:0">No hand</p>',
                unsafe_allow_html=True,
            )
            live_conf_placeholder.empty()
            progress_placeholder.progress(0.0)
            cv2.putText(annotated_frame, 'Show your hand',
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)

        if st.session_state.last_added_letter:
            last_added_placeholder.markdown(
                f'<p style="font-size:{fs+16}px;font-weight:700;color:{t["primary"]};margin:0">✅ {st.session_state.last_added_letter}</p>',
                unsafe_allow_html=True,
            )

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, width=None, use_container_width=True)

        if added_this_frame:
            cap.release()
            cv2.destroyAllWindows()
            st.rerun()

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# RECORD MODE
# ─────────────────────────────────────────────────────────────────────────────
def show_record_mode():
    fs = get_font()
    styled_header('💾 Record Training Data')
    styled_text('Add your own training samples to improve the system!', font_size=fs)

    if 'recording_detector' not in st.session_state:
        st.session_state.recording_detector = HandDetector()
    detector = st.session_state.recording_detector

    col1, col2 = st.columns(2)
    with col1:
        person_name = st.text_input('Your name:', placeholder='Enter your name')
    with col2:
        letter_to_record = st.selectbox('Letter to record:', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

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

    data_dir = 'data/reference'
    os.makedirs(data_dir, exist_ok=True)
    filename = f'{letter_to_record}_{person_name}.json'
    filepath = os.path.join(data_dir, filename)

    existing_samples = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing_samples = json.load(f)

    st.metric('Samples for this letter', len(existing_samples))
    styled_text(f'Ready to Record: {letter_to_record}', font_size=fs + 4)

    if st.button('🎬 Record New Sample', type='primary'):
        camera_placeholder    = st.empty()
        countdown_placeholder = st.empty()
        progress_placeholder  = st.empty()
        status_placeholder    = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('❌ Cannot open camera!')
            return

        for _ in range(5):
            cap.read()

        countdown_placeholder.warning('📹 Camera opened - Get ready!')

        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                landmarks, annotated_frame = detector.find_hands(frame)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                cv2.putText(frame_rgb, f'Starting in {countdown}...', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                camera_placeholder.image(frame_rgb, width=640)
            time.sleep(0.8)

        countdown_placeholder.empty()
        status_placeholder.warning('🔴 RECORDING NOW - Make the sign!')

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
            cv2.putText(frame_rgb, 'RECORDING', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame_rgb, f'{len(collected_frames)} frames captured', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            camera_placeholder.image(frame_rgb, width=640)
            progress_placeholder.progress((i + 1) / max_frames)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.2)
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

    st.write('---')
    styled_text('Instructions:', font_size=fs + 2)
    for step in [
        '1. Enter your name and select letter',
        '2. Click "Record New Sample"',
        '3. Camera opens with 3-second countdown',
        '4. Make the sign clearly',
        '5. Recording lasts 1-2 seconds',
        '6. Camera closes automatically',
        '7. Sample is saved',
        '8. Click again to add more samples',
    ]:
        styled_text(step, font_size=fs)


if __name__ == '__main__':
    main()