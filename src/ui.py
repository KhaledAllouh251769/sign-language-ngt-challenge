"""
Streamlit UI
Author: Person 3
"""

import streamlit as st

def main():
    st.set_page_config(page_title='NGT Recognition', page_icon='??')
    st.title('?? Dutch Sign Language Recognition')
    st.sidebar.header('Settings')
    mode = st.sidebar.radio('Mode', ['Learn', 'Practice', 'Record'])
    if mode == 'Learn':
        st.header('?? Learn Mode')
        letter = st.selectbox('Letter', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        st.info('Video coming soon')
    elif mode == 'Practice':
        st.header('?? Practice Mode')
        if st.button('Start Camera'):
            st.success('Camera started')
    else:
        st.header('?? Record Data')
        letter = st.text_input('Letter')
        if st.button('Record'):
            st.info(f'Recording {letter}')

if __name__ == '__main__':
    main()
