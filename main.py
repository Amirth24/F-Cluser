import streamlit as st
from core import process_image

st.set_page_config(
    layout='wide',
    page_icon='👁️'
)


uploaded_files = None
def run_process():
    if len(uploaded_files) <= 1:
        st.error('Upload more than one file to process.', icon='⚠️')
    else:
        process_image(uploaded_files)
        

uploaded_files = st.sidebar.file_uploader('Upload Images', accept_multiple_files=True, type=["png","jpg","jpeg"])

process_btn = st.sidebar.button('Process', on_click=run_process)
