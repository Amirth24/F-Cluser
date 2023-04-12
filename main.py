import streamlit as st





st.set_page_config(
    layout='wide',
    page_icon='👁️'
)




uploaded_files = st.sidebar.file_uploader('Upload Images', accept_multiple_files=True, type=["png","jpg","jpeg"])
