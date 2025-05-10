# app.py
import streamlit as st
import bollinger  # This assumes bollinger.py is in the same directory

def main():
    st.set_page_config(page_title="Bollinger Bands Viewer", layout="wide")
    bollinger.main()

if __name__ == "__main__":
    main()
