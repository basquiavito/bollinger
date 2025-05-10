# app.py
import streamlit as st

# ← MUST be the first Streamlit command in the file
st.set_page_config(page_title="Bollinger Bands Viewer", layout="wide")

import bollinger  # now safe to import your other modules

def main():
    bollinger.main()

if __name__ == "__main__":
    main()
