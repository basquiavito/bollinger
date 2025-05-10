# app.py
import streamlit as st

# ── THIS MUST BE THE FIRST Streamlit COMMAND IN THE FILE ──
st.set_page_config(page_title="Bollinger Bands Viewer", layout="wide")

import bollinger  # now safe to import your module

def main():
    bollinger.main()

if __name__ == "__main__":
    main()
