# app.py
import streamlit as st

# This must be the first Streamlit command
st.set_page_config(page_title="Bollinger Bands Viewer", layout="wide")

import bollinger  # your bollinger.py script

def main():
    try:
        bollinger.main()
    except Exception as e:
        st.error(f"⚠️ Error in `bollinger.main()`: {e}")

if __name__ == "__main__":
    main()
