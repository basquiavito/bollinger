import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.header("📘 Trade Logger")

# Input form
with st.form("log_form"):
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", placeholder="e.g. AAPL")
        direction = st.selectbox("Direction", ["Call", "Put"])
        result = st.selectbox("Result", ["Win", "Loss", "Breakeven"])
    with col2:
        entry = st.number_input("Entry Price", min_value=0.0, step=0.01)
        exit = st.number_input("Exit Price", min_value=0.0, step=0.01)
    
    notes = st.text_area("Notes / Setup / Angle / Reason")
    submitted = st.form_submit_button("💾 Log Trade")

# CSV log file path
csv_path = "trades_log.csv"

# Save the entry
if submitted:
    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker.upper(),
        "Direction": direction,
        "Entry": entry,
        "Exit": exit,
        "Result": result,
        "Notes": notes
    }

    df_entry = pd.DataFrame([log_entry])
    
    if os.path.exists(csv_path):
        df_entry.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(csv_path, index=False)

    st.success("Trade logged successfully!")

# Show last 5 trades
if os.path.exists(csv_path):
    st.subheader("📊 Recent Trades")
    df_log = pd.read_csv(csv_path)
    st.dataframe(df_log.tail(5))
