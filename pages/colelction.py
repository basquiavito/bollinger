import streamlit as st
import pandas as pd
import os
from datetime import datetime

OUTCOME_FILE = "trading_outcomes.csv"

st.title("📑 Trade Outcomes Tracker")

# --- Load outcomes if exists ---
if os.path.exists(OUTCOME_FILE):
    outcomes = pd.read_csv(OUTCOME_FILE)
else:
    outcomes = pd.DataFrame(columns=[
        "Date", "Ticker", 
        "Entry 1 Time", "Entry 1 Price", "Entry 1 Prototype",
        "Entry 2 Time", "Kijun Cross Type",
        "Entry 3 Time", "IB Line Cross Type",
        "Exit Price", "Change", "Total P&L",
        "Mirror 1 Time", "Mirror 1 Price", "Mirror 1 Prototype",
        "Mirror 2 Time", "Mirror 2 Kijun Cross Type",
        "Mirror 3 Time", "Mirror 3 IB Line Cross",
        "Mirror Exit Price", "Mirror Change", "Mirror P&L",
        "Notes"
    ])

# --- Upload to restore ---
uploaded = st.file_uploader("📤 Upload existing outcomes (CSV)", type="csv")
if uploaded is not None:
    outcomes = pd.read_csv(uploaded)
    outcomes.to_csv(OUTCOME_FILE, index=False)
    st.success("Outcomes restored from uploaded file ✅")

# --- Form to add new outcome ---
with st.form("outcome_entry"):
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date")
        ticker = st.text_input("Ticker (e.g. TSLA)").upper()

        e1_time = st.text_input("Entry 1 Time (e.g. 11:55 AM)")
        e1_price = st.number_input("Entry 1 Price", min_value=0.0, step=0.01)
        e1_proto = st.text_input("Entry 1 Prototype")

        e2_time = st.text_input("Entry 2 Time")
        kijun_type = st.text_input("Kijun Cross Type")

        e3_time = st.text_input("Entry 3 Time")
        ib_type = st.text_input("IB Line Cross Type")

        exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01)

    with col2:
        mirror1_time = st.text_input("Mirror 1 Time")
        mirror1_price = st.number_input("Mirror 1 Price", min_value=0.0, step=0.01)
        mirror1_proto = st.text_input("Mirror 1 Prototype")

        mirror2_time = st.text_input("Mirror 2 Time")
        mirror2_kijun = st.text_input("Mirror 2 Kijun Cross Type")

        mirror3_time = st.text_input("Mirror 3 Time")
        mirror3_ib = st.text_input("Mirror 3 IB Line Cross")

        mirror_exit = st.number_input("Mirror Exit Price", min_value=0.0, step=0.01)

        notes = st.text_area("Notes")

    submitted = st.form_submit_button("➕ Add Outcome")

if submitted:
    # Calculate changes and P&L
    change = exit_price - e1_price if e1_price > 0 else 0
    mirror_change = mirror_exit - mirror1_price if mirror1_price > 0 else 0
    total_pnl = change + mirror_change

    new_row = {
        "Date": date,
        "Ticker": ticker,
        "Entry 1 Time": e1_time, "Entry 1 Price": e1_price, "Entry 1 Prototype": e1_proto,
        "Entry 2 Time": e2_time, "Kijun Cross Type": kijun_type,
        "Entry 3 Time": e3_time, "IB Line Cross Type": ib_type,
        "Exit Price": exit_price, "Change": change,
        "Total P&L": total_pnl,
        "Mirror 1 Time": mirror1_time, "Mirror 1 Price": mirror1_price, "Mirror 1 Prototype": mirror1_proto,
        "Mirror 2 Time": mirror2_time, "Mirror 2 Kijun Cross Type": mirror2_kijun,
        "Mirror 3 Time": mirror3_time, "Mirror 3 IB Line Cross": mirror3_ib,
        "Mirror Exit Price": mirror_exit, "Mirror Change": mirror_change, "Mirror P&L": mirror_change,
        "Notes": notes
    }

    outcomes = pd.concat([outcomes, pd.DataFrame([new_row])], ignore_index=True)
    outcomes.to_csv(OUTCOME_FILE, index=False)
    st.success(f"Added outcome for {ticker} ✅")

# --- Show outcomes ---
st.subheader("📊 Recorded Outcomes")
st.dataframe(outcomes, use_container_width=True)

# --- Editable table ---
st.subheader("✏️ Edit Outcomes")
edited_df = st.data_editor(outcomes, num_rows="dynamic", use_container_width=True)

if st.button("💾 Save Edits"):
    edited_df.to_csv(OUTCOME_FILE, index=False)
    outcomes = edited_df
    st.success("Changes saved ✅")

# --- Delete row by index ---
delete_index = st.number_input("Row index to delete", min_value=0, max_value=len(outcomes)-1, step=1)
if st.button("🗑 Delete Row"):
    outcomes = outcomes.drop(delete_index).reset_index(drop=True)
    outcomes.to_csv(OUTCOME_FILE, index=False)
    st.warning(f"Row {delete_index} deleted ✅")

# --- Reset button ---
if st.button("🗑 Reset Outcomes"):
    outcomes = pd.DataFrame(columns=outcomes.columns)
    outcomes.to_csv(OUTCOME_FILE, index=False)
    st.warning("Outcomes cleared. File reset ✅")

# --- Download outcomes ---
csv = outcomes.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Outcomes (CSV)",
    data=csv,
    file_name="trading_outcomes.csv",
    mime="text/csv",
)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
