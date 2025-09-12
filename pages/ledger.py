import streamlit as st
import pandas as pd
import os

LEDGER_FILE = "trading_ledger.csv"
# --- Inventory (your tradeable tickers) ---
TICKERS = [
    "NVDA", "AMD", "AVGO", "MRVL", "MU", "SMCI", "QCOM",
    "MSFT", "AMZN", "AAPL", "GOOGL", "UBER", "PLTR", "META", "TSLA",
    "HOOD", "COIN", "C", "WFC", "JPM",
    "SPY", "QQQ"
]

st.title("📒 Trading Ledger")

# --- Load ledger if exists ---
if os.path.exists(LEDGER_FILE):
    ledger = pd.read_csv(LEDGER_FILE)
else:
    ledger = pd.DataFrame(columns=["Date", "Ticker", "Entry","Time", "Delta","Entry_Level","Enhancer",  "Ear", "Nose","Type", "StopLoss", "PnL", "Notes"])


# --- Upload to restore ---
uploaded = st.file_uploader("📤 Upload existing ledger (CSV)", type="csv")
if uploaded is not None:
    ledger = pd.read_csv(uploaded)
    ledger.to_csv(LEDGER_FILE, index=False)
    st.success("Ledger restored from uploaded file.")

# # --- Trade entry form ---
# with st.form("trade_entry"):
#     col1, col2 = st.columns(2)
#     with col1:
#         date = st.date_input("Date")
#         ticker = st.text_input("Ticker (e.g. TSLA)")
#         trade_type = st.selectbox("Type", ["Call", "Put"])
#     with col2:
#         entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01)
#         delta = st.number_input("Delta", min_value=0.0, step=0.01)
#         stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)

#     notes = st.text_area("Notes (optional)")
#     submitted = st.form_submit_button("➕ Add Trade")

# if submitted:
#     new_row = {
#         "Date": date,
#         "Ticker": ticker.upper(),
#         "Entry": entry_price,
#         "Delta": delta,
#         "Type": trade_type,
#         "StopLoss": stop_loss,
#         "Notes": notes
#     }
#     ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
#     ledger.to_csv(LEDGER_FILE, index=False)
#     st.success(f"Added {ticker} trade.")


# --- Trade entry form ---
with st.form("trade_entry"):
    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Date")
        entry_time = st.time_input("Entry Time")   # 👈 New field

        ticker = st.selectbox("Ticker", TICKERS)
        enhancer = st.checkbox("Enhancer present?")
        ear = st.checkbox("👂 Ear (Volume Memory crossed?)")
        nose = st.checkbox("👃 Nose (Time Memory crossed?)")

        trade_type = st.selectbox("Type", ["Call", "Put"])
        entry_level = st.selectbox(
        "Entry Level",
        ["Entry 1", "Entry 2", "Entry 3", "Mirror 1", "Mirror 2", "Mirror 3"]
    )

    with col2:
        entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01)
        delta = st.number_input("Delta", min_value=0.0, step=0.01)
        stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)
    with col3:
        exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01, format="%.2f")
        notes = st.text_area("Notes (optional)", height=50)
 
    submitted = st.form_submit_button("➕ Add Trade")

if submitted:
    pnl = exit_price - entry_price if exit_price > 0 else 0
    new_row = {
        "Date": date,
        "Ticker": ticker.upper(),
        "Time": entry_time.strftime("%H:%M:%S"),
        "Entry": entry_price,
        "Exit": exit_price,
        "Delta": delta,
        "Type": trade_type,
        "Entry_Level": entry_level,   # 👈 new column
        "StopLoss": stop_loss,
        "Enhancer": enhancer,   # 👈 boolean
        "Ear": ear,
        "Nose": nose,

        "PnL": pnl,
        "Notes": notes
    }
    ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
    ledger.to_csv(LEDGER_FILE, index=False)
    st.success(f"Added {ticker} trade.")



# --- Show ledger ---
st.subheader("📊 Current Ledger")
st.dataframe(ledger, use_container_width=True)

# --- Download ledger ---
csv = ledger.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Ledger (CSV)",
    data=csv,
    file_name="trading_ledger.csv",
    mime="text/csv",
)
