import streamlit as st
import pandas as pd
import os
import yfinance as yf
from datetime import datetime

GEX_FILE = "gex_levels.csv"

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


if os.path.exists(GEX_FILE):
    gex_df = pd.read_csv(GEX_FILE)
else:
    gex_df = pd.DataFrame(columns=["Ticker", "GEX Ceiling", "GEX Floor"])

with st.expander("📡 GEX Radar"):
    st.write("Premarket GEX Levels")

    selected_tickers = st.multiselect(
        "Select Tickers",
        TICKERS,
        default=gex_df["Ticker"].tolist()  # ✅ remembers last saved tickers
    )

    radar_data = []
    for ticker in selected_tickers:
        # Get saved values if they exist
        saved_ceiling = gex_df[gex_df["Ticker"] == ticker]["GEX Ceiling"].values
        saved_floor = gex_df[gex_df["Ticker"] == ticker]["GEX Floor"].values

        gex_ceiling = st.number_input(
            f"{ticker} GEX Ceiling", 
            step=0.1, 
            value=float(saved_ceiling[0]) if len(saved_ceiling) else 0.0, 
            key=f"{ticker}_ceiling"
        )
        gex_floor = st.number_input(
            f"{ticker} GEX Floor", 
            step=0.1, 
            value=float(saved_floor[0]) if len(saved_floor) else 0.0, 
            key=f"{ticker}_floor"
        )

        # --- Update GEX table ---
        new_levels = pd.DataFrame(
            [[ticker, gex_ceiling, gex_floor]],
            columns=["Ticker", "GEX Ceiling", "GEX Floor"]
        )
        gex_df = gex_df[gex_df["Ticker"] != ticker]  # remove old row if exists
        gex_df = pd.concat([gex_df, new_levels], ignore_index=True)
        gex_df.to_csv(GEX_FILE, index=False)

        # Get last price
        try:
            last_price = yf.Ticker(ticker).history(period="1d", interval="1m")["Close"].iloc[-1]
        except:
            last_price = None

        # Condition highlight
        status = "⚪ Between"
        if last_price and gex_ceiling and last_price > gex_ceiling:
            status = "🟢 Above Ceiling"
        elif last_price and gex_floor and last_price < gex_floor:
            status = "🔴 Below Floor"

        radar_data.append([ticker, gex_ceiling, gex_floor, last_price, status])

    # Build table
    if radar_data:
        df = pd.DataFrame(radar_data, columns=["Ticker", "GEX Ceiling", "GEX Floor", "Last Price", "Status"])
        st.dataframe(df, hide_index=True)
 # ✅ Add timestamp
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# --- Reset Ledger button ---
if st.button("🗑 Reset Ledger (Clear Trades)"):
    ledger = pd.DataFrame(columns=["Date", "Ticker", "Entry","Time", "Delta","Entry_Level",
                                   "Enhancer", "Ear", "Nose","Type", "StopLoss", "PnL", "Notes"])
    ledger.to_csv(LEDGER_FILE, index=False)
    st.warning("Ledger reset. GEX levels remain untouched ✅")



# --- Download ledger ---
csv = ledger.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Ledger (CSV)",
    data=csv,
    file_name="trading_ledger.csv",
    mime="text/csv",
)
