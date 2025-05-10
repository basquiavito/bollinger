# daily.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="Daily Candlestick Chart", layout="wide")
st.title("📆 Daily Candlestick Chart Viewer")

# ----------------------------
# User Inputs
# ----------------------------
ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
st.caption("Shows 60 days of daily candles.")

# ----------------------------
# Fetch & Plot
# ----------------------------
if ticker:
    try:
        with st.spinner(f"Fetching daily data for {ticker}..."):
            df = yf.download(ticker, period="60d", interval="1d", progress=False)

        if df.empty:
            st.warning(f"No data available for {ticker}.")
        else:
            # Clean columns and ensure 'Date'
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            if "Date" not in df.columns:
                df["Date"] = df.index
            df["Date"] = pd.to_datetime(df["Date"])

            # Plot candlesticks
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles"
            ))

            fig.update_layout(
                title=f"{ticker} – Daily Candlestick Chart (Last 60 Days)",
                height=800,
                xaxis_rangeslider_visible=False,
                margin=dict(l=30, r=30, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load data for {ticker}: {e}")
