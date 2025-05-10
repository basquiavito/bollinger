# bollinger.py
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

def calculate_bollinger_bands(df, window=20, num_std=2):
    df['MA20'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['MA20'] + num_std * df['STD']
    df['Lower'] = df['MA20'] - num_std * df['STD']
    return df

def plot_candlestick_with_bb(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Candles'
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1), name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='gray', width=1), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='blue', width=1), name='Lower Band'))

    fig.update_layout(
        title=f'{ticker} - Bollinger Bands (Intraday)',
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig

def main():
    st.title("📈 Intraday Bollinger Bands Viewer")

    ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
    interval = st.selectbox("Interval", ['5m', '15m', '1h'])
    today = datetime.today()

    default_start = today - timedelta(days=7)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')  # yfinance end is exclusive

    if ticker:
        with st.spinner("Fetching data..."):
            df = yf.download(ticker, start=start_str, end=end_str, interval=interval, progress=False)

        if not df.empty:
            df = calculate_bollinger_bands(df)
            fig = plot_candlestick_with_bb(df, ticker)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for this range.")

