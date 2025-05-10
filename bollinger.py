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

    default_start = today - timedelta(days=3)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    filter_rth = st.checkbox("Show Regular Trading Hours Only (9:30 AM – 4:00 PM)", value=True)

    if ticker:
        with st.spinner("Fetching data..."):
            df = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                interval=interval,
                progress=False
            )

        if df.empty:
            st.warning("No data found for this range.")
            return

        df.reset_index(inplace=True)

        # Fix datetime like your dashboard:
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

        if df["Date"].dtype == "datetime64[ns]":
            df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        else:
            df["Date"] = df["Date"].dt.tz_convert("America/New_York")

        df["Date"] = df["Date"].dt.tz_localize(None)

        if filter_rth:
            df = df[df["Date"].dt.time.between(datetime.strptime("09:30", "%H:%M").time(),
                                               datetime.strptime("16:00", "%H:%M").time())]

        df.set_index("Date", inplace=True)

        if df.shape[0] < 20:
            st.warning("Not enough bars to calculate Bollinger Bands.")
            st.dataframe(df)
            return

        df = calculate_bollinger_bands(df)
        fig = plot_candlestick_with_bb(df, ticker)
        st.plotly_chart(fig, use_container_width=True)


