# bollinger.py
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def calculate_bollinger_bands(df, window=20, num_std=2):
    df['MA20'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['Upper'] = df['MA20'] + num_std * df['STD']
    df['Lower'] = df['MA20'] - num_std * df['STD']
    return df

def plot_candlestick_with_bb(df, ticker):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Price'
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(200,0,0,0.5)', width=1), name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='gray', width=1), name='20MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(0,0,200,0.5)', width=1), name='Lower Band'))

    fig.update_layout(
        title=f'{ticker} - Bollinger Bands',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig

def main():
    st.title("Bollinger Bands Candlestick Chart")

    ticker = st.text_input("Enter Ticker", value='AAPL')
    interval = st.selectbox("Interval", ['1d', '1h', '30m', '15m', '5m'])
    period = st.selectbox("Period", ['7d', '14d', '30d', '60d'])

    if ticker:
        data = yf.download(ticker, period=period, interval=interval)
        if not data.empty:
            data = calculate_bollinger_bands(data)
            fig = plot_candlestick_with_bb(data, ticker)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for the ticker.")

if __name__ == "__main__":
    main()
