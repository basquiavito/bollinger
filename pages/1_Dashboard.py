import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date


# =================
# Page Config
# =================
st.set_page_config(
    page_title="Modern Day Trading Dashboard",
    layout="wide"
)


st.title("Options Wealth")

# ======================================
# Sidebar - User Inputs & Advanced Options
# ======================================
st.sidebar.header("Input Options")

default_tickers = ["NVDA", "AVGO","AMD","PLTR","MRVL","uber","AMZN","AAPL","googl","MSFT","META","tsla","sbux","nke","chwy","DKNG","GM","cmg","c","wfc","hood","coin","bac","jpm","PYPL","tgt"]
tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=default_tickers,
    default=["NVDA"]  # Start with one selected
)

# Date range inputs
start_date = st.sidebar.date_input("Start Date", value=date(2025, 5, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["2m","5m", "15m", "30m", "60m", "1d"],
    index=1  # Default to 5m
)

# # 🔥 Candlestick Chart Toggle (Place this here)
# show_candlestick = st.sidebar.checkbox("Show Candlestick Chart", value=False)



# Gap threshold slider
gap_threshold = st.sidebar.slider(
    "Gap Threshold (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="Sets the % gap threshold for UP or DOWN alerts."
)


# ======================================
# Helper function to detect "40ish" + reversal
# ======================================
def detect_40ish_reversal(intraday_df):
    """
    Flags reversals when F% is between 44% to 55% (up) or -55% to -44% (down),
    and the next row moves significantly in the opposite direction.
    """
    intraday_df["40ish"] = ""

    for i in range(len(intraday_df) - 1):
        current_val = intraday_df.loc[i, "F_numeric"]
        next_val = intraday_df.loc[i + 1, "F_numeric"]

        # 44% - 55% (Reversal Down) & -55% to -44% (Reversal Up)
        if 44 <= current_val <= 55 and next_val < current_val:
            intraday_df.loc[i, "40ish"] = "40ish UP & Reversed Down"
        elif -55 <= current_val <= -44 and next_val > current_val:
            intraday_df.loc[i, "40ish"] = "❄️ 40ish DOWN & Reversed Up"

    return intraday_df

# Momentum helper (for 2 and 7 periods)
def add_momentum(df, price_col="Close"):
    """
    Adds Momentum_2 and Momentum_7 columns:
      Momentum_2 = Close[t] - Close[t-2]
      Momentum_7 = Close[t] - Close[t-7]
    """
    df["Momentum_2"] = df[price_col].diff(periods=7)
    df["Momentum_7"] = df[price_col].diff(periods=14)
    return df


# ======================================
# Main Button to Run
# ======================================
if st.sidebar.button("Run Analysis"):
    main_tabs = st.tabs([f"Ticker: {t}" for t in tickers])

    for idx, t in enumerate(tickers):
        with main_tabs[idx]:



            try:
                # ================
                # 1) Fetch Previous Day's Data
                # ================
                daily_data = yf.download(
                    t,
                    end=start_date,
                    interval="1d",
                    progress=False,
                    threads=False
                )

                prev_close, prev_high, prev_low = None, None, None
                prev_close_str, prev_high_str, prev_low_str = "N/A", "N/A", "N/A"

                if not daily_data.empty:
                    if isinstance(daily_data.columns, pd.MultiIndex):
                        daily_data.columns = daily_data.columns.map(
                            lambda x: x[0] if isinstance(x, tuple) else x
                        )
                    prev_close = daily_data["Close"].iloc[-1]
                    prev_high = daily_data["High"].iloc[-1]
                    prev_low = daily_data["Low"].iloc[-1]

                    prev_close_str = f"{prev_close:.2f}"
                    prev_high_str = f"{prev_high:.2f}"
                    prev_low_str = f"{prev_low:.2f}"

                # ================
                # 2) Fetch Intraday Data
                # ================
                intraday = yf.download(
                    t,
                    start=start_date,
                    end=end_date,
                    interval=timeframe,
                    progress=False
                )

                if intraday.empty:
                    st.error(f"No intraday data for {t}.")
                    continue

                intraday.reset_index(inplace=True)
                if isinstance(intraday.columns, pd.MultiIndex):
                    intraday.columns = intraday.columns.map(
                        lambda x: x[0] if isinstance(x, tuple) else x
                    )




                if "Datetime" in intraday.columns:
                    intraday.rename(columns={"Datetime": "Date"}, inplace=True)

                # Convert to New York time
                if intraday["Date"].dtype == "datetime64[ns]":
                    intraday["Date"] = intraday["Date"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                else:
                    intraday["Date"] = intraday["Date"].dt.tz_convert("America/New_York")
                intraday["Date"] = intraday["Date"].dt.tz_localize(None)


                def adjust_marker_y_positions(data, column, base_offset=5):
                    """
                    Adjusts Y-axis positions dynamically to prevent symbol overlap.
                    - `column`: Column containing the markers (e.g., "TD REI Crossover", "VAS Transition").
                    - `base_offset`: Minimum gap between symbols.
                    """
                    y_positions = {}  # Dictionary to track adjustments

                    adjusted_y = []  # List to store adjusted Y-values
                    for i, time in enumerate(data["Time"]):
                        marker = data.at[data.index[i], column]

                        if pd.notna(marker) and marker != "":
                            # If multiple markers exist at the same time, increment the y-offset
                            if time in y_positions:
                                y_positions[time] -= base_offset  # Push down
                            else:
                                y_positions[time] = data.at[data.index[i], "F_numeric"]  # Start at F%

                            adjusted_y.append(y_positions[time])  # Assign adjusted position
                        else:
                            adjusted_y.append(data.at[data.index[i], "F_numeric"])  # Default to F%

                    return adjusted_y




                # Add a Time column (12-hour)
                intraday["Time"] = intraday["Date"].dt.strftime("%I:%M %p")
                # Keep only YYYY-MM-DD in Date column
                intraday["Date"] = intraday["Date"].dt.strftime("%Y-%m-%d")

                # Add a Range column
                intraday["Range"] = intraday["High"] - intraday["Low"]

                # ================
                # 3) Calculate Gap Alerts
                # ================
            # Ensure we have a previous close
                gap_alert = ""
                gap_type = None
                gap_threshold_decimal = gap_threshold / 100.0

                if prev_close is not None and not intraday.empty:
                    first_open = intraday["Open"].iloc[0]

                    # Ensure first_open is valid (not NaN)
                    if pd.isna(first_open):
                        first_open = prev_close  # Default to prev close if missing

                    # Calculate the gap percentage
                    gap_percentage = (first_open - prev_close) / prev_close

                    # **Corrected Logic**
                    if first_open > prev_high:  # Must open *above* previous high to count as gap up
                        if gap_percentage > gap_threshold_decimal:
                            gap_alert = "🚀 UP GAP ALERT"
                            gap_type = "UP"
                    elif first_open < prev_low:  # Must open *below* previous low to count as gap down
                        if gap_percentage < -gap_threshold_decimal:
                            gap_alert = "🔻 DOWN GAP ALERT"
                            gap_type = "DOWN"


                           # 4) High of Day / Low of Day
                # ================
                intraday["High of Day"] = ""
                for date_value, group_df in intraday.groupby("Date", as_index=False):
                    day_indices = group_df.index
                    current_high = -float("inf")
                    last_high_row = None

                    for i2 in day_indices:
                        row_high = intraday.loc[i2, "High"]
                        if row_high > current_high:
                            current_high = row_high
                            last_high_row = i2
                            intraday.loc[i2, "High of Day"] = f"{current_high:.2f}"
                        else:
                            offset = i2 - last_high_row
                            intraday.loc[i2, "High of Day"] = f"+{offset}"

                intraday["Low of Day"] = ""
                for date_value, group_df in intraday.groupby("Date", as_index=False):
                    day_indices = group_df.index
                    current_low = float("inf")
                    last_low_row = None

                    for i2 in day_indices:
                        row_low = intraday.loc[i2, "Low"]
                        if row_low < current_low:
                            current_low = row_low
                            last_low_row = i2
                            intraday.loc[i2, "Low of Day"] = f"{current_low:.2f}"
                        else:
                            offset = i2 - last_low_row
                            intraday.loc[i2, "Low of Day"] = f"+{offset}"

                # ================
                # 5) TD Open Column Example
                # ================
                def check_td_open(row):
                    # Simple example logic
                    if gap_type == "UP":
                        # If price reversed and touched previous day's high
                        if row["Low"] <= prev_high:
                            return "Sell SIGNAL (Reversed Down)"
                    elif gap_type == "DOWN":
                        # If price reversed and touched previous day's low
                        if row["High"] >= prev_low:
                            return "Buy SIGNAL (Reversed Up)"
                    return ""

                intraday["TD Open"] = intraday.apply(check_td_open, axis=1)

                # Get the first intraday open price
                first_open = intraday["Open"].iloc[0]

                def check_td_trap(row):
                    # Only consider TD Trap if the day opened within the previous day's range
                    if first_open > prev_low and first_open < prev_high:
                        # If price moves above previous high, it's a BUY trap signal
                        if row["High"] >= prev_high:
                            return "Buy SIGNAL (TD Trap)"
                        # If price falls below previous low, it's a SELL trap signal
                        elif row["Low"] <= prev_low:
                            return "Sell SIGNAL (TD Trap)"
                    return ""

                intraday["TD Trap"] = intraday.apply(check_td_trap, axis=1)









                prev_open = daily_data["Open"].iloc[-1]   # Yesterday's Open
                prev_close = daily_data["Close"].iloc[-1] # Yesterday's Close
                # Function to check TD CLoP conditions
                def check_td_clop(row):
                    """
                    Checks for TD CLoP signals using previous day's Open (prev_open) and Close (prev_close).
                    - Buy SIGNAL (TD CLoP): Current open < both prev_open & prev_close, then current high > both.
                    - Sell SIGNAL (TD CLoP): Current open > both prev_open & prev_close, then current low < both.
                    """
                    if row["Open"] < prev_open and row["Open"] < prev_close and row["High"] > prev_open and row["High"] > prev_close:
                        return "Buy SIGNAL (TD CLoP)"
                    elif row["Open"] > prev_open and row["Open"] > prev_close and row["Low"] < prev_open and row["Low"] < prev_close:
                        return "Sell SIGNAL (TD CLoP)"
                    return ""

                # Apply function properly
                intraday["TD CLoP"] = intraday.apply(check_td_clop, axis=1)



                # Now call the function outside the definition:
            # Compute F% numeric (ensure this is the final calculation)
                if prev_close is not None:
                    intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
                else:
                    intraday["F_numeric"] = 0  # fallback

                def determine_trap_status(open_price, p_high, p_low):
                    if open_price is None or pd.isna(open_price):
                        return ""
                    if p_high is None or p_low is None:
                        return "Unknown"
                    if open_price > p_high:
                        return "OUTSIDE (Above Prev High)"
                    elif open_price < p_low:
                        return "OUTSIDE (Below Prev Low)"
                    else:
                        return "WITHIN Range"

                intraday["Day Type"] = ""
                mask_930 = intraday["Time"] == "09:30 AM"
                intraday.loc[mask_930, "Day Type"] = intraday[mask_930].apply(
                    lambda row: determine_trap_status(row["Open"], prev_high, prev_low),
                    axis=1
                )




                # Ensure we have at least 5 rows for calculation
                if len(intraday) >= 5:
                    # 1) Calculate the 5-period moving average of volume
                    intraday["Avg_Vol_5"] = intraday["Volume"].rolling(window=5).mean()

                    # 2) Calculate Relative Volume (RVOL)
                    intraday["RVOL_5"] = intraday["Volume"] / intraday["Avg_Vol_5"]

                    # 3) Drop Avg_Vol_5 column since we only need RVOL_5
                    intraday.drop(columns=["Avg_Vol_5"], inplace=True)
                else:
                    # If not enough data, set RVOL_5 to "N/A"
                    intraday["RVOL_5"] = "N/A"
                # ================
                # 7) Calculate F%
                # ================
                def calculate_f_percentage(intraday_df, prev_close_val):
                    if prev_close_val is not None and not intraday_df.empty:
                        intraday_df["F%"] = ((intraday_df["Close"] - prev_close_val) / prev_close_val) * 10000
                        # Round to nearest integer
                        intraday_df["F%"] = intraday_df["F%"].round(0).astype(int).astype(str) + "%"
                    else:
                        intraday_df["F%"] = "N/A"
                    return intraday_df

                intraday = calculate_f_percentage(intraday, prev_close)




                import numpy as np




#**********************************************************************************************************************#**********************************************************************************************************************

                                #Bolinger Bands and BBW Volatility




                def calculate_f_std_bands(df, window=20):
                    if "F_numeric" in df.columns:
                        df["F% MA"] = df["F_numeric"].rolling(window=window, min_periods=1).mean()
                        df["F% Std"] = df["F_numeric"].rolling(window=window, min_periods=1).std()
                        df["F% Upper"] = df["F% MA"] + (2 * df["F% Std"])
                        df["F% Lower"] = df["F% MA"] - (2 * df["F% Std"])
                    return df

                # Apply it to the dataset BEFORE calculating BBW
                intraday = calculate_f_std_bands(intraday, window=20)





                def calculate_f_bbw(df, scale_factor=10):
                            """
                            Computes Bollinger Band Width (BBW) for F% and scales it down.
                            BBW = (Upper Band - Lower Band) / |Middle Band| * 100
                            The result is then divided by `scale_factor` to adjust its magnitude.
                            """
                            if "F% Upper" in df.columns and "F% Lower" in df.columns and "F% MA" in df.columns:
                                df["F% BBW"] = (((df["F% Upper"] - df["F% Lower"]) / df["F% MA"].abs().replace(0, np.nan)) * 100) / scale_factor
                                df["F% BBW"].fillna(0, inplace=True)
                            return df

                        # Apply the function with scaling (e.g., divide by 100)
                intraday = calculate_f_bbw(intraday, scale_factor=10)






                def detect_bbw_tight(df, window=5, percentile_threshold=10):
                    """
                    Detects BBW Tight Compression using dynamic threshold based on ticker's own BBW distribution.
                    Fires 🐝 when at least 3 of last 5 BBW values are below the Xth percentile.
                    """
                    if "F% BBW" not in df.columns:
                        return df

                    # Dynamic threshold: e.g., 10th percentile of all BBW values
                    dynamic_threshold = np.percentile(df["F% BBW"].dropna(), percentile_threshold)

                    # Mark bars where BBW is below threshold
                    df["BBW_Tight"] = df["F% BBW"] < dynamic_threshold

                    # Detect clusters: At least 3 of last 5 bars are tight
                    df["BBW_Tight_Emoji"] = ""
                    for i in range(window, len(df)):
                        recent = df["BBW_Tight"].iloc[i-window:i]
                        if recent.sum() >= 3:
                            df.at[df.index[i], "BBW_Tight_Emoji"] = "🐝"

                    return df

                intraday = detect_bbw_tight(intraday)






                lookback = 5
                intraday["BBW_Anchor"] = intraday["F% BBW"].shift(lookback)




                intraday["BBW_Ratio"] = intraday["F% BBW"] / intraday["BBW_Anchor"]

                def bbw_alert(row):
                        if pd.isna(row["BBW_Ratio"]):
                            return ""
                        if row["BBW_Ratio"] >= 3:
                            return "🔥"  # Triple Expansion
                        elif row["BBW_Ratio"] >= 2:
                            return "🔥"  # Double Expansion
                        return ""

                intraday["BBW Alert"] = intraday.apply(bbw_alert, axis=1)









#**********************************************************************************************************************#**********************************************************************************************************************





                def calculate_kijun_sen(df, period=26):
                    highest_high = df["High"].rolling(window=period, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()
                    df["Kijun_sen"] = (highest_high + lowest_low) / 2
                    return df

                intraday = calculate_kijun_sen(intraday, period=26)
                # Use the previous close (prev_close) from your daily data
                intraday["Kijun_F"] = ((intraday["Kijun_sen"] - prev_close) / prev_close) * 10000


                # Apply the function to your intraday data
                intraday = calculate_kijun_sen(intraday, period=26)

                def f_ichimoku_confirmation(row):
                    if row["Close"] > row["Kijun_sen"]:
                        # Price is above Kijun → bullish bias
                        if row["F_numeric"] > 0:
                            return "Confirmed Bullish"
                        else:
                            return "Bullish Price, but F% negative"
                    else:
                        # Price is below Kijun → bearish bias
                        if row["F_numeric"] < 0:
                            return "Confirmed Bearish"
                        else:
                            return "Bearish Price, but F% positive"

            # Apply this function row-wise
                intraday["F_Ichimoku_Confirmation"] = intraday.apply(f_ichimoku_confirmation, axis=1)








                def detect_cross(series, reference):
                    """
                    Returns a Series with:
                    - "up" if the series crosses above the reference (i.e. previous value below and current value at/above)
                    - "down" if it crosses below (previous value above and current value at/below)
                    - "" otherwise.
                    """
                    cross = []
                    for i in range(len(series)):
                        if i == 0:
                            cross.append("")
                        else:
                            if series.iloc[i-1] < reference.iloc[i-1] and series.iloc[i] >= reference.iloc[i]:
                                cross.append("up")
                            elif series.iloc[i-1] > reference.iloc[i-1] and series.iloc[i] <= reference.iloc[i]:
                                cross.append("down")
                            else:
                                cross.append("")
                    return pd.Series(cross, index=series.index)

                # Detect crosses of F_numeric over its middle band:
                intraday["Cross_Mid"] = detect_cross(intraday["F_numeric"], intraday["F% MA"])

                # Detect crosses of F_numeric over the Kijun_F line:
                intraday["Cross_Kijun"] = detect_cross(intraday["F_numeric"], intraday["Kijun_F"])


                def map_alert_mid(cross):
                    if cross == "up":
                        return "POMB"
                    elif cross == "down":
                        return "PUMB"
                    else:
                        return ""

                def map_alert_kijun(cross):
                    if cross == "up":
                        return "POK"
                    elif cross == "down":
                        return "PUK"
                    else:
                        return ""

                intraday["Alert_Mid"] = intraday["Cross_Mid"].apply(map_alert_mid)
                intraday["Alert_Kijun"] = intraday["Cross_Kijun"].apply(map_alert_kijun)




                import numpy as np

                def calculate_rsi(f_percent, period=14):
                    delta = f_percent.diff()

                    gain = np.where(delta > 0, delta, 0)
                    loss = np.where(delta < 0, -delta, 0)

                    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
                    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                    return rsi





                # After fetching intraday data and ensuring you have prev_close:
                if prev_close is not None:
                    intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
                else:
                    intraday["F_numeric"] = 0  # fallback

                # Now calculate RSI on numeric F%
                intraday["RSI_F%"] = calculate_rsi(intraday["F_numeric"])

                intraday["RSI_Signal"] = intraday["RSI_F%"].rolling(window=7, min_periods=1).mean()

                # Sample DataFrame
                # Ensure 'Time' is in datetime format and market starts at 9:30 AM
                intraday["Time"] = pd.to_datetime(intraday["Time"])

                # Define the morning session (first two hours)
                morning_mask = (intraday["Time"].dt.time >= pd.to_datetime("09:30").time()) & (intraday["Time"].dt.time <= pd.to_datetime("11:30").time())

                # Get highest and lowest price in the first two hours
                ctod_high = intraday.loc[morning_mask, "High"].max()
                ctod_low = intraday.loc[morning_mask, "Low"].min()

                # Add new columns for CTOD High and Low
                intraday["CTOD_High"] = ctod_high
                intraday["CTOD_Low"] = ctod_low

                # Generate Buy/Sell Alerts
                intraday["Buy_Alert"] = intraday["Close"] > intraday["CTOD_High"]
                intraday["Sell_Alert"] = intraday["Close"] < intraday["CTOD_Low"]

                # Convert boolean alerts to text
                intraday["CTOD Alert"] = intraday.apply(
                    lambda row: "Buy" if row["Buy_Alert"] else ("Sell" if row["Sell_Alert"] else ""), axis=1
                )


                # Drop boolean alert columns if not needed
                intraday.drop(columns=["Buy_Alert", "Sell_Alert"], inplace=True)


                # Ensure RSI Crossovers are calculated before Master Buy Signal
                intraday["RSI_Cross"] = ""

                for i in range(1, len(intraday)):
                    prev_rsi = intraday.loc[i - 1, "RSI_F%"]
                    prev_signal = intraday.loc[i - 1, "RSI_Signal"]
                    curr_rsi = intraday.loc[i, "RSI_F%"]
                    curr_signal = intraday.loc[i, "RSI_Signal"]

                    # RSI Crosses Above Signal Line → Bullish Crossover
                    if prev_rsi < prev_signal and curr_rsi > curr_signal:
                        intraday.loc[i, "RSI_Cross"] = "Up"

                    # RSI Crosses Below Signal Line → Bearish Crossover
                    elif prev_rsi > prev_signal and curr_rsi < curr_signal:
                        intraday.loc[i, "RSI_Cross"] = "Down"



                    def detect_wealth_signals(df, expiration_bars=12):
                            """
                            Wealth Trading Signals with Color Coding:
                            - Wealth Buy/Sell I triggers when RVOL_5 > 1.8 (Volume Spike).
                            - Different colors based on spike intensity:
                            - 🔴 Red: Extreme Volume (RVOL_5 > 1.8)
                            - 🟡 Yellow: Strong Volume (RVOL_5 > 1.5)
                            - 🌸 Pink: Moderate Volume (RVOL_5 > 1.2)
                            - Buy II & Sell II depend on Kijun crossovers.
                            - Buy III & Sell III confirm trend reversals.
                            - Buy IV & Sell IV confirm additional volume spikes in trend direction.
                            """

                            df["Wealth Signal"] = ""
                            volume_spike_active = False
                            buy_ii_active = False  # Track if Buy II has happened
                            sell_ii_active = False  # Track if Sell II has happened
                            volume_spike_index = None  # Track when volume spike happened
                            above_kijun = False  # Track if price is already above Kijun after Buy II
                            below_kijun = False  # Track if price is already below Kijun after Sell II

                            for i in range(1, len(df)):
                                # ✅ **Check for Volume Spike (Triggers Wealth Buy I / Wealth Sell I)**
                                if df.loc[i, "RVOL_5"] > 1.2:  # Any RVOL spike above 1.2 triggers a signal
                                    if df.loc[i, "RVOL_5"] > 1.8:
                                        color = "red"  # Extreme Volume → Default (Red for Sell, Green for Buy)
                                    elif df.loc[i, "RVOL_5"] > 1.5:
                                        color = "yellow"  # Strong Volume → Change to Yellow
                                    else:
                                        color = "pink"  # Moderate Volume → Change to Pink



                                    # ✅ **Continue with Volume Spike Activation**
                                    volume_spike_active = True
                                    buy_ii_active = False  # Reset buy tracking
                                    sell_ii_active = False  # Reset sell tracking
                                    volume_spike_index = i  # Track when it happened

                                # ✅ **Check if the signal should expire**
                                if volume_spike_active and volume_spike_index is not None:
                                    if i - volume_spike_index > expiration_bars:
                                        volume_spike_active = False  # Expire the signal
                                        buy_ii_active = False
                                        sell_ii_active = False
                                        volume_spike_index = None  # Reset tracking
                                        above_kijun = False  # Reset tracking
                                        below_kijun = False  # Reset tracking

                                # ✅ **If volume spike is active, check for confirmation signals**
                                if volume_spike_active:
                                    prev_f, curr_f = df.loc[i - 1, "F_numeric"], df.loc[i, "F_numeric"]
                                    prev_kijun, curr_kijun = df.loc[i - 1, "Kijun_F"], df.loc[i, "Kijun_F"]

                                    kijun_cross_up = prev_f < prev_kijun and curr_f >= curr_kijun
                                    kijun_cross_down = prev_f > prev_kijun and curr_f <= curr_kijun

                                    # ✅ **Handle first Kijun cross (Buy II / Sell II)**
                                    if not buy_ii_active and not sell_ii_active:
                                        if kijun_cross_up:  # ✅ **Only Kijun UP Cross**
                                            df.loc[i, "Wealth Signal"] = "Wealth Buy II"
                                            buy_ii_active = True
                                            above_kijun = True
                                        elif kijun_cross_down:  # ✅ **Only Kijun DOWN Cross**
                                            df.loc[i, "Wealth Signal"] = "Wealth Sell II"
                                            sell_ii_active = True
                                            below_kijun = True

                                    # ✅ **Handle second Kijun cross (Buy III / Sell III)**
                                    elif buy_ii_active:
                                        if kijun_cross_down:  # Second confirmation **ONLY Kijun**
                                            df.loc[i, "Wealth Signal"] = "Wealth Sell III"
                                            volume_spike_active = False  # Reset everything
                                            buy_ii_active = False
                                            sell_ii_active = False
                                            above_kijun = False  # Reset
                                    elif sell_ii_active:
                                        if kijun_cross_up:  # Second confirmation **ONLY Kijun**
                                            df.loc[i, "Wealth Signal"] = "Wealth Buy III"
                                            volume_spike_active = False  # Reset everything
                                            buy_ii_active = False
                                            sell_ii_active = False
                                            below_kijun = False  # Reset

                                    # ✅ **NEW: Handle Wealth Buy IV (Strength Confirmation)**
                                    elif above_kijun and df.loc[i, "RVOL_5"] > 1.8:
                                        df.loc[i, "Wealth Signal"] = "Wealth Buy IV (Strength Continuation)"
                                        above_kijun = False  # Prevent further signals

                                    # ✅ **NEW: Handle Wealth Sell IV (Continuation Below Kijun)**
                                    elif below_kijun and df.loc[i, "RVOL_5"] > 1.8:
                                        df.loc[i, "Wealth Signal"] = "Wealth Sell IV (Downtrend Strength)"
                                        below_kijun = False  # Prevent further signals

                            return df








                intraday = detect_wealth_signals(intraday)




                def generate_market_snapshot(df, current_time, current_price, prev_close, symbol):
                    """
                    Generates a concise market snapshot:
                    - Time and current price
                    - Opening price & where it stands now
                    - F% change in raw dollars
                    - Price position relative to Kijun and Bollinger Mid
                    - Latest Buy/Sell Signal
                    """

                    # Convert time to 12-hour format (e.g., "03:55 PM")
                    current_time_str = pd.to_datetime(current_time).strftime("%I:%M %p")

                    # Get today's opening price
                    open_price = df["Open"].iloc[0]

                    # Calculate today's price changes
                    price_change = current_price - prev_close
                    f_percent_change = (price_change / prev_close) * 10000  # F%

                    # Identify price position relative to Kijun and Bollinger Middle
                    last_kijun = df["Kijun_sen"].iloc[-1]
                    last_mid_band = df["F% MA"].iloc[-1]

                    position_kijun = "above Kijun" if current_price > last_kijun else "below Kijun"
                    position_mid = "above Mid Band" if current_price > last_mid_band else "below Mid Band"

                    # Get the latest Buy/Sell signal
                    latest_signal = df.loc[df["Wealth Signal"] != "", ["Wealth Signal"]].tail(1)
                    signal_text = latest_signal["Wealth Signal"].values[0] if not latest_signal.empty else "No Signal"

                    # Construct the message
                    snapshot = (
                        f"📌 {current_time_str} – **{symbol}** is trading at **${current_price:.2f}**\n\n"
                        f"• Opened at **${open_price:.2f}** and is now sitting at **${current_price:.2f}**\n"
                        f"• F% Change: **{f_percent_change:.0f} F%** (${price_change:.2f})\n"
                        f"• Price is **{position_kijun}** & **{position_mid}**\n"
                        f"• **Latest Signal**: {signal_text}\n"
                    )

                    return snapshot





                if not intraday.empty:
                    current_time = intraday["Time"].iloc[-1]
                    current_price = intraday["Close"].iloc[-1]
                    st.markdown(generate_market_snapshot(intraday, current_time, current_price, prev_close, symbol=t))
                else:
                    st.warning(f"No intraday data available for {t}.")

                def detect_kijun_f_cross(df):
                    """
                    Detects when F% crosses above or below Kijun_F%.
                    - "Buy Kijun Cross" → F_numeric crosses above Kijun_F
                    - "Sell Kijun Cross" → F_numeric crosses below Kijun_F
                    """
                    df["Kijun_F_Cross"] = ""

                    for i in range(1, len(df)):
                        prev_f = df.loc[i - 1, "F_numeric"]
                        prev_kijun = df.loc[i - 1, "Kijun_F"]
                        curr_f = df.loc[i, "F_numeric"]
                        curr_kijun = df.loc[i, "Kijun_F"]

                        # Bullish Cross (Buy Signal)
                        if prev_f < prev_kijun and curr_f >= curr_kijun:
                            df.loc[i, "Kijun_F_Cross"] = "Buy Kijun Cross"

                        # Bearish Cross (Sell Signal)
                        elif prev_f > prev_kijun and curr_f <= curr_kijun:
                            df.loc[i, "Kijun_F_Cross"] = "Sell Kijun Cross"

                    return df

                # Apply function to detect Kijun F% crosses
                intraday = detect_kijun_f_cross(intraday)





                def calculate_f_tenkan(df, period=9):
                    """
                    Computes the F% version of Tenkan-sen (Conversion Line).
                    Formula: (Tenkan-sen - Prev Close) / Prev Close * 10000
                    """
                    highest_high = df["High"].rolling(window=period, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()
                    df["Tenkan_sen"] = (highest_high + lowest_low) / 2

                    if "Prev_Close" in df.columns:
                        df["F% Tenkan"] = ((df["Tenkan_sen"] - df["Prev_Close"]) / df["Prev_Close"]) * 10000
                    else:
                        df["F% Tenkan"] = 0  # Fallback in case Prev_Close is missing

                    return df

                # Apply to intraday dataset
                intraday = calculate_f_tenkan(intraday, period=9)



   # Step 1: Calculate OBV
                def calculate_obv(df):
                    df["OBV"] = 0  # Initialize OBV column
                    df["OBV"] = np.where(df["Close"] > df["Close"].shift(1), df["Volume"],
                                        np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0)).cumsum()

                    # Normalize OBV to be in hundreds instead of thousands
                    df["OBV"] = df["OBV"] / 10000

                    return df

                # Step 2: Detect OBV Crossovers
                def detect_obv_crossovers(df):
                    df["OBV_Crossover"] = ""

                    for i in range(1, len(df)):
                        prev_obv = df.loc[i - 1, "OBV"]
                        curr_obv = df.loc[i, "OBV"]

                        if prev_obv < 0 and curr_obv >= 0:
                            df.loc[i, "OBV_Crossover"] = "🔈"  # Speaker (Bullish Volume Shift)
                        elif prev_obv > 0 and curr_obv <= 0:
                            df.loc[i, "OBV_Crossover"] = "🔇"  # Muted Speaker (Bearish Volume Weakness)

                    return df

                # Apply OBV & Crossover Detection
                intraday = calculate_obv(intraday)
                intraday = detect_obv_crossovers(intraday)

                def calculate_f_theta(df, scale_factor=100):
                    """
                    Computes tan(theta) of F% to detect sharp movements.
                    Formula: tan(theta) = F% change (approximate slope)
                    Scales result by scale_factor (default 100).
                    """
                    if "F_numeric" in df.columns:
                        df["F% Theta"] = np.degrees(np.arctan(df["F_numeric"].diff())) * scale_factor
                    else:
                        df["F% Theta"] = 0  # Fallback if column is missing
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_theta(intraday, scale_factor=100)  # Adjust scale_factor if needed

                def detect_theta_spikes(df):
                    """
                    Identifies large spikes in F% Theta automatically using standard deviation.
                    - Uses 2.5x standard deviation as a dynamic threshold.
                    - Detects both positive and negative spikes.
                    """
                    if "F% Theta" not in df.columns:
                        return df  # Avoid crash if missing column

                    theta_std = df["F% Theta"].std()  # Compute stock-specific volatility
                    threshold = 2 * theta_std  # Set dynamic threshold

                    df["Theta_Change"] = df["F% Theta"].diff()  # Compute directional change
                    df["Theta_Spike"] = df["Theta_Change"].abs() > threshold  # Detect both up/down spikes

                    return df
                intraday = detect_theta_spikes(intraday)





                def calculate_f_velocity_and_speed(df):
                    """
                    Computes:
                    - **F% Velocity** = directional rate of F% change per bar.
                    - **F% Speed** = absolute rate of F% change per bar (ignores direction).
                    """
                    if "F_numeric" in df.columns:
                        df["F% Velocity"] = df["F_numeric"].diff()  # Includes direction (+/-)
                        df["F% Speed"] = df["F% Velocity"].abs()    # Only magnitude, no direction
                    else:
                        df["F% Velocity"] = 0  # Fallback
                        df["F% Speed"] = 0      # Fallback
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_velocity_and_speed(intraday)


                def calculate_f_theta_cot(df, scale_factor=100):
                    """
                    Computes tan(theta) and cot(theta) of F% to detect sharp movements.
                    - tan(theta) = slope of F% movement
                    - cot(theta) = inverse of tan(theta) (sensitive to small changes)
                    - Results are scaled by `scale_factor` for readability.
                    """
                    if "F_numeric" in df.columns:
                        df["F% Theta"] = np.tan(np.radians(df["F_numeric"].diff())) * scale_factor

                        # Avoid division by zero
                        df["F% Cotangent"] = np.where(df["F% Theta"] != 0, 1 / df["F% Theta"], 0)
                    else:
                        df["F% Theta"] = 0  # Fallback
                        df["F% Cotangent"] = 0  # Fallback
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_theta_cot(intraday, scale_factor=100)




                def detect_f_tenkan_cross(df):
                    """
                    Detects F% Tenkan crosses over F% Kijun.
                    - Returns 'up' if F% Tenkan crosses above F% Kijun
                    - Returns 'down' if F% Tenkan crosses below F% Kijun
                    """
                    df["F% Tenkan Cross"] = ""

                    for i in range(1, len(df)):
                        prev_tenkan = df.loc[i - 1, "F% Tenkan"]
                        prev_kijun = df.loc[i - 1, "Kijun_F"]
                        curr_tenkan = df.loc[i, "F% Tenkan"]
                        curr_kijun = df.loc[i, "Kijun_F"]

                        if prev_tenkan < prev_kijun and curr_tenkan >= curr_kijun:
                            df.loc[i, "F% Tenkan Cross"] = "Up"
                        elif prev_tenkan > prev_kijun and curr_tenkan <= curr_kijun:
                            df.loc[i, "F% Tenkan Cross"] = "Down"

                    return df

                # Apply crossover detection
                intraday = detect_f_tenkan_cross(intraday)

                def track_ll_hh_streaks(df, min_streak=10):
                    """
                    Tracks consecutive occurrences of Low of Day (LL) and High of Day (HH).
                    - If LL or HH persists for at least `min_streak` rows, it gets labeled as "LL + X" or "HH + X".
                    """
                    df["LL_Streak"] = ""
                    df["HH_Streak"] = ""

                    # Track streaks
                    low_streak, high_streak = 0, 0

                    for i in range(len(df)):
                        if df.loc[i, "Low of Day"] != "":
                            low_streak += 1
                        else:
                            low_streak = 0

                        if df.loc[i, "High of Day"] != "":
                            high_streak += 1
                        else:
                            high_streak = 0

                        # Assign labels only if streaks exceed the minimum threshold
                        if low_streak >= min_streak:
                            df.loc[i, "LL_Streak"] = f"LL +{low_streak}"
                        if high_streak >= min_streak:
                            df.loc[i, "HH_Streak"] = f"HH +{high_streak}"

                    return df

                def calculate_td_sequential(data):
                        """
                        Calculates TD Sequential buy/sell setups while avoiding ambiguous
                        boolean errors by using NumPy arrays for comparisons.
                        """

                        # Initialize columns
                        data['Buy Setup'] = np.nan
                        data['Sell Setup'] = np.nan

                        # Convert Close prices to a NumPy array for guaranteed scalar access
                        close_vals = data['Close'].values

                        # Arrays to track consecutive buy/sell counts
                        buy_count = np.zeros(len(data), dtype=np.int32)
                        sell_count = np.zeros(len(data), dtype=np.int32)

                        # Iterate through the rows
                        for i in range(len(data)):
                            # We need at least 4 prior bars to do the comparison
                            if i < 4:
                                continue

                            # Compare scalars from the NumPy array (guaranteed single float)
                            is_buy = (close_vals[i] < close_vals[i - 4])
                            is_sell = (close_vals[i] > close_vals[i - 4])

                            # Update consecutive counts
                            if is_buy:
                                buy_count[i] = buy_count[i - 1] + 1  # increment
                                sell_count[i] = 0                   # reset sell
                            else:
                                buy_count[i] = 0

                            if is_sell:
                                sell_count[i] = sell_count[i - 1] + 1  # increment
                                buy_count[i] = 0                       # reset buy
                            else:
                                sell_count[i] = 0

                            # Assign setup labels if the count is nonzero or completed
                            if buy_count[i] == 9:
                                data.at[data.index[i], 'Buy Setup'] = 'Buy Setup Completed'
                                buy_count[i] = 0  # reset after completion
                            elif buy_count[i] > 0:
                                data.at[data.index[i], 'Buy Setup'] = f'Buy Setup {buy_count[i]}'

                            if sell_count[i] == 9:
                                data.at[data.index[i], 'Sell Setup'] = 'Sell Setup Completed'
                                sell_count[i] = 0  # reset after completion
                            elif sell_count[i] > 0:
                                data.at[data.index[i], 'Sell Setup'] = f'Sell Setup {sell_count[i]}'

                        return data
                intraday = calculate_td_sequential(intraday)

                def detect_king_signal(intraday):
                        """
                        Mike becomes King when:
                        - Buy Queen + F% +20 → 👑
                        - Sell Queen + F% -20 → 🔻👑
                        """
                        intraday["King_Signal"] = ""

                        # Green Kingdom 👑
                        queen_buy_indices = intraday.index[intraday["Kijun_F_Cross"] == "Buy Kijun Cross"].tolist()
                        for q_idx in queen_buy_indices:
                            f_start = intraday.loc[q_idx, "F_numeric"]
                            for i in range(q_idx + 1, len(intraday)):
                                f_now = intraday.loc[i, "F_numeric"]
                                if f_now - f_start >= 20:
                                    intraday.loc[i, "King_Signal"] = "👑"
                                    break

                        # Red Kingdom 🔻👑
                        queen_sell_indices = intraday.index[intraday["Kijun_F_Cross"] == "Sell Kijun Cross"].tolist()
                        for q_idx in queen_sell_indices:
                            f_start = intraday.loc[q_idx, "F_numeric"]
                            for i in range(q_idx + 1, len(intraday)):
                                f_now = intraday.loc[i, "F_numeric"]
                                if f_now - f_start <= -20:
                                    intraday.loc[i, "King_Signal"] = "🔻👑"
                                    break

                        return intraday


                intraday = detect_king_signal(intraday)

                def calculate_td_countdown(data):
                    """
                    Calculates TD Sequential Countdown after a Buy or Sell Setup completion.
                    """

                    # Initialize Countdown columns
                    data['Buy Countdown'] = np.nan
                    data['Sell Countdown'] = np.nan

                    # Convert Close prices to NumPy array for fast comparisons
                    close_vals = data['Close'].values

                    # Initialize countdown arrays
                    buy_countdown = np.zeros(len(data), dtype=np.int32)
                    sell_countdown = np.zeros(len(data), dtype=np.int32)

                    # Iterate through the dataset
                    for i in range(len(data)):
                        if i < 2:  # Need at least 2 prior bars for comparison
                            continue

                        # Start Buy Countdown after Buy Setup Completion
                        if data.at[data.index[i], 'Buy Setup'] == 'Buy Setup Completed':
                            buy_countdown[i] = 1  # Start countdown

                        # Increment Buy Countdown if conditions are met
                        if buy_countdown[i - 1] > 0 and close_vals[i] < close_vals[i - 2]:
                            buy_countdown[i] = buy_countdown[i - 1] + 1
                            data.at[data.index[i], 'Buy Countdown'] = f'Buy Countdown {buy_countdown[i]}'
                            if buy_countdown[i] == 13:
                                data.at[data.index[i], 'Buy Countdown'] = 'Buy Countdown Completed'

                        # Start Sell Countdown after Sell Setup Completion
                        if data.at[data.index[i], 'Sell Setup'] == 'Sell Setup Completed':
                            sell_countdown[i] = 1  # Start countdown

                        # Increment Sell Countdown if conditions are met
                        if sell_countdown[i - 1] > 0 and close_vals[i] > close_vals[i - 2]:
                            sell_countdown[i] = sell_countdown[i - 1] + 1
                            data.at[data.index[i], 'Sell Countdown'] = f'Sell Countdown {sell_countdown[i]}'
                            if sell_countdown[i] == 13:
                                data.at[data.index[i], 'Sell Countdown'] = 'Sell Countdown Completed'

                    return data

                intraday = calculate_td_countdown(intraday)





                def calculate_vas(data, signal_col="F_numeric", volatility_col="ATR", period=14):
                    """
                    Computes Volatility Adjusted Score (VAS) using the given signal and volatility measure.
                    Default: F% as signal, ATR as volatility.
                    """
                    if volatility_col == "ATR":
                        data["ATR"] = data["High"].rolling(window=period).max() - data["Low"].rolling(window=period).min()

                    elif volatility_col == "MAD":
                        data["MAD"] = data["Close"].rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)

                    elif volatility_col == "STD":
                        data["STD"] = data["Close"].rolling(window=period).std()

                    # Compute VAS using selected volatility measure
                    selected_vol = data[volatility_col].fillna(method="bfill")  # Avoid NaN errors
                    data["VAS"] = data[signal_col] / selected_vol
                    return data

                # Apply function to intraday data (defaulting to ATR)
                intraday = calculate_vas(intraday, signal_col="F_numeric", volatility_col="ATR", period=14)
                def calculate_tenkan_sen(df, period=9):
                    """
                    Computes Tenkan-sen for F% based on the midpoint of high/low over a rolling period.
                    """
                    highest_high = df["High"].rolling(window=period, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()
                    df["Tenkan_sen"] = (highest_high + lowest_low) / 2

                    # Convert to F% scale
                    df["Tenkan_F"] = ((df["Tenkan_sen"] - prev_close) / prev_close) * 10000
                    return df

                # Apply to intraday data
                intraday = calculate_tenkan_sen(intraday, period=9)


                def calculate_f_sine_cosine(df):
                    """
                    Computes sine and cosine of F% Theta:
                    - sin(θ) indicates how steep the price change is.
                    - cos(θ) indicates how stable the price trend is.
                    """
                    if "F% Theta" in df.columns:
                        df["F% Sine"] = np.sin(np.radians(df["F% Theta"]))
                        df["F% Cosine"] = np.cos(np.radians(df["F% Theta"]))
                    else:
                        df["F% Sine"] = 0  # Fallback
                        df["F% Cosine"] = 0  # Fallback
                    return df

                # Apply the function after calculating F% Theta
                intraday = calculate_f_sine_cosine(intraday)

                def calculate_chikou_span(df, period=26):
                    """
                    Computes the Chikou Span (Lagging Span) for Ichimoku.
                    Chikou Span is the closing price shifted back by `period` bars.
                    """
                    df["Chikou_Span"] = df["Close"].shift(-period)  # Shift forward
                    return df

                # Apply Chikou Span calculation
                intraday = calculate_chikou_span(intraday, period=26)

                def calculate_kumo(df, period_a=26, period_b=52, shift=26):
                    """
                    Computes Senkou Span A and Senkou Span B for Ichimoku Cloud (Kumo).
                    - Senkou Span A = (Tenkan-Sen + Kijun-Sen) / 2, shifted forward
                    - Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods, shifted forward
                    """
                    df["Senkou_Span_A"] = ((df["Tenkan_sen"] + df["Kijun_sen"]) / 2).shift(shift)

                    highest_high = df["High"].rolling(window=period_b, min_periods=1).max()
                    lowest_low = df["Low"].rolling(window=period_b, min_periods=1).min()
                    df["Senkou_Span_B"] = ((highest_high + lowest_low) / 2).shift(shift)

                    return df

                # Apply Kumo (Cloud) calculations
                intraday = calculate_kumo(intraday)

                def calculate_td_pressure(data):


                        # 1) Compute the price range per bar.
                        #    Where the range is zero, we'll get division by zero — so we handle that by assigning NaN.
                        price_range = data['High'] - data['Low']

                        # 2) Compute the "pressure ratio" for each bar.
                        #    ratio = ((Close - Open) / price_range) * Volume
                        #    If price_range == 0, replace with NaN to avoid inf or division by zero.
                        ratio = (data['Close'] - data['Open']) / price_range * data['Volume']
                        ratio[price_range == 0] = np.nan  # Mark division-by-zero cases as NaN

                        # 3) Compute absolute price difference per bar
                        abs_diff = (data['Close'] - data['Open']).abs()

                        # 4) Sum over a rolling 5-bar window using .rolling(5).
                        #    - rolling_ratio_sum: Sum of the 5-bar pressure ratios
                        #    - rolling_abs_diff_sum: Sum of the 5-bar absolute price differences
                        #    - min_periods=5 ensures we only output a valid sum starting at the 5th bar
                        rolling_ratio_sum = ratio.rolling(5, min_periods=5).sum()
                        rolling_abs_diff_sum = abs_diff.rolling(5, min_periods=5).sum()

                        # 5) Compute the normalized TD Pressure:
                        #    TD Pressure = (sum_of_5_bar_ratios / sum_of_5_bar_abs_diff) / 100000
                        #    If rolling_abs_diff_sum is 0, the result will be NaN (safe handling).
                        data['TD Pressure'] = (rolling_ratio_sum / rolling_abs_diff_sum) / 100000
                        data['TD Pressure'] = data['TD Pressure'].fillna(0)  # Replace NaNs with 0 or another suitable value
                        return data

                intraday = calculate_td_pressure(intraday)

                def calculate_td_rei(data):
                    """
                    Calculates the TD Range Expansion Index (TD REI).
                    TD REI measures the strength of price expansion relative to its range over the last 5 bars.
                    """

                    # Initialize TD REI column
                    data['TD REI'] = np.nan

                    # Convert High and Low prices to NumPy arrays for faster calculations
                    high_vals = data['High'].values
                    low_vals = data['Low'].values

                    # Iterate through the dataset, starting from the 5th row
                    for i in range(5, len(data)):
                        # Step 1: Calculate numerator (high_diff + low_diff)
                        high_diff = high_vals[i] - high_vals[i - 2]  # Current high - high two bars ago
                        low_diff = low_vals[i] - low_vals[i - 2]    # Current low - low two bars ago
                        numerator = high_diff + low_diff  # Sum of the differences

                        # Step 2: Calculate denominator (highest high - lowest low over the last 5 bars)
                        highest_high = np.max(high_vals[i - 4:i + 1])  # Highest high in the last 5 bars
                        lowest_low = np.min(low_vals[i - 4:i + 1])    # Lowest low in the last 5 bars
                        denominator = highest_high - lowest_low

                        # Step 3: Calculate TD REI, ensuring no division by zero
                        if denominator != 0:
                            td_rei_value = (numerator / denominator) * 100
                        else:
                            td_rei_value = 0  # Prevent division by zero

                        # **Fix for extreme values:** Ensure TD REI remains within [-100, 100]
                        td_rei_value = max(min(td_rei_value, 100), -100)

                        # Assign calculated TD REI to the DataFrame
                        data.at[data.index[i], 'TD REI'] = td_rei_value

                    return data
                intraday = calculate_td_rei(intraday)  # Compute TD REI



                def add_momentum(intraday, price_col="Close"):
                    """
                    Adds Momentum_2 and Momentum_7 columns:
                    Momentum_2 = Close[t] - Close[t-2]
                    Momentum_7 = Close[t] - Close[t-7]
                    """
                    intraday["Momentum_2"] = intraday[price_col].diff(periods=2)
                    intraday["Momentum_7"] = intraday[price_col].diff(periods=7)
                    return intraday

                intraday = add_momentum(intraday)  # Compute TD REI


                def add_momentum_shift_emojis(intraday):
                    """
                    Detects sign changes in Momentum_7:
                    - + to - → 🐎
                    - - to + → 🦭
                    """
                    intraday['Momentum_Shift'] = ''
                    intraday['Momentum_Shift_Y'] = np.nan

                    mom = intraday['Momentum_7']

                    shift_down = (mom.shift(1) > 0) & (mom <= 0)
                    shift_up = (mom.shift(1) < 0) & (mom >= 0)

                    intraday.loc[shift_down, 'Momentum_Shift'] = '🐎'
                    intraday.loc[shift_down, 'Momentum_Shift_Y'] = intraday['F_numeric'] + 144

                    intraday.loc[shift_up, 'Momentum_Shift'] = '🦭'
                    intraday.loc[shift_up, 'Momentum_Shift_Y'] = intraday['F_numeric'] + 144

                    return intraday

                intraday = add_momentum_shift_emojis(intraday)  # Compute TD REI




                import numpy as np

                def calculate_td_poq(data):
                    """
                    Computes TD POQ signals based on TD REI conditions and price action breakouts.
                    - Scenario 1 & 3: Buy Calls
                    - Scenario 2 & 4: Buy Puts
                    """
                    data["TD_POQ"] = np.nan  # Initialize column

                    for i in range(6, len(data)):  # Start at row 6 to account for prior bars
                        td_rei = data.at[data.index[i], "TD REI"]
                        close_1, close_2 = data.at[data.index[i - 1], "Close"], data.at[data.index[i - 2], "Close"]
                        open_i, high_i, low_i, close_i = data.at[data.index[i], "Open"], data.at[data.index[i], "High"], data.at[data.index[i], "Low"], data.at[data.index[i], "Close"]
                        high_1, low_1, high_2, low_2 = data.at[data.index[i - 1], "High"], data.at[data.index[i - 1], "Low"], data.at[data.index[i - 2], "High"], data.at[data.index[i - 2], "Low"]

                        # Scenario 1: Qualified TD POQ Upside Breakout — Buy Call
                        if (
                            not np.isnan(td_rei) and td_rei < -45 and  # TD REI in oversold condition
                            close_1 > close_2 and  # Previous close > close two bars ago
                            open_i <= high_1 and  # Current open <= previous high
                            high_i > high_1  # Current high > previous high
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 1: Buy Call"

                        # Scenario 2: Qualified TD POQ Downside Breakout — Buy Put
                        elif (
                            not np.isnan(td_rei) and td_rei > 45 and  # TD REI in overbought condition
                            close_1 < close_2 and  # Previous close < close two bars ago
                            open_i >= low_1 and  # Current open >= previous low
                            low_i < low_1  # Current low < previous low
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 2: Buy Put"

                        # Scenario 3: Alternative TD POQ Upside Breakout — Buy Call
                        elif (
                            not np.isnan(td_rei) and td_rei < -45 and  # TD REI in mild oversold condition
                            close_1 > close_2 and  # Previous close > close two bars ago
                            open_i > high_1 and open_i < high_2 and  # Current open > previous high but < high two bars ago
                            high_i > high_2 and  # Current high > high two bars ago
                            close_i > open_i  # Current close > current open
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 3: Buy Call"

                        # Scenario 4: Alternative TD POQ Downside Breakout — Buy Put
                        elif (
                            not np.isnan(td_rei) and td_rei > 45 and  # TD REI in mild overbought condition
                            close_1 < close_2 and  # Previous close < close two bars ago
                            open_i < low_1 and open_i > low_2 and  # Current open < previous low but > low two bars ago
                            low_i < low_2 and  # Current low < low two bars ago
                            close_i < open_i  # Current close < current open
                        ):
                            data.at[data.index[i], "TD_POQ"] = "Scenario 4: Buy Put"

                    return data

                # Apply function to intraday DataFrame
                intraday = calculate_td_poq(intraday)



                def calculate_f_trig(df):
                    """
                    Computes sine, cosine, cosecant, and secant of F% to detect oscillatory trends.
                    - sin(F%) and cos(F%) capture cyclic behavior.
                    - csc(F%) and sec(F%) detect extreme changes.
                    """
                    if "F_numeric" in df.columns:
                        df["F% Sine"] = np.sin(np.radians(df["F_numeric"]))
                        df["F% Cosine"] = np.cos(np.radians(df["F_numeric"]))

                        # Avoid division by zero
                        df["F% Cosecant"] = np.where(df["F% Sine"] != 0, 1 / df["F% Sine"], np.nan)
                        df["F% Secant"] = np.where(df["F% Cosine"] != 0, 1 / df["F% Cosine"], np.nan)
                    else:
                        df["F% Sine"] = df["F% Cosine"] = df["F% Cosecant"] = df["F% Secant"] = np.nan

                    return df

                def detect_td_rei_crossovers(data):
                    """
                    Identifies TD REI crossovers:
                    - 🧨 (Firecracker) when TD REI crosses from + to -
                    - 🔑 (Key) when TD REI crosses from - to +
                    """
                    data["TD REI Crossover"] = np.nan  # Initialize crossover column

                    for i in range(1, len(data)):  # Start from second row
                        prev_rei = data.at[data.index[i - 1], "TD REI"]
                        curr_rei = data.at[data.index[i], "TD REI"]

                        if pd.notna(prev_rei) and pd.notna(curr_rei):
                            # **From + to - (Bearish) → Firecracker 🧨**
                            if prev_rei > 0 and curr_rei < 0:
                                data.at[data.index[i], "TD REI Crossover"] = "🧨"

                            # **From - to + (Bullish) → Key 🔑**
                            elif prev_rei < 0 and curr_rei > 0:
                                data.at[data.index[i], "TD REI Crossover"] = "🔑"

                    return data
                intraday = detect_td_rei_crossovers(intraday)  # Detect TD REI crossovers


                def calculate_td_poq(data):
                    data['TD POQ'] = np.nan  # Use NaN for consistency

                    for i in range(5, len(data)):  # Start from the 6th row for sufficient prior data
                        if pd.notna(data['TD REI'].iloc[i]):  # Ensure TD REI is not NaN

                            # Buy POQ Logic: Qualified Upside Breakout
                            if (data['TD REI'].iloc[i] < -45 and
                                data['Close'].iloc[i - 1] > data['Close'].iloc[i - 2] and
                                data['Open'].iloc[i] <= data['High'].iloc[i - 1] and
                                data['High'].iloc[i] > data['High'].iloc[i - 1]):
                                data.loc[data.index[i], 'TD POQ'] = 'Buy POQ'

                            # Sell POQ Logic: Qualified Downside Breakout
                            elif (data['TD REI'].iloc[i] > 45 and
                                data['Close'].iloc[i - 1] < data['Close'].iloc[i - 2] and
                                data['Open'].iloc[i] >= data['Low'].iloc[i - 1] and
                                data['Low'].iloc[i] < data['Low'].iloc[i - 1]):
                                data.loc[data.index[i], 'TD POQ'] = 'Sell POQ'

                    return data
                intraday = calculate_td_poq(intraday)  # Detect TD REI crossovers



                def calculate_vas(data, signal_col="F_numeric", volatility_col="ATR", period=14):
                    """
                    Computes Volatility Adjusted Score (VAS) using the given signal and volatility measure.
                    Default: F% as signal, ATR as volatility.
                    """
                    if volatility_col == "ATR":
                        data["ATR"] = data["High"].rolling(window=period).max() - data["Low"].rolling(window=period).min()

                    elif volatility_col == "MAD":
                        data["MAD"] = data["Close"].rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)

                    elif volatility_col == "STD":
                        data["STD"] = data["Close"].rolling(window=period).std()

                    # Compute VAS using selected volatility measure
                    selected_vol = data[volatility_col].fillna(method="bfill")  # Avoid NaN errors
                    data["VAS"] = data[signal_col] / selected_vol
                    return data

                # Apply function to intraday data (defaulting to ATR)
                intraday = calculate_vas(intraday, signal_col="F_numeric", volatility_col="ATR", period=14)
                def calculate_stop_loss(df, prev_low, prev_high):
                    """
                    Calculates stop loss levels:
                    - **CALL Stop Loss**: 1/8 point **below** previous day's Low.
                    - **PUT Stop Loss**: 1/8 point **above** previous day's High.
                    """
                    df["Call Stop Loss"] = prev_low - 0.125  # ✅ Corrected for CALL stop loss
                    df["Put Stop Loss"] = prev_high + 0.125  # ✅ Corrected for PUT stop loss
                    return df

                # Apply the function to intraday data
                intraday = calculate_stop_loss(intraday, prev_low, prev_high)


                intraday["Opening Price Signal"] = intraday["Close"] - prev_open
                intraday["Net Price"] = intraday["Close"] - prev_close

                # Detect Net Price direction changes
                intraday["Net Price Direction"] = ""
                net_price_shifted = intraday["Net Price"].shift(1)

                # 🦅 Eagle: Net Price flips from negative to positive
                intraday.loc[(net_price_shifted < 0) & (intraday["Net Price"] >= 0), "Net Price Direction"] = "🦅"

                # 🦉 Owl: Net Price flips from positive to negative
                intraday.loc[(net_price_shifted > 0) & (intraday["Net Price"] <= 0), "Net Price Direction"] = "🦉"


                # Step 1: Calculate OBV
                def calculate_obv(df):
                    df["OBV"] = 0  # Initialize OBV column
                    df["OBV"] = np.where(df["Close"] > df["Close"].shift(1), df["Volume"],
                                        np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0)).cumsum()

                    # Normalize OBV to be in hundreds instead of thousands
                    df["OBV"] = df["OBV"] / 10000

                    return df

                # Step 2: Detect OBV Crossovers
                def detect_obv_crossovers(df):
                    df["OBV_Crossover"] = ""

                    for i in range(1, len(df)):
                        prev_obv = df.loc[i - 1, "OBV"]
                        curr_obv = df.loc[i, "OBV"]

                        if prev_obv < 0 and curr_obv >= 0:
                            df.loc[i, "OBV_Crossover"] = "🔈"  # Speaker (Bullish Volume Shift)
                        elif prev_obv > 0 and curr_obv <= 0:
                            df.loc[i, "OBV_Crossover"] = "🔇"  # Muted Speaker (Bearish Volume Weakness)

                    return df

                # Apply OBV & Crossover Detection
                intraday = calculate_obv(intraday)
                intraday = detect_obv_crossovers(intraday)


                # Detect Tenkan-Kijun Crosses
                intraday["Tenkan_Kijun_Cross"] = ""

                for i in range(1, len(intraday)):
                    prev_tenkan, prev_kijun = intraday.loc[i - 1, "Tenkan_F"], intraday.loc[i - 1, "Kijun_F"]
                    curr_tenkan, curr_kijun = intraday.loc[i, "Tenkan_F"], intraday.loc[i, "Kijun_F"]

                    # Bullish Cross (🌞)
                    if prev_tenkan < prev_kijun and curr_tenkan >= curr_kijun:
                        intraday.loc[i, "Tenkan_Kijun_Cross"] = "🌞"

                    # Bearish Cross (🌙)
                    elif prev_tenkan > prev_kijun and curr_tenkan <= curr_kijun:
                        intraday.loc[i, "Tenkan_Kijun_Cross"] = "🌙"





                intraday["F%_STD"] = intraday["F_numeric"].rolling(window=9).std()

                lookback_std = 5
                intraday["STD_Anchor"] = intraday["F% Std"].shift(lookback_std)
                intraday["STD_Ratio"] = intraday["F% Std"] / intraday["STD_Anchor"]

                def std_alert(row):
                    if pd.isna(row["STD_Ratio"]):
                        return ""
                    if row["STD_Ratio"] >= 3:
                        return "🐦‍🔥"  # Triple STD Expansion
                    elif row["STD_Ratio"] >= 2:
                        return "🐦‍🔥"  # Double STD Expansion
                    return ""

                intraday["STD_Alert"] = intraday.apply(std_alert, axis=1)





                # Convert previous day levels to F% scale
                intraday["Yesterday Open F%"] = ((prev_open - prev_close) / prev_close) * 10000
                intraday["Yesterday High F%"] = ((prev_high - prev_close) / prev_close) * 10000
                intraday["Yesterday Low F%"] = ((prev_low - prev_close) / prev_close) * 10000
                intraday["Yesterday Close F%"] = ((prev_close - prev_close) / prev_close) * 10000  # Always 0


                # Function to detect OPS transitions
                def detect_ops_transitions(df):
                    df["OPS Transition"] = ""

                    for i in range(1, len(df)):  # Start from second row to compare with previous
                        prev_ops = df.loc[i - 1, "Opening Price Signal"]
                        curr_ops = df.loc[i, "Opening Price Signal"]

                        if prev_ops > 0 and curr_ops <= 0:  # Bearish transition
                            df.loc[i, "OPS Transition"] = "🐻"
                        elif prev_ops < 0 and curr_ops >= 0:  # Bullish transition
                            df.loc[i, "OPS Transition"] = "🐼"

                    return df

                # Apply OPS transition detection
                intraday = detect_ops_transitions(intraday)

                def calculate_f_dmi(df, period=14):
                            """
                            Computes +DI, -DI, and ADX for F% instead of price.
                            - Uses the correct True Range logic for ATR calculation.
                            - Ensures +DM and -DM use absolute differences correctly.
                            """
                            # Compute F% movement between bars
                            df["F_Diff"] = df["F_numeric"].diff()

                            # Compute True Range for F% (ATR Equivalent)
                            df["TR_F%"] = np.abs(df["F_numeric"].diff())

                            # Compute Directional Movement
                            df["+DM"] = np.where(df["F_Diff"] > 0, df["F_Diff"], 0)
                            df["-DM"] = np.where(df["F_Diff"] < 0, -df["F_Diff"], 0)

                            # Ensure no double-counting
                            df["+DM"] = np.where(df["+DM"] > df["-DM"], df["+DM"], 0)
                            df["-DM"] = np.where(df["-DM"] > df["+DM"], df["-DM"], 0)

                            # Smooth using Wilder's Moving Average (EMA Approximation)
                            df["+DM_Smoothed"] = df["+DM"].rolling(window=period, min_periods=1).mean()
                            df["-DM_Smoothed"] = df["-DM"].rolling(window=period, min_periods=1).mean()
                            df["ATR_F%"] = df["TR_F%"].rolling(window=period, min_periods=1).mean()

                            # Compute Directional Indicators (Avoid divide-by-zero)
                            df["+DI_F%"] = (df["+DM_Smoothed"] / df["ATR_F%"]) * 100
                            df["-DI_F%"] = (df["-DM_Smoothed"] / df["ATR_F%"]) * 100

                            # Handle potential NaN or infinite values
                            df["+DI_F%"] = df["+DI_F%"].replace([np.inf, -np.inf], np.nan).fillna(0)
                            df["-DI_F%"] = df["-DI_F%"].replace([np.inf, -np.inf], np.nan).fillna(0)

                            # Compute DX (Directional Movement Index)
                            df["DX_F%"] = np.abs((df["+DI_F%"] - df["-DI_F%"]) / (df["+DI_F%"] + df["-DI_F%"])) * 100
                            df["ADX_F%"] = df["DX_F%"].rolling(window=period, min_periods=1).mean()

                            return df






                intraday = calculate_f_dmi(intraday, period=14)

                lookback_adx = 9  # or 4 for tighter sensitivity
                intraday["ADX_Anchor"] = intraday["ADX_F%"].shift(lookback_adx)
                intraday["ADX_Ratio"] = intraday["ADX_F%"] / intraday["ADX_Anchor"]

                def adx_expansion_alert(row):
                    if pd.isna(row["ADX_Ratio"]):
                        return ""
                    if row["ADX_Ratio"] >= 3:
                        return "🧨"  # Triple Expansion
                    elif row["ADX_Ratio"] >= 2:
                        return "♨️"  # Double Expansion
                    return ""

                intraday["ADX_Alert"] = intraday.apply(adx_expansion_alert, axis=1)




                def calculate_td_demand_supply_lines_fpercent(intraday):
                    """
                    Calculate TD Demand and Supply Lines using ringed lows/highs in F_numeric space.
                    This version aligns with your F% plot.
                    """

                    intraday['TD Demand Line F'] = np.nan
                    intraday['TD Supply Line F'] = np.nan

                    demand_points = []
                    supply_points = []

                    f_vals = intraday['F_numeric'].to_numpy()

                    for i in range(1, len(intraday) - 1):
                        # Ringed Low (Demand in F%)
                        if f_vals[i] < f_vals[i - 1] and f_vals[i] < f_vals[i + 1]:
                            demand_points.append(f_vals[i])
                            if len(demand_points) >= 2:
                                intraday.at[intraday.index[i], 'TD Demand Line F'] = max(demand_points[-2:])
                            else:
                                intraday.at[intraday.index[i], 'TD Demand Line F'] = demand_points[-1]

                        # Ringed High (Supply in F%)
                        if f_vals[i] > f_vals[i - 1] and f_vals[i] > f_vals[i + 1]:
                            supply_points.append(f_vals[i])
                            if len(supply_points) >= 2:
                                intraday.at[intraday.index[i], 'TD Supply Line F'] = min(supply_points[-2:])
                            else:
                                intraday.at[intraday.index[i], 'TD Supply Line F'] = supply_points[-1]

                    # Forward-fill both lines
                    intraday['TD Demand Line F'] = intraday['TD Demand Line F'].ffill()
                    intraday['TD Supply Line F'] = intraday['TD Supply Line F'].ffill()

                    return intraday

                intraday = calculate_td_demand_supply_lines_fpercent(intraday)



                def calculate_td_supply_cross_alert(intraday):
                    """
                    Detects crosses over and under the TD Supply Line F using F_numeric.
                    Adds a column 'tdSupplyCrossalert' with values 'cross', 'down', or None.
                    """

                    intraday["tdSupplyCrossalert"] = None

                    for i in range(1, len(intraday)):
                        prev_f = intraday["F_numeric"].iloc[i - 1]
                        curr_f = intraday["F_numeric"].iloc[i]

                        prev_supply = intraday["TD Supply Line F"].iloc[i - 1]
                        curr_supply = intraday["TD Supply Line F"].iloc[i]

                        # Cross up
                        if prev_f < prev_supply and curr_f >= curr_supply:
                            intraday.at[intraday.index[i], "tdSupplyCrossalert"] = "cross"

                        # Cross down
                        elif prev_f > prev_supply and curr_f <= curr_supply:
                            intraday.at[intraday.index[i], "tdSupplyCrossalert"] = "down"

                    return intraday
                intraday = calculate_td_supply_cross_alert(intraday)


                def calculate_clean_tdst(intraday):
                    """
                    TDST version that only assigns the first TDST value at setup completion,
                    and then blanks it until a new one is formed.
                    """

                    intraday['TDST'] = None
                    current_tdst = None

                    for i in range(9, len(intraday)):
                        # --- Buy Setup Completed ---
                        if intraday['Buy Setup'].iloc[i] == 'Buy Setup Completed':
                            bs1_high = intraday['High'].iloc[i - 8]
                            bs2_high = intraday['High'].iloc[i - 7]
                            current_tdst = f"Buy TDST: {round(max(bs1_high, bs2_high), 2)}"
                            intraday.at[intraday.index[i], 'TDST'] = current_tdst

                        # --- Sell Setup Completed ---
                        elif intraday['Sell Setup'].iloc[i] == 'Sell Setup Completed':
                            ss1_low = intraday['Low'].iloc[i - 8]
                            current_tdst = f"Sell TDST: {round(ss1_low, 2)}"
                            intraday.at[intraday.index[i], 'TDST'] = current_tdst

                        # --- Otherwise: blank until a new setup forms
                        else:
                            intraday.at[intraday.index[i], 'TDST'] = None

                    return intraday

                intraday = calculate_clean_tdst(intraday)



                def calculate_tdst_partial_f(intraday):
                            """
                            Calculates TDST Partial levels dynamically in F% space (F_numeric).
                            - Buy TDST Partial: max(F_numeric during setup + prior F%)
                            - Sell TDST Partial: min(F_numeric during setup + prior F%)
                            """

                            intraday['TDST_Partial_F'] = None  # New column for F% version

                            for i in range(9, len(intraday)):
                                # BUY TDST PARTIAL (F%)
                                if isinstance(intraday['Buy Setup'].iloc[i], str):
                                    start_idx = max(0, i - 8)
                                    setup_high_f = intraday['F_numeric'].iloc[start_idx:i+1].max()
                                    prior_f = intraday['F_numeric'].iloc[max(0, i - 9)]
                                    level = max(setup_high_f, prior_f)
                                    intraday.at[intraday.index[i], 'TDST_Partial_F'] = f"Buy TDST Partial F: {round(level, 2)}"

                                # SELL TDST PARTIAL (F%)
                                if isinstance(intraday['Sell Setup'].iloc[i], str):
                                    start_idx = max(0, i - 8)
                                    setup_low_f = intraday['F_numeric'].iloc[start_idx:i+1].min()
                                    prior_f = intraday['F_numeric'].iloc[max(0, i - 9)]
                                    level = min(setup_low_f, prior_f)
                                    intraday.at[intraday.index[i], 'TDST_Partial_F'] = f"Sell TDST Partial F: {round(level, 2)}"

                            return intraday


                def calculate_f_std_bands(df, window=20):
                    if "F_numeric" in df.columns:
                        df["F% MA"] = df["F_numeric"].rolling(window=window, min_periods=1).mean()
                        df["F% Std"] = df["F_numeric"].rolling(window=window, min_periods=1).std()
                        df["F% Upper"] = df["F% MA"] + (2 * df["F% Std"])
                        df["F% Lower"] = df["F% MA"] - (2 * df["F% Std"])
                    return df

                # Apply it to the dataset BEFORE calculating BBW
                intraday = calculate_f_std_bands(intraday, window=20)


                def detect_kijun_cross_emoji(df, lookahead=3, threshold=0.30):
                """
                Only crown the “🕊️” or “🐦‍⬛” if, after a true Kijun_F% cross,
                price closes at least `threshold` above/below the cross-bar close
                within the next `lookahead` bars.
                """
                df["Kijun_F_Cross_Emoji"] = ""
            
                for i in range(1, len(df) - lookahead):
                    prev_F, curr_F = df.loc[i-1, "F_numeric"], df.loc[i, "F_numeric"]
                    prev_K, curr_K = df.loc[i-1, "Kijun_F"],   df.loc[i,   "Kijun_F"]
                    cross_close    = df.loc[i,   "Close"]
            
                    # — BUY cross detected —
                    if prev_F < prev_K and curr_F >= curr_K:
                        # look for confirmation within the next lookahead bars
                        for j in range(1, lookahead + 1):
                            if df.loc[i+j, "Close"] > cross_close + threshold:
                                df.loc[i, "Kijun_F_Cross_Emoji"] = "🕊️"
                                break
            
                    # — SELL cross detected —
                    elif prev_F > prev_K and curr_F <= curr_K:
                        for j in range(1, lookahead + 1):
                            if df.loc[i+j, "Close"] < cross_close - threshold:
                                df.loc[i, "Kijun_F_Cross_Emoji"] = "🐦‍⬛"
                                break
            
                         return df

                # Then replace your plotting call with:
                intraday = detect_kijun_cross_emoji(intraday, lookahead=3, threshold=0.30)





# Ensure TD Supply Line F exists and is not NaN
                if "TD Supply Line F" in intraday.columns:
                        intraday["Heaven_Cloud"] = np.where(
                            intraday["F_numeric"] > intraday["TD Supply Line F"], "☁️", ""
                        )
                else:
                        intraday["Heaven_Cloud"] = ""






                # 🌧️ Drizzle Emoji when price crosses down below TD Demand Line
                intraday["Prev_F"] = intraday["F_numeric"].shift(1)
                intraday["Prev_Demand"] = intraday["TD Demand Line F"].shift(1)

                intraday["Drizzle_Emoji"] = np.where(
                    (intraday["Prev_F"] >= intraday["Prev_Demand"]) &
                    (intraday["F_numeric"] < intraday["TD Demand Line F"]),
                    "🌧️",
                    ""
                )


                def calculate_atr(df, period=14):
                    high = df['High']
                    low = df['Low']
                    close = df['Close']

                    tr1 = high - low
                    tr2 = (high - close.shift()).abs()
                    tr3 = (low - close.shift()).abs()

                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.ewm(span=period, adjust=False).mean()

                    df['ATR'] = atr
                    return df

                intraday = calculate_atr(intraday)  # This adds the ATR column to the intraday DataFrame





                def detect_atr_expansion(df, lookback=5):
                    """
                    Flags ATR expansion by comparing current ATR to ATR 'lookback' periods ago.
                    """
                    df["ATR_Lag"] = df["ATR"].shift(lookback)

                    df["ATR_Exp_Alert"] = np.select(
                        [
                            df["ATR"] >= 1.5 * df["ATR_Lag"],
                            df["ATR"] >= 1.2 * df["ATR_Lag"]
                        ],
                        [
                            "☄️",  # triple
                            "☄️"     # double
                        ],
                        default=""
                    )

                    return df





                intraday = detect_atr_expansion(intraday, lookback=5)
                # 🐢 Turtle condition: extremely slow velocity

                intraday["Tenkan"] = (intraday["High"].rolling(window=9).max() + intraday["Low"].rolling(window=9).min()) / 2
                intraday["Kijun"] = (intraday["High"].rolling(window=26).max() + intraday["Low"].rolling(window=26).min()) / 2
                intraday["SpanA"] = ((intraday["Tenkan"] + intraday["Kijun"]) / 2)
                intraday["SpanB"] = (intraday["High"].rolling(window=52).max() + intraday["Low"].rolling(window=52).min()) / 2
                # Fill early NaNs so cloud appears fully from 9:30 AM
                intraday["SpanA"] = intraday["SpanA"].bfill()
                intraday["SpanB"] = intraday["SpanB"].bfill()

                intraday["SpanA_F"] = ((intraday["SpanA"] - prev_close) / prev_close) * 10000
                intraday["SpanB_F"] = ((intraday["SpanB"] - prev_close) / prev_close) * 10000

                # Fill again after F%-conversion to guarantee values exist
                intraday["SpanA_F"] = intraday["SpanA_F"].bfill()
                intraday["SpanB_F"] = intraday["SpanB_F"].bfill()


                intraday["Chikou"] = intraday["Close"].shift(-26)

                def calculate_kumo_twist_alerts(intraday):
                    """
                    Adds a 'KumoTwistAlert' column to detect bullish 👼🏼 and bearish 👺 twists.
                    A bullish twist occurs when SpanA crosses above SpanB.
                    A bearish twist occurs when SpanA crosses below SpanB.
                    """
                    intraday["KumoTwistAlert"] = None

                    twist_bullish = (intraday["SpanA_F"] > intraday["SpanB_F"]) & (intraday["SpanA_F"].shift(1) <= intraday["SpanB_F"].shift(1))
                    twist_bearish = (intraday["SpanA_F"] < intraday["SpanB_F"]) & (intraday["SpanA_F"].shift(1) >= intraday["SpanB_F"].shift(1))

                    intraday.loc[twist_bullish, "KumoTwistAlert"] = "bullish"
                    intraday.loc[twist_bearish, "KumoTwistAlert"] = "bearish"

                    return intraday
                intraday = calculate_kumo_twist_alerts(intraday)


                def detect_y_close_cross(intraday):
                    intraday["Y_Close_Cross"] = ""

                    for i in range(1, len(intraday)):
                        prev_f = intraday["F_numeric"].iloc[i - 1]
                        curr_f = intraday["F_numeric"].iloc[i]
                        y_close_f = 0  # Always zero

                        if prev_f < y_close_f and curr_f >= y_close_f:
                            intraday.loc[intraday.index[i], "Y_Close_Cross"] = "🏎️"  # Cross above
                        elif prev_f > y_close_f and curr_f <= y_close_f:
                            intraday.loc[intraday.index[i], "Y_Close_Cross"] = "🚵‍♂️"  # Cross below

                    return intraday
                intraday = detect_y_close_cross(intraday)

                def detect_y_open_cross(intraday):
                    intraday["Y_Open_Cross"] = ""

                    for i in range(1, len(intraday)):
                        prev_f = intraday["F_numeric"].iloc[i - 1]
                        curr_f = intraday["F_numeric"].iloc[i]
                        y_open_f = intraday["Yesterday Open F%"].iloc[0]

                        if prev_f < y_open_f and curr_f >= y_open_f:
                            intraday.loc[intraday.index[i], "Y_Open_Cross"] = "🚶🏾"  # Crossed above open
                        elif prev_f > y_open_f and curr_f <= y_open_f:
                            intraday.loc[intraday.index[i], "Y_Open_Cross"] = "🏃🏾"  # Crossed below open

                    return intraday

                intraday = detect_y_open_cross(intraday)



                def detect_kijun_f_cross(df):
                    """
                    Detects when F% crosses above or below Kijun_F%.
                    - "Buy Kijun Cross" → F_numeric crosses above Kijun_F
                    - "Sell Kijun Cross" → F_numeric crosses below Kijun_F
                    """
                    df["Kijun_F_Cross"] = ""

                    for i in range(1, len(df)):
                        prev_f = df.loc[i - 1, "F_numeric"]
                        prev_kijun = df.loc[i - 1, "Kijun_F"]
                        curr_f = df.loc[i, "F_numeric"]
                        curr_kijun = df.loc[i, "Kijun_F"]

                        # Bullish Cross (Buy Signal)
                        if prev_f < prev_kijun and curr_f >= curr_kijun:
                            df.loc[i, "Kijun_F_Cross"] = "Buy Kijun Cross"

                        # Bearish Cross (Sell Signal)
                        elif prev_f > prev_kijun and curr_f <= curr_kijun:
                            df.loc[i, "Kijun_F_Cross"] = "Sell Kijun Cross"

                    return df

                # Apply function to detect Kijun F% crosses
                intraday = detect_kijun_f_cross(intraday)



                intraday = detect_kijun_f_cross(intraday)

                # Calculate VWAP
                intraday["TP"] = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3
                intraday["TPV"] = intraday["TP"] * intraday["Volume"]
                intraday["Cumulative_TPV"] = intraday["TPV"].cumsum()
                intraday["Cumulative_Volume"] = intraday["Volume"].cumsum()
                intraday["VWAP"] = intraday["Cumulative_TPV"] / intraday["Cumulative_Volume"]

                # Convert VWAP to F% scale
                intraday["VWAP_F"] = ((intraday["VWAP"] - prev_close) / prev_close) * 10000


                # Detect F% vs VWAP_F crossovers
                intraday["VWAP_Cross_Emoji"] = ""
                for i in range(1, len(intraday)):
                    prev_f = intraday.loc[i - 1, "F_numeric"]
                    curr_f = intraday.loc[i, "F_numeric"]
                    prev_vwap = intraday.loc[i - 1, "VWAP_F"]
                    curr_vwap = intraday.loc[i, "VWAP_F"]

                    if prev_f < prev_vwap and curr_f >= curr_vwap:
                        intraday.loc[i, "VWAP_Cross_Emoji"] = "🥁"
                    elif prev_f > prev_vwap and curr_f <= curr_vwap:
                        intraday.loc[i, "VWAP_Cross_Emoji"] = "🎻"


                def detect_vwap_cross_before_kijun(df):
                    """
                    Detects the last VWAP Cross before each Kijun Cross (Buy or Sell).
                    Marks it in a new column: 'VWAP_Before_Kijun' with ♘ (white) for Buy Kijun Cross,
                    and ♞ (black) for Sell Kijun Cross.
                    """
                    df["VWAP_Before_Kijun"] = ""

                    kijun_cross_indices = df.index[df["Kijun_F_Cross"] != ""].tolist()
                    vwap_cross_indices = df.index[df["VWAP_Cross_Emoji"] != ""]

                    for kijun_idx in kijun_cross_indices:
                        # Find all VWAP crosses BEFORE this Kijun cross
                        prior_vwap = [idx for idx in vwap_cross_indices if idx < kijun_idx]

                        if prior_vwap:
                            last_vwap_idx = prior_vwap[-1]
                            # Assign ♘ for Buy Kijun Cross, ♞ for Sell Kijun Cross
                            if df.loc[kijun_idx, "Kijun_F_Cross"] == "Buy Kijun Cross":
                                df.loc[last_vwap_idx, "VWAP_Before_Kijun"] = "♘"
                            elif df.loc[kijun_idx, "Kijun_F_Cross"] == "Sell Kijun Cross":
                                df.loc[last_vwap_idx, "VWAP_Before_Kijun"] = "♞"

                    return df

                intraday = detect_vwap_cross_before_kijun(intraday)

                def classify_rvol_alert(rvol_value):
                    """
                    Classify RVOL level into alert labels or emojis:
                    - Extreme: RVOL > 1.8 → 🔺 Extreme
                    - Strong: 1.5 ≤ RVOL < 1.8 → 🟧 Strong
                    - Moderate: 1.2 ≤ RVOL < 1.5 →  Moderate
                    - None: RVOL < 1.2 → ""
                    """
                    if rvol_value > 1.8:
                        return "🔺 Extreme"
                    elif rvol_value >= 1.5:
                        return "🟧 Strong"
                    elif rvol_value >= 1.2:
                        return "Moderate"
                    else:
                        return ""

                # Apply it like this to your intraday DataFrame:
                intraday["RVOL_Alert"] = intraday["RVOL_5"].apply(classify_rvol_alert)

                # Ensure F_numeric exists in intraday
                if "F_numeric" in intraday.columns:
                    # Compute High & Low of Day in F% scale
                    intraday["F% High"] = intraday["F_numeric"].cummax()  # Rolling highest F%
                    intraday["F% Low"] = intraday["F_numeric"].cummin()   # Rolling lowest F%

                    # Calculate Bounce (Recovery from Lows)
                    intraday["Bounce"] = ((intraday["F_numeric"] - intraday["F% Low"]) / intraday["F% Low"].abs()) * 100

                    # Calculate Retrace (Pullback from Highs)
                    intraday["Retrace"] = ((intraday["F% High"] - intraday["F_numeric"]) / intraday["F% High"].abs()) * 100

                    # Clean up: Replace infinities and NaN values
                    intraday["Bounce"] = intraday["Bounce"].replace([np.inf, -np.inf], 0).fillna(0).round(2)
                    intraday["Retrace"] = intraday["Retrace"].replace([np.inf, -np.inf], 0).fillna(0).round(2)





                # Identify the first OPS value at 9:30 AM
                first_ops_row = intraday[intraday["Time"] == "09:30 AM"]
                if not first_ops_row.empty:
                    first_ops_value = first_ops_row["Opening Price Signal"].iloc[0]
                    first_ops_time = first_ops_row["Time"].iloc[0]

                    # Determine if OPS started positive or negative
                    ops_label = "OPS 🔼" if first_ops_value > 0 else "OPS 🔽"
                    ops_color = "green" if first_ops_value > 0 else "red"



                # Apply the function after computing F_numeric
                intraday = calculate_f_trig(intraday)
                # Add numeric version of F% for plotting
                if prev_close is not None:
                    intraday["F_numeric"] = ((intraday["Close"] - prev_close) / prev_close) * 10000
                else:
                    intraday["F_numeric"] = 0  # fallback

                # ================
                # 8) 40ish Column & Reversal Detection
                # ================
                intraday = detect_40ish_reversal(intraday)

                # Add 2-bar momentum (existing example), plus a 7-bar momentum
                intraday = add_momentum(intraday, price_col="Close")  # => Momentum_2, Momentum_7

# 3) Now that intraday is fully processed,
                #    let's get the final row (which has all new columns).
                # =================================
                # AFTER all pipeline transformations
                # =================================
                # Fetch last 3 bars
                recent_rows = intraday.tail(3)




                def calculate_kijun_cross_returns(df, bars_forward=3):
                    """
                    Calculates $ return from Kijun cross (🕊️ or 🐦‍⬛) to N bars ahead.
                    Returns a DataFrame with direction, entry time, entry price, exit price, and return.
                    """
                    results = []
                    for i in range(len(df)):
                        emoji = df.loc[i, "Kijun_F_Cross_Emoji"]
                        if emoji in ["🕊️", "🐦‍⬛"]:
                            entry_time = df.loc[i, "Time"]
                            entry_price = df.loc[i, "Close"]
                            exit_index = i + bars_forward
                            if exit_index < len(df):
                                exit_price = df.loc[exit_index, "Close"]
                                dollar_return = round(exit_price - entry_price, 2)
                                direction = "Call (🕊️)" if emoji == "🕊️" else "Put (🐦‍⬛)"
                                results.append({
                                    "Direction": direction,
                                    "Entry Time": entry_time,
                                    "Entry Price": round(entry_price, 2),
                                    "Exit Price": round(exit_price, 2),
                                    "Return ($)": dollar_return
                                })
                    return pd.DataFrame(results)
                def detect_yesterday_high_cross(intraday):
                    """
                    Detects when F_numeric crosses above or below Yesterday's High (F%).
                    ✈️ for upward crosses, 🪂 for downward.
                    """
                    intraday["Y_High_Cross"] = ""
                    y_high_f = intraday["Yesterday High F%"].iloc[0]  # Static level for all rows

                    for i in range(1, len(intraday)):
                        prev_f = intraday["F_numeric"].iloc[i - 1]
                        curr_f = intraday["F_numeric"].iloc[i]

                        if prev_f < y_high_f and curr_f >= y_high_f:
                            intraday.loc[intraday.index[i], "Y_High_Cross"] = "✈️"
                        elif prev_f > y_high_f and curr_f <= y_high_f:
                            intraday.loc[intraday.index[i], "Y_High_Cross"] = "🪂"

                    return intraday
                intraday = detect_yesterday_high_cross(intraday)



                def detect_new_highs_above_yesterday_high(intraday):
                    """
                    Detects every new intraday high that is above Yesterday's High F%.
                    Adds 👨🏽‍🚀 emoji each time a new high is made above Yesterday High.
                    """
                    intraday["Astronaut_Emoji"] = ""
                    y_high_f = intraday["Yesterday High F%"].iloc[0]  # Static level

                    highest_so_far = y_high_f  # Start from yesterday's high

                    for i in range(1, len(intraday)):
                        curr_f = intraday["F_numeric"].iloc[i]

                        # Only if price is above yesterday's high
                        if curr_f > y_high_f:
                            # Check if it's a new high above highest recorded
                            if curr_f > highest_so_far:
                                intraday.loc[intraday.index[i], "Astronaut_Emoji"] = "👨🏽‍🚀"
                                highest_so_far = curr_f  # Update highest

                    return intraday

                intraday = detect_new_highs_above_yesterday_high(intraday)


                # Find the last astronaut (new high) row
                last_astronaut_idx = intraday[intraday["Astronaut_Emoji"] == "👨🏽‍🚀"].index.max()

                # If there was at least one astronaut
                if pd.notna(last_astronaut_idx):
                    intraday.loc[last_astronaut_idx, "Astronaut_Emoji"] = "🌒"


                def detect_f_takeoff_breakout(df):
                    """
                    Tracks the last ✈️ cross and issues a 🚀 breakout if F% crosses above that bar’s F%.
                    """
                    df["Takeoff_Level"] = None
                    df["Breakout_Emoji"] = ""

                    takeoff_level = None

                    for i in range(1, len(df)):
                        # Step 1: Store the ✈️ level
                        if df.loc[df.index[i], "Y_High_Cross"] == "✈️":
                            takeoff_level = df.loc[df.index[i], "F_numeric"]
                            df.loc[df.index[i], "Takeoff_Level"] = takeoff_level

                        # Step 2: Trigger 🚀 if price crosses above last takeoff level
                        elif takeoff_level is not None and df.loc[df.index[i], "F_numeric"] > takeoff_level:
                            df.loc[df.index[i], "Breakout_Emoji"] = "🚀"
                            takeoff_level = None  # Reset to avoid repeated 🚀

                    return df
                intraday = detect_f_takeoff_breakout(intraday)


                def detect_y_low_crosses(intraday):
                    """
                    Adds a column 'Y_Low_Cross' with:
                    - 🚣🏽 when F% crosses above yesterday's low (bullish recovery)
                    - 🛟 when F% crosses below yesterday's low (bearish breach)
                    """
                    intraday["Y_Low_Cross"] = ""
                    y_low_f = intraday["Yesterday Low F%"].iloc[0]

                    for i in range(1, len(intraday)):
                        prev_f = intraday["F_numeric"].iloc[i - 1]
                        curr_f = intraday["F_numeric"].iloc[i]

                        if prev_f < y_low_f and curr_f >= y_low_f:
                            intraday.loc[intraday.index[i], "Y_Low_Cross"] = "🚣🏽"  # Recovery Up
                        elif prev_f > y_low_f and curr_f <= y_low_f:
                            intraday.loc[intraday.index[i], "Y_Low_Cross"] = "🛟"  # Breach Down

                    return intraday

                intraday = detect_y_low_crosses(intraday)


                def calculate_f_velocity_and_speed(df):
                    """
                    Computes:
                    - **F% Velocity** = directional rate of F% change per bar.
                    - **F% Speed** = absolute rate of F% change per bar (ignores direction).
                    """
                    if "F_numeric" in df.columns:
                        df["F% Velocity"] = df["F_numeric"].diff()  # Includes direction (+/-)
                        df["F% Speed"] = df["F% Velocity"].abs()    # Only magnitude, no direction
                    else:
                        df["F% Velocity"] = 0  # Fallback
                        df["F% Speed"] = 0      # Fallback
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_velocity_and_speed(intraday)



                def detect_checkmate(df):
                    """
                    Detects Checkmate condition at end of session:
                    - If F% is above Kijun_F → CHECKMATE UP
                    - If F% is below Kijun_F → CHECKMATE DOWN
                    """
                    df["Checkmate"] = ""

                    last_idx = df.index[-1]
                    last_f = df.at[last_idx, "F_numeric"]
                    last_kijun = df.at[last_idx, "Kijun_F"]

                    if last_f > last_kijun:
                        df.at[last_idx, "Checkmate"] = "CHECKMATE UP"
                    elif last_f < last_kijun:
                        df.at[last_idx, "Checkmate"] = "CHECKMATE DOWN"

                    return df

                intraday = detect_checkmate(intraday)


                def detect_velocity_spikes(df):
                    """
                    Identifies large spikes in F% Velocity automatically using standard deviation.
                    - Uses 2.5x standard deviation as a dynamic threshold.
                    - Detects both positive and negative spikes.
                    """
                    if "F% Velocity" not in df.columns:
                        return df  # Avoid crash if missing column

                    velocity_std = df["F% Velocity"].std()  # Compute stock-specific volatility
                    threshold = 2.5 * velocity_std  # Set dynamic threshold (adjust multiplier if needed)

                    df["Velocity_Change"] = df["F% Velocity"].diff()  # Compute directional change
                    df["Velocity_Spike"] = df["Velocity_Change"].abs() > threshold  # Detect both up/down spikes

                # Detect extremely slow (stagnant) velocity
                    df["Slow_Velocity"] = df["F% Velocity"].abs() < (0.15 * velocity_std)
                    df["Slow_Velocity_Emoji"] = df["Slow_Velocity"].apply(lambda x: "🐢" if x else "")


                    return df

                # Apply function with user-defined threshold
                intraday = detect_velocity_spikes(intraday)





                def detect_kijun_cross_horses(df):
                    """
                    After F% crosses Kijun (buy or sell), look forward up to 9 bars.
                    Mark every bar that has RVOL_5 ≥ 1.2 with a horse emoji:
                    - ♘ for Buy Kijun Cross
                    - ♞ for Sell Kijun Cross
                    """
                    df["Kijun_Cross_Horse"] = ""

                    kijun_cross_indices = df.index[df["Kijun_F_Cross"].isin(["Buy Kijun Cross", "Sell Kijun Cross"])]

                    for idx in kijun_cross_indices:
                        start_idx = df.index.get_loc(idx) + 1
                        end_idx = start_idx + 21

                        future = df.iloc[start_idx:end_idx]

                        mask_high_rvol = future["RVOL_5"] >= 1.2

                        for future_idx in future[mask_high_rvol].index:
                            if df.at[idx, "Kijun_F_Cross"] == "Buy Kijun Cross":
                                df.at[future_idx, "Kijun_Cross_Horse"] = "♘"
                            elif df.at[idx, "Kijun_F_Cross"] == "Sell Kijun Cross":
                                df.at[future_idx, "Kijun_Cross_Horse"] = "♞"

                    return df




                intraday =  detect_kijun_cross_horses(intraday)




                def detect_tenkan_pawns(df):
                    """
                    Detect when F% crosses Tenkan.
                    Assign a pawn emoji:
                    - ♙ for upward cross (F% crosses above Tenkan)
                    - ♟️ for downward cross (F% crosses below Tenkan)
                    """
                    df["Tenkan_Pawn"] = ""

                    for i in range(1, len(df)):
                        prev_f = df.loc[i - 1, "F_numeric"]
                        prev_tenkan = df.loc[i - 1, "Tenkan_F"]
                        curr_f = df.loc[i, "F_numeric"]
                        curr_tenkan = df.loc[i, "Tenkan_F"]

                        # Upward Cross (♙)
                        if prev_f < prev_tenkan and curr_f >= curr_tenkan:
                            df.at[df.index[i], "Tenkan_Pawn"] = "♙"

                        # Downward Cross (♟️)
                        elif prev_f > prev_tenkan and curr_f <= curr_tenkan:
                            df.at[df.index[i], "Tenkan_Pawn"] = "♟️"

                    return df

                intraday = detect_tenkan_pawns(intraday)





                def detect_kijun_cross_bishops(df):
                    """
                    After F% crosses Kijun (buy or sell), look forward up to 9 bars.
                    Mark every bar that has a BBW Alert with a bishop emoji:
                    - ♗ for Buy Kijun Cross
                    - ♝ for Sell Kijun Cross
                    """
                    df["Kijun_Cross_Bishop"] = ""

                    kijun_cross_indices = df.index[df["Kijun_F_Cross"].isin(["Buy Kijun Cross", "Sell Kijun Cross"])]

                    for idx in kijun_cross_indices:
                        start_idx = df.index.get_loc(idx) + 1
                        end_idx = start_idx + 9

                        future = df.iloc[start_idx:end_idx]

                        mask_bbw_alert = future["BBW Alert"] != ""

                        for future_idx in future[mask_bbw_alert].index:
                            if df.at[idx, "Kijun_F_Cross"] == "Buy Kijun Cross":
                                df.at[future_idx, "Kijun_Cross_Bishop"] = "♗"
                            elif df.at[idx, "Kijun_F_Cross"] == "Sell Kijun Cross":
                                df.at[future_idx, "Kijun_Cross_Bishop"] = "♝"

                    return df

                intraday =  detect_kijun_cross_bishops(intraday)




                def calculate_f_theta(df, scale_factor=100):
                    """
                    Computes tan(theta) of F% to detect sharp movements.
                    Formula: tan(theta) = F% change (approximate slope)
                    Scales result by scale_factor (default 100).
                    """
                    if "F_numeric" in df.columns:
                        df["F% Theta"] = np.degrees(np.arctan(df["F_numeric"].diff())) * scale_factor
                    else:
                        df["F% Theta"] = 0  # Fallback if column is missing
                    return df

                # Apply function after calculating F_numeric
                intraday = calculate_f_theta(intraday, scale_factor=100)  # Adjust scale_factor if needed

                def detect_theta_spikes(df):
                    """
                    Identifies large spikes in F% Theta automatically using standard deviation.
                    - Uses 2.5x standard deviation as a dynamic threshold.
                    - Detects both positive and negative spikes.
                    """
                    if "F% Theta" not in df.columns:
                        return df  # Avoid crash if missing column

                    theta_std = df["F% Theta"].std()  # Compute stock-specific volatility
                    threshold = 2 * theta_std  # Set dynamic threshold

                    df["Theta_Change"] = df["F% Theta"].diff()  # Compute directional change
                    df["Theta_Spike"] = df["Theta_Change"].abs() > threshold  # Detect both up/down spikes

                    return df
                intraday = detect_theta_spikes(intraday)






                def detect_new_lows_below_yesterday_low(intraday):
                    """
                    Detects every new intraday low that is below Yesterday's Low F%.
                    Adds 🏊🏽‍♂️ emoji each time a new low is made below Yesterday Low.
                    """
                    intraday["Swimmer_Emoji"] = ""
                    y_low_f = intraday["Yesterday Low F%"].iloc[0]

                    lowest_so_far = y_low_f  # Start from yesterday's low

                    for i in range(1, len(intraday)):
                        curr_f = intraday["F_numeric"].iloc[i]

                        # Only if price is below yesterday's low
                        if curr_f < y_low_f:
                            # Check if it's a new low below the lowest recorded
                            if curr_f < lowest_so_far:
                                intraday.loc[intraday.index[i], "Swimmer_Emoji"] = "🏊🏽‍♂️"
                                lowest_so_far = curr_f  # Update lowest

                    return intraday

                intraday = detect_new_lows_below_yesterday_low(intraday)




                def detect_td_supply_cross_rooks(df):
                    """
                    Detect crosses above/below TD Supply Line and assign Rook emojis:
                    - ♖ for cross above
                    - ♜ for cross below
                    """
                    df["TD_Supply_Rook"] = ""

                    for i in range(1, len(df)):
                        prev_f = df.loc[i - 1, "F_numeric"]
                        curr_f = df.loc[i, "F_numeric"]
                        prev_supply = df.loc[i - 1, "TD Supply Line F"]
                        curr_supply = df.loc[i, "TD Supply Line F"]

                        if prev_f < prev_supply and curr_f >= curr_supply:
                            df.loc[i, "TD_Supply_Rook"] = "♖"  # Cross up → White rook

                        elif prev_f > prev_supply and curr_f <= curr_supply:
                            df.loc[i, "TD_Supply_Rook"] = "♜"  # Cross down → Black rook

                    return df

                intraday = detect_td_supply_cross_rooks(intraday)


                intraday["Event"] = None  # Initialize with None
                intraday.loc[42, "Event"] = "KING"  # Example: set a KING at index 42

                def mark_mike_check(intraday):
                    """
                    Adds a 'Check_Alert' column with '🔔' on the first TD Supply or TD Demand cross
                    in the opposite direction after a KING or QUEEN event.
                    Determines direction from colored Event names:
                    - bearish_events = ["RED_KING", "RED_QUEEN"] → look for next TD_Supply_Cross
                    - bullish_events = ["GREEN_KING", "GREEN_QUEEN"] → look for next TD_Demand_Cross
                    """
                    intraday = intraday.copy()
                    intraday["Check_Alert"] = None

                    bearish_events = ["RED_KING", "RED_QUEEN"]
                    bullish_events = ["GREEN_KING", "GREEN_QUEEN"]

                    # For each bearish KING/QUEEN, mark first subsequent TD Supply cross
                    for idx in intraday.index[intraday["Event"].isin(bearish_events)]:
                        sub = intraday.loc[idx + 1 :]
                        mask = sub["TD_Supply_Cross"] == True
                        if mask.any():
                            first_idx = mask.idxmax()
                            intraday.at[first_idx, "Check_Alert"] = "🔔"

                    # For each bullish KING/QUEEN, mark first subsequent TD Demand cross
                    for idx in intraday.index[intraday["Event"].isin(bullish_events)]:
                        sub = intraday.loc[idx + 1 :]
                        mask = sub["TD_Demand_Cross"] == True
                        if mask.any():
                            first_idx = mask.idxmax()
                            intraday.at[first_idx, "Check_Alert"] = "🔔"

                    return intraday

                # Usage:
                intraday = mark_mike_check(intraday)


                def detect_fortress_bee_clusters(df):
                    """
                    Detect clusters of 🐝 bees and add 🏰 fortress emoji only once per cluster.
                    """
                    df["Fortress_Emoji"] = ""
                    in_cluster = False

                    for i in range(10, len(df)):
                        recent_bees = (df["BBW_Tight_Emoji"].iloc[i-10:i] == "🐝").sum()

                        if recent_bees >= 3 and not in_cluster:
                            df.at[df.index[i], "Fortress_Emoji"] = "🏰"
                            in_cluster = True  # Once marked, wait for bees to disappear

                        elif recent_bees < 3:
                            in_cluster = False  # Reset: ready for next cluster

                    return df

                # Apply it
                intraday = detect_fortress_bee_clusters(intraday)
                def entryAlert(intraday, threshold=0.1, rvol_threshold=1.2, rvol_lookback=9):
                    """
                    Entry Alert System (Corrected):
                    - Step 1: Detect clean cross of F% through Kijun_F with buffer threshold.
                    - Step 2: Confirm with next bar continuation.
                    - Step 3: Require at least one RVOL_5 > threshold in the last rvol_lookback bars.
                    """

                    intraday["Entry_Alert_Short"] = False
                    intraday["Entry_Alert_Long"]  = False

                    for i in range(1, len(intraday) - 1):
                        prev_f = intraday.iloc[i-1]["F_numeric"]
                        prev_k = intraday.iloc[i-1]["Kijun_F"]
                        curr_f = intraday.iloc[i]["F_numeric"]
                        curr_k = intraday.iloc[i]["Kijun_F"]
                        next_f = intraday.iloc[i+1]["F_numeric"]

                        # ➡️ LONG CROSS
                        if (prev_f < prev_k - threshold) and (curr_f > curr_k + threshold):
                            if next_f >= curr_f:
                                intraday.at[intraday.index[i], "Entry_Alert_Long"] = True

                        # ⬅️ SHORT CROSS
                        if (prev_f > prev_k + threshold) and (curr_f < curr_k - threshold):
                            if next_f <= curr_f:
                                intraday.at[intraday.index[i], "Entry_Alert_Short"] = True

                    # 🔍 Second pass: check if at least one high RVOL_5
                    for i in range(1, len(intraday) - 1):
                        if intraday.iloc[i]["Entry_Alert_Long"] or intraday.iloc[i]["Entry_Alert_Short"]:
                            start_idx = max(0, i - rvol_lookback + 1)
                            rvol_window = intraday.iloc[start_idx:i+1]["RVOL_5"]

                            if (rvol_window > rvol_threshold).sum() == 0:
                                # 🛑 No bars with RVOL > threshold → kill alert
                                intraday.at[intraday.index[i], "Entry_Alert_Long"] = False
                                intraday.at[intraday.index[i], "Entry_Alert_Short"] = False

                    return intraday



                intraday = entryAlert(intraday, threshold=0.1)



                def entryAlertType2(intraday, threshold=0.1, rvol_threshold=1.2, rvol_lookforward=9):
                    """
                    Entry Type II (✅) - Delayed Volume Confirmation:
                    - Step 1: Detect Kijun crosses without immediate RVOL confirmation.
                    - Step 2: Look FORWARD up to rvol_lookforward bars to find RVOL > threshold.
                    - Step 3: If found, mark the first bar with volume as the alert bar.
                    """
                    intraday["Entry_Alert_Short_Delayed"] = False
                    intraday["Entry_Alert_Long_Delayed"] = False

                    for i in range(1, len(intraday) - 1):
                        prev_f = intraday.iloc[i-1]["F_numeric"]
                        prev_k = intraday.iloc[i-1]["Kijun_F"]
                        curr_f = intraday.iloc[i]["F_numeric"]
                        curr_k = intraday.iloc[i]["Kijun_F"]
                        curr_rvol = intraday.iloc[i]["RVOL_5"]

                        # --- Only proceed if no strong RVOL at the moment of cross
                        if curr_rvol > rvol_threshold:
                            continue

                        # --- LONG CROSS Candidate
                        if (prev_f < prev_k - threshold) and (curr_f > curr_k + threshold):
                            # Look forward for volume
                            forward_window = intraday.iloc[i+1 : i+1+rvol_lookforward]
                            rvol_found = forward_window[forward_window["RVOL_5"] > rvol_threshold]
                            if not rvol_found.empty:
                                first_idx = rvol_found.index[0]
                                intraday.at[first_idx, "Entry_Alert_Long_Delayed"] = True

                        # --- SHORT CROSS Candidate
                        if (prev_f > prev_k + threshold) and (curr_f < curr_k - threshold):
                            forward_window = intraday.iloc[i+1 : i+1+rvol_lookforward]
                            rvol_found = forward_window[forward_window["RVOL_5"] > rvol_threshold]
                            if not rvol_found.empty:
                                first_idx = rvol_found.index[0]
                                intraday.at[first_idx, "Entry_Alert_Short_Delayed"] = True

                    return intraday

                intraday = entryAlertType2(intraday, threshold=0.1)




                # def get_kijun_streak_log_with_dollar(df):
                #     """
                #     Returns list of Kijun streaks with their dollar change.
                #     Format: 'K+5 : $2.45', 'K-3 : $-1.20'
                #     """
                #     if "F_numeric" not in df.columns or "Kijun_F" not in df.columns or "Close" not in df.columns:
                #         return []

                #     streaks = []
                #     current_streak = 0
                #     current_state = None
                #     start_price = None

                #     for i in range(len(df)):
                #         f_val = df["F_numeric"].iloc[i]
                #         k_val = df["Kijun_F"].iloc[i]
                #         close_price = df["Close"].iloc[i]

                #         if pd.isna(f_val) or pd.isna(k_val) or pd.isna(close_price):
                #             continue

                #         is_above = f_val > k_val

                #         if current_state is None:
                #             current_state = is_above
                #             current_streak = 1
                #             start_price = close_price
                #         elif is_above == current_state:
                #             current_streak += 1
                #         else:
                #             end_price = df["Close"].iloc[i - 1]
                #             dollar_return = end_price - start_price
                #             label = f"K+{current_streak}" if current_state else f"K-{current_streak}"
                #             streaks.append(f"{label} : ${dollar_return:.2f}")
                #             current_state = is_above
                #             current_streak = 1
                #             start_price = close_price

                #     if current_streak > 0 and start_price is not None:
                #         end_price = df["Close"].iloc[-1]
                #         dollar_return = end_price - start_price
                #         label = f"K+{current_streak}" if current_state else f"K-{current_streak}"
                #         streaks.append(f"{label} : ${dollar_return:.2f}")

                #     return streaks



                # log_with_returns = get_kijun_streak_log_with_dollar(intraday)

                # st.markdown("### 📘 Full Kijun Streak Log with $ Returns:")
                # for line in log_with_returns:
                #     st.markdown(f"<div style='font-size:20px'>{line}</div>", unsafe_allow_html=True)


                # 1️⃣   compute θ on a lightly-smoothed F%
                intraday["F_smoothed"] = intraday["F_numeric"].ewm(span=3, adjust=False).mean()
                intraday["F_theta"]    = np.degrees(np.arctan(intraday["F_smoothed"].diff()))  # no scale factor

                # 2️⃣   detect spikes
                theta_std   = intraday["F_theta"].diff().abs().std()
                thr         = 2.5 * theta_std        # or 2.0 – experiment

                intraday["Theta_spike"] = intraday["F_theta"].diff().abs() > thr
                intraday["Theta_emoji"] = np.where(
                        intraday["Theta_spike"] & (intraday["F_theta"].diff() > 0), "🚡",
                        np.where(intraday["Theta_spike"] & (intraday["F_theta"].diff() < 0), "⚓️", "")
                )



                # Find the last swimmer (new low) row
                last_swimmer_idx = intraday[intraday["Swimmer_Emoji"] == "🏊🏽‍♂️"].index.max()

                # If there was at least one swimmer
                if pd.notna(last_swimmer_idx):
                    intraday.loc[last_swimmer_idx, "Swimmer_Emoji"] = "🦑"














                if gap_alert:
                    st.warning(gap_alert)


                from plotly.subplots import make_subplots
                import plotly.graph_objects as go



                # Calculate RVOL 5 (Relative Volume vs last 5 bars)
                intraday['RVOL_5'] = intraday['Volume'] / intraday['Volume'].rolling(5).mean()



                # UPPER WICK DETECTION (simple and lightweight)

                # Calculate candle parts
                intraday['Body'] = (intraday['Close'] - intraday['Open']).abs()
                intraday['UpperWick'] = intraday['High'] - intraday[['Close', 'Open']].max(axis=1)
                intraday['LowerWick'] = intraday[['Close', 'Open']].min(axis=1) - intraday['Low']

                # Define simple upper wick condition
                # Long upper wick if upper wick > 2x body
                intraday['LongUpperWick'] = intraday['UpperWick'] > 2.2 * intraday['Body']

                # Optional: Filter tiny bodies
                intraday['ValidBody'] = intraday['Body'] > 0

                # Final condition: big upper wick + valid body
                intraday['UpperWickFlag'] = intraday['LongUpperWick'] & intraday['ValidBody']

                # with st.expander("🕯️ Hidden Candlestick + Ichimoku View", expanded=True):
                #     # ❶ create a 2-row figure (70 % price, 30 % volume)
                #     fig = make_subplots(rows=2, cols=1,
                #                         shared_xaxes=True,
                #                         vertical_spacing=0.03,
                #                         row_heights=[0.7, 0.3])

                #     # ❷ ── price & Ichimoku (row 1) ──────────────────────────────────────
                #     fig.add_trace(go.Candlestick(
                #         x=intraday['Time'],
                #         open=intraday['Open'],
                #         high=intraday['High'],
                #         low=intraday['Low'],
                #         close=intraday['Close'],
                #         name='Candles'),
                #         row=1, col=1)

                #     fig.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Tenkan'],
                #                             line=dict(color='red'), name='Tenkan-sen'),
                #                 row=1, col=1)
                #     fig.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Kijun'],
                #                             line=dict(color='green'), name='Kijun-sen'),
                #                 row=1, col=1)
                #     fig.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanA'],
                #                             line=dict(color='yellow'), name='Span A'),
                #                 row=1, col=1)
                #     fig.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanB'],
                #                             line=dict(color='blue'), name='Span B'),
                #                 row=1, col=1)
                #     fig.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Chikou'],
                #                             line=dict(color='purple'), name='Chikou'),
                #                 row=1, col=1)

                #     # cloud
                #     fig.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanA'],
                #                             line=dict(width=0), showlegend=False),
                #                 row=1, col=1)
                #     fig.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanB'],
                #                             fill='tonexty',
                #                             fillcolor='rgba(128,128,128,0.2)',
                #                             line=dict(width=0),
                #                             showlegend=False),
                #                 row=1, col=1)

                #     # ❸ ── volume bars (row 2) ──────────────────────────────────────────
                #     fig.add_trace(go.Bar(
                #         x=intraday['Time'],
                #         y=intraday['Volume'],
                #         marker=dict(opacity=0.5),
                #         name='Volume'),
                #         row=2, col=1)

                #     # ❹ layout tweaks
                #     fig.update_layout(
                #         title="Ichimoku Candlestick Chart + Volume",
                #         xaxis_rangeslider_visible=False,  # rangeslider only once
                #         height=600,
                #         margin=dict(l=30, r=30, t=40, b=20)
                #     )
                #     if 'RVOL_5' in intraday.columns:
                #         # Initialize no markers first
                #         intraday['MarkerColor'] = None
                #         intraday['MarkerLabel'] = None

                #         # Prioritize higher RVOL first
                #         for thr, color, label in [(1.8, '#ff0000', 'RVOL ≥ 1.8'),
                #                                 (1.5, '#ffd700', 'RVOL ≥ 1.5'),
                #                                 (1.2, '#ff69b4', 'RVOL ≥ 1.2')]:
                #             mask = (intraday['RVOL_5'] >= thr) & (intraday['MarkerColor'].isna())
                #             intraday.loc[mask, 'MarkerColor'] = color
                #             intraday.loc[mask, 'MarkerLabel'] = label

                #         # Now plot only once per bar
                #         mask_final = intraday['MarkerColor'].notna()
                #         fig.add_trace(go.Scatter(
                #             x=intraday.loc[mask_final, 'Time'],
                #             y=intraday.loc[mask_final, 'Volume'],
                #             mode='markers',
                #             marker=dict(
                #                 symbol='triangle-up',
                #                 size=10,
                #                 color=intraday.loc[mask_final, 'MarkerColor']
                #             ),
                #             name='RVOL Marker',
                #             customdata=intraday.loc[mask_final, 'RVOL_5'],
                #             hovertemplate='%{x}<br>Vol: %{y}<br>RVOL_5: %{customdata:.2f}',
                #         ), row=2, col=1)




                #     # Plot an emoji for detected upper wicks
                #     intraday['UpperWickEmoji'] = intraday['UpperWickFlag'].apply(lambda x: "♟️" if x else "")

                #     mask_upperwick = intraday['UpperWickEmoji'] != ""

                #     fig.add_trace(go.Scatter(
                #         x=intraday.loc[mask_upperwick, 'Time'],
                #         y=intraday.loc[mask_upperwick, 'High'],
                #         mode='text',
                #         text=intraday.loc[mask_upperwick, 'UpperWickEmoji'],
                #         textposition='top center',
                #         textfont=dict(size=21),
                #         showlegend=False
                #     ), row=1, col=1)


                #     st.plotly_chart(fig, use_container_width=True)



                    #"Time", "Close","Opening Price Signal","OPS Transition","Net Price","Net Price Direction","Bounce","Retrace","BBW_Tight_Emoji",
                    #"RVOL_5","F% BBW", "Day Type", "High of Day",
                    #"Low of Day", "F%","TD Open","TD Trap","TD CLoP", "40ish","Tenkan_Kijun_Cross",
                    #"CTOD Alert","Alert_Kijun","Alert_Mid","Wealth Signal","BBW Alert"


                # with st.expander("Show/Hide Data Table",  expanded=False):
                #                 # Show data table, including new columns
                #     cols_to_show = [
                #                     "Time","RVOL_5","RVOL_Alert","F% Theta", "BBW_Tight_Emoji","BBW Alert","tdSupplyCrossalert", "Kijun_F_Cross","ADX_Alert","STD_Alert","ATR_Exp_Alert","Tenkan_Kijun_Cross","KumoTwistAlert"
                #                 ]

                #     st.dataframe(intraday[cols_to_show])

                ticker_tabs = st.tabs(["Interactive F% & Momentum", "Intraday Data Table"])

                with ticker_tabs[0]:
                    # -- Create Subplots: Row1=F%, Row2=Momentum
                    fig = make_subplots(
                        rows=1,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[30]  # Adjust row heights to your preference
                    )

                    # (A) F% over time as lines+markers
                    # ---------------------------------
                    max_abs_val = intraday["F_numeric"].abs().max()
                    scatter_f = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F_numeric"],
                        mode="lines+markers",
                        customdata=intraday["Close"],

                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Close: $%{customdata:.2f}<extra></extra>",

                        name="F% (scaled)",

                    )
                    fig.add_trace(scatter_f, row=1, col=1)

                    # (A.1) 40ish Reversal (star markers)
                    mask_40ish = intraday["40ish"] != ""
                    scatter_40ish = go.Scatter(
                        x=intraday.loc[mask_40ish, "Time"],
                        y=intraday.loc[mask_40ish, "F_numeric"] +22,
                        mode="markers",
                        marker_symbol="star",
                        marker_size=18,
                        marker_color="gold",
                        name="40ish Reversal",
                        text=intraday.loc[mask_40ish, "40ish"],

                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )
                    fig.add_trace(scatter_40ish, row=1, col=1)

                    # (A.2) Dashed horizontal line at 0
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        row=1, col=1,
                        annotation_text="0%",
                        annotation_position="top left"
                    )



                    intraday["Tenkan"] = (intraday["High"].rolling(window=9).max() + intraday["Low"].rolling(window=9).min()) / 2
                    intraday["Kijun"] = (intraday["High"].rolling(window=26).max() + intraday["Low"].rolling(window=26).min()) / 2
                    intraday["SpanA"] = ((intraday["Tenkan"] + intraday["Kijun"]) / 2)
                    intraday["SpanB"] = (intraday["High"].rolling(window=52).max() + intraday["Low"].rolling(window=52).min()) / 2
                    # Fill early NaNs so cloud appears fully from 9:30 AM
                    intraday["SpanA"] = intraday["SpanA"].bfill()
                    intraday["SpanB"] = intraday["SpanB"].bfill()

                    intraday["SpanA_F"] = ((intraday["SpanA"] - prev_close) / prev_close) * 10000
                    intraday["SpanB_F"] = ((intraday["SpanB"] - prev_close) / prev_close) * 10000

                    # Fill again after F%-conversion to guarantee values exist
                    intraday["SpanA_F"] = intraday["SpanA_F"].bfill()
                    intraday["SpanB_F"] = intraday["SpanB_F"].bfill()


                    intraday["Chikou"] = intraday["Close"].shift(-26)


                    # Chikou moved ABOVE price (🕵🏻‍♂️) — signal at time when it actually happened
                    chikou_above_mask = (intraday["Chikou"] > intraday["Close"]).shift(26)
                    chikou_above = intraday[chikou_above_mask.fillna(False)]

                    # Chikou moved BELOW price (👮🏻‍♂️)
                    chikou_below_mask = (intraday["Chikou"] < intraday["Close"]).shift(26)
                    chikou_below = intraday[chikou_below_mask.fillna(False)]



                   # Calculate Chikou (lagging span) using Close price shifted BACKWARD
                    intraday["Chikou"] = intraday["Close"].shift(-26)

                    # Calculate Chikou_F using shifted price, keeping Time as-is
                    intraday["Chikou_F"] = ((intraday["Chikou"] - prev_close) / prev_close) * 10000

                    # Drop rows where Chikou_F is NaN (due to shifting)
                    chikou_plot = intraday.dropna(subset=["Chikou_F"])

                    # Plot without shifting time
                    chikou_line = go.Scatter(
                        x=chikou_plot["Time"],
                        y=chikou_plot["Chikou_F"],
                        mode="lines",
                        name="Chikou (F%)",
                        line=dict(color="purple", dash="dash")
                    )
                    fig.add_trace(chikou_line, row=1, col=1)


 # 🟢   SPAN A & SPAN B




                    intraday["SpanA_F"] = ((intraday["SpanA"] - prev_close) / prev_close) * 10000
                    intraday["SpanB_F"] = ((intraday["SpanB"] - prev_close) / prev_close) * 10000



                                        # Span A – Yellow Line
                    span_a_line = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["SpanA_F"],
                        mode="lines",
                        line=dict(color="yellow", width=2),
                        name="Span A (F%)"
                    )
                    fig.add_trace(span_a_line, row=1, col=1)

                    # Span B – Blue Line
                    span_b_line = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["SpanB_F"],
                        mode="lines",
                        line=dict(color="blue", width=2),
                        name="Span B (F%)"
                    )
                    fig.add_trace(span_b_line, row=1, col=1)

                    # Invisible SpanA for cloud base
                    fig.add_trace(go.Scatter(
                        x=intraday["Time"],
                        y=intraday["SpanA_F"],
                        line=dict(width=0),
                        mode='lines',
                        showlegend=False
                    ), row=1, col=1)

                    # SpanB with fill → grey Kumo
                    fig.add_trace(go.Scatter(
                        x=intraday["Time"],
                        y=intraday["SpanB_F"],
                        fill='tonexty',
                        fillcolor='rgba(128, 128, 128, 0.25)',  # transparent grey
                        line=dict(width=0),
                        mode='lines',
                        name='Kumo Cloud'
                    ), row=1, col=1)



#                     # Kumo Twist Bullish (👼🏼): SpanA crosses above SpanB
#                     twist_bullish = (intraday["SpanA_F"] > intraday["SpanB_F"]) & (intraday["SpanA_F"].shift(1) <= intraday["SpanB_F"].shift(1))

#                     scatter_twist_bullish = go.Scatter(
#                         x=intraday.loc[twist_bullish, "Time"],
#                         y=intraday.loc[twist_bullish, "SpanA_F"] + 89,
#                         mode="text",
#                         text=["👼🏼"] * twist_bullish.sum(),
#                         textposition="top center",
#                         textfont=dict(size=34),
#                         name="Kumo Twist Bullish (👼🏼)",
#                         hovertemplate="Time: %{x}<br>👼🏼 SpanA crossed above SpanB<extra></extra>"
#                     )

#                     # Kumo Twist Bearish (👺): SpanA crosses below SpanB
#                     twist_bearish = (intraday["SpanA_F"] < intraday["SpanB_F"]) & (intraday["SpanA_F"].shift(1) >= intraday["SpanB_F"].shift(1))

#                     scatter_twist_bearish = go.Scatter(
#                         x=intraday.loc[twist_bearish, "Time"],
#                         y=intraday.loc[twist_bearish, "SpanA_F"] - 89,
#                         mode="text",
#                         text=["👺"] * twist_bearish.sum(),
#                         textposition="bottom center",
#                         textfont=dict(size=34),
#                         name="Kumo Twist Bearish (👺)",
#                         hovertemplate="Time: %{x}<br>👺 SpanA crossed below SpanB<extra></extra>"
#                     )

#                     # Add to the F% plot
#                     fig.add_trace(scatter_twist_bullish, row=1, col=1)
#                     fig.add_trace(scatter_twist_bearish, row=1, col=1)




#                                     # Calculate Chikou relation to current price
#                     intraday["Chikou_Position"] = np.where(intraday["Chikou"] > intraday["Close"], "above",
#                                                 np.where(intraday["Chikou"] < intraday["Close"], "below", "equal"))

#                     # Detect changes in Chikou relation
#                     intraday["Chikou_Change"] = intraday["Chikou_Position"].ne(intraday["Chikou_Position"].shift())

#                     # Filter first occurrence and changes
#                     chikou_shift_mask = intraday["Chikou_Change"] & (intraday["Chikou_Position"] != "equal")

#                     # Assign emojis for only these changes
#                     intraday["Chikou_Emoji"] = np.where(intraday["Chikou_Position"] == "above", "👨🏻‍✈️",
#                                                 np.where(intraday["Chikou_Position"] == "below", "👮🏻‍♂️", ""))

#                     mask_chikou_above = chikou_shift_mask & (intraday["Chikou_Position"] == "above")

#                     fig.add_trace(go.Scatter(
#                         x=intraday.loc[mask_chikou_above, "Time"],
#                         y=intraday.loc[mask_chikou_above, "F_numeric"] + 55,
#                         mode="text",
#                         text=["👨🏻‍✈️"] * mask_chikou_above.sum(),
#                         textposition="top center",
#                         textfont=dict(size=34),
#                         name="Chikou Above Price",
#                         hovertemplate="Time: %{x}<br>F%: %{y}<br>Chikou moved above<extra></extra>"
#                     ), row=1, col=1)

#                     mask_chikou_below = chikou_shift_mask & (intraday["Chikou_Position"] == "below")

#                     fig.add_trace(go.Scatter(
#                         x=intraday.loc[mask_chikou_below, "Time"],
#                         y=intraday.loc[mask_chikou_below, "F_numeric"] - 55,
#                         mode="text",
#                         text=["👮🏿‍♂️"] * mask_chikou_below.sum(),
#                         textposition="bottom center",
#                         textfont=dict(size=34),
#                         name="Chikou Below Price",
#                         hovertemplate="Time: %{x}<br>F%: %{y}<br>Chikou moved below<extra></extra>"
#                     ), row=1, col=1)



                    kijun_line = go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Kijun_F"],
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Kijun (F% scale)"
                )
                    fig.add_trace(kijun_line, row=1, col=1)




#                     # (C) CTOD Buy/Sell Triggers (Red & Green Dots)
#                     # ----------------------------------------------
#                     mask_ctod_buy = intraday["CTOD Alert"] == "Buy"
#                     mask_ctod_sell = intraday["CTOD Alert"] == "Sell"

#                     # Buy Alert (Green Dot)
#                     scatter_ctod_buy = go.Scatter(
#                         x=intraday.loc[mask_ctod_buy, "Time"],
#                         y=intraday.loc[mask_ctod_buy, "F_numeric"],
#                         mode="markers",
#                         marker=dict(symbol="circle", size=5, color="green"),
#                         name="CTOD Buy Signal",
#                         text="Buy Triggered",
#                         hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
#                     )
#                     fig.add_trace(scatter_ctod_buy, row=1, col=1)

#                     # Sell Alert (Red Dot)
#                     scatter_ctod_sell = go.Scatter(
#                         x=intraday.loc[mask_ctod_sell, "Time"],
#                         y=intraday.loc[mask_ctod_sell, "F_numeric"],
#                         mode="markers",
#                         marker=dict(symbol="circle", size=5, color="red"),
#                         name="CTOD Sell Signal",
#                         text="Sell Triggered",
#                         hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
#                     )
#                     fig.add_trace(scatter_ctod_sell, row=1, col=1)




                                    # Mask for different RVOL thresholds
                    mask_rvol_extreme = intraday["RVOL_5"] > 1.8
                    mask_rvol_strong = (intraday["RVOL_5"] >= 1.5) & (intraday["RVOL_5"] < 1.8)
                    mask_rvol_moderate = (intraday["RVOL_5"] >= 1.2) & (intraday["RVOL_5"] < 1.5)

                    # Scatter plot for extreme volume spikes (red triangle)
                    scatter_rvol_extreme = go.Scatter(
                        x=intraday.loc[mask_rvol_extreme, "Time"],
                        y=intraday.loc[mask_rvol_extreme, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=8, color="red"),
                        name="RVOL > 1.8 (Extreme Surge)",
                        text="Extreme Volume",

                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for strong volume spikes (orange triangle)
                    scatter_rvol_strong = go.Scatter(
                        x=intraday.loc[mask_rvol_strong, "Time"],
                        y=intraday.loc[mask_rvol_strong, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=8, color="orange"),
                        name="RVOL 1.5-1.79 (Strong Surge)",
                        text="Strong Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for moderate volume spikes (pink triangle)
                    scatter_rvol_moderate = go.Scatter(
                        x=intraday.loc[mask_rvol_moderate, "Time"],
                        y=intraday.loc[mask_rvol_moderate, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=8, color="pink"),
                        name="RVOL 1.2-1.49 (Moderate Surge)",
                        text="Moderate Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Add to the F% plot (Row 1)
                    fig.add_trace(scatter_rvol_extreme, row=1, col=1)
                    fig.add_trace(scatter_rvol_strong, row=1, col=1)
                    fig.add_trace(scatter_rvol_moderate, row=1, col=1)










                    # (B) Upper Band
                    upper_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Upper"],
                        mode="lines",
                        line=dict(dash="solid", color="grey"),
                        name="Upper Band"
                    )

                    # (C) Lower Band
                    lower_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Lower"],
                        mode="lines",
                        line=dict(dash="solid", color="grey"),
                        name="Lower Band"
                    )

                    # (D) Moving Average (Middle Band)
                    middle_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% MA"],
                        mode="lines",
                        line=dict(color="white", dash="dash"),  # Set dash style
                        name="Middle Band (14-MA)"
                    )

                    # Add all traces

                    fig.add_trace(upper_band, row=1, col=1)
                    fig.add_trace(lower_band, row=1, col=1)
                    fig.add_trace(middle_band, row=1, col=1)



                    tenkan_line = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["Tenkan_F"],
                        mode="lines",
                        line=dict(color="red", width=2, dash="dot"),
                        name="Tenkan (F%)"
                    )
                    fig.add_trace(tenkan_line, row=1, col=1)


# # ------------------------------------------------------------------------------------------------------------------------


                    # Mask for Tenkan-Kijun Crosses
                    mask_tk_sun = intraday["Tenkan_Kijun_Cross"] == "🌞"
                    mask_tk_moon = intraday["Tenkan_Kijun_Cross"] == "🌙"

                    # 🌞 Bullish Tenkan-Kijun Cross (Sun Emoji)
                    scatter_tk_sun = go.Scatter(
                        x=intraday.loc[mask_tk_sun, "Time"],
                        y=intraday.loc[mask_tk_sun, "F_numeric"] + 89,  # Offset for visibility
                        mode="text",
                        text="🌞",
                        textposition="top center",
                        textfont=dict(size=34),
                        name="Tenkan-Kijun Bullish Cross",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Above Kijun<extra></extra>"
                    )

                    # 🌙 Bearish Tenkan-Kijun Cross (Moon Emoji)
                    scatter_tk_moon = go.Scatter(
                        x=intraday.loc[mask_tk_moon, "Time"],
                        y=intraday.loc[mask_tk_moon, "F_numeric"] - 89,  # Offset for visibility
                        mode="text",
                        text="🌙",
                        textposition="bottom center",
                        textfont=dict(size=34),
                        name="Tenkan-Kijun Bearish Cross",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Below Kijun<extra></extra>"
                    )

                    # Add to the F% Plot
                    fig.add_trace(scatter_tk_sun, row=1, col=1)
                    fig.add_trace(scatter_tk_moon, row=1, col=1)

                    # ✅ Yesterday's Open - Grey Dashed Line (F% Scale)
                    y_open_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday Open F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="grey", dash="dash"),
                        name="Yesterday Open (F%)"
                    )

                    # ✅ Yesterday's High - Blue Dashed Line (F% Scale)
                    y_high_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday High F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="green", dash="dash"),
                        name="Yesterday High (F%)"
                    )

                    # ✅ Yesterday's Low - Green Dashed Line (F% Scale)
                    y_low_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday Low F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Yesterday Low (F%)"
                    )

                    # ✅ Yesterday's Close - Red Dashed Line (F% Scale) (Always at 0)
                    y_close_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[0] * len(intraday),
                        mode="lines",
                        line=dict(color="blue", dash="dash"),
                        name="Yesterday Close (F%)"
                    )


#                   # (D) TD Trap Arrows - Only First Sell TD Trap
#                     # ----------------------------------------------
#                     td_trap_mask = intraday["TD Trap"].str.contains("Sell", na=False)  # Only Sell TD Traps

#                     if not td_trap_mask.empty and td_trap_mask.any():
#                         first_sell_trap_idx = td_trap_mask.idxmax()  # Get first occurrence index
#                         first_sell_trap_time = intraday.loc[first_sell_trap_idx, "Time"]
#                         first_sell_trap_value = intraday.loc[first_sell_trap_idx, "F_numeric"]

#                         # Add annotation for first Sell TD Trap (Short = ST)
#                         fig.add_annotation(
#                             x=first_sell_trap_time,
#                             y=first_sell_trap_value - 10,  # Offset to avoid overlap
#                             text="ST",  # Short label instead of full text
#                             showarrow=True,
#                             arrowhead=2,
#                             arrowsize=1.5,
#                             arrowcolor="yellow",
#                             font=dict(size=12, color="red", family="Arial Black"),
#                         )

#                     fig.update_xaxes(title_text="Time", row=1, col=1)
#                     fig.update_yaxes(title_text="F% Scale", row=1, col=1)



           # BBW Tight Compression (🐝) – 3 of last 5 bars under threshold
                    mask_bbw_tight = intraday["BBW_Tight_Emoji"] == "🐝"
                    scatter_bbw_tight = go.Scatter(
                        x=intraday.loc[mask_bbw_tight, "Time"],
                        y=intraday.loc[mask_bbw_tight, "F_numeric"] + 13,  # Offset upward
                        mode="text",
                        text=["🐝"] * mask_bbw_tight.sum(),
                        textposition="top center",
                        textfont=dict(size=21),
                        name="BBW Tight Compression (🐝)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>BBW Tight Compression (🐝)<extra></extra>"
                    )

                    fig.add_trace(scatter_bbw_tight, row=1, col=1)




 # 🟢 BBW Expansion

                    mask_bbw_alert = intraday["BBW Alert"] != ""

                    scatter_bbw_alert = go.Scatter(
                        x=intraday.loc[mask_bbw_alert, "Time"],
                        y=intraday.loc[mask_bbw_alert, "F_numeric"] + 8,  # Offset above F%
                        mode="text",
                        text=intraday.loc[mask_bbw_alert, "BBW Alert"],
                        textposition="top center",
                        textfont=dict(size=13),
                        name="BBW Expansion Alert",
                        hovertemplate="Time: %{x}<br>BBW Ratio: %{customdata:.2f}<extra></extra>",
                        customdata=intraday.loc[mask_bbw_alert, "BBW_Ratio"]
                    )

                    fig.add_trace(scatter_bbw_alert, row=1, col=1)


 #  🟢 ADX Expansion


          # Mask for ADX Alerts (♨️, 🧨)
                    mask_adx_alert = intraday["ADX_Alert"] != ""

                    scatter_adx_alert = go.Scatter(
                        x=intraday.loc[mask_adx_alert, "Time"],
                        y=intraday.loc[mask_adx_alert, "F_numeric"] + 55,  # Offset for visibility
                        mode="text",
                        text=intraday.loc[mask_adx_alert, "ADX_Alert"],
                        textposition="bottom center",
                        textfont=dict(size=13),
                        name="ADX Expansion Alert",
                        hovertemplate="Time: %{x}<br>ADX Ratio: %{customdata:.2f}<extra></extra>",
                        customdata=intraday.loc[mask_adx_alert, "ADX_Ratio"]
                    )

                    fig.add_trace(scatter_adx_alert, row=1, col=1)



# 🟢  STD Expansion  (🐦‍🔥)
                    mask_std_alert = intraday["STD_Alert"] != ""

                    scatter_std_alert = go.Scatter(
                        x=intraday.loc[mask_std_alert, "Time"],
                        y=intraday.loc[mask_std_alert, "F_numeric"] + 55,  # Offset above F%
                        mode="text",
                        text=intraday.loc[mask_std_alert, "STD_Alert"],
                        textposition="top center",
                        textfont=dict(size=21),
                        name="F% STD Expansion",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>STD Alert: %{text}<extra></extra>"
                    )

                    fig.add_trace(scatter_std_alert, row=1, col=1)

#  🟢   ATR Expansion
                    mask_atr_alert = intraday["ATR_Exp_Alert"] != ""

                    atr_alert_scatter = go.Scatter(
                        x=intraday.loc[mask_atr_alert, "Time"],
                        y=intraday.loc[mask_atr_alert, "F_numeric"]  + 34,  # place below F%
                        mode="text",
                        text=intraday.loc[mask_atr_alert, "ATR_Exp_Alert"],
                        textfont=dict(size=21),
                        name="ATR Expansion",
                        hoverinfo="text",
                        hovertext=intraday.loc[mask_atr_alert, "ATR_Exp_Alert"]
                    )

                    fig.add_trace(atr_alert_scatter, row=1, col=1)

# 🟢 TD SUPPLY

          # 🟤 TD Supply Line (F%)
                    fig.add_trace(
                        go.Scatter(
                            x=intraday['Time'],
                            y=intraday['TD Supply Line F'],
                            mode='lines',
                            line=dict(width=1, color="#a9a9a9", dash='dot'),
                            name='TD Supply F%',
                            hovertemplate="Time: %{x}<br>Supply (F%): %{y:.2f}"
                        ),
                        row=1, col=1
                    )



#🟢 TD DEMAND


                    # 🔵 TD Demand Line (F%)
                    fig.add_trace(
                        go.Scatter(
                            x=intraday['Time'],
                            y=intraday['TD Demand Line F'],
                            mode='lines',
                            line=dict(width=1, color="DarkSeaGreen", dash='dot'),
                            name='TD Demand F%',
                            hovertemplate="Time: %{x}<br>Demand (F%): %{y:.2f}"
                        ),
                        row=1, col=1
                    )





               # Extract only the rows where TDST just formed
                    tdst_points = intraday["TDST"].notna()

                    tdst_buy_mask = intraday["TDST"].str.contains("Buy TDST", na=False)
                    tdst_sell_mask = intraday["TDST"].str.contains("Sell TDST", na=False)



                    tdst_buy_mask = intraday["TDST"].str.contains("Buy TDST", na=False)
                    tdst_sell_mask = intraday["TDST"].str.contains("Sell TDST", na=False)


                    # Buy TDST marker (⎯)
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[tdst_buy_mask, "Time"],
                            y=intraday.loc[tdst_buy_mask, "F_numeric"],
                            mode="text",
                            text=["⎯"] * tdst_buy_mask.sum(),
                            textposition="middle center",
                            textfont=dict(size=55, color="green"),
                            name="Buy TDST",
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                        ),
                        row=1, col=1
                    )

                    # Sell TDST marker (⎯)
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[tdst_sell_mask, "Time"],
                            y=intraday.loc[tdst_sell_mask, "F_numeric"],
                            mode="text",
                            text=["⎯"] * tdst_sell_mask.sum(),
                            textposition="middle center",
                            textfont=dict(size=55, color="red"),
                            name="Sell TDST",
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                        ),
                        row=1, col=1
                    )




                    cloud_mask = intraday["Heaven_Cloud"] == "☁️"

                    fig.add_trace(go.Scatter(
                        x=intraday.loc[cloud_mask, "Time"],
                        y=intraday.loc[cloud_mask, "F_numeric"] + 89,
                        mode="text",
                        text=intraday.loc[cloud_mask, "Heaven_Cloud"],
                        textposition="top center",
                        textfont=dict(size=34),
                        name="Heaven ☁️",
                        hovertemplate="Time: %{x}<br>Price above TD Supply Line<extra></extra>"
                    ), row=1, col=1)

                    # Generate continuous 🌧️ drizzle emojis while F% is below TD Demand Line F
                    intraday["Drizzle_Emoji"] = None
                    below_demand = False

                    for i in range(1, len(intraday)):
                        f = intraday["F_numeric"].iloc[i]
                        demand = intraday["TD Demand Line F"].iloc[i]

                        if pd.notna(demand) and f < demand:
                            below_demand = True
                        elif pd.notna(demand) and f >= demand:
                            below_demand = False

                        if below_demand:
                            intraday.at[intraday.index[i], "Drizzle_Emoji"] = "🌧️"



                    # Plot 🌧️ Drizzle Emoji on F% chart when price crosses down TD Demand Line
                    drizzle_mask = intraday["Drizzle_Emoji"] == "🌧️"

                    fig.add_trace(go.Scatter(
                        x=intraday.loc[drizzle_mask, "Time"],
                        y=intraday.loc[drizzle_mask, "F_numeric"] + 89,  # Position below the bar
                        mode="text",
                        text=intraday.loc[drizzle_mask, "Drizzle_Emoji"],
                        textposition="bottom center",
                        textfont=dict(size=32),
                        name="Price Dropped Below Demand 🌧️",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Crossed Below Demand<extra></extra>"
                    ), row=1, col=1)




                    # 🪫 Emoji at LOD (Low of Day)
                    lod_index = intraday["Low"].idxmin()  # Find the index of the lowest low
                    lod_time = intraday.loc[lod_index, "Time"]
                    lod_value = intraday.loc[lod_index, "F_numeric"]

                    fig.add_trace(go.Scatter(
                        x=[lod_time],
                        y=[lod_value - 55],  # offset below the actual low
                        mode="text",
                        text=["🪫"],
                        textposition="bottom center",
                        textfont=dict(size=21),
                        name="Low of Day (🪫)",
                        hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                    ))

                    # 🔋 Emoji at HOD (High of Day)
                    hod_index = intraday["High"].idxmax()  # Find the index of the highest high
                    hod_time = intraday.loc[hod_index, "Time"]
                    hod_value = intraday.loc[hod_index, "F_numeric"]

                    fig.add_trace(go.Scatter(
                        x=[hod_time],
                        y=[hod_value + 55],  # offset above the actual high
                        mode="text",
                        text=["🔋"],
                        textposition="top center",
                        textfont=dict(size=21),
                        name="High of Day (🔋)",
                        hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                    ))


                    intraday["F_shift"] = intraday["F_numeric"].shift(1)

#                     tdst_buy_mask = intraday["TDST"].str.contains("Buy TDST", na=False)
#                     tdst_sell_mask = intraday["TDST"].str.contains("Sell TDST", na=False)

#                                         # Step 1: For each Buy TDST bar, get the F% level
#                     buy_tdst_levels = intraday.loc[tdst_buy_mask, "F_numeric"]

#                     # Step 2: Loop through each Buy TDST and track from that point forward
#                     for buy_idx, tdst_level in buy_tdst_levels.items():
#                         # Get index location of the TDST signal
#                         i = intraday.index.get_loc(buy_idx)

#                         # Look at all bars forward from the TDST bar
#                         future = intraday.iloc[i+1:].copy()

#                         # Find where F% crosses and stays above the TDST level for 2 bars
#                         above = future["F_numeric"] > tdst_level
#                         two_bar_hold = above & above.shift(-1)

#                         # Find the first time this happens
#                         if two_bar_hold.any():
#                             ghost_idx = two_bar_hold[two_bar_hold].index[0]  # first valid bar

#                             # Plot 👻 emoji on the first bar
#                             fig.add_trace(
#                                 go.Scatter(
#                                     x=[intraday.at[ghost_idx, "Time"]],
#                                     y=[intraday.at[ghost_idx, "F_numeric"] + 144],
#                                     mode="text",
#                                     text=["👻"],
#                                     textposition="middle center",
#                                     textfont=dict(size=40, color="purple"),
#                                     name="Confirmed Buy TDST Breakout",
#                                     hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
#                                 ),
#                                 row=1, col=1
#                             )


#                     # Step 1: Get all Sell TDST points (each defines its own world)
#                     sell_tdst_levels = intraday.loc[tdst_sell_mask, "F_numeric"]
#                     sell_tdst_indices = list(sell_tdst_levels.index) + [intraday.index[-1]]  # add end of session as last boundary

#                     # Step 2: Loop through each Sell TDST "world"
#                     for i in range(len(sell_tdst_levels)):
#                         tdst_idx = sell_tdst_levels.index[i]
#                         tdst_level = sell_tdst_levels.iloc[i]

#                         # Define the domain: from this Sell TDST until the next one (or end of day)
#                         domain_start = intraday.index.get_loc(tdst_idx) + 1
#                         domain_end = intraday.index.get_loc(sell_tdst_indices[i+1])  # next TDST or end

#                         domain = intraday.iloc[domain_start:domain_end]

#                         # Condition: F% crosses below and stays below for 2 bars
#                         below = domain["F_numeric"] < tdst_level
#                         confirmed = below & below.shift(-1)

#                         if confirmed.any():
#                             ghost_idx = confirmed[confirmed].index[0]
#                             fig.add_trace(
#                                 go.Scatter(
#                                     x=[intraday.at[ghost_idx, "Time"]],
#                                     y=[intraday.at[ghost_idx, "F_numeric"] - 144],
#                                     mode="text",
#                                     text=["🫥"],
#                                     textposition="middle center",
#                                     textfont=dict(size=40, color="purple"),
#                                     name="Confirmed Sell TDST Breakdown",
#                                     hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
#                                 ),
#                                 row=1, col=1
#                             )



#                     # ✈️ Upward Crosses ABOVE Yesterday High
#                     up_high_mask = intraday["Y_High_Cross"] == "✈️"
#                     up_high_trace = go.Scatter(
#                         x=intraday.loc[up_high_mask, "Time"],
#                         y=intraday.loc[up_high_mask, "F_numeric"] + 144,
#                         mode="text",
#                         text=intraday.loc[up_high_mask, "Y_High_Cross"],
#                         textposition="top center",
#                         textfont=dict(size=55),
#                         name="Cross Above Y-High (✈️)"
#                     )

#                     # 🪂 Downward Crosses BELOW Yesterday High
#                     down_high_mask = intraday["Y_High_Cross"] == "🪂"
#                     down_high_trace = go.Scatter(
#                         x=intraday.loc[down_high_mask, "Time"],
#                         y=intraday.loc[down_high_mask, "F_numeric"] - 144,
#                         mode="text",
#                         text=intraday.loc[down_high_mask, "Y_High_Cross"],
#                         textposition="bottom center",
#                         textfont=dict(size=55),
#                         name="Cross Below Y-High (🪂)"
#                     )

#                     # Add to figure
#                     fig.add_trace(up_high_trace, row=1, col=1)
#                     fig.add_trace(down_high_trace, row=1, col=1)







                    astronaut_points = intraday[intraday["Astronaut_Emoji"] == "👨🏽‍🚀"]

                    scatter_astronaut = go.Scatter(
                        x=astronaut_points["Time"],
                        y=astronaut_points["F_numeric"] + 188,  # Higher offset
                        mode="text",
                        text=astronaut_points["Astronaut_Emoji"],
                        textposition="top center",
                        name="New Highs 👨🏽‍🚀",
                        textfont=dict(size=34),
                        showlegend=False
                    )

                    fig.add_trace(scatter_astronaut, row=1, col=1)



                    # Filter where the Astronaut or Moon emoji exist
                    astronaut_points = intraday[intraday["Astronaut_Emoji"] != ""]

                    scatter_astronaut = go.Scatter(
                        x=astronaut_points["Time"],
                        y=astronaut_points["F_numeric"] + 188,  # Offset so it floats higher
                        mode="text",
                        text=astronaut_points["Astronaut_Emoji"],  # Either 👨🏽‍🚀 or 🌒
                        textposition="top center",
                        name="New Highs 🌒",
                        textfont=dict(size=34),
                        showlegend=False
                    )

                    fig.add_trace(scatter_astronaut, row=1, col=1)



                    # Filter where Swimmer or Squid exist
                    swimmer_points = intraday[intraday["Swimmer_Emoji"] != ""]

                    scatter_swimmer = go.Scatter(
                        x=swimmer_points["Time"],
                        y=swimmer_points["F_numeric"] - 188,  # Offset downward so it floats below price
                        mode="text",
                        text=swimmer_points["Swimmer_Emoji"],  # Either 🏊🏽‍♂️ or 🦑
                        textposition="bottom center",
                        name="New Lows 🏊🏽‍♂️🦑",
                        textfont=dict(size=55),
                        showlegend=False
                    )

                    fig.add_trace(scatter_swimmer, row=1, col=1)



                    mask_green_king = intraday["King_Signal"] == "👑"
                    scatter_green_king = go.Scatter(
                        x=intraday.loc[mask_green_king, "Time"],
                        y=intraday.loc[mask_green_king, "F_numeric"] + 89,
                        mode="text",
                        text=["♔"] * mask_green_king.sum(),
                        textfont=dict(size=55, color="green"),
                        name="Green King Signal (♔)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>👑 Green Kingdom Crowned ♔<extra></extra>"
                    )


                    mask_red_king = intraday["King_Signal"] == "🔻👑"
                    scatter_red_king = go.Scatter(
                        x=intraday.loc[mask_red_king, "Time"],
                        y=intraday.loc[mask_red_king, "F_numeric"] - 89,
                        mode="text",
                        text=["♔"] * mask_red_king.sum(),
                        textfont=dict(size=55, color="red"),
                        name="Red King Signal (♔)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>🔻👑 Red Kingdom Crowned ♔<extra></extra>"
                    )


                    fig.add_trace(scatter_green_king, row=1, col=1)
                    fig.add_trace(scatter_red_king, row=1, col=1)





                    threshold = 0.5  # or even 1.0 depending on your scaling
                    intraday["Kijun_F_Cross_Emoji"] = np.where(
                        (intraday["F_numeric"] > intraday["Kijun_F"] + threshold) & (intraday["F_shift"] < intraday["Kijun_F"] - threshold),
                        "♕",
                        np.where(
                            (intraday["F_numeric"] < intraday["Kijun_F"] - threshold) & (intraday["F_shift"] > intraday["Kijun_F"] + threshold),
                            "♛",
                            ""
                        )
                    )



                 # # Create separate masks for upward and downward crosses:
                    mask_kijun_up = intraday["Kijun_F_Cross_Emoji"] == "♕"
                    mask_kijun_down = intraday["Kijun_F_Cross_Emoji"] == "♛"

                    # Upward Cross Trace (♕)
                    up_cross_trace = go.Scatter(
                        x=intraday.loc[mask_kijun_up, "Time"],
                        y=intraday.loc[mask_kijun_up, "F_numeric"] + 55,  # Offset upward (adjust as needed)
                        mode="text",
                        text=intraday.loc[mask_kijun_up, "Kijun_F_Cross_Emoji"],
                        textposition="top center",  # Positioned above the point
                        textfont=dict(size=55, color="green"),
                        name="Kijun Cross Up (♕)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Upward Cross: %{text}<extra></extra>"
                    )

                    # Downward Cross Trace (♛)
                    down_cross_trace = go.Scatter(
                        x=intraday.loc[mask_kijun_down, "Time"],
                        y=intraday.loc[mask_kijun_down, "F_numeric"] - 55,  # Offset downward
                        mode="text",
                        text=intraday.loc[mask_kijun_down, "Kijun_F_Cross_Emoji"],
                        textposition="bottom center",  # Positioned below the point
                        textfont=dict(size=55, color="red"),
                        name="Kijun Cross Down (♛)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Downward Cross: %{text}<extra></extra>"
                    )


                    fig.add_trace(up_cross_trace,   row=1, col=1)
                    fig.add_trace(down_cross_trace, row=1, col=1)






                # Mask for each horse
                mask_horse_buy = intraday["Kijun_Cross_Horse"] == "♘"
                mask_horse_sell = intraday["Kijun_Cross_Horse"] == "♞"

                # Buy Horse (♘) → normal above
                scatter_horse_buy = go.Scatter(
                    x=intraday.loc[mask_horse_buy, "Time"],
                    y=intraday.loc[mask_horse_buy, "F_numeric"] + 34,
                    mode="text",
                    text=["♘"] * mask_horse_buy.sum(),
                    textposition="top center",
                    textfont=dict(size=34, color="green"),  # You can make it white if you want
                    name="Horse After Buy Kijun Cross",
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>♘ Horse after Buy<extra></extra>"
                )

                # Sell Horse (♞) → below and red
                scatter_horse_sell = go.Scatter(
                    x=intraday.loc[mask_horse_sell, "Time"],
                    y=intraday.loc[mask_horse_sell, "F_numeric"] - 34,
                    mode="text",
                    text=["♞"] * mask_horse_sell.sum(),
                    textposition="bottom center",
                    textfont=dict(size=34, color="red"),
                    name="Horse After Sell Kijun Cross",
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>♞ Horse after Sell<extra></extra>"
                )

                fig.add_trace(scatter_horse_buy, row=1, col=1)
                fig.add_trace(scatter_horse_sell, row=1, col=1)



                mask_bishop_up = intraday["Kijun_Cross_Bishop"] == "♗"
                mask_bishop_down = intraday["Kijun_Cross_Bishop"] == "♝"

                # Bishop Up (♗)
                scatter_bishop_up = go.Scatter(
                    x=intraday.loc[mask_bishop_up, "Time"],
                    y=intraday.loc[mask_bishop_up, "F_numeric"] + 34,
                    mode="text",
                    text=intraday.loc[mask_bishop_up, "Kijun_Cross_Bishop"],
                    textposition="top center",
                    textfont=dict(size=34, color="green"),
                    name="Kijun Cross Bishop (Buy ♗)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Volatility Support ♗<extra></extra>"
                )

                # Bishop Down (♝)
                scatter_bishop_down = go.Scatter(
                    x=intraday.loc[mask_bishop_down, "Time"],
                    y=intraday.loc[mask_bishop_down, "F_numeric"] - 34,
                    mode="text",
                    text=intraday.loc[mask_bishop_down, "Kijun_Cross_Bishop"],
                    textposition="bottom center",
                    textfont=dict(size=34, color="red"),
                    name="Kijun Cross Bishop (Sell ♝)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Volatility Resistance ♝<extra></extra>"
                )

                fig.add_trace(scatter_bishop_up, row=1, col=1)
                fig.add_trace(scatter_bishop_down, row=1, col=1)



                # Mask the rook crosses
                mask_rook_up = intraday["TD_Supply_Rook"] == "♖"
                mask_rook_down = intraday["TD_Supply_Rook"] == "♜"

                # White rook (up cross)
                scatter_rook_up = go.Scatter(
                    x=intraday.loc[mask_rook_up, "Time"],
                    y=intraday.loc[mask_rook_up, "F_numeric"] + 21,  # Offset upward
                    mode="text",
                    text=intraday.loc[mask_rook_up, "TD_Supply_Rook"],
                    textposition="top center",
                    textfont=dict(size=21,  color="green"),
                    name="TD Supply Cross Up (♖)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>TD Supply Crossed Up ♖<extra></extra>"
                )

                # Black rook (down cross)
                scatter_rook_down = go.Scatter(
                    x=intraday.loc[mask_rook_down, "Time"],
                    y=intraday.loc[mask_rook_down, "F_numeric"] - 21,  # Offset downward
                    mode="text",
                    text=intraday.loc[mask_rook_down, "TD_Supply_Rook"],
                    textposition="bottom center",
                    textfont=dict(size=21,  color="red"),
                    name="TD Supply Cross Down (♜)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>TD Supply Crossed Down ♜<extra></extra>"
                )

                # Add both to figure
                fig.add_trace(scatter_rook_up, row=1, col=1)
                fig.add_trace(scatter_rook_down, row=1, col=1)

#                     # 1) build Boolean masks (you can tighten with a small buffer if you like)
#                 upper_tag = intraday["F_numeric"] >= intraday["F% Upper"]
#                 lower_tag = intraday["F_numeric"] <= intraday["F% Lower"]

#                 # 2) scatter for upper‑band tags
#                 fig.add_trace(
#                     go.Scatter(
#                         x=intraday.loc[upper_tag, "Time"],
#                         y=intraday.loc[upper_tag, "F_numeric"] +13,       # nudge up a bit
#                         mode="text",
#                         text=["🏷️"] * upper_tag.sum(),
#                         textposition="top center",
#                         textfont=dict(size=18),
#                         name="BB Upper Tag (🏷️)",
#                         hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Tagged Upper Band<extra></extra>"
#                     ),
#                     row=1, col=1
#                 )

#                 # 3) scatter for lower‑band tags
#                 fig.add_trace(
#                     go.Scatter(
#                         x=intraday.loc[lower_tag, "Time"],
#                         y=intraday.loc[lower_tag, "F_numeric"] - 13,       # nudge down a bit
#                         mode="text",
#                         text=["🏷️"] * lower_tag.sum(),
#                         textposition="bottom center",
#                         textfont=dict(size=18),
#                         name="BB Lower Tag (🏷️)",
#                         hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Tagged Lower Band<extra></extra>"
#                     ),
#                     row=1, col=1
#                 )





                # correct masks
                mask_pawn_up   = intraday["Tenkan_Pawn"] == "♙"
                mask_pawn_down = intraday["Tenkan_Pawn"] == "♟️"     # <-- changed ♙ → ♟️

                # ♙ Upward pawn
                pawn_up = go.Scatter(
                    x=intraday.loc[mask_pawn_up, "Time"],
                    y=intraday.loc[mask_pawn_up, "F_numeric"] + 13,
                    mode="text",
                    text=intraday.loc[mask_pawn_up, "Tenkan_Pawn"],
                    textposition="top center",
                    textfont=dict(size=21, color="green"),            # green for up
                    name="Pawn Up (Tenkan Cross)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>♙ Upward Tenkan Cross<extra></extra>"
                )

                # ♟️ Downward pawn
                pawn_down = go.Scatter(
                    x=intraday.loc[mask_pawn_down, "Time"],
                    y=intraday.loc[mask_pawn_down, "F_numeric"] - 13,
                    mode="text",
                    text=intraday.loc[mask_pawn_down, "Tenkan_Pawn"],
                    textposition="bottom center",
                    textfont=dict(size=21, color="red"),             # red for down
                    name="Pawn Down (Tenkan Cross)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>♟️ Downward Tenkan Cross<extra></extra>"
                )

                fig.add_trace(pawn_up,   row=1, col=1)
                fig.add_trace(pawn_down, row=1, col=1)



#                 short_entry_trace = go.Scatter(
#                     x=intraday.loc[intraday["Entry_Alert_Short"], "Time"],
#                     y=intraday.loc[intraday["Entry_Alert_Short"], "F_numeric"] - 8,
#                     mode="text",
#                     text=[" ✅"] * intraday["Entry_Alert_Short"].sum(),
#                     textposition="bottom center",
#                     textfont=dict(size=21, color="lime"),
#                     name="Short Entry (✅)"
#                 )
#                 fig.add_trace(short_entry_trace, row=1, col=1)






#                 long_entry_trace = go.Scatter(
#                     x=intraday.loc[intraday["Entry_Alert_Long"], "Time"],
#                     y=intraday.loc[intraday["Entry_Alert_Long"], "F_numeric"] + 8,
#                     mode="text",
#                     text=[" ✅"] * intraday["Entry_Alert_Long"].sum(),
#                     textposition="top center",
#                     textfont=dict(size=21, color="lime"),
#                     name="Long Entry (✅)"
#                 )
#                 fig.add_trace(long_entry_trace, row=1, col=1)




#                 # # Mask where Entry Type II (delayed volume confirmation) is True
#                 # mask_long_delayed = intraday["Entry_Alert_Long_Delayed"]
#                 # mask_short_delayed = intraday["Entry_Alert_Short_Delayed"]

#                 # # ☑️ Long Entry after delayed volume
#                 # scatter_long_delayed = go.Scatter(
#                 #     x=intraday.loc[mask_long_delayed, "Time"],
#                 #     y=intraday.loc[mask_long_delayed, "F_numeric"] + 8,  # Offset upward
#                 #     mode="text",
#                 #     text=["☑️"] * mask_long_delayed.sum(),
#                 #     textposition="top center",
#                 #     textfont=dict(size=13, color="lime"),
#                 #     name="Delayed Long Entry (☑️)",
#                 #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>☑️ Delayed Long Entry<extra></extra>"
#                 # )

#                 # # ☑️ Short Entry after delayed volume
#                 # scatter_short_delayed = go.Scatter(
#                 #     x=intraday.loc[mask_short_delayed, "Time"],
#                 #     y=intraday.loc[mask_short_delayed, "F_numeric"] - 8,  # Offset downward
#                 #     mode="text",
#                 #     text=["☑️"] * mask_short_delayed.sum(),
#                 #     textposition="bottom center",
#                 #     textfont=dict(size=13, color="red"),
#                 #     name="Delayed Short Entry (☑️)",
#                 # #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>☑️ Delayed Short Entry<extra></extra>"
#                 # )





#                 # Add to F% plot
#                 mask_ops_bear = intraday["OPS Transition"] == "🐻"
#                 mask_ops_panda = intraday["OPS Transition"] == "🐼"

#                 scatter_ops_bear = go.Scatter(
#                     x=intraday.loc[mask_ops_bear, "Time"],
#                     y=intraday.loc[mask_ops_bear, "F_numeric"] - 7,  # Offset to avoid overlap
#                     mode="text",
#                     text="🐻",
#                     textposition="bottom center",
#                     textfont=dict(size=22, color="red"),
#                     name="OPS Bearish Flip",
#                     hovertemplate="Time: %{x}<br>F%: %{y}<br>OPS Turned Bearish<extra></extra>"
#                 )

#                 scatter_ops_panda = go.Scatter(
#                     x=intraday.loc[mask_ops_panda, "Time"],
#                     y=intraday.loc[mask_ops_panda, "F_numeric"] + 7,  # Offset to avoid overlap
#                     mode="text",
#                     text="🐼",
#                     textposition="top center",
#                     textfont=dict(size=22, color="green"),
#                     name="OPS Bullish Flip",
#                     hovertemplate="Time: %{x}<br>F%: %{y}<br>OPS Turned Bullish<extra></extra>"
#                 )

#                 # Add to the F% plot
#                 fig.add_trace(scatter_ops_bear, row=1, col=1)
#                 fig.add_trace(scatter_ops_panda, row=1, col=1)


# # #*******************************************************************************************************************************************************************************

# # #  Cotangent Spikes (Skull 💀) - Catches both >3 and <-3
# #                 mask_cotangent_spike = intraday["F% Cotangent"].abs() > 3


# #                 scatter_cotangent_spike = go.Scatter(
# #                     x=intraday.loc[mask_cotangent_spike, "Time"],
# #                     y=intraday.loc[mask_cotangent_spike, "F_numeric"] - 89,  # Slightly offset for visibility
# #                     mode="text",
# #                     text="💀",
# #                     textposition="top center",
# #                     textfont=dict(size=18),  # Larger for emphasis
# #                     name="Cotangent Spike",
# #                     hovertext=intraday.loc[mask_cotangent_spike, "F% Cotangent"].round(2),  # Display rounded cotangent value
# #                     hovertemplate="Time: %{x}<br>F%: %{y}<br>Cotangent: %{hovertext}<extra></extra>"
# #                 )

# #     # Add to the F% plot (Row 1)
# #                 fig.add_trace(scatter_cotangent_spike, row=1, col=1)

# # #-------------------------------------------------------------------------------------

# #     #  Cosecant Spikes (Lightning ⚡) - Detects |F% Cosecant| > 20
# #                 mask_cosecant_spike = intraday["F% Cosecant"].abs() > 20

# #                 scatter_cosecant_spike = go.Scatter(
# #                     x=intraday.loc[mask_cosecant_spike, "Time"],
# #                     y=intraday.loc[mask_cosecant_spike, "F_numeric"] + 20,  # Offset for visibility
# #                     mode="text",
# #                     text="⚡",
# #                     textposition="top center",
# #                     textfont=dict(size=18, color="orange"),  # Larger and orange for emphasis
# #                     name="Cosecant Spike",
# #                     hovertext=intraday.loc[mask_cosecant_spike, "F% Cosecant"].round(2),  # Display rounded cosecant value
# #                     hovertemplate="Time: %{x}<br>F%: %{y}<br>Cosecant: %{hovertext}<extra></extra>"
# #                 )

# #                 # Add to the F% plot (Row 1)
# #                 fig.add_trace(scatter_cosecant_spike, row=1, col=1)



# # #---------------------------------------------------------------------------------------





#                 # # 🔵 Secant Spikes (Tornado 🌪) - Detects |F% Secant| > 3
#                 # mask_secant_spike = intraday["F% Secant"].abs() > 5

#                 # scatter_secant_spike = go.Scatter(
#                 #     x=intraday.loc[mask_secant_spike, "Time"],
#                 #     y=intraday.loc[mask_secant_spike, "F_numeric"] + 20,  # Offset for visibility
#                 #     mode="text",
#                 #     text="🌪",
#                 #     textposition="top center",
#                 #     textfont=dict(size=18, color="blue"),  # Large and blue for emphasis
#                 #     name="Secant Spike",
#                 #     hovertext=intraday.loc[mask_secant_spike, "F% Secant"].round(2),  # Display rounded secant value
#                 #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Secant: %{hovertext}<extra></extra>"
#                 # )

#                 # # Add to the F% plot (Row 1)
#                 # fig.add_trace(scatter_secant_spike, row=1, col=1)

# #  TRIGONOMETRIC COTAGENT / SECANT / TENKAN
# #**************************************************************************************************************************************************************************



#                 # 🎯 Add all lines to the F% plot
#                 fig.add_trace(y_open_f_line, row=1, col=1)
#                 fig.add_trace(y_high_f_line, row=1, col=1)
#                 fig.add_trace(y_low_f_line, row=1, col=1)
#                 fig.add_trace(y_close_f_line, row=1, col=1)





                # # 1. Calculate 1.618x Fib targets for both TD lines
                # intraday["Supply_Target"] = intraday["TD Supply Line F"] * 1.618
                # intraday["Demand_Target"] = intraday["TD Demand Line F"] * 1.618

                # # 2. Check if both F_numeric and the corresponding TD line confirm the breakout
                # bullish_mask = (intraday["F_numeric"] > intraday["Supply_Target"]) & (intraday["TD Supply Line F"] > 0)
                # bearish_mask = (intraday["F_numeric"] < intraday["Demand_Target"]) & (intraday["TD Demand Line F"] < 0)

                # # 3. Add emojis to the plot
                # fig.add_trace(go.Scatter(
                #     x=intraday.loc[bullish_mask, "Time"],
                #     y=intraday.loc[bullish_mask, "F_numeric"] + 144,
                #     mode="text",
                #     text=["🏝️"] * bullish_mask.sum(),
                #     textposition="top center",
                #     textfont=dict(size=34),
                #     name="Bullish Fib Target (🏝️)",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Breakout above Fib Target<extra></extra>"
                # ))

                # fig.add_trace(go.Scatter(
                #     x=intraday.loc[bearish_mask, "Time"],
                #     y=intraday.loc[bearish_mask, "F_numeric"] - 143,
                #     mode="text",
                #     text=["🌋"] * bearish_mask.sum(),
                #     textposition="bottom center",
                #     textfont=dict(size=21),
                #     name="Bearish Fib Target (🌋)",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Breakdown below Fib Target<extra></extra>"
                # ))
                # # 🏇 Mike rides real VWAP up
                # mask_vwap_up = intraday["VWAP_Cross_Emoji"] == "🥁"
                # scatter_vwap_up = go.Scatter(
                #     x=intraday.loc[mask_vwap_up, "Time"],
                #     y=intraday.loc[mask_vwap_up, "F_numeric"] +89,
                #     mode="text",
                #     text="🏇",
                #     textposition="top center",
                #     textfont=dict(size=55, color="blue"),
                #     name="Mike Rides VWAP 🏇",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>True Bullish VWAP Cross 🏇<extra></extra>"
                # )

                # # 🎠 Carousel fake-out (bearish VWAP cross)
                # mask_vwap_down = intraday["VWAP_Cross_Emoji"] == "🎻"
                # scatter_vwap_down = go.Scatter(
                #     x=intraday.loc[mask_vwap_down, "Time"],
                #     y=intraday.loc[mask_vwap_down, "F_numeric"] - 89,
                #     mode="text",
                #     text="🎠",
                #     textposition="bottom center",
                #     textfont=dict(size=55, color="deeppink"),
                #     name="Carousel Trap 🎠",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Bearish VWAP Cross — Carousel Trap 🎠<extra></extra>"
                # )

                # fig.add_trace(scatter_vwap_up, row=1, col=1)
                # fig.add_trace(scatter_vwap_down, row=1, col=1)


                # Update layout overall
                fig.update_layout(
                    title=f"{t} – Pure Demark",
                    margin=dict(l=30, r=30, t=50, b=30),
                    height=1000,  # Increase overall figure height (default ~450-600)

                    showlegend=True
                )





                st.plotly_chart(fig, use_container_width=True)



#"Time","RVOL_5", "BBW_Tight_Emoji","BBW Alert","tdSupplyCrossalert", "Kijun_F_Cross","ADX_Alert","STD_Alert","ATR_Exp_Alert","Tenkan_Kijun_Cross","KumoTwistAlert"
# "Time", "Close","Opening Price Signal","OPS Transition","Net Price","Net Price Direction","Bounce","Retrace","BBW_Tight_Emoji","RVOL_5","F% BBW", "Day Type", "High of Day","Low of Day", "F%","TD Open","TD Trap","TD CLoP", "40ish","Tenkan_Kijun_Cross","CTOD Alert","Alert_Kijun","Alert_Mid","Wealth Signal","BBW Alert","tdSupplyCrossalert"


            except Exception as e:
                st.error(f"Error fetching data for {t}: {e}")









    # with st.expander("📆 Daily Chart Overview", expanded=True):
    #     try:
    #         daily_chart_data = yf.download(t, period="60d", interval="1d", progress=False)

    #         if not daily_chart_data.empty:
    #             # Reset index and ensure proper column names
    #             daily_chart_data.reset_index(inplace=True)
    #             if isinstance(daily_chart_data.columns, pd.MultiIndex):
    #                 daily_chart_data.columns = [col[0] if isinstance(col, tuple) else col for col in daily_chart_data.columns]

    #             # Rename 'index' to 'Date' if needed
    #             if "Date" not in daily_chart_data.columns:
    #                 if "index" in daily_chart_data.columns:
    #                     daily_chart_data.rename(columns={"index": "Date"}, inplace=True)
    #                 else:
    #                     daily_chart_data["Date"] = daily_chart_data.index

    #             # Ensure datetime format
    #             daily_chart_data["Date"] = pd.to_datetime(daily_chart_data["Date"])

    #             # Build Plotly Candlestick chart
    #             fig_daily = go.Figure(data=[go.Candlestick(
    #                 x=daily_chart_data["Date"],
    #                 open=daily_chart_data["Open"],
    #                 high=daily_chart_data["High"],
    #                 low=daily_chart_data["Low"],
    #                 close=daily_chart_data["Close"],
    #                 name="Daily"
    #             )])

    #             fig_daily.update_layout(
    #                 title=f"{t} – Daily Candlestick Chart (Past 60 Days)",
    #                 height=2000,
    #                 xaxis_rangeslider_visible=False,
    #                 margin=dict(l=30, r=30, t=40, b=20)
    #             )





    #             st.plotly_chart(fig_daily, use_container_width=True)
    #         else:
    #             st.warning("No daily data available.")
    #     except Exception as e:
    #         st.error(f"Failed to load daily chart for {t}: {e}")


















        #     with st.expander("🕯️ Hidden Candlestick + Ichimoku View", expanded=True):
        #         fig_ichimoku = go.Figure()

        #         fig_ichimoku.add_trace(go.Candlestick(
        #             x=intraday['Time'],
        #             open=intraday['Open'],
        #             high=intraday['High'],
        #             low=intraday['Low'],
        #             close=intraday['Close'],
        #             name='Candles'
        #         ))

        #         fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Tenkan'], line=dict(color='red'), name='Tenkan-sen'))
        #         fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Kijun'], line=dict(color='green'), name='Kijun-sen'))
        #         fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanA'], line=dict(color='yellow'), name='Span A'))
        #         fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanB'], line=dict(color='blue'), name='Span B'))
        #         fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Chikou'], line=dict(color='purple'), name='Chikou'))

        #         fig_ichimoku.add_trace(go.Scatter(
        #             x=intraday['Time'],
        #             y=intraday['SpanA'],
        #             line=dict(width=0),
        #             showlegend=False
        #         ))

        #         fig_ichimoku.add_trace(go.Scatter(
        #             x=intraday['Time'],
        #             y=intraday['SpanB'],
        #             fill='tonexty',
        #             fillcolor='rgba(128, 128, 128, 0.2)',
        #             line=dict(width=0),
        #             showlegend=False
        #         ))

        #         fig_ichimoku.update_layout(
        #             title="Ichimoku Candlestick Chart",
        #             height=450,
        #             width=450,
        #             xaxis_rangeslider_visible=False,
        #             margin=dict(l=30, r=30, t=40, b=20)
        #         )

        #         st.plotly_chart(fig_ichimoku, use_container_width=True)
        # st.write("✅ Ichimoku Expander Rendered")


