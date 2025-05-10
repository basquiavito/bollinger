
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date


st.set_page_config(page_title="Dashboard", layout="wide")


st.title("📈 VST Dashboard")




# ======================================
# Sidebar - User Inputs & Advanced Options
# ======================================
st.sidebar.header("Input Options")

default_tickers = ["SPY","QQQ","NVDA", "AVGO","AMD","PLTR","MRVL","uber","AMZN","AAPL","googl","MSFT","META","tsla","sbux","nke","chwy","DKNG","GM","cmg","c","wfc","hood","coin","bac","jpm","PYPL","tgt","wmt","elf"]
# Get query param if provided
query_params = st.query_params
preselected_symbol = query_params.get("symbol", [default_tickers[0]])[0]

tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=default_tickers,
    default=[preselected_symbol] if preselected_symbol in default_tickers else ["SPY"]
)

# Date range inputs
start_date = st.sidebar.date_input("Start Date", value=date(2025, 4, 3))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["2m", "5m", "15m", "30m", "60m", "1d"],
    index=1  # Default to 5m
)

# # 🔥 Candlestick Chart Toggle (Place this here)
# show_candlestick = st.sidebar.checkbox("Show Candlestick Chart", value=true)



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


def fetch_last_two_daily_bars(symbol: str) -> dict:
        """
        Step 1 – Grab the two most‑recent daily bars.

        Returns a dict with:
            • yesterday_high   – yesterday’s High
            • yesterday_close  – yesterday’s Close
            • today_open       – today’s Open

        Raises if fewer than two daily bars are available.
        """
        # Pull a small safety window (5 days) to survive weekends & holidays
        daily = yf.download(
            symbol,
            period="5d",
            interval="1d",
            progress=False,
            prepost=True          # include pre‑/post‑market so today’s open is populated early
        )

        if daily.empty or len(daily) < 2:
            raise ValueError(f"Not enough daily data for {symbol}")

        # Flatten possible multi‑index columns (yfinance sometimes returns OHLCV as tuples)
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

        yesterday = daily.iloc[-2]
        today     = daily.iloc[-1]

        return {
            "yesterday_high":  float(yesterday["High"]),
            "yesterday_close": float(yesterday["Close"]),
            "today_open":      float(today["Open"])
        }




# ======================================
# Main Button to Run
# ======================================
if st.sidebar.button("Run Analysis"):
    main_tabs = st.tabs([f"Ticker: {t}" for t in tickers])

    for idx, t in enumerate(tickers):
        with main_tabs[idx]:

        # Example usage inside your loop:
            levels       = fetch_last_two_daily_bars(t)  # t is the current ticker
            prev_high    = levels["yesterday_high"]
            prev_close   = levels["yesterday_close"]
            today_open   = levels["today_open"]

            # Compute change
            dollar_change = today_open - prev_close
            percent_change = (dollar_change / prev_close) * 100

            # Format
            change_color = "green" if dollar_change >= 0 else "red"
            change_str = f"<span style='color:{change_color}'><b>{dollar_change:+.2f} ({percent_change:+.2f}%)</b></span>"
            price_str = f"<b>{t}</b> Last Close: {prev_close:.2f} → Open: {today_open:.2f} | {change_str}"

            try:
                # ================
                # 1) Fetch Previous Day's Data
                # ================
                daily_data = yf.download(
                    t,
                    end=start_date,
                    interval="1d",
                    progress=False,
                    prepost=True
                )

                prev_close, prev_high, prev_low = None, None, None
                prev_close_str, prev_high_str, prev_low_str = "N/A", "N/A", "N/A"
                if daily_data is not None and not daily_data.empty:
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














                marquee_lines = []

                for t in tickers:
                    try:
                        levels = fetch_last_two_daily_bars(t)
                        prev_close = levels["yesterday_close"]
                        today_open = levels["today_open"]

                        dollar_change = today_open - prev_close
                        percent_change = (dollar_change / prev_close) * 100
                        color = "green" if dollar_change >= 0 else "red"




                        change_info = f"<span style='color:{color}'><b>{dollar_change:+.2f} ({percent_change:+.2f}%)</b></span>"
                        marquee_lines.append(f"<b>{t}</b>: {prev_close:.2f} → {today_open:.2f} | {change_info}")

                    except Exception as e:
                        marquee_lines.append(f"{t}: Error")

                # Final string
                marquee_text = " | ".join(marquee_lines)

                st.markdown(
                    f"""
                    <marquee behavior="scroll" direction="left" scrollamount="5" style="color: white; font-size:20px;">
                        {marquee_text}
                    </marquee>
                    """,
                    unsafe_allow_html=True
                )


































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


                # ================
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




                # --- Step 2 : Larry Range & Expansion bands ---------------------------
                larry_range      = prev_high - prev_close          # Highᵧₑₛₜ − Closeᵧₑₛₜ
                expansion_up     = today_open + larry_range
                expansion_down   = today_open - larry_range

                # store once for every intraday row (handy for plotting / alerts)
                intraday["Larry_Range"]     = larry_range
                intraday["Expansion_Up"]    = expansion_up
                intraday["Expansion_Down"]  = expansion_down


                # --- Step 3 : CALL / PUT alerts when price crosses the bands -----------
                intraday["Larry_Alert"] = ""          # empty by default

                for i in range(1, len(intraday)):     # start at 1 so we can look back one bar
                    prev_close = intraday.loc[i-1, "Close"]
                    curr_close = intraday.loc[i,   "Close"]

                    # Up‑side break → CALL alert (only the first bar that crosses)
                    if prev_close <= expansion_up and curr_close > expansion_up:
                        intraday.loc[i, "Larry_Alert"] = "CALL ALERT"

                    # Down‑side break → PUT alert
                    elif prev_close >= expansion_down and curr_close < expansion_down:
                        intraday.loc[i, "Larry_Alert"] = "PUT ALERT"



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


                def detect_walk_the_band(df):
                    df["Walk_Up"] = (df["F_numeric"] > df["F% Upper"]).astype(int)
                    df["Walk_Down"] = (df["F_numeric"] < df["F% Lower"]).astype(int)

                    df["Walk_Up_Streak"] = df["Walk_Up"].rolling(window=3).sum()
                    df["Walk_Down_Streak"] = df["Walk_Down"].rolling(window=3).sum()

                    df["Walk_Up_Emoji"] = np.where(df["Walk_Up_Streak"] >= 3, "🚶🏻", "")
                    df["Walk_Down_Emoji"] = np.where(df["Walk_Down_Streak"] >= 3, "🧎‍♂️", "")
                    return df

                intraday = detect_walk_the_band(intraday)





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

                intraday["Cum_PriceVolume"] = (intraday["Close"] * intraday["Volume"]).cumsum()
                intraday["Cum_Volume"] = intraday["Volume"].cumsum()
                intraday["VWAP"] = intraday["Cum_PriceVolume"] / intraday["Cum_Volume"]

                intraday["VWAP_F"] = ((intraday["VWAP"] - prev_close) / prev_close) * 10000




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

                intraday["Chikou"] = intraday["Close"].shift(-26)


                intraday["Opening Price Signal"] = intraday["Close"] - prev_open
                intraday["Net Price"] = intraday["Close"] - prev_close



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



                def detect_adx_trend_once(df, adx_threshold=20):
                    """
                    Emits 📈 or 📉 only once when ADX > threshold AND direction changes from previous signal.
                    """
                    df["ADX_Trend_Emoji"] = ""
                    previous_trend = ""

                    for i in range(1, len(df)):
                        adx = df.loc[i, "ADX_F%"]
                        f_now = df.loc[i, "F_numeric"]

                        if adx > adx_threshold:
                            if f_now > 0 and previous_trend != "bull":
                                df.loc[i, "ADX_Trend_Emoji"] = "📈"
                                previous_trend = "bull"
                            elif f_now < 0 and previous_trend != "bear":
                                df.loc[i, "ADX_Trend_Emoji"] = "📉"
                                previous_trend = "bear"
                        else:
                            previous_trend = ""  # Reset when ADX is weak

                    return df


                # Apply it
                intraday = detect_adx_trend_once(intraday)



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



                def add_fib1618_after_kijun_cross(df):
                    """
                    • Detects each time F_numeric crosses Kijun_F (up or down).
                    • From that bar forward it projects a 1.618 “Fib” target in F‑% space.
                    • Adds two new columns:
                        – Fib_1618 : numeric target (float)
                        – Fib_Emoji: 🦅 for bullish hits, 🦇 for bearish hits
                    """

                    df["Fib_1618"]  = np.nan
                    df["Fib_Emoji"] = ""

                    last_cross_idx = None   # where the most‑recent cross happened
                    direction      = None   # "up"  or "down"

                    for i in range(1, len(df)):
                        # --- detect fresh cross -------------------------------------------
                        prev_f,  curr_f  = df.loc[i-1, "F_numeric"], df.loc[i, "F_numeric"]
                        prev_k,  curr_k  = df.loc[i-1, "Kijun_F"  ], df.loc[i, "Kijun_F"  ]

                        if prev_f < prev_k and curr_f >= curr_k:          # bullish cross
                            last_cross_idx = i
                            direction      = "up"
                        elif prev_f > prev_k and curr_f <= curr_k:        # bearish cross
                            last_cross_idx = i
                            direction      = "down"

                        # --- project Fib 1.618 from that cross ----------------------------
                        if last_cross_idx is not None:
                            base = df.loc[last_cross_idx, "F_numeric"]    # F% at the cross
                            if direction == "up":
                                fib = base + 1.618 + abs(curr_f - base)
                                if curr_f >= fib and df.loc[i, "Fib_Emoji"] == "":
                                    df.at[df.index[i], "Fib_Emoji"] = "🦅"   # bullish hit
                            else:  # direction == "down"
                                fib = base - 1.618 + abs(curr_f - base)
                                if curr_f <= fib and df.loc[i, "Fib_Emoji"] == "":
                                    df.at[df.index[i], "Fib_Emoji"] = "🦇"   # bearish hit

                            df.at[df.index[i], "Fib_1618"] = fib

                    return df

                intraday = add_fib1618_after_kijun_cross(intraday)

                def get_guru_signals(intraday):
                    recent = intraday.tail(9)
                    latest = recent.iloc[-1]

                    # ===== Guru 1: Price above/below Tenkan
                    emoji_level1 = "🟩" if latest["F_numeric"] > latest["Tenkan_F"] else "🟥"

                    # ===== Guru 2: Tenkan slope
                    slope = latest["Tenkan_F"] - recent.iloc[-2]["Tenkan_F"]
                    emoji_level2 = "📈" if slope > 0 else "📉"

                    # ===== Guru 3: Price vs MA and Tenkan
                    emoji_level3 = "🚀" if (latest["F_numeric"] >= latest["F% MA"] or latest["F_numeric"] >= latest["Tenkan_F"]) else "⚠️"

                    # ===== Guru 4: Price vs Kijun and cross check
                    kijun_f = latest["Kijun_F"]
                    tenkan_f = latest["Tenkan_F"]
                    tenkan_prev = recent.iloc[-2]["Tenkan_F"]
                    kijun_prev = recent.iloc[-2]["Kijun_F"]

                    cross_up = tenkan_prev < kijun_prev and tenkan_f > kijun_f
                    cross_down = tenkan_prev > kijun_prev and tenkan_f < kijun_f

                    emoji_level4 = ""
                    if latest["F_numeric"] > kijun_f:
                        emoji_level4 += "🟩K"
                    elif latest["F_numeric"] < kijun_f:
                        emoji_level4 += "🟥K"
                    if cross_up:
                        emoji_level4 += "☀️"
                    elif cross_down:
                        emoji_level4 += "🌙"

                    return emoji_level1, emoji_level2, emoji_level3, emoji_level4

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
                velocity_std = intraday["F% Velocity"].std()
                intraday["Turtle"] = intraday["F% Velocity"].abs() < (0.15 * velocity_std)

                # 📈 Volatility expansion: F% BBW or ADX_F% rising
                intraday["BBW_Rising"] = intraday["F% BBW"] > intraday["F% BBW"].shift(3)
                intraday["ADX_Rising"] = intraday["ADX_F%"] > intraday["ADX_F%"].shift(3)
                intraday["Volatility_Expanding"] = intraday["BBW_Rising"] | intraday["ADX_Rising"]

                # 🔁 RVOL curl
                intraday["RVOL_Curl"] = intraday["RVOL_5"] > intraday["RVOL_5"].shift(3)

                # 🐢 streak tracker (past 3 bars)
                intraday["Turtle_Streak"] = intraday["Turtle"].rolling(window=3).sum()

                # 🎯 Ambush Setup: turtle streak + volatility expansion + RVOL curl
                intraday["Ambush_Setup"] = (intraday["Turtle_Streak"] >= 2) & intraday["Volatility_Expanding"] & intraday["RVOL_Curl"]
                intraday["Ambush_Emoji"] = intraday["Ambush_Setup"].apply(lambda x: "🎯" if x else "")


                def calculate_td_sequential(intraday):
                    """
                    Calculates TD Sequential buy/sell setups while avoiding ambiguous
                    boolean errors by using NumPy arrays for comparisons.
                    """

                    # Initialize columns
                    intraday['Buy Setup'] = np.nan
                    intraday['Sell Setup'] = np.nan

                    # Convert Close prices to a NumPy array for guaranteed scalar access
                    close_vals = intraday['Close'].values

                    # Arrays to track consecutive buy/sell counts
                    buy_count = np.zeros(len(intraday), dtype=np.int32)
                    sell_count = np.zeros(len(intraday), dtype=np.int32)

                    # Iterate through the rows
                    for i in range(len(intraday)):
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
                            intraday.at[intraday.index[i], 'Buy Setup'] = 'Buy Setup Completed'
                            buy_count[i] = 0  # reset after completion
                        elif buy_count[i] > 0:
                            intraday.at[intraday.index[i], 'Buy Setup'] = f'Buy Setup {buy_count[i]}'

                        if sell_count[i] == 9:
                            intraday.at[intraday.index[i], 'Sell Setup'] = 'Sell Setup Completed'
                            sell_count[i] = 0  # reset after completion
                        elif sell_count[i] > 0:
                            intraday.at[intraday.index[i], 'Sell Setup'] = f'Sell Setup {sell_count[i]}'

                    return intraday

                intraday = calculate_td_sequential(intraday)


                def calculate_td_countdown(intraday):
                    intraday['Buy Countdown'] = None
                    intraday['Sell Countdown'] = None

                    buy_countdown_active = False
                    sell_countdown_active = False
                    buy_count = 0
                    sell_count = 0

                    for i in range(len(intraday)):
                        if i >= 2:  # Ensure at least 2 prior bars exist
                            # Activate Buy Countdown after Buy Setup completion
                            if intraday['Buy Setup'].iloc[i] == 'Buy Setup Completed':
                                buy_countdown_active = True
                                sell_countdown_active = False
                                buy_count = 0  # Reset Buy Countdown

                            # Activate Sell Countdown after Sell Setup completion
                            if intraday['Sell Setup'].iloc[i] == 'Sell Setup Completed':
                                sell_countdown_active = True
                                buy_countdown_active = False
                                sell_count = 0  # Reset Sell Countdown

                            # Buy Countdown: Close <= Low[2 bars ago]
                            if buy_countdown_active and intraday['Close'].iloc[i] <= intraday['Low'].iloc[i - 2]:
                                buy_count += 1
                                intraday.at[intraday.index[i], 'Buy Countdown'] = buy_count
                                if buy_count == 13:  # Complete Countdown at 13
                                    buy_countdown_active = False

                            # Sell Countdown: Close >= High[2 bars ago]
                            if sell_countdown_active and intraday['Close'].iloc[i] >= intraday['High'].iloc[i - 2]:
                                sell_count += 1
                                intraday.at[intraday.index[i], 'Sell Countdown'] = sell_count
                                if sell_count == 13:  # Complete Countdown at 13
                                    sell_countdown_active = False

                    return intraday


                intraday = calculate_td_countdown(intraday)


                def calculate_td_combo_countdown(data):
                    data['Buy Combo Countdown'] = None
                    data['Sell Combo Countdown'] = None

                    buy_countdown_active = False
                    sell_countdown_active = False
                    buy_count = 0
                    sell_count = 0

                    for i in range(len(data)):
                        if i >= 2:  # Ensure at least 2 prior bars exist
                            # Activate Buy Combo Countdown after Buy Setup completion
                            if data['Buy Setup'].iloc[i] == 'Buy Setup Completed':
                                buy_countdown_active = True
                                sell_countdown_active = False
                                buy_count = 0  # Reset Buy Combo Countdown

                            # Activate Sell Combo Countdown after Sell Setup completion
                            if data['Sell Setup'].iloc[i] == 'Sell Setup Completed':
                                sell_countdown_active = True
                                buy_countdown_active = False
                                sell_count = 0  # Reset Sell Combo Countdown

                            # Buy Combo Countdown: Close < Low[2 bars ago]
                            if buy_countdown_active and data['Close'].iloc[i] < data['Low'].iloc[i - 2]:
                                buy_count += 1
                                data.at[data.index[i], 'Buy Combo Countdown'] = buy_count
                                if buy_count == 13:  # Complete Combo Countdown at 13
                                    buy_countdown_active = False

                            # Sell Combo Countdown: Close > High[2 bars ago]
                            if sell_countdown_active and data['Close'].iloc[i] > data['High'].iloc[i - 2]:
                                sell_count += 1
                                data.at[data.index[i], 'Sell Combo Countdown'] = sell_count
                                if sell_count == 13:  # Complete Combo Countdown at 13
                                    sell_countdown_active = False

                    return data















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



                intraday =  calculate_td_pressure(intraday)


                def add_td_pressure_alert(intraday):
                    """
                    Adds a more meaningful TD Pressure Alert based on 5-bar pressure expansion.
                    Triggers alerts when TD Pressure increases 2x or 3x compared to 5 bars ago.
                    """

                    intraday['TD Pressure Alert'] = ''

                    for i in range(5, len(intraday)):
                        current_pressure = abs(intraday['TD Pressure'].iloc[i])
                        prev_pressure = abs(intraday['TD Pressure'].iloc[i - 5])

                        # Avoid divide-by-zero and false triggers
                        if prev_pressure == 0:
                            continue

                        ratio = current_pressure / prev_pressure

                        if ratio >= 4:
                            intraday.at[intraday.index[i], 'TD Pressure Alert'] = '🔥 Pressure 3x'
                        elif ratio >= 3:
                            intraday.at[intraday.index[i], 'TD Pressure Alert'] = '⚠️ Pressure 2x'

                    return intraday

                intraday =  add_td_pressure_alert(intraday)

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



                def calculate_td_rei(data):
                    """
                    Calculates TD REI as per the formula outlined in the reference image.

                    Steps:
                    1. Add the difference between the high of the current price bar and the high two bars earlier
                    to the difference between the low of the current price bar and the low two bars earlier.
                    2. Divide the result by the absolute price move over the 5-bar period (highest high - lowest low).
                    3. Multiply by 100 to normalize the result.
                    """

                    # Initialize TD REI column
                    data['TD REI'] = np.nan

                    for i in range(4, len(data)):
                        # Step 1: Calculate numerator (high_diff + low_diff)
                        high_diff = data['High'].iloc[i] - data['High'].iloc[i - 2]
                        low_diff = data['Low'].iloc[i] - data['Low'].iloc[i - 2]
                        numerator = high_diff + low_diff

                        # Step 2: Calculate denominator (absolute price move over the 5-bar period)
                        highest_high = data['High'].iloc[i - 4:i + 1].max()
                        lowest_low = data['Low'].iloc[i - 4:i + 1].min()
                        denominator = highest_high - lowest_low

                        # Avoid division by zero
                        if denominator == 0:
                            td_rei_value = 0
                        else:
                            # Step 3: Calculate TD REI
                            td_rei_value = (numerator / denominator) * 100

                        # Assign the calculated value
                        data.at[data.index[i], 'TD REI'] = td_rei_value

                    return data

                intraday = calculate_td_rei(intraday)

                def refine_td_rei_qualifiers(intraday):
                    """
                    Calculates TD REI using 5-bar high/low normalization,
                    then applies TD POQ-based logic to refine Buy/Sell signals.
                    """
                    # Initialize TD REI and POQ columns
                    intraday['TD REI'] = np.nan
                    intraday['TD POQ Signal'] = ''

                    # Step 1: Calculate TD REI
                    for i in range(4, len(intraday)):
                        high_diff = intraday['High'].iloc[i] - intraday['High'].iloc[i - 2]
                        low_diff = intraday['Low'].iloc[i] - intraday['Low'].iloc[i - 2]
                        numerator = high_diff + low_diff

                        highest_high = intraday['High'].iloc[i - 4:i + 1].max()
                        lowest_low = intraday['Low'].iloc[i - 4:i + 1].min()
                        denominator = highest_high - lowest_low

                        td_rei_value = (numerator / denominator) * 100 if denominator != 0 else 0
                        intraday.at[intraday.index[i], 'TD REI'] = td_rei_value

                    # Step 2: Refine TD REI qualifiers using POQ rules
                    for i in range(6, len(intraday) - 1):  # Start at 6 to ensure enough prior data
                        td_rei = intraday['TD REI'].iloc[i]
                        prev_rei = intraday['TD REI'].iloc[i - 1]

                        # Buy Signal Conditions
                        if td_rei < -40 and intraday['Close'].iloc[i] < intraday['Close'].iloc[i - 1]:
                            if (
                                intraday['Open'].iloc[i + 1] <= max(intraday['High'].iloc[i - 2:i]) and
                                intraday['High'].iloc[i + 1] > max(intraday['High'].iloc[i - 2:i])
                            ):
                                intraday.at[intraday.index[i], 'TD POQ Signal'] = 'Buy Signal'

                        # Sell Signal Conditions
                        if td_rei > 40 and intraday['Close'].iloc[i] > intraday['Close'].iloc[i - 1]:
                            if (
                                intraday['Open'].iloc[i + 1] >= min(intraday['Low'].iloc[i - 2:i]) and
                                intraday['Low'].iloc[i + 1] < min(intraday['Low'].iloc[i - 2:i])
                            ):
                                intraday.at[intraday.index[i], 'TD POQ Signal'] = 'Sell Signal'

                    return intraday

                intraday = refine_td_rei_qualifiers(intraday)



                def add_td_rei_alert(intraday):
                    """
                    Adds a TD REI Alert based on lookback expansion:
                    - "⚠️ REI 2x" if REI increases or decreases 2x vs 5 bars ago
                    - "🔥 REI 3x" if REI increases or decreases 3x vs 5 bars ago
                    """
                    intraday['TD REI Alert'] = ''

                    for i in range(5, len(intraday)):
                        current_rei = abs(intraday['TD REI'].iloc[i])
                        prev_rei = abs(intraday['TD REI'].iloc[i - 5])

                        if prev_rei == 0:
                            continue

                        ratio = current_rei / prev_rei

                        if ratio >= 3:
                            intraday.at[intraday.index[i], 'TD REI Alert'] = '🔥 REI 3x'
                        elif ratio >= 2:
                            intraday.at[intraday.index[i], 'TD REI Alert'] = '⚠️ REI 2x'

                    return intraday

                intraday = add_td_rei_alert(intraday)

                # Walk-the-Band logic 🚶🏻
                # Detect 3-bar walks outside Bollinger Bands
                def detect_walk_the_band(df):
                    df["Walk_Up"] = (df["F_numeric"] > df["F% Upper"]).astype(int)
                    df["Walk_Down"] = (df["F_numeric"] < df["F% Lower"]).astype(int)

                    df["Walk_Up_Streak"] = df["Walk_Up"].rolling(window=3).sum()
                    df["Walk_Down_Streak"] = df["Walk_Down"].rolling(window=3).sum()

                    df["Walk_Emoji"] = np.where(
                        df["Walk_Up_Streak"] >= 3, "🚶🏻",
                        np.where(df["Walk_Down_Streak"] >= 3, "🚶🏻", "")
                    )
                    return df

                # Apply the function
                intraday = detect_walk_the_band(intraday)



                def calculate_td_poq(intraday):
                    intraday['TD POQ'] = None

                    for i in range(5, len(intraday)):  # Start from the 6th row for sufficient prior data
                        # Buy POQ Logic: Qualified Upside Breakout
                        if (intraday['TD REI'].iloc[i] < -45 and  # TD REI in oversold condition
                            intraday['Close'].iloc[i - 1] > intraday['Close'].iloc[i - 2] and  # Previous close > close 2 bars ago
                            intraday['Open'].iloc[i] <= intraday['High'].iloc[i - 1] and  # Current open <= previous high
                            intraday['High'].iloc[i] > intraday['High'].iloc[i - 1]):  # Current high > previous high
                            intraday.at[intraday.index[i], 'TD POQ'] = 'Buy POQ'

                        # Sell POQ Logic: Qualified Downside Breakout
                        elif (intraday['TD REI'].iloc[i] > 45 and  # TD REI in overbought condition
                            intraday['Close'].iloc[i - 1] < intraday['Close'].iloc[i - 2] and  # Previous close < close 2 bars ago
                            intraday['Open'].iloc[i] >= intraday['Low'].iloc[i - 1] and  # Current open >= previous low
                            intraday['Low'].iloc[i] < intraday['Low'].iloc[i - 1]):  # Current low < previous low
                            intraday.at[intraday.index[i], 'TD POQ'] = 'Sell POQ'

                    return intraday

                intraday = calculate_td_poq(intraday)


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

                # Extract only the rows where TDST just formed
                tdst_points = intraday["TDST"].notna()


                def extract_tdst_level(intraday):
                    """
                    Extract numeric TDST level from 'TDST' string column.
                    Creates a new column 'TDST_Level'.
                    """
                    intraday['TDST_Level'] = intraday['TDST'].apply(
                        lambda x: float(x.split(":")[1].strip()) if isinstance(x, str) and ":" in x else np.nan
                    )
                    return intraday
                intraday = extract_tdst_level(intraday)


                def detect_sell_tdst_sniper(intraday):
                    """
                    Once a 'Sell TDST' is marked, track forward.
                    If price ever closes below that level, trigger a sniper 🎯 only once.
                    """
                    intraday['Sell TDST Sniper 🎯'] = ''
                    tdst_level = None
                    active = False

                    for i in range(len(intraday)):
                        tdst = intraday['TDST'].iloc[i]

                        # Look for a Sell TDST
                        if isinstance(tdst, str) and "Sell TDST" in tdst:
                            tdst_level = intraday['TDST_Level'].iloc[i]
                            active = True

                        # If active, wait for first cross below
                        if active and not pd.isna(tdst_level):
                            close = intraday['Close'].iloc[i]
                            if close < tdst_level:
                                intraday.at[intraday.index[i], 'Sell TDST Sniper 🎯'] = 'Sell TDST Sniper 🎯'
                                active = False  # Only trigger once

                    return intraday
                intraday = detect_sell_tdst_sniper(intraday)
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


                intraday =  calculate_tdst_partial_f(intraday)

                def detect_kijun_f_cross(intraday):
                    """
                    Detects when F_numeric crosses above or below Kijun_F.
                    Adds a new column 'Kijun_F_Cross' with values:
                    - 'Buy Kijun Cross' for upward cross
                    - 'Sell Kijun Cross' for downward cross
                    """

                    intraday['Kijun_F_Cross'] = ''

                    for i in range(1, len(intraday)):
                        prev_f = intraday['F_numeric'].iloc[i - 1]
                        prev_kijun = intraday['Kijun_F'].iloc[i - 1]
                        curr_f = intraday['F_numeric'].iloc[i]
                        curr_kijun = intraday['Kijun_F'].iloc[i]

                        # 📈 Cross above → Buy Kijun Cross
                        if prev_f < prev_kijun and curr_f > curr_kijun:
                            intraday.at[intraday.index[i], 'Kijun_F_Cross'] = 'Buy Kijun Cross'

                        # 📉 Cross below → Sell Kijun Cross
                        elif prev_f > prev_kijun and curr_f < curr_kijun:
                            intraday.at[intraday.index[i], 'Kijun_F_Cross'] = 'Sell Kijun Cross'

                    return intraday

                    intraday = detect_kijun_f_cross(intraday)
                def calculate_td_fib_range_fpercent(intraday):
                    """
                    Calculate Fib-style bands around F% using recent F% volatility.
                    Creates TD_Fib_Up_F and TD_Fib_Down_F columns.
                    """
                    intraday['TD_Fib_Up_F'] = np.nan
                    intraday['TD_Fib_Down_F'] = np.nan

                    for i in range(4, len(intraday)):
                        # True range style logic, but applied to F%
                        prev_close_f = intraday['F_numeric'].iloc[i - 1]
                        current_range = max(intraday['F_numeric'].iloc[i], prev_close_f) - \
                                        min(intraday['F_numeric'].iloc[i], prev_close_f)

                        prev_ranges = []
                        for j in range(i - 3, i):
                            prev_range = max(intraday['F_numeric'].iloc[j], intraday['F_numeric'].iloc[j - 1]) - \
                                        min(intraday['F_numeric'].iloc[j], intraday['F_numeric'].iloc[j - 1])
                            prev_ranges.append(prev_range)

                        avg_range = np.nanmean(prev_ranges)
                        max_range = max(current_range, avg_range)

                        fib_target = 1.618 * max_range

                        intraday.at[intraday.index[i], 'TD_Fib_Up_F'] = round(prev_close_f + fib_target, 2)
                        intraday.at[intraday.index[i], 'TD_Fib_Down_F'] = round(prev_close_f - fib_target, 2)

                    return intraday

                intraday = calculate_td_fib_range_fpercent(intraday)


                def detect_kijun_cross_emoji(df):
                    """
                    Detects when F_numeric crosses above or below Kijun_F and
                    assigns an emoji accordingly:
                    - "🕊️" when F_numeric crosses above Kijun_F (upward cross)
                    - "🐦‍⬛" when F_numeric crosses below Kijun_F (downward cross)
                    The result is stored in a new column 'Kijun_F_Cross_Emoji'.
                    """
                    df["Kijun_F_Cross_Emoji"] = ""
                    for i in range(1, len(df)):
                        prev_F = df.loc[i-1, "F_numeric"]
                        prev_K = df.loc[i-1, "Kijun_F"]
                        curr_F = df.loc[i, "F_numeric"]
                        curr_K = df.loc[i, "Kijun_F"]

                        # Upward cross: Was below the Kijun, now at or above
                        if prev_F < prev_K and curr_F >= curr_K:
                            df.loc[i, "Kijun_F_Cross_Emoji"] = "🕊️"
                        # Downward cross: Was above the Kijun, now at or below
                        elif prev_F > prev_K and curr_F <= curr_K:
                            df.loc[i, "Kijun_F_Cross_Emoji"] = "🐦‍⬛"
                    return df

                intraday = detect_kijun_cross_emoji(intraday)




                def detect_tenkan_cross_emoji(df):
                    """
                    Detects when Tenkan_F crosses above or below F_numeric.
                    Assigns:
                        🦢 for upward cross (Tenkan_F crosses above F_numeric)
                        🦜 for downward cross (Tenkan_F crosses below F_numeric)
                    """
                    df["Tenkan_F_Cross_Emoji"] = ""
                    for i in range(1, len(df)):
                        prev_t = df.loc[i-1, "Tenkan_F"]
                        prev_f = df.loc[i-1, "F_numeric"]
                        curr_t = df.loc[i, "Tenkan_F"]
                        curr_f = df.loc[i, "F_numeric"]

                        # Upward cross 🦢
                        if prev_t < prev_f and curr_t >= curr_f:
                            df.loc[i, "Tenkan_F_Cross_Emoji"] = "🦢"
                        # Downward cross 🦜
                        elif prev_t > prev_f and curr_t <= curr_f:
                            df.loc[i, "Tenkan_F_Cross_Emoji"] = "🦜"
                    return df
                intraday = detect_tenkan_cross_emoji(intraday)


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





                # Convert previous day levels to F% scale
                intraday["Yesterday Open F%"] = ((prev_open - prev_close) / prev_close) * 10000
                intraday["Yesterday High F%"] = ((prev_high - prev_close) / prev_close) * 10000
                intraday["Yesterday Low F%"] = ((prev_low - prev_close) / prev_close) * 10000
                intraday["Yesterday Close F%"] = ((prev_close - prev_close) / prev_close) * 10000  # Always 0




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
















                def calculate_clean_tdst_f(intraday):
                    """
                    TDST version using F_numeric instead of raw price.
                    Assigns first TDST value when Buy/Sell Setup completes, then blanks until a new one.
                    """

                    intraday['TDST_F'] = None
                    current_tdst = None

                    for i in range(9, len(intraday)):
                        # --- Buy Setup Completed ---
                        if intraday['Buy Setup'].iloc[i] == 'Buy Setup Completed':
                            bs1_f = intraday['F_numeric'].iloc[i - 8]  # Bar 1
                            bs2_f = intraday['F_numeric'].iloc[i - 7]  # Bar 2
                            current_tdst = f"Buy TDST F: {round(max(bs1_f, bs2_f), 2)}"
                            intraday.at[intraday.index[i], 'TDST_F'] = current_tdst

                        # --- Sell Setup Completed ---
                        elif intraday['Sell Setup'].iloc[i] == 'Sell Setup Completed':
                            ss1_f = intraday['F_numeric'].iloc[i - 8]  # Bar 1
                            current_tdst = f"Sell TDST F: {round(ss1_f, 2)}"
                            intraday.at[intraday.index[i], 'TDST_F'] = current_tdst

                        # --- Otherwise: blank until new setup forms
                        else:
                            intraday.at[intraday.index[i], 'TDST_F'] = None

                    return intraday
                intraday = calculate_clean_tdst_f(intraday)



                def detect_range_periods(df, window=3, threshold_pct=1.0):
                    """
                    Detects range-bound periods where the high-low range is within a percentage threshold
                    over a given rolling window.

                    Parameters:
                    - df: DataFrame with 'High' and 'Low' columns
                    - window: number of consecutive bars to qualify as ranging
                    - threshold_pct: max allowed range percentage (e.g., 1.0 = 1%)

                    Returns:
                    - A Boolean Series where True marks the last bar of each ranging segment.
                    """

                    # Calculate range % per bar
                    df["range_pct"] = ((df["High"] - df["Low"]) / df["Close"]) * 100

                    # Rolling window: count how many bars are within threshold
                    within_threshold = df["range_pct"] <= threshold_pct
                    rolling_sum = within_threshold.rolling(window=window).sum()

                    # Only mark the *last* bar of the ranging window
                    range_mask = (rolling_sum == window) & (within_threshold)

                    return range_mask

                intraday = calculate_clean_tdst_f(intraday)




                # intraday = add_inverted_honey_after_bee(intraday)





                def add_dual_honey_after_bee(df,
                                            bee_col="BBW_Tight_Emoji",
                                            honey1_col="BBW_Honey_Emoji",
                                            honey2_col="BBW_Honey_Emoji_2",
                                            f_col="F_numeric",
                                            supply_col="TD Supply Line F",
                                            kijun_col="Kijun_F"):

                    df[honey1_col] = ""
                    df[honey2_col] = ""

                    bee_indices = df.index[df[bee_col] == "🐝"]

                    for idx in bee_indices:
                        current_f = df.at[idx, f_col]
                        current_td = df.at[idx, supply_col]
                        bias = "above" if current_f > current_td else "below"

                        next_3 = df.loc[idx:].iloc[1:4]
                        if len(next_3) < 1:
                            continue

                        # 🍯 1: Trap (cross TD in opposite direction)
                        if bias == "above":
                            fakeout_mask = next_3[f_col] < next_3[supply_col]
                        else:
                            fakeout_mask = next_3[f_col] > next_3[supply_col]

                        if fakeout_mask.any():
                            honey1_idx = fakeout_mask.idxmax()
                            df.at[honey1_idx, honey1_col] = "🍯"

                            # 🍯 2: Recovery breakout → cross Kijun in original direction
                            rest = df.loc[honey1_idx:].iloc[1:10]  # look up to 10 bars after fakeout
                            if len(rest) < 1:
                                continue

                            if bias == "above":
                                recovery_mask = rest[f_col] > rest[kijun_col]
                            else:
                                recovery_mask = rest[f_col] < rest[kijun_col]

                            if recovery_mask.any():
                                honey2_idx = recovery_mask.idxmax()
                                df.at[honey2_idx, honey2_col] = "🍯"

                            break  # only one bee sequence allowed

                    return df

                intraday = add_dual_honey_after_bee(intraday)




                def add_hell_honey(df,
                                bee_col="BBW_Tight_Emoji",
                                honey1_col="BBW_Honey_Emoji",
                                honey2_col="BBW_Honey_Emoji_2",
                                honey3_col="BBW_Honey_Emoji_3",
                                f_col="F_numeric",
                                supply_col="TD Supply Line F",
                                kijun_col="Kijun_F"):

                    df[honey1_col] = ""
                    df[honey2_col] = ""
                    df[honey3_col] = ""

                    bee_indices = df.index[df[bee_col] == "🐝"]

                    for idx in bee_indices:
                        current_f = df.at[idx, f_col]
                        current_td = df.at[idx, supply_col]
                        bias = "above" if current_f > current_td else "below"

                        next_3 = df.loc[idx:].iloc[1:4]
                        if len(next_3) < 1:
                            continue

                        if bias == "above":
                            fakeout_mask = next_3[f_col] < next_3[supply_col]
                        else:
                            fakeout_mask = next_3[f_col] > next_3[supply_col]

                        if fakeout_mask.any():
                            honey1_idx = fakeout_mask.idxmax()
                            df.at[honey1_idx, honey1_col] = "🍯"

                            # Recovery breakout: cross Kijun in original bias
                            rest = df.loc[honey1_idx:].iloc[1:10]
                            if len(rest) < 1:
                                continue

                            if bias == "above":
                                recover_mask = rest[f_col] > rest[kijun_col]
                            else:
                                recover_mask = rest[f_col] < rest[kijun_col]

                            if recover_mask.any():
                                honey2_idx = recover_mask.idxmax()
                                df.at[honey2_idx, honey2_col] = "🍯"

                                # 🧨 Watch next 3 bars after Kijun breakout
                                trap_check = df.loc[honey2_idx:].iloc[1:4]
                                if len(trap_check) < 1:
                                    continue

                                if bias == "above":
                                    collapse = (trap_check[f_col] < trap_check[supply_col]) & (trap_check[f_col] < trap_check[kijun_col])
                                else:
                                    collapse = (trap_check[f_col] > trap_check[supply_col]) & (trap_check[f_col] > trap_check[kijun_col])

                                if collapse.any():
                                    collapse_idx = collapse.idxmax()
                                    df.at[collapse_idx, honey3_col] = "🍯🍯"
                                    break  # Only one full sequence

                    return df

                intraday = add_hell_honey(intraday)


                def add_hell_honey(df,
                                bee_col="BBW_Tight_Emoji",
                                honey1_col="BBW_Honey_Emoji",
                                honey2_col="BBW_Honey_Emoji_2",
                                honey3_col="BBW_Honey_Emoji_3",
                                f_col="F_numeric",
                                supply_col="TD Supply Line F",
                                kijun_col="Kijun_F"):

                    df[honey1_col] = ""
                    df[honey2_col] = ""
                    df[honey3_col] = ""

                    bee_indices = df.index[df[bee_col] == "🐝"]

                    for idx in bee_indices:
                        current_f = df.at[idx, f_col]
                        current_td = df.at[idx, supply_col]
                        bias = "above" if current_f > current_td else "below"

                        next_3 = df.loc[idx:].iloc[1:4]
                        if len(next_3) < 1:
                            continue

                        # Step 1: Trap 🍯
                        if bias == "above":
                            fakeout_mask = next_3[f_col] < next_3[supply_col]
                        else:
                            fakeout_mask = next_3[f_col] > next_3[supply_col]

                        if fakeout_mask.any():
                            honey1_idx = fakeout_mask.idxmax()
                            df.at[honey1_idx, honey1_col] = "🍯"

                            # Step 2: Breakout 🍯
                            rest = df.loc[honey1_idx:].iloc[1:10]
                            if len(rest) < 1:
                                continue

                            if bias == "above":
                                recover_mask = rest[f_col] > rest[kijun_col]
                            else:
                                recover_mask = rest[f_col] < rest[kijun_col]

                            if recover_mask.any():
                                honey2_idx = recover_mask.idxmax()
                                df.at[honey2_idx, honey2_col] = "🍯"

                                # Step 3: Betrayal 🍯🍯 — Look ahead 7 bars after second 🍯
                                betrayal_window = df.loc[honey2_idx:].iloc[1:8]
                                if len(betrayal_window) < 1:
                                    continue

                                if bias == "above":
                                    collapse_mask = (betrayal_window[f_col] < betrayal_window[supply_col]) & \
                                                    (betrayal_window[f_col] < betrayal_window[kijun_col])
                                else:
                                    collapse_mask = (betrayal_window[f_col] > betrayal_window[supply_col]) & \
                                                    (betrayal_window[f_col] > betrayal_window[kijun_col])

                                if collapse_mask.any():
                                    collapse_idx = collapse_mask.index[-1]  # 🧠 LAST occurrence
                                    df.at[collapse_idx, honey3_col] = "🍯🍯"
                                    break  # Full sequence completed

                    return df
















                # Initialize column
                intraday["Call_Progression"] = ""

                # Step tracker
                progress = 0
                highest_high_after_entry = None

                for i in range(1, len(intraday)):
                    f_now = intraday.loc[i, "F_numeric"]
                    supply_now = intraday.loc[i, "TD Supply Line F"]
                    kijun_now = intraday.loc[i, "Kijun_F"]
                    tenkan_now = intraday.loc[i, "Tenkan_F"]
                    ssa_now = intraday.loc[i, "Senkou_Span_A"]
                    ssb_now = intraday.loc[i, "Senkou_Span_B"]

                    if pd.isna(f_now) or pd.isna(supply_now):
                        continue

                    # Reset condition: F% falls below TD Supply → reset everything
                    if progress > 0 and f_now < supply_now:
                        progress = 0
                        highest_high_after_entry = None
                        continue

                    # Step 1: Entry if F% crosses TD Supply
                    if progress == 0 and f_now >= supply_now:
                        progress = 1
                        intraday.at[intraday.index[i], "Call_Progression"] = "1"
                        highest_high_after_entry = intraday.loc[i, "High"]
                        continue

                    # Step 2: Cross Kijun
                    if progress == 1 and f_now >= kijun_now:
                        progress = 2
                        intraday.at[intraday.index[i], "Call_Progression"] = "2"
                        highest_high_after_entry = max(highest_high_after_entry, intraday.loc[i, "High"])
                        continue

                    # Step 3: Tenkan crosses Kijun
                    if progress == 2 and tenkan_now >= kijun_now:
                        progress = 3
                        intraday.at[intraday.index[i], "Call_Progression"] = "3"
                        highest_high_after_entry = max(highest_high_after_entry, intraday.loc[i, "High"])
                        continue

                    # Step 4: SSA crosses SSB
                    if progress == 3 and ssa_now >= ssb_now:
                        progress = 4
                        intraday.at[intraday.index[i], "Call_Progression"] = "4"
                        highest_high_after_entry = max(highest_high_after_entry, intraday.loc[i, "High"])
                        continue

                    # 🟥 PUT Progression (Mirror of Call)
                    put_stage = 0
                    put_progression_y = []
                    put_progression_text = []
                    put_progression_x = []

                    for i in range(1, len(intraday)):
                        f_now = intraday.loc[i, "F_numeric"]
                        f_prev = intraday.loc[i - 1, "F_numeric"]
                        demand_now = intraday.loc[i, "TD Demand Line F"]
                        demand_prev = intraday.loc[i - 1, "TD Demand Line F"]
                        kijun_now = intraday.loc[i, "Kijun_F"]
                        kijun_prev = intraday.loc[i - 1, "Kijun_F"]
                        tenkan_now = intraday.loc[i, "Tenkan_F"]
                        tenkan_prev = intraday.loc[i - 1, "Tenkan_F"]
                        ssa_now = intraday.loc[i, "Senkou_Span_A"]
                        ssb_now = intraday.loc[i, "Senkou_Span_B"]

                        # Reset if F% goes back above TD Demand
                        if put_stage > 0 and f_now > demand_now:
                            put_stage = 0

                        # Stage 1: F% crosses below TD Demand Line
                        if put_stage == 0 and f_prev >= demand_prev and f_now < demand_now:
                            put_stage = 1
                            put_progression_x.append(intraday.loc[i, "Time"])
                            put_progression_y.append(f_now - 122)
                            put_progression_text.append("1")

                        # Stage 2: F% crosses below Kijun
                        elif put_stage == 1 and f_prev >= kijun_prev and f_now < kijun_now:
                            put_stage = 2
                            put_progression_x.append(intraday.loc[i, "Time"])
                            put_progression_y.append(f_now - 122)
                            put_progression_text.append("2")

                        # Stage 3: Tenkan crosses below Kijun
                        elif put_stage == 2 and tenkan_prev >= kijun_prev and tenkan_now < kijun_now:
                            put_stage = 3
                            put_progression_x.append(intraday.loc[i, "Time"])
                            put_progression_y.append(f_now - 122)
                            put_progression_text.append("3")

                        # Stage 4: SSA crosses below SSB
                        elif put_stage == 3 and ssa_now < ssb_now:
                            put_stage = 4
                            put_progression_x.append(intraday.loc[i, "Time"])
                            put_progression_y.append(f_now - 122)
                            put_progression_text.append("4")


                def tag_bb_pct_emoji(df):
                    """
                    Adds 🕸️ emoji when %b is near outer Bollinger Bands (upper ≥ 0.8 or lower ≤ 0.2).
                    """
                    if "F% Upper" in df.columns and "F% Lower" in df.columns and "F_numeric" in df.columns:
                        df["BB_pct"] = (
                            (df["F_numeric"] - df["F% Lower"]) /
                            (df["F% Upper"] - df["F% Lower"])
                        ).clip(0, 1)  # Ensure values stay between 0 and 1

                        df["BB_Web_Emoji"] = ""
                        df.loc[df["BB_pct"] >= 0.89, "BB_Web_Emoji"] = "🕸️"
                        df.loc[df["BB_pct"] <= 0.13, "BB_Web_Emoji"] = "🕸️"
                    return df
                intraday = tag_bb_pct_emoji(intraday)

                def add_tdst_ghost_crosses(intraday):
                    """
                    Add 👻 emoji when F% crosses TDST level:
                    - 👻 appears at F% +188 when crossing up during Buy Setup.
                    - 👻 appears at F% -188 when crossing down during Sell Setup.
                    """
                    intraday['TDST_Ghost 👻'] = ''
                    intraday['TDST_Ghost_y'] = np.nan

                    for i in range(1, len(intraday)):
                        # Skip if TDST level is not valid
                        if pd.isna(intraday['TDST_Level'].iloc[i]):
                            continue

                        f_prev = intraday['F%'].iloc[i-1]
                        f_curr = intraday['F%'].iloc[i]
                        tdst = intraday['TDST_Level'].iloc[i]

                        # 👻 Buy setup cross UP
                        if (
                            f_prev <= tdst and f_curr > tdst and
                            'Buy Setup' in str(intraday['TDST'].iloc[i])
                        ):
                            intraday.at[intraday.index[i], 'TDST_Ghost 👻'] = '👻'
                            intraday.at[intraday.index[i], 'TDST_Ghost_y'] = f_curr + 188

                        # 👻 Sell setup cross DOWN
                        elif (
                            f_prev >= tdst and f_curr < tdst and
                            'Sell Setup' in str(intraday['TDST'].iloc[i])
                        ):
                            intraday.at[intraday.index[i], 'TDST_Ghost 👻'] = '👻'
                            intraday.at[intraday.index[i], 'TDST_Ghost_y'] = f_curr - 188

                    return intraday




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






                # Extract only the rows where TDST just formed
                tdst_points = intraday["TDST"].notna()

                # Ensure F_numeric exists in intraday
                if "F_numeric" in intraday.columns:
                    # Compute High & Low of Day in F% scale
                    intraday["F% High"] = intraday["F_numeric"].cummax()  # Rolling highest F%
                    intraday["F% Low"] = intraday["F_numeric"].cummin()   # Rolling lowest F%


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


                intraday = detect_40ish_reversal(intraday)


                intraday = add_momentum(intraday, price_col="Close")  # => Momentum_2, Momentum_7


                recent_rows = intraday.tail(3)






                intraday = calculate_td_combo_countdown(intraday)







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

                # --- Assume intraday DataFrame is already loaded with required columns ---

                # 0) Normalize column names (replace non‑breaking spaces)
                intraday.columns = intraday.columns.str.replace("\xa0", " ")

                # 1) System 1: TD Supply/Demand breakout
                intraday["System 1"] = ""
                supply_cross = (
                    (intraday["F_numeric"] > intraday["TD Supply Line F"]) &
                    (intraday["F_numeric"].shift(1) <= intraday["TD Supply Line F"].shift(1))
                )
                demand_cross = (
                    (intraday["F_numeric"] < intraday["TD Demand Line F"]) &
                    (intraday["F_numeric"].shift(1) >= intraday["TD Demand Line F"].shift(1))
                )
                intraday.loc[supply_cross, "System 1"] = "UP"
                intraday.loc[demand_cross, "System 1"] = "DOWN"

                # 2) System 2: Kijun trend continuation (2‑bar sustain)
                intraday["System 2"] = ""
                for idx in intraday.index[intraday["System 1"] != ""]:
                    direction = intraday.at[idx, "System 1"]
                    pos = intraday.index.get_loc(idx)
                    window = intraday.iloc[pos+1:pos+3]  # next 2 bars
                    if direction == "UP" and (window["F_numeric"] > window["Kijun_F"]).all():
                        intraday.at[idx, "System 2"] = "🕊️"
                    elif direction == "DOWN" and (window["F_numeric"] < window["Kijun_F"]).all():
                        intraday.at[idx, "System 2"] = "🐦‍⬛"






#*************************************************************************************************************************************************************************************
#Volatility Booster
                # 5) Volatility Booster Pre: return all volatility signals before TD/Kijun cross
                intraday["Volatility Booster Pre"] = ""

                vol_cols = ["ATR_Exp_Alert", "ADX_Alert", "BBW_Tight_Emoji", "BBW Alert"]

                cross_mask = (
                    ((intraday["F_numeric"] > intraday["TD Supply Line F"]) & (intraday["F_numeric"].shift(1) <= intraday["TD Supply Line F"].shift(1))) |
                    ((intraday["F_numeric"] < intraday["TD Demand Line F"]) & (intraday["F_numeric"].shift(1) >= intraday["TD Demand Line F"].shift(1))) |
                    ((intraday["F_numeric"] > intraday["Kijun_F"]) & (intraday["F_numeric"].shift(1) <= intraday["Kijun_F"].shift(1))) |
                    ((intraday["F_numeric"] < intraday["Kijun_F"]) & (intraday["F_numeric"].shift(1) >= intraday["Kijun_F"].shift(1)))
                )
                cross_indices = intraday.index[cross_mask]

                for idx in cross_indices:
                    pos = intraday.index.get_loc(idx)
                    bw = intraday.iloc[max(0, pos-10):pos]  # lookback window

                    found = set()
                    for j in bw.index:
                        for col in vol_cols:
                            val = intraday.at[j, col]
                            if isinstance(val, str) and val.strip():
                                found.add(val)

                    if found:
                        intraday.at[idx, "Volatility Booster Pre"] = ''.join(found)

#*************************************************************************************************************************************************************************************

                            # 6) Volatility Booster Post: collect all volatility signals AFTER TD/Kijun cross
                    intraday["Volatility Booster Post"] = ""

                    vol_cols = ["ATR_Exp_Alert", "ADX_Alert", "BBW_Tight_Emoji", "BBW Alert"]

                    cross_mask = (
                        ((intraday["F_numeric"] > intraday["TD Supply Line F"]) & (intraday["F_numeric"].shift(1) <= intraday["TD Supply Line F"].shift(1))) |
                        ((intraday["F_numeric"] < intraday["TD Demand Line F"]) & (intraday["F_numeric"].shift(1) >= intraday["TD Demand Line F"].shift(1))) |
                        ((intraday["F_numeric"] > intraday["Kijun_F"]) & (intraday["F_numeric"].shift(1) <= intraday["Kijun_F"].shift(1))) |
                        ((intraday["F_numeric"] < intraday["Kijun_F"]) & (intraday["F_numeric"].shift(1) >= intraday["Kijun_F"].shift(1)))
                    )
                    cross_indices = intraday.index[cross_mask]

                    for idx in cross_indices:
                        pos = intraday.index.get_loc(idx)
                        fw = intraday.iloc[pos+1:pos+10]  # look ahead

                        found = set()
                        for j in fw.index:
                            for col in vol_cols:
                                val = intraday.at[j, col]
                                if isinstance(val, str) and val.strip():
                                    found.add(val)

                        if found:
                            intraday.at[idx, "Volatility Booster Post"] = ''.join(found)



#*************************************************************************************************************************************************************************************
                                # Ensure TK cross column is created first
                        tk_diff = intraday["Tenkan_F"] - intraday["Kijun_F"]
                        prev_tk_diff = tk_diff.shift(1)
                        bull_tk = (tk_diff >= 0) & (prev_tk_diff < 0)    # 🌞
                        bear_tk = (tk_diff <= 0) & (prev_tk_diff > 0)    # 🌙
                        intraday["Tenkan_Kijun_Cross"] = ""
                        intraday.loc[bull_tk, "Tenkan_Kijun_Cross"] = "🌞"
                        intraday.loc[bear_tk, "Tenkan_Kijun_Cross"] = "🌙"

                        # System 3 logic: Check if TK cross occurs within next 3 bars of System 2
                        intraday["System 3"] = ""

                        system2_idx = intraday.index[intraday["System 2"] != ""]

                        for idx in system2_idx:
                            pos = intraday.index.get_loc(idx)
                            fw = intraday.iloc[pos+1 : pos+4]  # Look forward 3 bars
                            tk_cross = fw["Tenkan_Kijun_Cross"]
                            if tk_cross.isin(["🌞", "🌙"]).any():
                                cross_emoji = tk_cross[tk_cross.isin(["🌞", "🌙"])].iloc[0]
                                intraday.at[idx, "System 3"] = cross_emoji  # 👈 assign to original System 2 bar



#*************************************************************************************************************************************************************************************
                                        # 8) System 4 – SSA vs SSB cross, triggered only after System 2 is active
                        intraday["System 4"] = ""

                        # Ensure numeric and fill early NaNs
                        intraday["SpanA_F"] = pd.to_numeric(intraday["SpanA_F"], errors="coerce").bfill()
                        intraday["SpanB_F"] = pd.to_numeric(intraday["SpanB_F"], errors="coerce").bfill()

                        # Get indices where System 2 triggered
                        s2_indices = intraday.index[intraday["System 2"] != ""]

                        # For each S2 point, look ahead for SSA/SSB cross
                        for idx in s2_indices:
                            pos = intraday.index.get_loc(idx)
                            fw = intraday.iloc[pos+1 : pos+10]  # next 9 bars

                            ssa_diff = fw["SpanA_F"] - fw["SpanB_F"]
                            prev_ssa_diff = ssa_diff.shift(1)

                            bull_ssa = (ssa_diff > 0) & (prev_ssa_diff <= 0)
                            bear_ssa = (ssa_diff < 0) & (prev_ssa_diff >= 0)

                            if bull_ssa.any():
                                intraday.at[idx, "System 4"] = "👼🏼"  # Bullish SSA cross
                            elif bear_ssa.any():
                                intraday.at[idx, "System 4"] = "👺"  # Bearish SSA cross



#*************************************************************************************************************************************************************************************


                    # 3) Volume Booster Pre: RVOL_5 spikes before breakout/Kijun cross
                intraday["Volume Booster Pre"] = ""
                for spike_idx in intraday.index[intraday["RVOL_5"] >= 1.2]:
                    pos = intraday.index.get_loc(spike_idx)
                    bw = intraday.iloc[max(0, pos-10):pos]  # look back up to 6 bars
                    cond = (
                        ((bw["F_numeric"] > bw["TD Supply Line F"]) &
                        (bw["F_numeric"].shift(1) <= bw["TD Supply Line F"].shift(1))).any() or
                        ((bw["F_numeric"] < bw["TD Demand Line F"]) &
                        (bw["F_numeric"].shift(1) >= bw["TD Demand Line F"].shift(1))).any() or
                        ((bw["F_numeric"] > bw["Kijun_F"]) &
                        (bw["F_numeric"].shift(1) <= bw["Kijun_F"].shift(1))).any() or
                        ((bw["F_numeric"] < bw["Kijun_F"]) &
                        (bw["F_numeric"].shift(1) >= bw["Kijun_F"].shift(1))).any()
                    )
                    if cond:
                        r = intraday.at[spike_idx, "RVOL_5"]
                        intraday.at[spike_idx, "Volume Booster Pre"] = "💣" if r>=1.8 else "🔥" if r>=1.5 else "🔺"

#*************************************************************************************************************************************************************************************
                # 4) Volume Booster Post: RVOL_5 spikes after breakout/Kijun cross
                intraday["Volume Booster Post"] = ""
                system1_idx = intraday.index[intraday["System 1"] != ""]
                k_up = (
                    (intraday["F_numeric"] > intraday["Kijun_F"]) &
                    (intraday["F_numeric"].shift(1) <= intraday["Kijun_F"].shift(1))
                )
                k_down = (
                    (intraday["F_numeric"] < intraday["Kijun_F"]) &
                    (intraday["F_numeric"].shift(1) >= intraday["Kijun_F"].shift(1))
                )
                kijun_idx = intraday.index[k_up | k_down]
                trigger_idx = system1_idx.union(kijun_idx)

                for t in trigger_idx:
                    pos = intraday.index.get_loc(t)
                    fw = intraday.iloc[pos+1:pos+10]
                    for j in fw.index:
                        r = intraday.at[j, "RVOL_5"]
                        intraday.at[j, "Volume Booster Post"] = "💣" if r>=1.8 else "🔥" if r>=1.5 else "🔺"
#*************************************************************************************************************************************************************************************


      # --------------------------------------------------------
                # ❶  create all blanks ONCE, then copy, then fill
                intraday = intraday.assign(
                    **{
                        "System 3": "",
                        "System 4": "",
                        "Volume Booster Pre": "",
                        "Volume Booster Post": ""
                    }
                )

                # defragment immediately, before any .at writes
                intraday = intraday.copy()
                # --------------------------------------------------------
                # … now run all the .at / .loc filling logic below …


                if gap_alert:
                    st.warning(gap_alert)






                # st.subheader(f"Live Market Snapshot - Last 5 Rows for {t}")
                # snapshot_cols = [                          "Time", "Close","VWAP", "F%", "VWAP_F","VWAP_Cross_Emoji",
                #        "RVOL_5", "Day Type", "High of Day",
                #         "Low of Day","TD Open","TD Trap","TD CLoP", "40ish","Tenkan_Kijun_Cross",
                #  "CTOD Alert","Alert_Kijun","Alert_Mid","Wealth Signal","Theta_Spike","F% Velocity","F% Cotangent","F% BBW","BBW_Ratio","F%_STD","+DI_F%","-DI_F%","ADX_F%","ADX_Trend_Emoji","ADX_Alert","OBV","OBV_Crossover",]  # Adjust as needed
                # snapshot_df = intraday[snapshot_cols].tail(5)

                # st.dataframe(snapshot_df, use_container_width=True)







                with st.expander("Show/Hide Data Table", expanded=False):
                    # Show data table, including new columns
                    cols_to_show = [
                          "Time", "Close","VWAP", "F%", "VWAP_F","VWAP_Cross_Emoji",
                       "RVOL_5", "Day Type", "High of Day",
                        "Low of Day", "TD Open","TD Trap","TD CLoP", "40ish","Tenkan_Kijun_Cross",
                 "CTOD Alert","Alert_Kijun","Alert_Mid","Wealth Signal","F% Velocity","F% Cotangent","F% BBW","BBW_Ratio","F%_STD","+DI_F%","-DI_F%","ADX_F%","ADX_Trend_Emoji","ADX_Alert","OBV","OBV_Crossover",'ATR',"ATR_Exp_Alert","Larry_Alert","Velocity_Spike","Ambush_Setup","Ambush_Emoji",'TD REI','TD REI Alert','TD POQ Signal','Buy Setup','Sell Setup','Buy Countdown','Sell Countdown','Buy Combo Countdown','Sell Combo Countdown','TD Demand Line F','TD Supply Line F','TDST','TDST_Level','TD Pressure','TD Pressure Alert','TD_Fib_Up_F','TD_Fib_Down_F','TDST_Partial_F','TD POQ',"BBW_Tight_Emoji","Volatility Booster Pre","SpanA_F","SpanB_F"
                    ]
                    sorted_df = intraday.sort_values("Time", ascending=False)

                    st.dataframe(sorted_df[cols_to_show], use_container_width=True)

                # # 5) Display in Streamlit
                # st.dataframe(
                #     intraday.loc[
                #         intraday["System 1"] != "",
                #         ["Time", "Volume Booster Pre","Volatility Booster Pre", "System 1", "System 2", "Volume Booster Post","Volatility Booster Post"]
                #     ].reset_index(drop=True),
                #     use_container_width=True
                # )

                ticker_tabs = st.tabs(["Interactive F% & Momentum", "Intraday Data Table"])





                with ticker_tabs[0]:
                    # -- Create Subplots: Row1=F%, Row2=Momentum
                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.002,
                         subplot_titles=("F% Chart","Candlestick"),
                        row_heights=[8, 1.2,0.8]  # 👈 60% for F%, 20% each for the rest


                    )






                                                        # 🟢 TRIGONOMETRIC COTAGENT / SECANT / TENKAN

                                                        # 🟢 F%

#**************************************************************************************************************************************************************************


  # (A) F% over time as lines+markers

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


#**************************************************************************************************************************************************************************


                                                        # 🟢 40% RETRACEMENT




                    # (A.1) 40ish Reversal (star markers)
                    mask_40ish = intraday["40ish"] != ""
                    scatter_40ish = go.Scatter(
                        x=intraday.loc[mask_40ish, "Time"],
                        y=intraday.loc[mask_40ish, "F_numeric"] + 233,
                        mode="markers",
                        marker_symbol="star",
                        marker_size=18,
                        marker_color="gold",
                        name="40ish Reversal",
                        text=intraday.loc[mask_40ish, "40ish"],

                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )
                    fig.add_trace(scatter_40ish, row=1, col=1)




#**************************************************************************************************************************************************************************

                    # # (A.2) Dashed horizontal line at 0
                    # fig.add_hline(
                    #     y=0,
                    #     line_dash="dash",
                    #     row=1, col=1,
                    #     annotation_text="0%",
                    #     annotation_position="top left"
                    # )


                                                        # 🟢 Wealth Buy


#**************************************************************************************************************************************************************************
                    # 🟢 Wealth Buy & Sell Signals (with Emojis 🦅 / 🦉 based on RVOL)
                    mask_buy_signal = intraday["Wealth Signal"].str.contains("Wealth Buy", na=False)
                    mask_sell_signal = intraday["Wealth Signal"].str.contains("Wealth Sell", na=False)

                    # Define emoji based on RVOL levels
                    def assign_emoji(row, signal_type):
                        rvol = row["RVOL_5"]
                        if rvol > 1.8:
                            return "🦅" if signal_type == "buy" else "🦉"
                        elif rvol >= 1.5:
                            return "🦅" if signal_type == "buy" else "🦉"
                        elif rvol >= 1.2:
                            return "🦅" if signal_type == "buy" else "🦉"
                        return ""

                    # Assign emojis to the DataFrame
                    intraday["Wealth_Emoji"] = ""
                    intraday.loc[mask_buy_signal, "Wealth_Emoji"] = intraday[mask_buy_signal].apply(lambda row: assign_emoji(row, "buy"), axis=1)
                    intraday.loc[mask_sell_signal, "Wealth_Emoji"] = intraday[mask_sell_signal].apply(lambda row: assign_emoji(row, "sell"), axis=1)

                    # Plot Wealth Buy (🦅)
                    scatter_buy_emoji = go.Scatter(
                        x=intraday.loc[mask_buy_signal, "Time"],
                        y=intraday.loc[mask_buy_signal, "F_numeric"] + 377,
                        mode="text",
                        text=intraday.loc[mask_buy_signal, "Wealth_Emoji"],
                        textposition="top center",
                        textfont=dict(size=38),
                        name="Wealth Buy Signal (🦅)",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Plot Wealth Sell (🦉)
                    scatter_sell_emoji = go.Scatter(
                        x=intraday.loc[mask_sell_signal, "Time"],
                        y=intraday.loc[mask_sell_signal, "F_numeric"] - 377,
                        mode="text",
                        text=intraday.loc[mask_sell_signal, "Wealth_Emoji"],
                        textposition="bottom center",
                        textfont=dict(size=38),
                        name="Wealth Sell Signal (🦉)",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Add to figure
                    fig.add_trace(scatter_buy_emoji, row=1, col=1)
                    fig.add_trace(scatter_sell_emoji, row=1, col=1)



                                                        # 🟢 YESTERDAY OHLC

#**************************************************************************************************************************************************************************





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

                    # 🎯 Add all lines to the F% plot
                    fig.add_trace(y_open_f_line, row=1, col=1)
                    fig.add_trace(y_high_f_line, row=1, col=1)
                    fig.add_trace(y_low_f_line, row=1, col=1)
                    fig.add_trace(y_close_f_line, row=1, col=1)




                                                        # 🟢 TRIGONOMETRIC COTAGENT / SECANT / TENKAN


#**************************************************************************************************************************************************************************

                    y_min = intraday["F_numeric"].min()
                    y_high = intraday["Yesterday High F%"].iloc[0]
                    y_max = intraday["F_numeric"].max()
                    y_low = round(intraday["Yesterday Low F%"].iloc[0], 4)
                    y_high = intraday["Yesterday High F%"].iloc[0]  # ensure this is already defined
                    y_high = intraday["Yesterday High F%"].iloc[0]  # ensure this is already defined
                    y_open = intraday["Yesterday Open F%"].iloc[0]

                    fig.add_hrect(
                        y0=y_high,
                        y1=y_max,
                        line_width=0,
                        fillcolor="#0ff",  # Strong white
                        opacity=0.35,       # Stronger than the usual 0.18
                        layer="below",
                        annotation_text="HEAVEN",
                        annotation_position="left",
                        annotation=dict(
                            font_size=10,
                            font_color="black"
                        )
                    )




                    # Define EARTH from Yesterday Close (0) to Yesterday Low
                    fig.add_hrect(
                        y0=y_open,

                        y1=0,             # yesterday's close (F% = 0)
                        line_width=0,
                        fillcolor="rgba(169, 169, 169, 0.25)",  # light grey (similar to LightGray)
                        opacity=0.25,
                        layer="below",
                        annotation_text="EARTH",
                        annotation_position="left",
                        annotation=dict(
                            font_size=10,
                            font_color="black"
                        )
                    )


                    fig.add_hrect(
                        y0=0,               # Yesterday's close (F% = 0)
                        y1=y_high - 1e-6,   # Just below yesterday's high
                        line_width=0,
                        fillcolor="#87CEEB",  # LightBlue
                        opacity=0.25,
                        layer="below",
                        annotation_text="SKY",
                        annotation_position="left",
                        annotation=dict(
                            font_size=10,
                            font_color="black"
                        )
                    )




           # Use a fixed offset for y0 if today's prices never go below yesterday’s low
                    # For instance, you could compute a value based on historical data or pick a fixed number.
                    # Here, I'll assume we want Hell to extend a fixed amount below yesterday's low.

                    hell_extension = 50  # change 50 to whatever value works best in your F% scale
                    y_hell_upper = y_low - 1e-6  # just slightly below yesterday's low
                    y_hell_lower = y_low - hell_extension

                 # Define HELL from Yesterday Low all the way to the bottom of the chart
                    fig.add_hrect(
                        y0=y_low + 1e-6,
                        y1=y_open,  # Start just below the red dashed line
                        line_width=0,
                        fillcolor="rgba(255, 100, 100, 0.3)",  # Gentle but firm red
                        opacity=0.3,
                        layer="below",
                        annotation_text="HELL",
                        annotation_position="left",
                        annotation=dict(
                            font_size=10,
                            font_color="black"
                        )
                    )


                    # ✅ Frozen Wasteland → from Yesterday Low to the chart’s minimum (ninth circle)
                    fig.add_hrect(
                        y0=y_min,
                        y1=y_low - 1e-6,  # just beneath the low
                        line_width=0,
                        fillcolor="rgba(180, 220, 255, 0.25)",  # Icy blue, evokes a frozen realm
                        opacity=0.25,
                        layer="below",
                        annotation_text="FROZEN",
                        annotation_position="left",
                        annotation=dict(
                            font_size=10,
                            font_color="black"
                        )
                    )

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

                    # 🎯 Add all lines to the F% plot
                    fig.add_trace(y_open_f_line, row=1, col=1)
                    fig.add_trace(y_high_f_line, row=1, col=1)
                    fig.add_trace(y_low_f_line, row=1, col=1)
                    fig.add_trace(y_close_f_line, row=1, col=1)




                                                        # 🟢 TRIGONOMETRIC COTAGENT / SECANT / TENKAN



#*******************************************************************************************************************************************************************************

    #  Cotangent Spikes (Skull 💀) - Catches both >3 and <-3
                    mask_cotangent_spike = intraday["F% Cotangent"].abs() > 3


                    scatter_cotangent_spike = go.Scatter(
                        x=intraday.loc[mask_cotangent_spike, "Time"],
                        y=intraday.loc[mask_cotangent_spike, "F_numeric"] - 89,  # Slightly offset for visibility
                        mode="text",
                        text="💀",
                        textposition="top center",
                        textfont=dict(size=18),  # Larger for emphasis
                        name="Cotangent Spike",
                        hovertext=intraday.loc[mask_cotangent_spike, "F% Cotangent"].round(2),  # Display rounded cotangent value
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Cotangent: %{hovertext}<extra></extra>"
                    )

         # Add to the F% plot (Row 1)
                    fig.add_trace(scatter_cotangent_spike, row=1, col=1)

#-------------------------------------------------------------------------------------

          #  Cosecant Spikes (Lightning ⚡) - Detects |F% Cosecant| > 20
                    mask_cosecant_spike = intraday["F% Cosecant"].abs() > 20

                    scatter_cosecant_spike = go.Scatter(
                        x=intraday.loc[mask_cosecant_spike, "Time"],
                        y=intraday.loc[mask_cosecant_spike, "F_numeric"] + 20,  # Offset for visibility
                        mode="text",
                        text="⚡",
                        textposition="top center",
                        textfont=dict(size=18, color="orange"),  # Larger and orange for emphasis
                        name="Cosecant Spike",
                        hovertext=intraday.loc[mask_cosecant_spike, "F% Cosecant"].round(2),  # Display rounded cosecant value
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Cosecant: %{hovertext}<extra></extra>"
                    )

                    # Add to the F% plot (Row 1)
                    fig.add_trace(scatter_cosecant_spike, row=1, col=1)



#---------------------------------------------------------------------------------------





                    # 🔵 Secant Spikes (Tornado 🌪) - Detects |F% Secant| > 3
                    mask_secant_spike = intraday["F% Secant"].abs() > 5

                    scatter_secant_spike = go.Scatter(
                        x=intraday.loc[mask_secant_spike, "Time"],
                        y=intraday.loc[mask_secant_spike, "F_numeric"] + 20,  # Offset for visibility
                        mode="text",
                        text="🌪",
                        textposition="top center",
                        textfont=dict(size=18, color="blue"),  # Large and blue for emphasis
                        name="Secant Spike",
                        hovertext=intraday.loc[mask_secant_spike, "F% Secant"].round(2),  # Display rounded secant value
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Secant: %{hovertext}<extra></extra>"
                    )

                    # Add to the F% plot (Row 1)
                    fig.add_trace(scatter_secant_spike, row=1, col=1)

    #  TRIGONOMETRIC COTAGENT / SECANT / TENKAN
#**************************************************************************************************************************************************************************

















                                                            #EXPANSION GROUP
#**************************************************************************************************************************************************************************
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






#  🟢   ATR Expansion
                    mask_atr_alert = intraday["ATR_Exp_Alert"] != ""

                    atr_alert_scatter = go.Scatter(
                        x=intraday.loc[mask_atr_alert, "Time"],
                        y=intraday.loc[mask_atr_alert, "F_numeric"]  + 89,  # place below F%
                        mode="text",
                        text=intraday.loc[mask_atr_alert, "ATR_Exp_Alert"],
                        textfont=dict(size=21),
                        name="ATR Expansion",
                        hoverinfo="text",
                        hovertext=intraday.loc[mask_atr_alert, "ATR_Exp_Alert"]
                    )

                    fig.add_trace(atr_alert_scatter, row=1, col=1)


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



#**************************************************************************************************************************************************************************






















                                                                    #VOLUME
#**************************************************************************************************************************************************************************


#---------------------------------------------------------------------------------------

# 🟢 RVOL
                                    # Mask for different RVOL thresholds
                    mask_rvol_extreme = intraday["RVOL_5"] > 1.8
                    mask_rvol_strong = (intraday["RVOL_5"] >= 1.5) & (intraday["RVOL_5"] < 1.8)
                    mask_rvol_moderate = (intraday["RVOL_5"] >= 1.2) & (intraday["RVOL_5"] < 1.5)

                    # Scatter plot for extreme volume spikes (red triangle)
                    scatter_rvol_extreme = go.Scatter(
                        x=intraday.loc[mask_rvol_extreme, "Time"],
                        y=intraday.loc[mask_rvol_extreme, "F_numeric"] + 5,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=13, color="red"),
                        name="RVOL > 1.8 (Extreme Surge)",
                        text="Extreme Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for strong volume spikes (orange triangle)
                    scatter_rvol_strong = go.Scatter(
                        x=intraday.loc[mask_rvol_strong, "Time"],
                        y=intraday.loc[mask_rvol_strong, "F_numeric"] + 8,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=13, color="orange"),
                        name="RVOL 1.5-1.79 (Strong Surge)",
                        text="Strong Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for moderate volume spikes (pink triangle)
                    scatter_rvol_moderate = go.Scatter(
                        x=intraday.loc[mask_rvol_moderate, "Time"],
                        y=intraday.loc[mask_rvol_moderate, "F_numeric"] + 8,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=13, color="pink"),
                        name="RVOL 1.2-1.49 (Moderate Surge)",
                        text="Moderate Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Add to the F% plot (Row 1)
                    fig.add_trace(scatter_rvol_extreme, row=1, col=1)
                    fig.add_trace(scatter_rvol_strong, row=1, col=1)
                    fig.add_trace(scatter_rvol_moderate, row=1, col=1)















                                                                                    #DEMARKER
#**************************************************************************************************************************************************************************
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



# # 🟢 TDST PARTIAL

#                     # Extract the dynamic partial TDST level (if not already done)
#                     intraday['TDST_Partial_F_Level'] = intraday['TDST_Partial_F'].apply(
#                         lambda x: float(x.split(": ")[1]) if isinstance(x, str) else np.nan
#                     )

#                     # # Plot TDST Partial F% Line
#                     fig.add_trace(
#                         go.Scatter(
#                              x=intraday['Time'],
#                             y=intraday['TDST_Partial_F_Level'],
#                             mode='lines',
#                             line=dict(width=1, color='#F8F8FF', dash='solid'),
#                             name='TDST Partial F%',
#                            hovertemplate="Time: %{x}<br>TDST Partial (F%): %{y:.2f}"
#                      ),
#                          row=1, col=1
#                      )

# # 🟢 TD REI

#                     # # ⚠️ Moderate REI Expansion (2x)
#                     rei_2x = intraday["TD REI Alert"] == "⚠️ REI 2x"
#                     fig.add_trace(
#                         go.Scatter(
#                             x=intraday.loc[rei_2x, "Time"],
#                             y=intraday.loc[rei_2x, "F_numeric"] - 10,  # Lower placement
#                             mode="text",
#                             text=["🦊"] * rei_2x.sum(),
#                             textposition="bottom center",
#                             textfont=dict(size=16),
#                             name="TD REI 2x",
#                             hovertemplate="Time: %{x}<br>F%: %{y}<br>TD REI 2x Expansion"
#                         ),
#                         row=1, col=1
#                     )

#                     # 🔥 Strong REI Expansion (3x)
#                     rei_3x = intraday["TD REI Alert"] == "🔥 REI 3x"
#                     fig.add_trace(
#                         go.Scatter(
#                             x=intraday.loc[rei_3x, "Time"],
#                             y=intraday.loc[rei_3x, "F_numeric"] - 15,  # Slightly lower to avoid overlap
#                             mode="text",
#                             text=["🦁"] * rei_3x.sum(),
#                             textposition="bottom center",
#                             textfont=dict(size=18),
#                             name="TD REI 3x",
#                             hovertemplate=(
#                             "Time: %{x}<br>"
#                             "F%: %{y:.2f}<br>"
#                             "TD REI: %{customdata:.2f}"
#                             "<extra></extra>"
#         ),
#                         ),
#                         row=1, col=1
#                     )
# # 🟢 TD PRESSURE
                    # ⚠️ Moderate Pressure Surge (2x)
                    pressure_2x = intraday["TD Pressure Alert"] == "⚠️ Pressure 2x"
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[pressure_2x, "Time"],
                            y=intraday.loc[pressure_2x, "F_numeric"] - 34,
                            mode="text",
                            text=["🚀"] * pressure_2x.sum(),
                            textposition="top center",
                            textfont=dict(size=13),
                            name="TD Pressure 2x",
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>TD Pressure 2x Surge"
                        ),
                        row=1, col=1
                    )

                    #  Strong Pressure Surge (3x)
                    pressure_3x = intraday["TD Pressure Alert"] == "🔥 Pressure 3x"
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[pressure_3x, "Time"],
                            y=intraday.loc[pressure_3x, "F_numeric"] - 34,
                            mode="text",
                            text=["🚀"] * pressure_3x.sum(),
                            textposition="top center",
                            textfont=dict(size=13),
                            name="TD Pressure 3x",
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>TD Pressure 3x Surge"
                        ),
                        row=1, col=1
                    )

# 🟢 TD TRAP

                  # (D) TD Trap Arrows - Only First Sell TD Trap
                    # ----------------------------------------------
                    td_trap_mask = intraday["TD Trap"].str.contains("Sell", na=False)  # Only Sell TD Traps

                    if not td_trap_mask.empty and td_trap_mask.any():
                        first_sell_trap_idx = td_trap_mask.idxmax()  # Get first occurrence index
                        first_sell_trap_time = intraday.loc[first_sell_trap_idx, "Time"]
                        first_sell_trap_value = intraday.loc[first_sell_trap_idx, "F_numeric"]

                        # Add annotation for first Sell TD Trap (Short = ST)
                        fig.add_annotation(
                            x=first_sell_trap_time,
                            y=first_sell_trap_value - 10,  # Offset to avoid overlap
                            text="ST",  # Short label instead of full text
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowcolor="yellow",
                            font=dict(size=12, color="red", family="Arial Black"),
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

                    # ✅ TD Buy Setup Complete – Green "9"
                    buy9_mask = intraday["Buy Setup"] == "Buy Setup Completed"
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[buy9_mask, "Time"],
                        y=intraday.loc[buy9_mask, "F_numeric"] - 89,
                        mode="text",
                        text=["9️⃣"] * buy9_mask.sum(),
                        textfont=dict(size=24),
                        textposition="bottom center",
                        name="TD Buy Setup 9",
                        hovertemplate="Time: %{x}<br>TD Sequential: Buy Setup 9<extra></extra>"
                    ), row=1, col=1)

                    # ✅ TD Sell Setup Complete – Red "9"
                    sell9_mask = intraday["Sell Setup"] == "Sell Setup Completed"
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[sell9_mask, "Time"],
                        y=intraday.loc[sell9_mask, "F_numeric"] + 89,
                        mode="text",
                        text=["9️⃣"] * sell9_mask.sum(),
                        textfont=dict(size=24, color="red"),
                        textposition="top center",
                        name="TD Sell Setup 9",
                        hovertemplate="Time: %{x}<br>TD Sequential: Sell Setup 9<extra></extra>"
                    ), row=1, col=1)







                    cloud_mask = intraday["Heaven_Cloud"] == "☁️"

                    fig.add_trace(go.Scatter(
                        x=intraday.loc[cloud_mask, "Time"],
                        y=intraday.loc[cloud_mask, "F_numeric"] + 233,
                        mode="text",
                        text=intraday.loc[cloud_mask, "Heaven_Cloud"],
                        textposition="top center",
                        textfont=dict(size=34),
                        name="Heaven ☁️",
                        hovertemplate="Time: %{x}<br>Price above TD Supply Line<extra></extra>"
                    ), row=1, col=1)




                    # Plot 🌧️ Drizzle Emoji on F% chart when price crosses down TD Demand Line
                    drizzle_mask = intraday["Drizzle_Emoji"] == "🌧️"

                    fig.add_trace(go.Scatter(
                        x=intraday.loc[drizzle_mask, "Time"],
                        y=intraday.loc[drizzle_mask, "F_numeric"] + 233,  # Position below the bar
                        mode="text",
                        text=intraday.loc[drizzle_mask, "Drizzle_Emoji"],
                        textposition="bottom center",
                        textfont=dict(size=32),
                        name="Price Dropped Below Demand 🌧️",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Crossed Below Demand<extra></extra>"
                    ), row=1, col=1)

#**************************************************************************************************************************************************************************








                                                                        #ICHIMOKU CLOUD

#**************************************************************************************************************************************************************************






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

 # 🟢   Tekan Something


                    tenkan_line = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["Tenkan_F"],
                        mode="lines",
                        line=dict(color="red", width=2, dash="dot"),
                        name="Tenkan (F%)"
                    )
                    fig.add_trace(tenkan_line, row=1, col=1)



 # 🟢   Kijun Something


                    kijun_line = go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Kijun_F"],
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Kijun (F% scale)"
                )
                    fig.add_trace(kijun_line, row=1, col=1)



 #🟢  Tenkan Cross
                    mask_tenkan_up = intraday["F% Tenkan Cross"] == "Up"
                    mask_tenkan_down = intraday["F% Tenkan Cross"] == "Down"

                    # 🟩 Tenkan Cross Up (Green "T")
                    scatter_tenkan_up = go.Scatter(
                        x=intraday.loc[mask_tenkan_up, "Time"],
                        y=intraday.loc[mask_tenkan_up, "F_numeric"] + 89,  # Offset to avoid overlap
                        mode="text",
                        text="T",
                        textposition="top center",
                        textfont=dict(size=21, color="green"),
                        name="F% Tenkan Up",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Cross Up"
                    )

                    # 🟥 Tenkan Cross Down (Red "T")
                    scatter_tenkan_down = go.Scatter(
                        x=intraday.loc[mask_tenkan_down, "Time"],
                        y=intraday.loc[mask_tenkan_down, "F_numeric"] - 89,  # Offset to avoid overlap
                        mode="text",
                        text="T",
                        textposition="bottom center",
                        textfont=dict(size=21, color="red"),
                        name="F% Tenkan Down",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Cross Down"
                    )

                    # Step 2: Add to the F% Plot (Row 1)
                    fig.add_trace(scatter_tenkan_up, row=1, col=1)  # Bullish Tenkan Cross
                    fig.add_trace(scatter_tenkan_down, row=1, col=1)  # Bearish Tenkan Cross


 # 🟢   Kijun Cross




                    mask_kijun_up = intraday["Kijun_F_Cross"] == "Up"
                    mask_kijun_down = intraday["Kijun_F_Cross"] == "Down"


                    # 🟢 Bullish Kijun Cross (Green "K")
                    mask_kijun_buy = intraday["Kijun_F_Cross"] == "Buy Kijun Cross"
                    scatter_kijun_buy = go.Scatter(
                        x=intraday.loc[mask_kijun_buy, "Time"],
                        y=intraday.loc[mask_kijun_buy, "F_numeric"] + 8,  # Offset slightly above
                        mode="text",
                        text="K",
                        textposition="top center",
                        textfont=dict(size=100, color="green"),
                        hovertemplate="Time: %{x}<br>Kijun Cross: Buy<extra></extra>",

                        name="Buy Kijun Cross"
                    )

                    # 🔴 Bearish Kijun Cross (Red "K")
                    mask_kijun_sell = intraday["Kijun_F_Cross"] == "Sell Kijun Cross"
                    scatter_kijun_sell = go.Scatter(
                        x=intraday.loc[mask_kijun_sell, "Time"],
                        y=intraday.loc[mask_kijun_sell, "F_numeric"] - 8,  # Offset slightly below
                        mode="text",
                        text="K",
                        textposition="bottom center",
                        textfont=dict(size=100, color="red"),
                        hovertemplate="Time: %{x}<br>Kijun Cross: Sell<extra></extra>",

                        name="Sell Kijun Cross"
                    )




 # 🟢   TK CROSS



                    # Mask for Tenkan-Kijun Crosses
                    mask_tk_sun = intraday["Tenkan_Kijun_Cross"] == "🌞"
                    mask_tk_moon = intraday["Tenkan_Kijun_Cross"] == "🌙"

                    # 🌞 Bullish Tenkan-Kijun Cross (Sun Emoji)
                    scatter_tk_sun = go.Scatter(
                        x=intraday.loc[mask_tk_sun, "Time"],
                        y=intraday.loc[mask_tk_sun, "F_numeric"] + 377,  # Offset for visibility
                        mode="text",
                        text="🌞",
                        textposition="top center",
                        textfont=dict(size=55),
                        name="Tenkan-Kijun Bullish Cross",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Above Kijun<extra></extra>"
                    )

                    # 🌙 Bearish Tenkan-Kijun Cross (Moon Emoji)
                    scatter_tk_moon = go.Scatter(
                        x=intraday.loc[mask_tk_moon, "Time"],
                        y=intraday.loc[mask_tk_moon, "F_numeric"] - 377,  # Offset for visibility
                        mode="text",
                        text="🌙",
                        textposition="bottom center",
                        textfont=dict(size=55),
                        name="Tenkan-Kijun Bearish Cross",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Below Kijun<extra></extra>"
                    )

                    # Add to the F% Plot
                    fig.add_trace(scatter_tk_sun, row=1, col=1)
                    fig.add_trace(scatter_tk_moon, row=1, col=1)





 # 🟢   Chikou



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





                 # Create separate masks for upward and downward crosses:
                    mask_kijun_up = intraday["Kijun_F_Cross_Emoji"] == "🕊️"
                    mask_kijun_down = intraday["Kijun_F_Cross_Emoji"] == "🐦‍⬛"

                    # Upward Cross Trace (🕊️)
                    up_cross_trace = go.Scatter(
                        x=intraday.loc[mask_kijun_up, "Time"],
                        y=intraday.loc[mask_kijun_up, "F_numeric"] + 144,  # Offset upward (adjust as needed)
                        mode="text",
                        text=intraday.loc[mask_kijun_up, "Kijun_F_Cross_Emoji"],
                        textposition="top center",  # Positioned above the point
                        textfont=dict(size=34),
                        name="Kijun Cross Up (🕊️)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Upward Cross: %{text}<extra></extra>"
                    )

                    # Downward Cross Trace (🐦‍⬛)
                    down_cross_trace = go.Scatter(
                        x=intraday.loc[mask_kijun_down, "Time"],
                        y=intraday.loc[mask_kijun_down, "F_numeric"] - 144,  # Offset downward
                        mode="text",
                        text=intraday.loc[mask_kijun_down, "Kijun_F_Cross_Emoji"],
                        textposition="bottom center",  # Positioned below the point
                        textfont=dict(size=34),
                        name="Kijun Cross Down (🐦‍⬛)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Downward Cross: %{text}<extra></extra>"
                    )

                    # Add the two traces to your figure (assuming fig is your Plotly Figure and row 1, col 1 is your F% subplot)
                    fig.add_trace(up_cross_trace, row=1, col=1)
                    fig.add_trace(down_cross_trace, row=1, col=1)




                    # # Masks
                    # mask_tenkan_up = intraday["Tenkan_F_Cross_Emoji"] == "🦢"
                    # mask_tenkan_down = intraday["Tenkan_F_Cross_Emoji"] == "🦜"

                    # # 🦢 Tenkan Up Cross
                    # scatter_tenkan_up = go.Scatter(
                    #     x=intraday.loc[mask_tenkan_up, "Time"],
                    #     y=intraday.loc[mask_tenkan_up, "F_numeric"] + 89,
                    #     mode="text",
                    #     text=["🦢"] * mask_tenkan_up.sum(),
                    #     textposition="top center",
                    #     textfont=dict(size=24),
                    #     name="Tenkan Cross Up (🦢)",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Above<extra></extra>"
                    # )

                    # # 🦜 Tenkan Down Cross
                    # scatter_tenkan_down = go.Scatter(
                    #     x=intraday.loc[mask_tenkan_down, "Time"],
                    #     y=intraday.loc[mask_tenkan_down, "F_numeric"] - 89,
                    #     mode="text",
                    #     text=["🦜"] * mask_tenkan_down.sum(),
                    #     textposition="bottom center",
                    #     textfont=dict(size=24),
                    #     name="Tenkan Cross Down (🦜)",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Below<extra></extra>"
                    # )

                    # # Add to plot
                    # fig.add_trace(scatter_tenkan_up, row=1, col=1)
                    # fig.add_trace(scatter_tenkan_down, row=1, col=1)



                    # Kumo Twist Bullish (👼🏼): SpanA crosses above SpanB
                    twist_bullish = (intraday["SpanA_F"] > intraday["SpanB_F"]) & (intraday["SpanA_F"].shift(1) <= intraday["SpanB_F"].shift(1))

                    scatter_twist_bullish = go.Scatter(
                        x=intraday.loc[twist_bullish, "Time"],
                        y=intraday.loc[twist_bullish, "SpanA_F"] + 377,
                        mode="text",
                        text=["👼🏼"] * twist_bullish.sum(),
                        textposition="top center",
                        textfont=dict(size=55),
                        name="Kumo Twist Bullish (👼🏼)",
                        hovertemplate="Time: %{x}<br>👼🏼 SpanA crossed above SpanB<extra></extra>"
                    )

                    # Kumo Twist Bearish (👺): SpanA crosses below SpanB
                    twist_bearish = (intraday["SpanA_F"] < intraday["SpanB_F"]) & (intraday["SpanA_F"].shift(1) >= intraday["SpanB_F"].shift(1))

                    scatter_twist_bearish = go.Scatter(
                        x=intraday.loc[twist_bearish, "Time"],
                        y=intraday.loc[twist_bearish, "SpanA_F"] - 377,
                        mode="text",
                        text=["👺"] * twist_bearish.sum(),
                        textposition="bottom center",
                        textfont=dict(size=55),
                        name="Kumo Twist Bearish (👺)",
                        hovertemplate="Time: %{x}<br>👺 SpanA crossed below SpanB<extra></extra>"
                    )

                    # Add to the F% plot
                    fig.add_trace(scatter_twist_bullish, row=1, col=1)
                    fig.add_trace(scatter_twist_bearish, row=1, col=1)



                    #                 # Calculate Chikou relation to current price
                    # intraday["Chikou_Position"] = np.where(intraday["Chikou"] > intraday["Close"], "above",
                    #                             np.where(intraday["Chikou"] < intraday["Close"], "below", "equal"))

                    # # Detect changes in Chikou relation
                    # intraday["Chikou_Change"] = intraday["Chikou_Position"].ne(intraday["Chikou_Position"].shift())

                    # # Filter first occurrence and changes
                    # chikou_shift_mask = intraday["Chikou_Change"] & (intraday["Chikou_Position"] != "equal")

                    # # Assign emojis for only these changes
                    # intraday["Chikou_Emoji"] = np.where(intraday["Chikou_Position"] == "above", "🕵🏻‍♂️",
                    #                             np.where(intraday["Chikou_Position"] == "below", "👮🏻‍♂️", ""))

                    # mask_chikou_above = chikou_shift_mask & (intraday["Chikou_Position"] == "above")

                    # fig.add_trace(go.Scatter(
                    #     x=intraday.loc[mask_chikou_above, "Time"],
                    #     y=intraday.loc[mask_chikou_above, "F_numeric"] + 233,
                    #     mode="text",
                    #     text=["🕵🏻‍♂️"] * mask_chikou_above.sum(),
                    #     textposition="top center",
                    #     textfont=dict(size=34),
                    #     name="Chikou Above Price",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Chikou moved above<extra></extra>"
                    # ), row=1, col=1)

                    # mask_chikou_below = chikou_shift_mask & (intraday["Chikou_Position"] == "below")

                    # fig.add_trace(go.Scatter(
                    #     x=intraday.loc[mask_chikou_below, "Time"],
                    #     y=intraday.loc[mask_chikou_below, "F_numeric"] - 233,
                    #     mode="text",
                    #     text=["👮🏻‍♂️"] * mask_chikou_below.sum(),
                    #     textposition="bottom center",
                    #     textfont=dict(size=34),
                    #     name="Chikou Below Price",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Chikou moved below<extra></extra>"
                    # ), row=1, col=1)





#**************************************************************************************************************************************************************************




#**************************************************************************************************************************************************************************
                                                                                #ADX


# #  # 🟢   ADX TREND



#                     # Mask for ADX trend changes
#                     mask_adx_up = intraday["ADX_Trend_Emoji"] == "📈"
#                     mask_adx_down = intraday["ADX_Trend_Emoji"] == "📉"

#                     # 📈 Bullish ADX Trend
#                     scatter_adx_up = go.Scatter(
#                         x=intraday.loc[mask_adx_up, "Time"],
#                         y=intraday.loc[mask_adx_up, "F_numeric"] + 60,
#                         mode="text",
#                         text="📈",
#                         textposition="top center",
#                         textfont=dict(size=22),
#                         name="ADX Bullish Trend",
#                         hovertemplate="Time: %{x}<br>F%: %{y}<br>Trend: Strong Bull<extra></extra>"
#                     )

#                     # 📉 Bearish ADX Trend
#                     scatter_adx_down = go.Scatter(
#                         x=intraday.loc[mask_adx_down, "Time"],
#                         y=intraday.loc[mask_adx_down, "F_numeric"] - 60,
#                         mode="text",
#                         text="📉",
#                         textposition="bottom center",
#                         textfont=dict(size=22),
#                         name="ADX Bearish Trend",
#                         hovertemplate="Time: %{x}<br>F%: %{y}<br>Trend: Strong Bear<extra></extra>"
#                     )

#                     # Add both to the F% plot (Row 1)
#                     fig.add_trace(scatter_adx_up, row=1, col=1)
#                     fig.add_trace(scatter_adx_down, row=1, col=1)















 #Special Group
#**************************************************************************************************************************************************************************
       #🟢  CTOD SIGNAL

                    # (C) CTOD Buy/Sell Triggers (Red & Green Dots)
                    # ----------------------------------------------
                    mask_ctod_buy = intraday["CTOD Alert"] == "Buy"
                    mask_ctod_sell = intraday["CTOD Alert"] == "Sell"

                    # Buy Alert (Green Dot)
                    scatter_ctod_buy = go.Scatter(
                        x=intraday.loc[mask_ctod_buy, "Time"],
                        y=intraday.loc[mask_ctod_buy, "F_numeric"],
                        mode="markers",
                        marker=dict(symbol="circle", size=10, color="green"),
                        name="CTOD Buy Signal",
                        text="Buy Triggered",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )
                    fig.add_trace(scatter_ctod_buy, row=1, col=1)

                    # Sell Alert (Red Dot)
                    scatter_ctod_sell = go.Scatter(
                        x=intraday.loc[mask_ctod_sell, "Time"],
                        y=intraday.loc[mask_ctod_sell, "F_numeric"],
                        mode="markers",
                        marker=dict(symbol="circle", size=10, color="red"),
                        name="CTOD Sell Signal",
                        text="Sell Triggered",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )
                    fig.add_trace(scatter_ctod_sell, row=1, col=1)


                    # Possibly add a dashed horizontal line at 0 momentum as well
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        row=1, col=1,
                        annotation_text="0 (Momentum)",
                        annotation_position="top left"
                    )

# ------------------------------------------------------------------------------------------------------------------------



        # #🟢 Velocity

                   # 🐆 Speed spike (Cheetah)
                    mask_cheetah = intraday["Velocity_Spike"] == True
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[mask_cheetah, "Time"],
                        y=intraday.loc[mask_cheetah, "F_numeric"]+ 21,
                        mode="text",  # <- no markers, only emoji
                        text=["🐆"] * mask_cheetah.sum(),
                        textposition="middle center",
                        textfont=dict(size=13),
                        name="Fast Velocity"
                    ), row=1, col=1)

                    # 🐢 Slow velocity (Turtle)
                    mask_turtle = intraday["Slow_Velocity_Emoji"] == "🐢"
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[mask_turtle, "Time"],
                        y=intraday.loc[mask_turtle, "F_numeric"] - 21,
                        mode="text",  # <- no markers
                        text=["🐢"] * mask_turtle.sum(),
                        textposition="middle center",
                        textfont=dict(size=13),
                        name="Slow Velocity"
                    ), row=1, col=1)
# ------------------------------------------------------------------------------------------------------------------------

#🟢  Ambush Signal: Slow Velocity + volatility


                    # 👁️ Ambush Marker
                    scatter_ambush = go.Scatter(
                        x=intraday.loc[intraday["Ambush_Setup"], "Time"],
                        y=intraday.loc[intraday["Ambush_Setup"], "F_numeric"] - 55,  # Place below F% for spacing
                        mode="text",
                        text=["👁️"] * intraday["Ambush_Setup"].sum(),
                        textposition="bottom center",
                        textfont=dict(size=21),
                        name="Ambush Watch",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Ambush Setup"
                    )

                    # Add to F% plot (Row 1)
                    fig.add_trace(scatter_ambush, row=1, col=1)






# ------------------------------------------------------------------------------------------------------------------------


      #🟢  Larry Signal For Entry



                                # Larry CALL  🛩️
                    call_mask = intraday["Larry_Alert"] == "CALL ALERT"
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[call_mask, "Time"],
                            y=intraday.loc[call_mask, "F_numeric"]+ 144,
                            mode="text",
                            text=["🛩️"] * call_mask.sum(),          # one bull per point
                            textposition="top center",
                            textfont=dict(size=21, color="green"),
                            name="Larry CALL"
                        ),
                        row=1, col=1
                    )

                    # Larry PUT  🛥️
                    put_mask = intraday["Larry_Alert"] == "PUT ALERT"
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[put_mask, "Time"],
                            y=intraday.loc[put_mask, "F_numeric"] -4,
                            mode="text",
                            text=["🛥️"] * put_mask.sum(),           # one bear per point
                            textposition="bottom center",
                            textfont=dict(size=21, color="red"),
                            name="Larry PUT"
                        ),
                        row=1, col=1
                    )









#**************************************************************************************************************************************************************************

                                                                # BOLLinger BAND


                    # (B) Upper Band
                    upper_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Upper"],
                        mode="lines",
                        line=dict(dash="dot", color="grey"),
                        name="Upper Band"
                    )

                    # (C) Lower Band
                    lower_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Lower"],
                        mode="lines",
                        line=dict(dash="dot", color="grey"),
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


                    # Ensure you filter out early NaNs due to rolling window
                    boll_df = intraday.dropna(subset=["F% Upper", "F% Lower", "F% MA"])

                    # Upper Band
                    fig.add_trace(go.Scatter(
                        x=boll_df["Time"],
                        y=boll_df["F% Upper"],
                        mode="lines",
                        line=dict(dash="dot", color="grey"),
                        name="Upper Band"
                    ), row=1, col=1)

                    # Lower Band
                    fig.add_trace(go.Scatter(
                        x=boll_df["Time"],
                        y=boll_df["F% Lower"],
                        mode="lines",
                        line=dict(dash="dot", color="grey"),
                        name="Lower Band"
                    ), row=1, col=1)

                    # Middle Band
                    fig.add_trace(go.Scatter(
                        x=boll_df["Time"],
                        y=boll_df["F% MA"],
                        mode="lines",
                        line=dict(color="white", dash="dash"),
                        name="Middle Band (14-MA)"
                    ), row=1, col=1)


                    # # F% Price crossing Middle Band (🦩)

                    # # Cross above middle band
                    # middle_up = (intraday["F_numeric"] > intraday["F% MA"]) & (intraday["F_numeric"].shift(1) <= intraday["F% MA"].shift(1))
                    # # Cross below middle band
                    # middle_down = (intraday["F_numeric"] < intraday["F% MA"]) & (intraday["F_numeric"].shift(1) >= intraday["F% MA"].shift(1))

                    # # 🦩 Upward Cross
                    # fig.add_trace(go.Scatter(
                    #     x=intraday.loc[middle_up, "Time"],
                    #     y=intraday.loc[middle_up, "F_numeric"] + 89,
                    #     mode="text",
                    #     text=["🦩"] * middle_up.sum(),
                    #     textposition="top center",
                    #     textfont=dict(size=34),
                    #     name="Middle Band Cross Up",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Crossed Above Middle Band 🦩<extra></extra>"
                    # ), row=1, col=1)

                    # # 🦩 Downward Cross
                    # fig.add_trace(go.Scatter(
                    #     x=intraday.loc[middle_down, "Time"],
                    #     y=intraday.loc[middle_down, "F_numeric"] - 89,
                    #     mode="text",
                    #     text=["🦤"] * middle_down.sum(),
                    #     textposition="bottom center",
                    #     textfont=dict(size=34),
                    #     name="Middle Band Cross Down",
                    #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Crossed Below Middle Band 🦩<extra></extra>"
                    # ), row=1, col=1)



                    # BBW Tight Compression (🐝) – 3 of last 5 bars under threshold
                    mask_bbw_tight = intraday["BBW_Tight_Emoji"] == "🐝"
                    scatter_bbw_tight = go.Scatter(
                        x=intraday.loc[mask_bbw_tight, "Time"],
                        y=intraday.loc[mask_bbw_tight, "F_numeric"] + 34,  # Offset upward
                        mode="text",
                        text=["🐝"] * mask_bbw_tight.sum(),
                        textposition="top center",
                        textfont=dict(size=21),
                        name="BBW Tight Compression (🐝)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>BBW Tight Compression (🐝)<extra></extra>"
                    )

                    fig.add_trace(scatter_bbw_tight, row=1, col=1)

                    # 🚶🏻 Walk-Upper Band
                    walk_up_mask = intraday["Walk_Up_Emoji"] == "🚶🏻"
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[walk_up_mask, "Time"],
                            y=intraday.loc[walk_up_mask, "F_numeric"] + 89,
                            mode="text",
                            text=["🚶🏻"] * walk_up_mask.sum(),
                            textposition="top center",
                            textfont=dict(size=34),
                            name="Walk Upper Band (🚶🏻)",
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>Walking Up Band 🚶🏻<extra></extra>"
                        ),
                        row=1, col=1
                    )

                    # 🧎‍♂️ Walk-Lower Band
                    walk_down_mask = intraday["Walk_Down_Emoji"] == "🧎‍♂️"
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[walk_down_mask, "Time"],
                            y=intraday.loc[walk_down_mask, "F_numeric"] - 89,
                            mode="text",
                            text=["🏃🏾‍♂️"] * walk_down_mask.sum(),
                            textposition="bottom center",
                            textfont=dict(size=34),
                            name="Walk Lower Band (🧎‍♂️)",
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>Walking Down Band 🧎‍♂️<extra></extra>"
                        ),
                        row=1, col=1
                    )





                    # honey_mask = intraday["BBW_Honey_Emoji"] == "🍯"
                    # fig.add_trace(go.Scatter(
                    #     x=intraday.loc[honey_mask, "Time"],
                    #     y=intraday.loc[honey_mask, "F_numeric"] - 34,
                    #     mode="text",
                    #     text=["🍯"] * honey_mask.sum(),
                    #     textposition="top center",
                    #     textfont=dict(size=22),
                    #     name="Post-Compression Honey (🍯)",
                    #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Honey zone 🍯<extra></extra>"
                    # ), row=1, col=1)





                    # First honey (trap alert)
                    h1 = intraday["BBW_Honey_Emoji"] == "🍯"
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[h1, "Time"],
                        y=intraday.loc[h1, "F_numeric"] + 34,
                        mode="text",
                        text=["🍯"] * h1.sum(),
                        textposition="top center",
                        textfont=dict(size=22),
                        name="🍯 Fakeout Trap"
                    ), row=1, col=1)

                    # Second honey (breakout confirm)
                    h2 = intraday["BBW_Honey_Emoji_2"] == "🍯"
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[h2, "Time"],
                        y=intraday.loc[h2, "F_numeric"] + 68,
                        mode="text",
                        text=["🍯"] * h2.sum(),
                        textposition="top center",
                        textfont=dict(size=22),
                        name="🍯 Confirmed Breakout"
                    ), row=1, col=1)



                    h3 = intraday["BBW_Honey_Emoji_3"] == "🍯🍯"
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[h3, "Time"],
                        y=intraday.loc[h3, "F_numeric"] - 102,
                        mode="text",
                        text=["🌶️🌶️"] * h3.sum(),
                        textposition="top center",
                        textfont=dict(size=22),
                        name="Double Fake Collapse 🍯🍯",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>🍯🍯 Post-Recovery Collapse<extra></extra>"
                    ), row=1, col=1)



                    # # ──────────────────────────────────────────────────────────────
                    # # 🏷️  Bollinger‑Band “Tag” emoji (price touches a band)
                    # #     – upper‑band tag  → 🏷️ shown just above the point
                    # #     – lower‑band tag  → 🏷️ shown just below the point
                    # # ──────────────────────────────────────────────────────────────
                    # # 1) build Boolean masks (you can tighten with a small buffer if you like)
                    # upper_tag = intraday["F_numeric"] >= intraday["F% Upper"]
                    # lower_tag = intraday["F_numeric"] <= intraday["F% Lower"]

                    # # 2) scatter for upper‑band tags
                    # fig.add_trace(
                    #     go.Scatter(
                    #         x=intraday.loc[upper_tag, "Time"],
                    #         y=intraday.loc[upper_tag, "F_numeric"] +55,       # nudge up a bit
                    #         mode="text",
                    #         text=["🏷️"] * upper_tag.sum(),
                    #         textposition="top center",
                    #         textfont=dict(size=18),
                    #         name="BB Upper Tag (🏷️)",
                    #         hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Tagged Upper Band<extra></extra>"
                    #     ),
                    #     row=1, col=1
                    # )

                    # # 3) scatter for lower‑band tags
                    # fig.add_trace(
                    #     go.Scatter(
                    #         x=intraday.loc[lower_tag, "Time"],
                    #         y=intraday.loc[lower_tag, "F_numeric"] - 55,       # nudge down a bit
                    #         mode="text",
                    #         text=["🏷️"] * lower_tag.sum(),
                    #         textposition="bottom center",
                    #         textfont=dict(size=18),
                    #         name="BB Lower Tag (🏷️)",
                    #         hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Tagged Lower Band<extra></extra>"
                    #     ),
                    #     row=1, col=1
                    # )
                    # ──────────────────────────────────────────────────────────────



                # # 🕸️ BB% Tag – walks near outer bands
                #     mask_web = intraday["BB_Web_Emoji"] == "🕸️"
                #     web_trace = go.Scatter(
                #         x = intraday.loc[mask_web, "Time"],
                #         y = intraday.loc[mask_web, "F_numeric"] + np.where(intraday.loc[mask_web, "BB_pct"] >= 0.8, 55, -55),
                #         mode = "text",
                #         text = intraday.loc[mask_web, "BB_Web_Emoji"],
                #         textposition = "middle center",
                #         textfont = dict(size=20),
                #         name = "BB Tag (🕸️)",
                #         hovertemplate = "Time: %{x}<br>F%: %{y:.2f}<br>%b near edge<extra></extra>",
                #         showlegend = False
                #     )
                #     fig.add_trace(web_trace, row=1, col=1)


#**************************************************************************************************************************************************************************




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
                        textfont=dict(size=34),
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
                        textfont=dict(size=34),
                        name="High of Day (🔋)",
                        hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                    ))

                    # Add progression markers to the plot
                    mask_call = intraday["Call_Progression"] != ""
                    fig.add_trace(go.Scatter(
                        x=intraday.loc[mask_call, "Time"],
                        y=intraday.loc[mask_call, "F_numeric"] + 122,
                        mode="text",
                        text=intraday.loc[mask_call, "Call_Progression"],
                        textposition="top center",
                        textfont=dict(size=21, color="white"),
                        name="Call Progression"
                    ))

                    # ✅ Add to plot
                    fig.add_trace(go.Scatter(
                        x=put_progression_x,
                        y=put_progression_y,
                        mode="text",
                        text=put_progression_text,
                        textposition="bottom center",
                        textfont=dict(size=18, color="red"),
                        name="Put Progression 📉",
                        hovertemplate="Stage %{text} (PUT)<extra></extra>"
                    ), row=1, col=1)





                    # # Step 1: Find Low of Day
                    # lod_index = intraday["Low"].idxmin()
                    # lod_time = intraday.loc[lod_index, "Time"]
                    # lod_f = intraday.loc[lod_index, "F_numeric"]

                    # # Step 2: After LOD, find first bar where F% crosses above TD Supply Line F
                    # stage1_mask = (intraday.index > lod_index) & (
                    #     intraday["F_numeric"] > intraday["TD Supply Line F"]
                    # )

                    # if stage1_mask.any():
                    #     entry1_index = intraday[stage1_mask].index[0]
                    #     entry1_time = intraday.loc[entry1_index, "Time"]
                    #     entry1_f = intraday.loc[entry1_index, "F_numeric"]

                    #     # Plot the "1" emoji
                    #     fig.add_trace(go.Scatter(
                    #         x=[entry1_time],
                    #         y=[entry1_f + 18],  # small offset
                    #         mode="text",
                    #         text=["1"],
                    #         textposition="top center",
                    #         textfont=dict(size=22),
                    #         name="Stage 1 Entry",
                    #         hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                    #     ))


                    # # Step 3: After "1", find first bar where F_numeric crosses above Kijun_F
                    # if stage1_mask.any():
                    #     # Find the index of the Stage 1 breakout
                    #     entry1_index = intraday[stage1_mask].index[0]

                    #     # Mask for bars after Stage 1
                    #     stage2_mask = (intraday.index > entry1_index) & (
                    #         intraday["F_numeric"] > intraday["Kijun_F"]
                    #     )

                    #     if stage2_mask.any():
                    #         entry2_index = intraday[stage2_mask].index[0]
                    #         entry2_time = intraday.loc[entry2_index, "Time"]
                    #         entry2_f = intraday.loc[entry2_index, "F_numeric"]

                    #         # Plot the "2" emoji
                    #         fig.add_trace(go.Scatter(
                    #             x=[entry2_time],
                    #             y=[entry2_f + 22],  # Offset a bit above the value
                    #             mode="text",
                    #             text=["2"],
                    #             textposition="top center",
                    #             textfont=dict(size=22),
                    #             name="Stage 2 Entry",
                    #             hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                    #         ))




                    # max_lookahead = 4  # how many bars forward to look
                    # start_idx = intraday.index.get_loc(entry2_index)
                    # end_idx = min(start_idx + max_lookahead, len(intraday) - 1)

                    # # Slice only the rows within that window
                    # window_df = intraday.iloc[start_idx+1 : end_idx+1]

                    # cross_candidates = window_df[window_df["F% Tenkan Cross"] == "Up"]

                    # if not cross_candidates.empty:
                    #     entry3_index = cross_candidates.index[0]
                    #     entry3_time = intraday.loc[entry3_index, "Time"]
                    #     entry3_f = intraday.loc[entry3_index, "F_numeric"]

                    #     fig.add_trace(go.Scatter(
                    #         x=[entry3_time],
                    #         y=[entry3_f + 22],
                    #         mode="text",
                    #         text=["3"],
                    #         textposition="top center",
                    #         textfont=dict(size=22),
                    #         name="Stage 3 Entry",
                    #         hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Cross<extra></extra>"
                    #     ))

#**************************************************************************************************************************************************************************

                    # intraday["F_shift"] = intraday["F_numeric"].shift(1)

                    # tdst_buy_mask = intraday["TDST"].str.contains("Buy TDST", na=False)
                    # tdst_sell_mask = intraday["TDST"].str.contains("Sell TDST", na=False)

                    #                     # Step 1: For each Buy TDST bar, get the F% level
                    # buy_tdst_levels = intraday.loc[tdst_buy_mask, "F_numeric"]

                    # # Step 2: Loop through each Buy TDST and track from that point forward
                    # for buy_idx, tdst_level in buy_tdst_levels.items():
                    #     # Get index location of the TDST signal
                    #     i = intraday.index.get_loc(buy_idx)

                    #     # Look at all bars forward from the TDST bar
                    #     future = intraday.iloc[i+1:].copy()

                    #     # Find where F% crosses and stays above the TDST level for 2 bars
                    #     above = future["F_numeric"] > tdst_level
                    #     two_bar_hold = above & above.shift(-1)

                    #     # Find the first time this happens
                    #     if two_bar_hold.any():
                    #         ghost_idx = two_bar_hold[two_bar_hold].index[0]  # first valid bar

                    #         # Plot 👻 emoji on the first bar
                    #         fig.add_trace(
                    #             go.Scatter(
                    #                 x=[intraday.at[ghost_idx, "Time"]],
                    #                 y=[intraday.at[ghost_idx, "F_numeric"] + 144],
                    #                 mode="text",
                    #                 text=["👻"],
                    #                 textposition="middle center",
                    #                 textfont=dict(size=40, color="purple"),
                    #                 name="Confirmed Buy TDST Breakout",
                    #                 hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    #             ),
                    #             row=1, col=1
                    #         )


                    # # Step 1: Get all Sell TDST points (each defines its own world)
                    # sell_tdst_levels = intraday.loc[tdst_sell_mask, "F_numeric"]
                    # sell_tdst_indices = list(sell_tdst_levels.index) + [intraday.index[-1]]  # add end of session as last boundary

                    # # Step 2: Loop through each Sell TDST "world"
                    # for i in range(len(sell_tdst_levels)):
                    #     tdst_idx = sell_tdst_levels.index[i]
                    #     tdst_level = sell_tdst_levels.iloc[i]

                    #     # Define the domain: from this Sell TDST until the next one (or end of day)
                    #     domain_start = intraday.index.get_loc(tdst_idx) + 1
                    #     domain_end = intraday.index.get_loc(sell_tdst_indices[i+1])  # next TDST or end

                    #     domain = intraday.iloc[domain_start:domain_end]

                    #     # Condition: F% crosses below and stays below for 2 bars
                    #     below = domain["F_numeric"] < tdst_level
                    #     confirmed = below & below.shift(-1)

                    #     if confirmed.any():
                    #         ghost_idx = confirmed[confirmed].index[0]
                    #         fig.add_trace(
                    #             go.Scatter(
                    #                 x=[intraday.at[ghost_idx, "Time"]],
                    #                 y=[intraday.at[ghost_idx, "F_numeric"] - 144],
                    #                 mode="text",
                    #                 text=["🫥"],
                    #                 textposition="middle center",
                    #                 textfont=dict(size=40, color="purple"),
                    #                 name="Confirmed Sell TDST Breakdown",
                    #                 hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    #             ),
                    #             row=1, col=1
                    #         )









                    # Update layout overall
                    fig.update_layout(
                        title=f"{t} – Day Trading Dashboard",
                        margin=dict(l=30, r=30, t=50, b=30),
                        height=1500,  # Increase overall figure height (default ~450-600)
                        width=1600,

                        showlegend=True
                    )


                    fig.update_yaxes(title_text="F% Scale", row=1, col=1)







                    fig.update_yaxes(title_text="F%", row=1, col=1)




                    st.plotly_chart(fig, use_container_width=True)






                # 📆 Daily Chart Overview (Fix)
                with st.expander("📆 Daily Chart Overview", expanded=True):
                    try:
                        daily_chart_data = yf.download(t, period="60d", interval="1d", progress=False)

                        if not daily_chart_data.empty:
                            daily_chart_data.reset_index(inplace=True)

                            if isinstance(daily_chart_data.columns, pd.MultiIndex):
                                daily_chart_data.columns = daily_chart_data.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

                            if "Date" not in daily_chart_data.columns:
                                daily_chart_data.rename(columns={"index": "Date"}, inplace=True)

                            daily_chart_data["Date"] = pd.to_datetime(daily_chart_data["Date"])

                            fig_daily = go.Figure()
                            fig_daily.add_trace(go.Candlestick(
                                x=daily_chart_data["Date"],
                                open=daily_chart_data["Open"],
                                high=daily_chart_data["High"],
                                low=daily_chart_data["Low"],
                                close=daily_chart_data["Close"],
                                name="Daily Candles"
                            ))

                            fig_daily.update_layout(
                                title=f"{t} – Daily Candlestick Chart (Past 60 Days)",
                                 height=2000,                # taller canvas
                                 width=1800,
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=30, r=30, t=40, b=20)
                            )

                            st.plotly_chart(fig_daily, use_container_width=True)
                        else:
                            st.warning("No daily data available.")
                    except Exception as e:
                        st.error(f"Failed to load daily chart: {e}")




            except Exception as e:
                st.error(f"Error fetching data for {t}: {e}")

