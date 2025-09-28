 

import streamlit as st
import numpy as np
import string       
import matplotlib.pyplot as plt
from math import log, sqrt
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
from datetime import timedelta, datetime
import io
import numbers
import base64  # <-- add this at the top of your file (with other imports)

import uuid
import hashlib
from typing import List
           

def compute_value_area(
        df: pd.DataFrame,
        mike_col: str | None = None,
        target_bins: int = 20,       # how many price buckets to aim for
        min_bin_width: float = 0.5   # never make a bucket wider than this
) -> tuple[float, float, pd.DataFrame]:
    """
    Full Market-Profile / Value-Area engine.
    Returns: (va_min, va_max, profile_df)

    • Works with Mike, F_numeric, or any price column you pass in `mike_col`.
    • Auto-adjusts bin width so prints never collapse to one bucket.
    """
    import numpy as np, pandas as pd, string

    # 0️⃣ pick column ────────────────────────────────────────────────────────
    if mike_col is None:
        if "Mike" in df.columns:
            mike_col = "Mike"
        elif "F_numeric" in df.columns:
            mike_col = "F_numeric"
        else:
            raise ValueError("Need a Mike or F_numeric column.")

    # 1️⃣ adaptive bin array ────────────────────────────────────────────────
    lo, hi = df[mike_col].min(), df[mike_col].max()
    price_range = max(hi - lo, 1e-6)            # avoid zero divide
    step = max(price_range / target_bins, min_bin_width)
    f_bins = np.arange(lo - step, hi + step, step)

    df = df.copy()
    df["F_Bin"] = pd.cut(
        df[mike_col],
        bins=f_bins,
        labels=[str(x) for x in f_bins[:-1]]
    )

    # 2️⃣ letter assignment (15-min alphabet) ──────────────────────────────
    if "Time" not in df.columns:
        df["Letter"] = "X"          # fallback if no intraday clock
    else:
        df = df[df["Time"].notna()]
        df["TimeIndex"] = pd.to_datetime(
            df["Time"], format="%I:%M %p", errors="coerce"
        )
        df = df[df["TimeIndex"].notna()]
        df["LetterIndex"] = (
            (df["TimeIndex"].dt.hour * 60 + df["TimeIndex"].dt.minute) // 15
        ).astype(int)
        df["LetterIndex"] -= df["LetterIndex"].min()

        letters = string.ascii_uppercase
        df["Letter"] = df["LetterIndex"].apply(
            lambda n: letters[n] if n < 26
            else letters[(n // 26) - 1] + letters[n % 26]
        )

    # 3️⃣ build Market-Profile dict ────────────────────────────────────────
    profile = {}
    for b in f_bins[:-1]:
        key = str(b)
        lets = df.loc[df["F_Bin"] == key, "Letter"].dropna().unique()
        if len(lets):
            profile[key] = "".join(sorted(lets))

    profile_df = (pd.DataFrame(profile.items(),
                               columns=["F% Level", "Letters"])
                    .astype({"F% Level": float}))
    profile_df["Letter_Count"] = profile_df["Letters"].str.len().fillna(0)

    # 4️⃣ 70 % value-area calc ─────────────────────────────────────────────
    tot = profile_df["Letter_Count"].sum()
    target = tot * 0.7
    poc_sorted = profile_df.sort_values("Letter_Count", ascending=False)

    cumulative, va_levels = 0, []
    for _, row in poc_sorted.iterrows():
        cumulative += row["Letter_Count"]
        va_levels.append(row["F% Level"])
        if cumulative >= target:
            break

    va_min, va_max = min(va_levels), max(va_levels)

    # # (optional) flag collapse
    # if va_min == va_max:
    #     st.warning("⚠️ Value area collapsed to one level – "
    #                "range too narrow, even after adaptive binning.")

    return va_min, va_max, profile_df

 
# =================
# Page Config
# =================
st.set_page_config(
    page_title="Volmike.com",
    layout="wide"
)


# ======================================
# Sidebar - User Inputs & Advanced Options
# ======================================
st.sidebar.header("Input Options")

default_tickers = ["SPY","VIXY","SOXX","NVDA","AMZN","MU","NQ=F","AMD","QCOM","SMCI","MSFT","uber", "AVGO","MRVL","QQQ","PLTR","AAPL","GOOGL","META","XLY","TSLA","nke","GM","c","DKNG","CHWY","ETSY","CART","W","KBE","wfc","hood","PYPL","coin","bac","jpm","BTC-USD","ETH-USD","XRP-USD","ADA-USD","SOL-USD","DOGE-USD"]
tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=default_tickers,
    default=["NVDA"]  # Start with one selected
)

# Date range inputs
start_date = st.sidebar.date_input("Start Date", value=date(2025, 8, 1))
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

                               # ➕ Add Yesterday's Range
                    yesterday_range = prev_high - prev_low
                    yesterday_range_str = f"{yesterday_range:.2f}"

                    # ───────── 1.5) Yesterday's intraday for Value Area ─────────
                
        
                              # ───────── 1.5) Yesterday's intraday for Value Area ─────────
                # Ensure start_date is a date object
                if isinstance(start_date, str):
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                else:
                    start_dt = start_date                       # e.g., 2025-05-01
                
                yesterday_date = start_dt - timedelta(days=1)
                
                intraday_yesterday = yf.download(
                    t,
                    start=yesterday_date.strftime("%Y-%m-%d"),
                    end=start_dt.strftime("%Y-%m-%d"),          # up to but not including start_dt
                    interval=timeframe,                         # same 2 m / 5 m / etc.
                    progress=False
                )
                
                # ── Flatten multi-index columns so 'Close' is a Series ──
                if isinstance(intraday_yesterday.columns, pd.MultiIndex):
                    intraday_yesterday.columns = intraday_yesterday.columns.map(
                        lambda c: c[0] if isinstance(c, tuple) else c
                    )
                
                # Default values
                yva_min = yva_max = None
                
                if not intraday_yesterday.empty:
                
                    # Bring Datetime out of the index
                    intraday_yesterday = intraday_yesterday.reset_index()
                
                    # Build 'Time' column for letter coding (e.g., 09:35 AM)
                    intraday_yesterday["Time"] = intraday_yesterday["Datetime"].dt.strftime("%I:%M %p")
                
                    # Choose price column for VA: F_numeric → Mike → fallback Close
                    if "F_numeric" in intraday_yesterday.columns:
                        mike_col_va = "F_numeric"
                    elif "Mike" in intraday_yesterday.columns:
                        mike_col_va = "Mike"
                    else:
                        mike_col_va = "Close"


           
                    try:
                        yva_min, yva_max, y_profile_df = compute_value_area(
                            intraday_yesterday,
                            mike_col="Close" ,
                  
                        )


                      # 🔁 Convert price VA into F%
                        if prev_close:
                            yva_min_f = round((yva_min - prev_close) / prev_close * 100, 1)
                            yva_max_f = round((yva_max - prev_close) / prev_close * 100, 1)
                        else:
                            yva_min_f = yva_min
                            yva_max_f = yva_max
                    except Exception as e:
                        st.warning(f"Could not compute yesterday VA for {t}: {e}")
                else:
                    st.warning(f"No intraday data for yesterday on {t}.")

            

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



              # 1) Define a helper function to calculate *historical* annualized volatility
             
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

                    # # **Corrected Logic**
                    # if first_open > prev_high:  # Must open *above* previous high to count as gap up
                    #     if gap_percentage > gap_threshold_decimal:
                    #         gap_alert = "🚀 UP GAP ALERT"
                    #         gap_type = "UP"
                    # elif first_open < prev_low:  # Must open *below* previous low to count as gap down
                    #     if gap_percentage < -gap_threshold_decimal:
                    #         gap_alert = "🔻 DOWN GAP ALERT"
                    #         gap_type = "DOWN"


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
 
              

                def add_unit_percentage(intraday_df):
                    if intraday_df.empty:
                        intraday_df["Unit%"] = ""
                        return intraday_df
                
                    intraday_df = intraday_df.copy()
                    intraday_df["Unit%"] = ""
                
                    for i in range(len(intraday_df)):
                        open_price = intraday_df.iloc[i]["Open"]
                        close_price = intraday_df.iloc[i]["Close"]
                
                        if open_price > 0:
                            unit_mike = ((close_price - open_price) / open_price) * 10000
                            unit_mike = round(unit_mike, 0)
                            unit_mike_str = f"{int(unit_mike)}%"
                            intraday_df.at[i, "Unit%"] = unit_mike_str
                        else:
                            intraday_df.at[i, "Unit%"] = ""
                
                    return intraday_df
                intraday = add_unit_percentage(intraday)

                def calculate_vector_percentage(intraday_df):
                    if intraday_df.empty:
                        intraday_df["Vector%"] = ""
                        return intraday_df
                
                    intraday_df = intraday_df.copy()
                    intraday_df["Vector%"] = ""  # ← blank instead of "N/A"
                
                    total_rows = len(intraday_df)
                    num_vectors = total_rows // 3
                
                    for i in range(num_vectors):
                        i0 = i * 3
                        i2 = i0 + 2
                
                        open_price = intraday_df.iloc[i0]["Open"]
                        close_price = intraday_df.iloc[i2]["Close"]
                
                        vector_mike = ((close_price - open_price) / open_price) * 10000
                        vector_mike = round(vector_mike, 0)
                        vector_mike_str = f"{int(vector_mike)}%"
                
                        # Assign only to the third (last) bar of the vector
                        intraday_df.at[i2, "Vector%"] = vector_mike_str
                
                    return intraday_df

                intraday = calculate_vector_percentage(intraday)

                def add_unit_velocity(intraday_df):
                    if intraday_df.empty or "Unit%" not in intraday_df.columns:
                        intraday_df["Unit Velocity"] = ""
                        return intraday_df
                
                    intraday_df = intraday_df.copy()
                    intraday_df["Unit Velocity"] = ""
                
                    for i in range(1, len(intraday_df)):
                        prev = intraday_df.iloc[i - 1]["Unit%"]
                        curr = intraday_df.iloc[i]["Unit%"]
                
                        if isinstance(prev, str) and prev.strip().endswith("%") and \
                           isinstance(curr, str) and curr.strip().endswith("%"):
                
                            prev_val = int(prev.strip().replace("%", ""))
                            curr_val = int(curr.strip().replace("%", ""))
                            velocity = curr_val - prev_val
                
                            intraday_df.at[i, "Unit Velocity"] = f"{velocity:+d}%"
                        else:
                            intraday_df.at[i, "Unit Velocity"] = ""
                
                    # First row has no previous unit, so remains blank
                    intraday_df.at[0, "Unit Velocity"] = ""
                
                    return intraday_df

                intraday = add_unit_velocity(intraday)
                def add_vector_velocity(intraday_df):
                    if intraday_df.empty or "Vector%" not in intraday_df.columns:
                        intraday_df["Velocity"] = ""
                        return intraday_df
                
                    intraday_df = intraday_df.copy()
                    intraday_df["Velocity"] = ""
                
                    previous_vector_val = None
                
                    for i in range(len(intraday_df)):
                        vector_val = intraday_df.iloc[i]["Vector%"]
                
                        if isinstance(vector_val, str) and vector_val.strip().endswith("%") and vector_val.strip() != "":
                            current_val = int(vector_val.strip().replace("%", ""))
                
                            if previous_vector_val is not None:
                                velocity = current_val - previous_vector_val
                                intraday_df.at[i, "Velocity"] = f"{velocity:+d}%"
                            else:
                                intraday_df.at[i, "Velocity"] = ""
                
                            previous_vector_val = current_val
                
                    return intraday_df

                intraday = add_vector_velocity(intraday)

                
               

                def add_unit_acceleration(intraday_df):
                    if intraday_df.empty or "Unit Velocity" not in intraday_df.columns:
                        intraday_df["Unit Acceleration"] = ""
                        return intraday_df
                
                    intraday_df = intraday_df.copy()
                    intraday_df["Unit Acceleration"] = ""
                
                    for i in range(2, len(intraday_df)):
                        prev = intraday_df.iloc[i - 1]["Unit Velocity"]
                        curr = intraday_df.iloc[i]["Unit Velocity"]
                
                        if isinstance(prev, str) and prev.strip().endswith("%") and \
                           isinstance(curr, str) and curr.strip().endswith("%"):
                
                            prev_val = int(prev.strip().replace("%", ""))
                            curr_val = int(curr.strip().replace("%", ""))
                            accel = curr_val - prev_val
                
                            intraday_df.at[i, "Unit Acceleration"] = f"{accel:+d}%"
                        else:
                            intraday_df.at[i, "Unit Acceleration"] = ""
                
                    return intraday_df

                intraday = add_unit_acceleration(intraday)


                
                def add_vector_acceleration(intraday_df):
                    if intraday_df.empty or "Velocity" not in intraday_df.columns:
                        intraday_df["Acceleration"] = ""
                        return intraday_df
                
                    intraday_df = intraday_df.copy()
                    intraday_df["Acceleration"] = ""
                
                    last_vector_row = None
                    last_velocity_val = None
                
                    for i in range(len(intraday_df)):
                        velocity_str = intraday_df.iloc[i]["Velocity"]
                
                        if isinstance(velocity_str, str) and velocity_str.strip().endswith("%") and velocity_str.strip() != "":
                            current_velocity = int(velocity_str.strip().replace("%", ""))
                
                            if last_velocity_val is not None:
                                acceleration = current_velocity - last_velocity_val
                                intraday_df.at[i, "Acceleration"] = f"{acceleration:+d}%"
                            else:
                                intraday_df.at[i, "Acceleration"] = ""
                
                            last_velocity_val = current_velocity
                            last_vector_row = i
                
                    return intraday_df

                intraday = add_vector_acceleration(intraday)
                def add_integrated_unit_acceleration(df):
                    """
                    Adds a column: 'Integrated_Unit_Acceleration'
                    which is the cumulative sum of Unit Acceleration.
                    """
                    df = df.copy()
                
                    # Clean Unit Acceleration
                    unit_accel_clean = (
                        df["Unit Acceleration"]
                        .fillna("0%")
                        .replace("", "0%")
                        .astype(str)
                        .str.replace("%", "", regex=False)
                    )
                
                    df["Unit_Acceleration_numeric"] = pd.to_numeric(unit_accel_clean, errors="coerce").fillna(0)
                
                    # Cumulative sum
                    df["Integrated_Unit_Acceleration"] = df["Unit_Acceleration_numeric"].cumsum()
                
                    return df
                
                # ✅ Apply to your dataframe
                intraday = add_integrated_unit_acceleration(intraday)

                def detect_acceleration_bursts(df, column="Acceleration", window=5, accel_threshold=15):
                    """
                    Detects clusters of acceleration bursts.
                    Flags 🔥 if ≥3 of the last `window` acceleration values exceed `accel_threshold`.
                    """
                    if column not in df.columns:
                        return df
                
                    # Create empty alert column
                    df["Acceleration_Alert"] = ""
                
                    # Clean and convert to numeric
                    accel_numeric = df[column].str.replace("%", "").replace("", np.nan).astype("float")
                    df["Accel_Spike"] = accel_numeric.abs() >= accel_threshold
                
                    # Rolling window cluster logic
                    for i in range(window, len(df)):
                        recent = df["Accel_Spike"].iloc[i - window:i]
                        if recent.sum() >= 5:
                            df.at[df.index[i], "Acceleration_Alert"] = "🔥"
                
                    return df

                intraday = detect_acceleration_bursts(intraday)

    
    
    

              
                def add_dual_jerk(df):
                  """
                  Adds:
                    - Jerk_Unit   = Δ(Unit Acceleration) on every bar
                    - Jerk_Vector = Δ(Vector Acceleration) ONLY on the last bar of each 3-bar vector
                  """
                  df = df.copy()
              
                  # --- Parse Unit Acceleration to numeric ---
                  ua = (
                      df["Unit Acceleration"]
                      .fillna("")           # blanks → ""
                      .replace("", "0%")    # interpret blank as 0%
                      .astype(str)
                      .str.replace("%", "", regex=False)
                  )
                  df["Unit_Acc_num"] = pd.to_numeric(ua, errors="coerce").fillna(0)
              
                  # --- Parse Vector Acceleration to numeric ---
                  va = (
                      df["Acceleration"]
                      .fillna("")           # blanks → ""
                      .replace("", "0%")
                      .astype(str)
                      .str.replace("%", "", regex=False)
                  )
                  df["Vector_Acc_num"] = pd.to_numeric(va, errors="coerce").fillna(0)
              
                  # --- Unit Jerk: Δ(Unit Acceleration) every bar ---
                  df["Jerk_Unit"] = df["Unit_Acc_num"].diff().fillna(0)
              
                  # --- Full Vector Jerk diff series ---
                  full_vec_diff = df["Vector_Acc_num"].diff().fillna(0)
              
                  # --- Vector Jerk only on 3rd bar of each vector (0-based idx 2,5,8...) ---
                  vec_jerk = pd.Series(0.0, index=df.index)
                  for i in range(len(df)):
                      if i % 3 == 2:
                          vec_jerk.iloc[i] = full_vec_diff.iloc[i]
                  df["Jerk_Vector"] = vec_jerk
                  df["Snap"] = df["Jerk_Vector"].diff()
 

                  return df

                intraday = add_dual_jerk(intraday)
                def mark_threshold_crosses(series, threshold=100):
                    """
                    Returns a boolean Series where:
                    - True at first bar where value crosses above `threshold`
                    - Ignores sustained values above threshold
                    - Re-arms after value falls back below
                    """
                    crosses = pd.Series(False, index=series.index)
                    armed = True
                
                    for i in range(1, len(series)):
                        curr = series.iloc[i]
                        prev = series.iloc[i - 1]
                
                        if armed and curr > threshold and prev <= threshold:
                            crosses.iloc[i] = True
                            armed = False
                        elif not armed and curr <= threshold:
                            armed = True
                
                    return crosses
                # Apply threshold logic on Jerk_Vector
                intraday["Jerk_Spike_Alert"] = mark_threshold_crosses(intraday["Jerk_Vector"], threshold=100)

                        
       
        
                def add_market_capacitance(df):
                    """
                    Computes a vector-style Capacitance:
                      - Charge = sum of RVOL_5 over 3 bars × direction of Velocity
                      - Voltage = abs(Vector Velocity)
                      - Capacitance = Charge / Voltage × 100 (scaled)
                
                    Assumes:
                      - 'RVOL_5' is bar-level volume
                      - 'Velocity' is already 3-bar vector velocity (in "%")
                    """
                    df = df.copy()
                
                    # Parse Velocity into numeric and sign
                    velocity_str = df["Velocity"].astype(str).str.strip()
                    df["Velocity_Sign"] = velocity_str.str[0].map({"+": 1, "-": -1}).fillna(0)
                    df["Voltage"] = pd.to_numeric(velocity_str.str.replace("%", "", regex=False), errors="coerce").abs()
                
                    # Initialize columns
                    df["Vector_Charge"] = 0.0
                    df["Vector_Capacitance"] = 0.0
                
                    # Only calculate every 3rd row (assuming Velocity is vector-style)
                    for i in range(2, len(df), 3):
                        charge_sum = df.loc[i-2:i, "RVOL_5"].sum()
                        sign = df.at[i, "Velocity_Sign"]
                        voltage = df.at[i, "Voltage"]
                
                        signed_charge = charge_sum * sign
                
                        df.at[i, "Vector_Charge"] = signed_charge
                        df.at[i, "Vector_Capacitance"] = (signed_charge / voltage * 100) if voltage not in [0, None, float("nan")] else 0
                
                    return df
                intraday =  add_market_capacitance(intraday)


                def add_charge_polarity(df):
                    """
                    Adds 'Charge_Polarity' and 'Charge_Bias' columns:
                      - Charge_Bias: 3-bar rolling sum of (Velocity_sign * RVOL_5)
                      - Charge_Polarity:
                          '🔴 Protonic'   → positive (bullish charge)
                          '🔵 Electronic' → negative (bearish charge)
                          '⚪ Neutral'     → near zero bias
                    """
                    df = df.copy()
                
                    # Ensure RVOL_5 is numeric
                    df["RVOL_5"] = pd.to_numeric(df["RVOL_5"], errors="coerce").fillna(0)
                
                    # Get sign of vector velocity (from column 'Velocity' as string with '%')
                    velocity_str = df["Velocity"].astype(str).str.strip()
                    velocity_sign = velocity_str.str[0].map({"+": 1, "-": -1}).fillna(0)
                
                    # Compute signed charge (1 bar)
                    df["Signed_Charge"] = velocity_sign * df["RVOL_5"]
                
                    # 3-bar rolling sum = Charge Bias
                    df["Charge_Bias"] = df["Signed_Charge"].rolling(window=3, min_periods=1).sum()
                
                    # Interpret polarity
                    def classify_bias(val, threshold=0.5):
                        if val > threshold:
                            return "🔵"
                        elif val < -threshold:
                            return "🔴"
                        else:
                            return "⚪"
                
                    df["Charge_Polarity"] = df["Charge_Bias"].apply(classify_bias)
                
                    return df
                intraday = add_charge_polarity(intraday)



              
                def add_unit_momentum_rvol(df):
                    if df.empty or "Unit Velocity" not in df.columns or "RVOL_5" not in df.columns:
                        df["Unit Momentum"] = ""
                        return df
                
                    df = df.copy()
                    df["Unit Momentum"] = ""
                
                    for i in range(len(df)):
                        v_str  = df.iloc[i]["Unit Velocity"]
                        rvol   = df.iloc[i]["RVOL_5"]
                
                        if isinstance(v_str, str) and v_str.strip().endswith("%") and isinstance(rvol, numbers.Number):
                            try:
                                v_val = int(v_str.strip().replace("%", ""))
                                df.at[i, "Unit Momentum"] = v_val * rvol
                            except ValueError:
                                df.at[i, "Unit Momentum"] = ""
                
                    return df
                intraday = add_unit_momentum_rvol(intraday)

                def add_vector_momentum_rvol(df):
                    if df.empty or "Velocity" not in df.columns or "RVOL_5" not in df.columns:
                        df["Vector Momentum"] = ""
                        return df
                
                    df = df.copy()
                    df["Vector Momentum"] = ""
                
                    for i in range(2, len(df), 3):  # Every third bar
                        v_str = df.iloc[i]["Velocity"]
                        rvol_sum = df.iloc[i-2:i+1]["RVOL_5"].sum()
                
                        if isinstance(v_str, str) and v_str.strip().endswith("%") and isinstance(rvol_sum, numbers.Number):
                            try:
                                v_val = int(v_str.strip().replace("%", ""))
                                df.at[i, "Vector Momentum"] = v_val * rvol_sum
                            except ValueError:
                                df.at[i, "Vector Momentum"] = ""
                
                    return df

                intraday = add_vector_momentum_rvol(intraday)

                def add_unit_force(intraday_df):
                      if intraday_df.empty or "Unit Acceleration" not in intraday_df.columns or "RVOL_5" not in intraday_df.columns:
                          intraday_df["Unit Force"] = ""
                          return intraday_df
                  
                      intraday_df = intraday_df.copy()
                      intraday_df["Unit Force"] = ""
                  
                      for i in range(len(intraday_df)):
                          accel_str = intraday_df.iloc[i]["Unit Acceleration"]
                          rvol = intraday_df.iloc[i]["RVOL_5"]
                  
                          if isinstance(accel_str, str) and accel_str.strip().endswith("%") and isinstance(rvol, numbers.Number):
                              try:
                                  accel_val = int(accel_str.strip().replace("%", ""))
                                  force = accel_val * rvol
                                  intraday_df.at[i, "Unit Force"] = force
                              except ValueError:
                                  intraday_df.at[i, "Unit Force"] = ""
                          else:
                              intraday_df.at[i, "Unit Force"] = ""
  
                      return intraday_df
                intraday = add_unit_force(intraday)
  
                def add_vector_force(intraday_df):
                  if intraday_df.empty or "Acceleration" not in intraday_df.columns or "RVOL_5" not in intraday_df.columns:
                      intraday_df["Vector Force"] = ""
                      return intraday_df
              
                  intraday_df = intraday_df.copy()
                  intraday_df["Vector Force"] = ""
              
                  for i in range(2, len(intraday_df), 3):
                      accel_str = intraday_df.iloc[i]["Acceleration"]
                      rvol_sum = intraday_df.iloc[i - 2:i + 1]["RVOL_5"].sum()
              
                      if isinstance(accel_str, str) and accel_str.strip().endswith("%") and isinstance(rvol_sum, numbers.Number):
                          try:
                              accel_val = int(accel_str.strip().replace("%", ""))
                              force = accel_val * rvol_sum
                              intraday_df.at[i, "Vector Force"] = force
                          except ValueError:
                              intraday_df.at[i, "Vector Force"] = ""
                      else:
                          intraday_df.at[i, "Vector Force"] = ""
              
                  return intraday_df

                intraday = add_vector_force(intraday)

                def add_mike_power(df):
                    df = df.copy()
                
                    if "Vector Force" not in df.columns or "Velocity" not in df.columns:
                        df["Power"] = ""
                        return df
                
                    # --- Clean velocity column ---
                    velocity_clean = (
                        df["Velocity"]
                        .fillna("0%")              # fill NaNs
                        .replace("", "0%")         # replace empty strings
                        .str.replace("%", "", regex=False)
                        .astype(float)
                    )
                
                    # --- Clean vector force column ---
                    force_clean = pd.to_numeric(df["Vector Force"], errors="coerce").fillna(0)
                
                    # --- Compute power ---
                    df["Power"] = force_clean * velocity_clean
                
                    return df

                intraday = add_mike_power(intraday)


              
                
                              
                def add_unit_energy(df):
                    if df.empty or "Unit Velocity" not in df.columns or "RVOL_5" not in df.columns:
                        df["Unit Energy"] = ""
                        return df
                
                    df = df.copy()
                    df["Unit Energy"] = ""
                
                    for i in range(len(df)):
                        v_str = df.iloc[i]["Unit Velocity"]
                        rvol = df.iloc[i]["RVOL_5"]
                
                        if isinstance(v_str, str) and v_str.strip().endswith("%") and isinstance(rvol, numbers.Number):
                            try:
                                v_val = int(v_str.strip().replace("%", ""))
                                energy = rvol * (v_val ** 2)
                                df.at[i, "Unit Energy"] = energy
                            except ValueError:
                                df.at[i, "Unit Energy"] = ""
                        else:
                            df.at[i, "Unit Energy"] = ""
                
                    return df

                intraday = add_unit_energy(intraday)
                def add_vector_energy(df):
                    if df.empty or "Velocity" not in df.columns or "RVOL_5" not in df.columns:
                        df["Vector Energy"] = ""
                        return df
                
                    df = df.copy()
                    df["Vector Energy"] = ""
                
                    for i in range(2, len(df), 3):
                        v_str = df.iloc[i]["Velocity"]
                        rvol_sum = df.iloc[i - 2:i + 1]["RVOL_5"].sum()
                
                        if isinstance(v_str, str) and v_str.strip().endswith("%") and isinstance(rvol_sum, numbers.Number):
                            try:
                                v_val = int(v_str.strip().replace("%", ""))
                                energy = rvol_sum * (v_val ** 2)
                                df.at[i, "Vector Energy"] = energy
                            except ValueError:
                                df.at[i, "Vector Energy"] = ""
                        else:
                            df.at[i, "Vector Energy"] = ""
                
                    return df

                intraday = add_vector_energy(intraday)



 

                def add_force_efficiency(df):
                    """
                    Adds two columns:
                      - Force_per_Range: Vector Force divided by last bar's Range
                      - Force_per_3bar_Range: Vector Force divided by 3-bar cumulative Range
                    """
                    if df.empty or "Vector Force" not in df.columns or "Range" not in df.columns:
                        df["Force_per_Range"] = ""
                        df["Force_per_3bar_Range"] = ""
                        return df
                
                    df = df.copy()
                    df["Force_per_Range"] = ""
                    df["Force_per_3bar_Range"] = ""
                
                    for i in range(2, len(df), 3):  # Only vector rows
                        force = df.iloc[i]["Vector Force"]
                        last_range = df.iloc[i]["Range"]
                        three_bar_range = df.iloc[i - 2:i + 1]["Range"].sum()
                
                        # Force / last bar range
                        if isinstance(force, numbers.Number) and isinstance(last_range, numbers.Number) and last_range != 0:
                            df.at[i, "Force_per_Range"] = force / last_range
                
                        # Force / 3-bar range
                        if isinstance(force, numbers.Number) and isinstance(three_bar_range, numbers.Number) and three_bar_range != 0:
                            df.at[i, "Force_per_3bar_Range"] = force / three_bar_range
                
                    return df
                
                # Apply it
                intraday = add_force_efficiency(intraday)


                
                def add_energy_efficiency(df):
                    """
                    Adds:
                    - Unit_Energy_per_Range: Unit Energy ÷ Range (each bar)
                    - Vector_Energy_per_3bar_Range: Vector Energy ÷ 3-bar cumulative Range (every 3rd bar)
                    """
                    df = df.copy()
                
                    # Initialize new columns
                    df["Unit_Energy_per_Range"] = ""
                    df["Vector_Energy_per_3bar_Range"] = ""
                
                    # --- Unit Energy per single-bar Range ---
                    for i in range(len(df)):
                        energy = df.iloc[i].get("Unit Energy")
                        rng = df.iloc[i].get("Range")
                
                        if isinstance(energy, numbers.Number) and isinstance(rng, numbers.Number) and rng != 0:
                            df.at[i, "Unit_Energy_per_Range"] = energy / rng
                
                    # --- Vector Energy per 3-bar Range ---
                    for i in range(2, len(df), 3):
                        energy = df.iloc[i].get("Vector Energy")
                        rng_sum = df.iloc[i - 2:i + 1]["Range"].sum()
                
                        if isinstance(energy, numbers.Number) and isinstance(rng_sum, numbers.Number) and rng_sum != 0:
                            df.at[i, "Vector_Energy_per_3bar_Range"] = energy / rng_sum
                
                    return df
                
                # Apply to intraday
                intraday = add_energy_efficiency(intraday)




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

             

                intraday["Acceleration_numeric"] = pd.to_numeric(intraday["Acceleration"].str.replace("%", ""), errors="coerce")
 
                            # Convert Unit% to numeric
                intraday["Unit%_Numeric"] = (
                    intraday["Unit%"].str.replace("%", "", regex=False).replace("", "0").astype(float)
                )
                
                # Calculate cumulative sum
                intraday["Cumulative_Unit"] = intraday["Unit%_Numeric"].cumsum()


                def add_kijun_displacement(df, period=26):
                    """
                    Adds 'Kijun_Cumulative' column to df:
                    Midpoint of highest and lowest Cumulative_Unit over a rolling period.
                    This is the Ichimoku Kijun-sen, but computed in displacement space.
                    """
                    df = df.copy()
                
                    # Ensure Cumulative_Unit is numeric
                    df["Cumulative_Unit"] = pd.to_numeric(df["Cumulative_Unit"], errors="coerce")
                
                    # Rolling midpoint of displacement
                    df["Kijun_Cumulative"] = (
                        df["Cumulative_Unit"]
                        .rolling(window=period, min_periods=1)
                        .apply(lambda x: (x.max() + x.min()) / 2)
                    )
                
                    return df
                
                # Apply to your intraday DataFrame
                intraday = add_kijun_displacement(intraday)

                def add_wave_intensity(df):
                    """
                    Adds 'Intensity' column to represent wave energy intensity:
                        Intensity = Power / Distance
                    where:
                        - Power is from your physics engine (already computed)
                        - Distance is the absolute change in Cumulative_Unit (price movement)
                
                    Notes:
                        - Handles division by zero or missing values gracefully.
                        - Outputs 0 when Power is 0 or no movement occurred.
                    """
                    df = df.copy()
                
                    # Ensure required columns are present and clean
                    df["Power"] = pd.to_numeric(df["Power"], errors="coerce").fillna(0)
                    df["Cumulative_Unit"] = pd.to_numeric(df["Cumulative_Unit"], errors="coerce")
                
                    # Calculate distance = movement of price (like amplitude)
                    df["Distance"] = df["Cumulative_Unit"].diff().abs().fillna(0)
                
                    # Avoid divide-by-zero by replacing 0 distances with np.nan temporarily
                    df["Intensity"] = df.apply(
                        lambda row: row["Power"] / row["Distance"] if row["Distance"] != 0 else 0,
                        axis=1
                    )
                
                    return df
                
                intraday = add_wave_intensity(intraday)

                def add_market_field_force(df, resistance_col="Kijun_F"):
                            df = df.copy()
                            
                            # Voltage = Velocity (numeric)
                            df["V_numeric"] = pd.to_numeric(df["Velocity"].str.replace("%", ""), errors="coerce")
                            
                            # Charge = RVOL_5
                            Q = df["RVOL_5"]
                            
                            # Distance = |resistance - current level|
                            d = (df[resistance_col] - df["Cumulative_Unit"]).abs().replace(0, np.nan)
                            
                            # Field = V / d
                            df["Field_Intensity"] = df["V_numeric"] / d
                            
                            # Force = Q * (V / d)
                            df["Electric_Force"] = Q * df["Field_Intensity"]
                            
                            return df
                        
                                  
                intraday = add_market_field_force(intraday)
     
                def add_wave_intensity(df):
                    """
                    Adds 'Wave_Intensity' = Power / Range, representing how much power is delivered per unit price movement.
                    Assumes 'Power' and 'Range' columns are already in df.
                    """
                    df = df.copy()
                    
                    # Ensure numeric values
                    power = pd.to_numeric(df["Power"], errors="coerce")
                    range_ = pd.to_numeric(df["Range"], errors="coerce").replace(0, np.nan)  # avoid div-by-zero
                
                    # Compute intensity
                    df["Wave_Intensity"] = power / range_
                    
                    return df
                
                intraday = add_wave_intensity(intraday)

                def add_integrated_accelerations(df):
                    """
                    Adds cumulative (integrated) acceleration:
                      - 'Acceleration_numeric' and 'Unit_Acceleration_numeric' = cleaned % values
                      - 'Integrated_Vector_Acceleration' = cumsum of vector acceleration
                      - 'Integrated_Unit_Acceleration' = cumsum of unit acceleration
                    """
                    df = df.copy()
                
                    # --- Clean Vector Acceleration ---
                    vec_accel = (
                        df["Acceleration"]
                        .fillna("0%")
                        .replace("", "0%")
                        .astype(str)
                        .str.replace("%", "", regex=False)
                    )
                    df["Acceleration_numeric"] = pd.to_numeric(vec_accel, errors="coerce").fillna(0)
                
                    # --- Clean Unit Acceleration ---
                    unit_accel = (
                        df["Unit Acceleration"]
                        .fillna("0%")
                        .replace("", "0%")
                        .astype(str)
                        .str.replace("%", "", regex=False)
                    )
                    df["Unit_Acceleration_numeric"] = pd.to_numeric(unit_accel, errors="coerce").fillna(0)
                
                    # --- Integrals (Cumulative Sums) ---
                    df["Integrated_Vector_Acceleration"] = df["Acceleration_numeric"].cumsum()
                    df["Integrated_Unit_Acceleration"] = df["Unit_Acceleration_numeric"].cumsum()
                
                    return df
                
                # ✅ Apply it
                intraday = add_integrated_accelerations(intraday)
                

 
                def add_volatility_composite(df, window=10, alpha=1.0, beta=1.0, gamma=1.0):
                    """
                    Adds rolling volatility measures and a composite Volatility_Composite to the DataFrame.
                    
                    New columns:
                      - Range                    = High − Low
                      - Acceleration_numeric     = cleaned acceleration values as float
                      - Jerk_numeric             = cleaned jerk values as float
                      - Volatility_Range         = rolling std of Range
                      - Volatility_Acceleration  = rolling std of Acceleration_numeric
                      - Volatility_Jerk          = rolling std of Jerk_numeric
                      - Volatility_Composite     = alpha*Volatility_Range + beta*Volatility_Acceleration + gamma*Volatility_Jerk
                    """
                    df = df.copy()
                    
                    # 1) Compute Range
                    df["Range"] = pd.to_numeric(df["High"], errors="coerce") - pd.to_numeric(df["Low"], errors="coerce")
                    
                    # 2) Prepare Acceleration_numeric
                    if "Acceleration_numeric" in df.columns:
                        accel_raw = df["Acceleration_numeric"].astype(float)
                    else:
                        accel_raw = df.get("Acceleration", pd.Series("0%", index=df.index))
                    # Clean and convert to float
                    accel_series = (
                        accel_raw
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .replace("", "0")
                        .astype(float)
                    )
                    df["Acceleration_numeric"] = accel_series.fillna(0)
                    
                    # 3) Prepare Jerk_numeric
                    jerk_raw = df.get("Jerk", pd.Series("0%", index=df.index))
                    jerk_series = (
                        jerk_raw
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .replace("", "0")
                        .astype(float)
                    )
                    df["Jerk_numeric"] = jerk_series.fillna(0)
                    
                    # 4) Rolling standard deviations
                    df["Volatility_Range"] = df["Range"].rolling(window, min_periods=1).std()
                    df["Volatility_Acceleration"] = df["Acceleration_numeric"].rolling(window, min_periods=1).std()
                    df["Volatility_Jerk"] = df["Jerk_numeric"].rolling(window, min_periods=1).std()
                    
                    # 5) Composite score
                    df["Volatility_Composite"] = (
                        alpha * df["Volatility_Range"]
                        + beta * df["Volatility_Acceleration"]
                        + gamma * df["Volatility_Jerk"]
                    )
                    
                    return df

# Example usage:
# intraday = add_volatility_composite(intraday, window=10, alpha=1.0, beta=1.0, gamma=1.0)

                # Apply to intraday
                intraday = add_volatility_composite(intraday, window=10, alpha=1.0, beta=1.0, gamma=1.0)

                def add_gravity_break_signal(df, threshold=9.8, source_col="Volatility_Composite"):
                    """
                    Detects bars where the volatility composite jumps by more than the gravity threshold (default 9.8).
                    
                    Adds:
                      - Volatility_Composite_Diff: change from previous bar
                      - Gravity_Break_Alert: emoji 🪂 if diff > threshold
                    """
                    df = df.copy()
                
                    # Calculate bar-to-bar difference
                    df["Volatility_Composite_Diff"] = df[source_col].diff()
                
                    # Mark gravity break events
                    df["Gravity_Break_Alert"] = df["Volatility_Composite_Diff"].apply(
                        lambda x: "🪂" if x > threshold else ""
                    )
                
                    return df
                intraday = add_gravity_break_signal(intraday)





              
                def compute_option_value(df, premium=64, contracts=100):
                    """
                    Adds realistic Call and Put option simulation columns based on dynamic strike (K).
                    Delta and Gamma change based on moneyness.
                    """
                    # 1️⃣ Baseline
                    spot_open = df.iloc[0]["Close"]
                    f_open    = df.iloc[0]["F_numeric"]
                
                    # 2️⃣ Strike offset based on spot price
                    if spot_open < 50:
                        strike_offset = 1
                    elif spot_open < 100:
                        strike_offset = 2
                    elif spot_open < 250:
                        strike_offset = 4
                    else:
                        strike_offset = 6
                
                    call_strike = spot_open + strike_offset
                    put_strike  = spot_open - strike_offset
                
                    # 3️⃣ Price moves
                    df["F%_Move"]            = df["F_numeric"] - f_open
                    # df["Dollar_Move_From_F"] = (df["F%_Move"] / 10_000) * spot_open
                    df["Dollar_Move_From_F"] = (df["F%_Move"] / 10_000) * prev_close

                    df["Sim_Spot"]           = spot_open + df["Dollar_Move_From_F"]
                
                    # 4️⃣ Moneyness
                    df["Call_Moneyness"] = df["Sim_Spot"] - call_strike
                    df["Put_Moneyness"]  = put_strike - df["Sim_Spot"]
                
                    # 5️⃣ Dynamic Δ and Γ (clipped for realism)
                    df["Call_Delta"] = df["Call_Moneyness"].apply(lambda x: max(min(0.5 + 0.05 * x, 1), 0))
                    df["Put_Delta"]  = df["Put_Moneyness"].apply(lambda x: max(min(0.5 + 0.05 * x, 1), 0))
                
                    df["Call_Gamma"] = df["Call_Delta"] * (1 - df["Call_Delta"]) * 0.4
                    df["Put_Gamma"]  = df["Put_Delta"]  * (1 - df["Put_Delta"])  * 0.4
                
                    # 6️⃣ Option Values
                    df["Call_Option_Value"] = (
                        df["Call_Delta"] * df["Dollar_Move_From_F"] +
                        0.5 * df["Call_Gamma"] * df["Dollar_Move_From_F"]**2
                    ) * contracts
                
                    df["Put_Option_Value"] = (
                        -df["Put_Delta"] * df["Dollar_Move_From_F"] +
                        0.5 * df["Put_Gamma"] * df["Dollar_Move_From_F"]**2
                    ) * contracts
                
                    # 7️⃣ Returns
                    df["Call_Return_%"] = ((df["Call_Option_Value"] - premium) / premium) * 100
                    df["Put_Return_%"]  = ((df["Put_Option_Value"] - premium) / premium) * 100

                    # 8️⃣ Smooth the raw option values before further logic
                    df["Call_Option_Smooth"] = df["Call_Option_Value"].rolling(window=3).mean()
                    df["Put_Option_Smooth"]  = df["Put_Option_Value"].rolling(window=3).mean()
                    
                    # 9️⃣ Bollinger Band Logic (used internally, not plotted)
                    df["Call_MA"] = df["Call_Option_Smooth"].rolling(window=20).mean()
                    df["Call_STD"] = df["Call_Option_Smooth"].rolling(window=20).std()
                    df["Call_BB_Upper"] = df["Call_MA"] + 2 * df["Call_STD"]
                    df["Call_BB_Lower"] = df["Call_MA"] - 2 * df["Call_STD"]
                    df["Call_BB_Tag"] = np.where(df["Call_Option_Smooth"] > df["Call_BB_Upper"], "🧃", "")
                    
                    df["Put_MA"] = df["Put_Option_Smooth"].rolling(window=20).mean()
                    df["Put_STD"] = df["Put_Option_Smooth"].rolling(window=20).std()
                    df["Put_BB_Upper"] = df["Put_MA"] + 2 * df["Put_STD"]
                    df["Put_BB_Lower"] = df["Put_MA"] - 2 * df["Put_STD"]
                    df["Put_BB_Tag"] = np.where(df["Put_Option_Smooth"] > df["Put_BB_Upper"], "💨", "")
                    
                                      
                    # 🔬 10️⃣ BBW: Bollinger Band Width for Option Flow
                    
                    # -- Call Option BBW
                    df["Call_BBW"] = (df["Call_BB_Upper"] - df["Call_BB_Lower"]) / df["Call_MA"]
                    
                    # -- Put Option BBW
                    df["Put_BBW"] = (df["Put_BB_Upper"] - df["Put_BB_Lower"]) / df["Put_MA"]
                    
                    # 🐝 Detect Tightness: below 10th percentile of last 50 bars
                    call_bbw_thresh = df["Call_BBW"].rolling(50).quantile(0.10)
                    put_bbw_thresh  = df["Put_BBW"].rolling(50).quantile(0.10)
                    
                    df["Call_BBW_Is_Tight"] = df["Call_BBW"] < call_bbw_thresh
                    df["Put_BBW_Is_Tight"]  = df["Put_BBW"]  < put_bbw_thresh
                    
                    # 🐝 Mark tight zones if 3 of last 5 are tight
                    df["Call_BBW_Tight_Emoji"] = df["Call_BBW_Is_Tight"].rolling(3).apply(lambda x: x.sum() >= 3).fillna(0).astype(bool).map({True: "🐝", False: ""})
                    df["Put_BBW_Tight_Emoji"]  = df["Put_BBW_Is_Tight"] .rolling(3).apply(lambda x: x.sum() >= 3).fillna(0).astype(bool).map({True: "🐝", False: ""})

                    # 1️⃣ Raw speed of the call option value
                    df["Call_Option_Speed"] = df["Call_Option_Value"].diff()
                    # 1️⃣ Raw speed of the call option value
                    df["Put_Option_Speed"] = df["Put_Option_Value"].diff()


                    intraday["COV_Change"] = intraday["Call_Option_Value"].diff()
                    intraday["COV_Accel"] = intraday["COV_Change"].diff()


                                    # 2️⃣ Smoothed 3-bar trend of that speed
                    df["Call_Vol_Explosion"] = df["Call_Option_Speed"].rolling(3).mean()
                    df["Put_Vol_Explosion"]  = df["Put_Option_Speed"].rolling(3).mean()
                    
                    # 3️⃣ Normalize to opening premium
                    call_premium = df["Call_Option_Value"].iloc[0]
                    put_premium  = df["Put_Option_Value"].iloc[0]
                    
                    df["Call_Vol_Explosion_%"] = (df["Call_Vol_Explosion"] / call_premium) * 100
                    df["Put_Vol_Explosion_%"]  = (df["Put_Vol_Explosion"]  / put_premium)  * 100
                    
                    # 4️⃣ Signal when smoothed speed crosses 10% threshold
                    df["Call_Vol_Surge_Signal"] = df["Call_Vol_Explosion_%"] > 33
                    df["Put_Vol_Surge_Signal"]  = df["Put_Vol_Explosion_%"]  > 33

                    df["Call_Vol_Explosion_Emoji"] = np.where(df["Call_Vol_Surge_Signal"], "💥", "")
                    df["Put_Vol_Explosion_Emoji"]  = np.where(df["Put_Vol_Surge_Signal"],  "💥", "")
                    intraday["Tiger"] = np.where(intraday["COV_Change"] > 30, "🐅", "")




                  
                    intraday["CallPut_Cross"] = (
                        (intraday["Call_Option_Smooth"] > intraday["Put_Option_Smooth"]) &
                        (intraday["Call_Option_Smooth"].shift(1) <= intraday["Put_Option_Smooth"].shift(1))
                    ) | (
                        (intraday["Call_Option_Smooth"] < intraday["Put_Option_Smooth"]) &
                        (intraday["Call_Option_Smooth"].shift(1) >= intraday["Put_Option_Smooth"].shift(1)))
                    



                    intraday["CallPut_Cross"] = (
                        # 🟣 Call crosses ABOVE Put
                        (intraday["Call_Option_Smooth"] > intraday["Put_Option_Smooth"]) &
                        (intraday["Call_Option_Smooth"].shift(1) <= intraday["Put_Option_Smooth"].shift(1))
                    ) | (
                        # 🔵 Put crosses ABOVE Call
                        (intraday["Call_Option_Smooth"] < intraday["Put_Option_Smooth"]) &
                        (intraday["Call_Option_Smooth"].shift(1) >= intraday["Put_Option_Smooth"].shift(1))
                    )

                    intraday["Magic_Emoji"] = np.where(intraday["CallPut_Cross"], "🪄", "")
                    intraday["Magic_Y"] = np.where(
                        intraday["Call_Option_Smooth"] > intraday["Put_Option_Smooth"],
                        intraday["Call_Option_Smooth"] + 10,
                        intraday["Put_Option_Smooth"] - 10)

                  
                    # 🔁 Force starting values
                    df.at[df.index[0], "Call_Option_Value"] = premium
                    df.at[df.index[0], "Put_Option_Value"]  = premium
                    df.at[df.index[0], "Call_Return_%"]     = 0
                    df.at[df.index[0], "Put_Return_%"]      = 0

                    # 👁️ Mark sharp jumps in option value
                    df["Call_Δ"] = df["Call_Option_Value"].diff()
                    df["Put_Δ"]  = df["Put_Option_Value"].diff()
                
                    df["Call_Eye"] = np.where(df["Call_Δ"] >= 10, "👁️", "")
                    df["Put_Eye"]  = np.where(df["Put_Δ"]  >= 10, "👀", "")

                    # 🔭 Detect "Wake-Up" Phase After Cross
                                    #  ---- CROSS FLAGS (unchanged) ----
                    df["Call_Smooth_Cross"] = (df["Call_Option_Smooth"] > df["Put_Option_Smooth"]) & \
                                              (df["Call_Option_Smooth"].shift(1) <= df["Put_Option_Smooth"].shift(1))
                    
                    df["Put_Smooth_Cross"]  = (df["Put_Option_Smooth"] > df["Call_Option_Smooth"]) & \
                                              (df["Put_Option_Smooth"].shift(1) <= df["Call_Option_Smooth"].shift(1))
                    
                    #  ---- TRACKING, no limit ----
                    df["Call_Tracking"] = df["Call_Smooth_Cross"].replace(False, np.nan).ffill()
                    df["Put_Tracking"]  = df["Put_Smooth_Cross"] .replace(False, np.nan).ffill()
                    
                    #  ---- BASELINE AT CROSS ----
                    df["Call_Cross_Base"] = df["Call_Option_Smooth"].where(df["Call_Smooth_Cross"]).ffill()
                    df["Put_Cross_Base"]  = df["Put_Option_Smooth"] .where(df["Put_Smooth_Cross"]).ffill()
                    
                    #  ---- RISE SINCE CROSS ----
                    df["Call_Rise_Since_Cross"] = df["Call_Option_Smooth"] - df["Call_Cross_Base"]
                    df["Put_Rise_Since_Cross"]  = df["Put_Option_Smooth"]  - df["Put_Cross_Base"]
                    
                    #  ---- EMOJI ----
                    df["Call_Wake_Emoji"] = np.where(df["Call_Rise_Since_Cross"] >= 12, "👁️", "")
                    df["Put_Wake_Emoji"]  = np.where(df["Put_Rise_Since_Cross"]  >= 12, "🦉", "")
                    
             # 🔥 New: Detect strong option gain WITHOUT any cross
                    
                    # 12% gain from original premium (entry-based, not cross-based)
                    df["Call_Pure_Gain"] = ((df["Call_Option_Value"] - df["Call_Option_Value"].iloc[0]) / df["Call_Option_Value"].iloc[0]) * 100
                    df["Put_Pure_Gain"]  = ((df["Put_Option_Value"]  - df["Put_Option_Value"].iloc[0]) / df["Put_Option_Value"].iloc[0]) * 100
                    
                    # 🔥 Use flame emoji for strong rise without needing a cross
                    df["Call_Flame_Emoji"] = np.where((df["Call_Pure_Gain"] >= 12) & (~df["Call_Tracking"].notna()), "🔥", "")
                    df["Put_Flame_Emoji"]  = np.where((df["Put_Pure_Gain"]  >= 12) & (~df["Put_Tracking"].notna()), "🔥", "")

                                      # 👁️ Nueva lógica: "Ojo sin cruce"
                                  # Avance en F% (ej. 12 F% = 0.12 si tu escala es decimal, o 12.0 si ya es en F_numeric)
                    f_advance_threshold = 12.0  # o ajusta según tu escala
                    
                    # Lógica 👁️ Call Eye Solo (sin cruce pero avanza 12 F%)
                    df["Call_Change_4"] = df["Call_Option_Smooth"] - df["Call_Option_Smooth"].shift(4)
                    df["Put_Change_4"]  = df["Put_Option_Smooth"]  - df["Put_Option_Smooth"].shift(4)
                    
                    df["Call_Eye_Solo"] = np.where(
                        (df["Call_Change_4"] > f_advance_threshold) &    # Avanza 12+ F%
                        (df["Put_Change_4"] < f_advance_threshold / 2) & # Put no acompaña
                        (~df["Call_Smooth_Cross"]),                      # Sin cruce
                        "👁️", ""
                    )
                    
                    # Lógica 🦉 Put Eye Solo (sin cruce pero avanza 12 F%)
                    df["Put_Eye_Solo"] = np.where(
                        (df["Put_Change_4"] < -f_advance_threshold) &     # Avanza put hacia abajo (es decir, sube en valor)
                        (df["Call_Change_4"] > -f_advance_threshold / 2) & # Call no acompaña hacia arriba
                        (~df["Put_Smooth_Cross"]),                        # Sin cruce
                        "🦉", ""
                    )

                    return df


                  



                intraday = compute_option_value(intraday)      


  

                def compute_option_price_elasticity(
                    intraday,
                    fcol="F_numeric",
                    call_col="Call_Option_Smooth",
                    put_col="Put_Option_Smooth",
                    smooth_window=3,
                    median_window=50,
                    threshold_scale=1.2,
                    eps_replace_zero=True,
                ):
                    """
                    PE = ΔOption  /  ΔF_numeric_points
                         (cents)    (1 pt = 0.01 % move in underlying)
                
                    The smoothed PE is multiplied by 100 so a value like 23.4 means:
                    ~23 ¢ option move for every 1-point change in F_numeric.
                    """
                
                    intraday = intraday.copy()
                
                    # 1⃣  ΔF as point change in F_numeric
                    intraday["dF"] = intraday[fcol].diff().abs()
                    if eps_replace_zero:
                        intraday["dF"] = intraday["dF"].replace(0, np.nan)
                
                    # 2⃣  raw elasticity
                    intraday["Call_PE_raw"] = intraday[call_col].diff() / intraday["dF"]
                    intraday["Put_PE_raw"]  = intraday[put_col].diff()  / intraday["dF"]
                
                    # 3⃣  smoothed PE
                    intraday["Call_PE"] = intraday["Call_PE_raw"].rolling(smooth_window, min_periods=1).mean()
                    intraday["Put_PE"]  = intraday["Put_PE_raw"].rolling(smooth_window, min_periods=1).mean()
                
                    # 4⃣  **readable PE** (×100) ―-> option-cents per F-point
                    intraday["Call_PE_x100"] = (intraday["Call_PE"] * 100).round(2)
                    intraday["Put_PE_x100"]  = (intraday["Put_PE"]  * 100).round(2)
                
                    # 5⃣  rolling-median gates on the ×100 series
                    call_med = intraday["Call_PE_x100"].rolling(median_window, min_periods=1).median()
                    put_med  = intraday["Put_PE_x100"].rolling(median_window, min_periods=1).median()
                
                    intraday["call_ok"] = intraday["Call_PE_x100"] > call_med * threshold_scale
                    intraday["put_ok"]  = intraday["Put_PE_x100"]  > put_med  * threshold_scale
                
                    intraday["call_ok"] = intraday["call_ok"].fillna(False)
                    intraday["put_ok"]  = intraday["put_ok"].fillna(False)

                    flat = intraday["dF"] < intraday["dF"].rolling(20).quantile(0.25)
                    
                    intraday["Call_LF"] = (intraday["Call_Option_Smooth"].diff().rolling(3).mean() > 0) & flat
                    intraday["Put_LF"]  = (intraday["Put_Option_Smooth"].diff().rolling(3).mean() > 0) & flat
                    # Boolean crossovers
                    intraday["PE_Cross_Bull"] = (intraday["Call_PE"] > intraday["Put_PE"]) & \
                                                (intraday["Call_PE"].shift(1) <= intraday["Put_PE"].shift(1))
                    
                    intraday["PE_Cross_Bear"] = (intraday["Put_PE"] > intraday["Call_PE"]) & \
                                                (intraday["Put_PE"].shift(1) <= intraday["Call_PE"].shift(1))


                    conditions_call = (intraday["Call_PE"] > intraday["Put_PE"] * 1.5) & (intraday["Call_PE"] > 0.50)
                    conditions_put  = (intraday["Put_PE"] > intraday["Call_PE"] * 1.5) & (intraday["Put_PE"] > 0.50)
                    
                    intraday["PE_Kill_Shot"] = np.select(
                        [conditions_call, conditions_put],
                        ["🟢", "🔴"],
                        default=None
                    )

                    intraday["PE_Same_Side"] = (
                        (intraday["Call_PE"] > intraday["Put_PE"]) == (intraday["Call_PE"].shift(1) > intraday["Put_PE"].shift(1))
                    )
                    intraday["PE_Streak"] = intraday["PE_Same_Side"].rolling(3).sum() >= 3
                    intraday["PE_Timer"] = np.where(intraday["PE_Streak"], "🔁", "")
                    intraday["PE_PCR"] = intraday["Put_PE"] / intraday["Call_PE"]



                    intraday["PE_PCR"] = intraday["PE_PCR"].clip(upper=5)
                     # Calculate PE-based PCR and clip for extreme stability
                    intraday["PE_PCR"] = (intraday["Put_PE"] / intraday["Call_PE"]).clip(upper=5)

                    return intraday


                intraday = compute_option_price_elasticity(intraday)      

                 
                
                # --- make sure we have numeric versions (no % strings) ---
                intraday["Vector_Energy_num"] = pd.to_numeric(intraday.get("Vector Energy"), errors="coerce").fillna(0)
                intraday["Jerk_Vector_num"]   = pd.to_numeric(intraday.get("Jerk_Vector"),  errors="coerce").fillna(0)
                intraday["VolComp_num"]       = pd.to_numeric(intraday.get("Volatility_Composite"), errors="coerce").fillna(0)
                
                def winsorize(s, lo=0.01, hi=0.99):
                    qlo, qhi = s.quantile(lo), s.quantile(hi)
                    return s.clip(qlo, qhi)
                
                def zscore(s):
                    s = s.astype(float)
                    mu = s.mean()
                    sd = s.std(ddof=0)
                    return (s - mu) / (sd if sd > 0 else 1.0)
                
                # --- clip outliers then z-score (session standardization) ---
                vecE_w = winsorize(intraday["Vector_Energy_num"])
                jerk_w = winsorize(intraday["Jerk_Vector_num"])
                vcomp_w= winsorize(intraday["VolComp_num"])
                
                intraday["z_vecE"]  = zscore(vecE_w).clip(-4, 4)
                intraday["z_jerk"]  = zscore(jerk_w).clip(-4, 4)
                intraday["z_vcomp"] = zscore(vcomp_w).clip(-4, 4)

                def armed_cross_down(series, down=-2.0, up=-1.0):
                    hit = np.zeros(len(series), dtype=bool)
                    armed = True
                    for i in range(1, len(series)):
                        prev, curr = series.iat[i-1], series.iat[i]
                        if armed and curr < down and prev >= down:
                            hit[i] = True
                            armed = False
                        elif not armed and curr > up:
                            armed = True
                    return hit

                intraday["BRAKE"] = armed_cross_down(intraday["z_jerk"], down=-2.0, up=-1.0)

       
                rng3 = intraday["Range"].rolling(3, min_periods=3).sum()
                intraday["Force_per_3bar_Range"] = (
                      pd.to_numeric(intraday.get("Vector Force"), errors="coerce").fillna(0) / rng3.replace(0, np.nan)
                  )
              
                vec_idx = intraday.index[intraday.index % 3 == 2]  # your 3rd bars
                eff = intraday.loc[vec_idx, "Force_per_3bar_Range"].astype(float)

                def detect_option_speed_explosion(df, lookback=3, strong_ratio=2.0, mild_ratio=1.5, percentile=90):
                    """
                    Detects call/put speed explosions using a ratio test and percentile filter.
                    Only flags emojis if current speed is both:
                    - Greater than X times the lagged speed
                    - Within the top N percentile of recent speeds
                    """
                    # ─── Lagged speed (baseline) ─────────────────
                    df["Call_Speed_Lag"] = df["Call_Option_Speed"].shift(lookback)
                    df["Put_Speed_Lag"]  = df["Put_Option_Speed"].shift(lookback)
                
                    # ─── Avoid division by 0 ─────────────────────
                    df["Call_Speed_Lag"].replace(0, np.nan, inplace=True)
                    df["Put_Speed_Lag"] .replace(0, np.nan, inplace=True)
                
                    # ─── Speed ratio ─────────────────────────────
                    df["Call_Speed_Ratio"] = df["Call_Option_Speed"] / df["Call_Speed_Lag"]
                    df["Put_Speed_Ratio"]  = df["Put_Option_Speed"]  / df["Put_Speed_Lag"]
                
                    # ─── Percentile threshold ─────────────────────
                    call_thresh = np.percentile(df["Call_Option_Speed"].dropna(), percentile)
                    put_thresh  = np.percentile(df["Put_Option_Speed"] .dropna(), percentile)
                
                    # ─── Flags based on ratio and percentile ─────
                    call_strong = (df["Call_Speed_Ratio"] >= strong_ratio) & (df["Call_Option_Speed"] >= call_thresh)
                    call_mild   = (df["Call_Speed_Ratio"] >= mild_ratio)   & (df["Call_Option_Speed"] >= call_thresh * 0.8)
                
                    put_strong  = (df["Put_Speed_Ratio"]  >= strong_ratio) & (df["Put_Option_Speed"]  >= put_thresh)
                    put_mild    = (df["Put_Speed_Ratio"]  >= mild_ratio)   & (df["Put_Option_Speed"]  >= put_thresh * 0.8)
                
                    # ─── Emoji assignment ────────────────────────
                    df["Call_Speed_Explosion"] = np.select(
                        [call_strong, call_mild],
                        ["🏎️", "🚗"],
                        default=""
                    )
                
                    df["Put_Speed_Explosion"] = np.select(
                        [put_strong, put_mild],
                        ["🏎️", "🚗"],
                        default=""
                    )
                
                    return df
  

                intraday =  detect_option_speed_explosion(intraday)      

                
                # def compute_option_value(
                #         df, *,               # keyword-only for clarity
                #         delta: float   = 0.50,
                #         gamma: float   = 0.05,
                #         premium: float = 64,
                #         contracts: int = 100
                # ):
                #     """
                #     Add simulated ATM option columns to *df*.
                
                #     Columns created
                #     ---------------
                #     F%_Move          – intraday F-point move from the open
                #     Dollar_Move_From_F
                #     Option_Value     – Δ + ½Γ approximation, scaled by *contracts*
                #     Option_Return_%  – percentage PnL vs. fixed premium
                #     """
                #     # 1️⃣  Baseline values at the open
                #     spot_open  = df.iloc[0]["Close"]
                #     f_open     = df.iloc[0]["F_numeric"]
                
                #     # 2️⃣  Translate F-move → $-move
                #     df["F%_Move"]           = df["F_numeric"] - f_open
                #     df["Dollar_Move_From_F"] = (df["F%_Move"] / 10_000) * spot_open
                
                #     # 3️⃣  Option value: Δ + ½Γ *contracts*
                #     df["Option_Value"] = (
                #         delta * df["Dollar_Move_From_F"]
                #         + 0.5 * gamma * df["Dollar_Move_From_F"]**2
                #     ) * contracts


                #     df.at[df.index[0], "Option_Value"] = premium

                #     # 4️⃣  Return (%) relative to premium
                #     df["Option_Return_%"] = ((df["Option_Value"] - premium) / premium) * 100
                
                #     return df
                
                
                # # --- call it right after intraday is ready -----------------
                # intraday = compute_option_value(intraday)

            
 
  

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



                def calculate_distensibility(df):
                    """
                    Distensibility = (BBW - BBW_Anchor) / (RVOL_5 * BBW_Anchor)
                    """
                    df["Distensibility"] = np.nan
                    if {"F% BBW", "BBW_Anchor", "RVOL_5"}.issubset(df.columns):
                        delta_volume = df["F% BBW"] - df["BBW_Anchor"]
                        pressure = df["RVOL_5"].replace(0, np.nan)
                        original_volume = df["BBW_Anchor"]
                        df["Distensibility"] = delta_volume / (pressure * original_volume)
                    return df
                intraday = calculate_distensibility(intraday)


                # Initialize column
                intraday["Distensibility Alert"] = ""
                
                # Skip first 6 bars and apply threshold
                distensible_subset = intraday.iloc[6:].copy()
                valid = distensible_subset[distensible_subset["Distensibility"] > 0.05]
                
                # Get top 3 high-distensibility bars (if any)
                top_distensible = valid["Distensibility"].nlargest(3)
                
                # Mark them with the window emoji
                intraday.loc[top_distensible.index, "Distensibility Alert"] = "🪟"

                def armed_cross_up(series, up=2.0, down=1.0):
                    hit = np.zeros(len(series), dtype=bool)
                    armed = True
                    for i in range(1, len(series)):
                        prev, curr = series.iat[i-1], series.iat[i]
                        if armed and curr > up and prev <= up:
                            hit[i] = True
                            armed = False
                        elif not armed and curr < down:
                            armed = True
                    return hit
                
                intraday["IGNITION"] = armed_cross_up(intraday["z_vecE"], up=2.0, down=1.0)
              
                def calculate_compliance(df):
                    """
                    Compliance = (BBW - BBW_Anchor) / RVOL_5
                    """
                    df["Compliance"] = np.nan
                    if {"F% BBW", "BBW_Anchor", "RVOL_5"}.issubset(df.columns):
                        delta_volume = df["F% BBW"] - df["BBW_Anchor"]
                        pressure = df["RVOL_5"].replace(0, np.nan)
                        df["Compliance"] = delta_volume / pressure
                    return df

                intraday = calculate_compliance(intraday)



                def detect_compliance_shift(df):
                    """
                    Adds 🫧 emoji where Compliance shifts from negative to positive.
                    """
                    df["Compliance Shift"] = ""
                    for i in range(1, len(df)):
                        prev = df["Compliance"].iloc[i - 1]
                        curr = df["Compliance"].iloc[i]
                        if pd.notna(prev) and pd.notna(curr):
                            if prev < 0 and curr >= 0:
                                df.at[df.index[i], "Compliance Shift"] = "🫧"
                    return df
  
                intraday = detect_compliance_shift(intraday)


                def detect_compliance_expansion(df):
                  df["Compliance Anchor"] = np.nan
                  df["Compliance Surge"] = ""
                  
                  anchor = None
                  for i in range(1, len(df)):
                      prev = df["Compliance"].iloc[i - 1]
                      curr = df["Compliance"].iloc[i]
              
                      if pd.notna(prev) and pd.notna(curr):
                          # Shift detected
                          if prev < 0 and curr >= 0:
                              anchor = curr
                              df.at[df.index[i], "Compliance Anchor"] = anchor
                          elif anchor is not None and curr >= 0:
                              ratio = curr / anchor if anchor != 0 else 0
              
                              if ratio >= 10:
                                  df.at[df.index[i], "Compliance Surge"] = "🚀"
                              elif ratio >= 5:
                                  df.at[df.index[i], "Compliance Surge"] = "💥"
                              elif ratio >= 2:
                                  df.at[df.index[i], "Compliance Surge"] = "⚡"
                          else:
                              anchor = None  # reset when compliance drops again
                  return df
                intraday = detect_compliance_expansion(intraday)

       

                def calculate_stroke_metrics_continuous(df, lookahead=10):
                    """
                    Continuously computes Stroke Volume and Stroke Efficiency for each bar 
                    from the Compliance Shift 🫧 until the regime ends (2 consecutive negatives).
                    """
                    # Ensure numeric types
                    df["F_numeric"]  = pd.to_numeric(df["F_numeric"], errors="coerce")
                    df["Compliance"] = pd.to_numeric(df["Compliance"], errors="coerce")
                
                    # Initialize columns
                    df["Stroke Volume"]     = np.nan
                    df["Stroke Efficiency"] = np.nan
                
                    shift_f = None
                    shift_comp = None
                    in_regime = False
                    neg_count = 0
                
                    for i in range(len(df)):
                        comp   = df["Compliance"].iloc[i]
                        f_val  = df["F_numeric"].iloc[i]
                        prev   = df["Compliance"].iloc[i-1] if i > 0 else np.nan
                
                        # 1) Detect start of positive‐compliance regime (negative → positive)
                        if not in_regime and prev < 0 and comp >= 0:
                            in_regime  = True
                            shift_f    = f_val
                            shift_comp = comp if comp != 0 else np.nan
                            neg_count  = 0
                
                        # 2) While in the regime, track until 2 consecutive negatives
                        if in_regime:
                            if comp < 0:
                                neg_count += 1
                                if neg_count >= 2:
                                    # Mark regime end on the last positive bar
                                    in_regime  = False
                                    shift_f    = None
                                    shift_comp = None
                                    neg_count  = 0
                                    continue
                            else:
                                neg_count = 0
                
                            # 3) Compute stroke metrics for this bar
                            stroke      = f_val - shift_f
                            efficiency  = stroke / shift_comp if shift_comp else np.nan
                
                            df.at[df.index[i], "Stroke Volume"]     = stroke
                            df.at[df.index[i], "Stroke Efficiency"] = efficiency
                
                    return df
                
                # Apply to your intraday DataFrame:
                intraday = calculate_stroke_metrics_continuous(intraday, lookahead=10)

              
                def mark_star_growth(df, threshold=10):
                    df["Stroke Growth ⭐"] = ""
                    i = 0
                    while i < len(df):
                        if df["Compliance Shift"].iloc[i] == "🫧":
                            retrace_count = 0
                            last_sv = df["Stroke Volume"].iloc[i]
                            j = i + 1
                            while j < len(df):
                                # Detect retrace in F%
                                if df["F%"].iloc[j] < df["F%"].iloc[j - 1]:
                                    retrace_count += 1
                                else:
                                    retrace_count = 0
                
                                # Add ⭐ if stroke volume increases by more than threshold
                                current_sv = df["Stroke Volume"].iloc[j]
                                if pd.notnull(current_sv) and pd.notnull(last_sv):
                                    if current_sv - last_sv > threshold:
                                        df.at[df.index[j], "Stroke Growth ⭐"] = "⭐"
                                        last_sv = current_sv  # update for next comparison
                
                                if retrace_count >= 2:
                                    break
                
                                j += 1
                            i = j
                        else:
                            i += 1
                    return df

                intraday = mark_star_growth(intraday)


                intraday["O2 Quality"] = ""
                def define_oxygen_quality(df, range_column="Range", output_column="O2 Quality", window=5):
                  """
                  Assigns oxygen quality based on how each bar's range compares to its rolling average.
                  
                  Parameters:
                      df (pd.DataFrame): The intraday data with 'Range' column.
                      range_column (str): The name of the column containing range values.
                      output_column (str): The name of the new column to assign oxygen emojis.
                      window (int): Rolling window size for average range.
              
                  Returns:
                      pd.DataFrame: Original dataframe with new column for O2 Quality.
                  """
                  df[output_column] = ""
                  atr = df[range_column].rolling(window).mean()
              
                  for i in range(len(df)):
                      bar_range = df.loc[i, range_column]
                      avg_range = atr[i]
              
                      if pd.isna(avg_range):
                          continue
              
                      if bar_range < 0.5 * avg_range:
                          df.loc[i, output_column] = "😮‍💨"  # low oxygen
                      elif bar_range > 1.5 * avg_range:
                          df.loc[i, output_column] = "🫁"    # rich oxygen
                      else:
                          df.loc[i, output_column] = "😐"    # normal oxygen
              
                  return df

                intraday = define_oxygen_quality(intraday)

             
                

                
                     


              
                def detect_marengo(df):
                    """
                    Detects North Marengo:
                    - Mike (F_numeric) touches or exceeds F% Upper band
                    - RVOL_5 > 1.2
                    Places 🐎 in 'Marengo' column when both conditions are met.
                    """
                
                    df["Marengo"] = ""
                    df["South_Marengo"] = ""

                    for i in range(len(df)):
                        if (
                            "F_numeric" in df.columns
                            and "F% Upper" in df.columns
                            and "F% Lower" in df.columns

                            and "RVOL_5" in df.columns
                        ):
                            mike = df.loc[i, "F_numeric"]
                            upper = df.loc[i, "F% Upper"]
                            lower = df.loc[i, "F% Lower"]

                            rvol = df.loc[i, "RVOL_5"]
                
                            if pd.notna(mike) and pd.notna(upper) and pd.notna(lower) and pd.notna(rvol):
                                if mike >= upper and rvol > 1.2:
                                    df.at[i, "Marengo"] = "🐎"
                                elif mike <= lower and rvol > 1.2:
                                    df.at[i, "South_Marengo"] = "🐎"  # South Marengo
                    return df

                intraday = detect_marengo(intraday)

 
            
                
              
                # def calculate_bollinger_band_angles(df, band_col="F% Upper", angle_col="Upper Angle", window=1):
                #     """
                #     Calculates the angle (in degrees) of the specified Bollinger Band using tan(θ) = Δy / Δx,
                #     where Δx = 1 bar (time), so angle = atan(Δy). This gives a sense of slope/steepness.
                    
                #     Parameters:
                #         df: DataFrame with Bollinger Band columns.
                #         band_col: Column name to calculate angle from.
                #         angle_col: Output column to store angle in degrees.
                #         window: How many bars back to compare against (1 = adjacent bar).
                #     """
                #     if band_col in df.columns:
                #         delta_y = df[band_col].diff(periods=window)
                #         angle_rad = np.arctan(delta_y)  # since delta_x = 1
                #         df[angle_col] = np.degrees(angle_rad)
                #     else:
                #         df[angle_col] = np.nan
                
                #     return df
                #     # Calculate angles for both bands
                # intraday = calculate_bollinger_band_angles(intraday, band_col="F% Upper", angle_col="Upper Angle")
                # intraday = calculate_bollinger_band_angles(intraday, band_col="F% Lower", angle_col="Lower Angle")
                            



                def calculate_smoothed_band_angle(df, band_col="F% Upper", angle_col="Upper Angle", window=5):
                    """
                    Calculates the angle (in degrees) of a Bollinger Band over a smoothed n-bar window.
                    This reduces noise by measuring trend over time instead of bar-to-bar jitter.
                
                    Args:
                        df (DataFrame): Must include the band_col
                        band_col (str): Column to calculate angle on (e.g., 'F% Upper')
                        angle_col (str): Name of the new output angle column
                        window (int): Number of bars for smoothing
                    """
                    if band_col in df.columns:
                        slope = (df[band_col] - df[band_col].shift(window)) / window
                        angle_rad = np.arctan(slope)
                        df[angle_col] = np.degrees(angle_rad)
                    else:
                        df[angle_col] = np.nan
                    return df
                intraday = calculate_smoothed_band_angle(intraday, band_col="F% Upper", angle_col="Upper Angle", window=5)
                intraday = calculate_smoothed_band_angle(intraday, band_col="F% Lower", angle_col="Lower Angle", window=5)


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




                # def generate_market_snapshot(df, current_time, current_price, prev_close, symbol):
                #     """
                #     Generates a concise market snapshot:
                #     - Time and current price
                #     - Opening price & where it stands now
                #     - F% change in raw dollars
                #     - Price position relative to Kijun and Bollinger Mid
                #     - Latest Buy/Sell Signal
                #     """

                #     # Convert time to 12-hour format (e.g., "03:55 PM")
                #     current_time_str = pd.to_datetime(current_time).strftime("%I:%M %p")

                #     # Get today's opening price
                #     open_price = df["Open"].iloc[0]

                #     # Calculate today's price changes
                #     price_change = current_price - prev_close
                #     f_percent_change = (price_change / prev_close) * 10000  # F%

                #     # Identify price position relative to Kijun and Bollinger Middle
                #     last_kijun = df["Kijun_sen"].iloc[-1]
                #     last_mid_band = df["F% MA"].iloc[-1]

                #     position_kijun = "above Kijun" if current_price > last_kijun else "below Kijun"
                #     position_mid = "above Mid Band" if current_price > last_mid_band else "below Mid Band"

                #     # Get the latest Buy/Sell signal
                #     latest_signal = df.loc[df["Wealth Signal"] != "", ["Wealth Signal"]].tail(1)
                #     signal_text = latest_signal["Wealth Signal"].values[0] if not latest_signal.empty else "No Signal"

                #     # Construct the message
                #     snapshot = (
                #         f"📌 {current_time_str} – **{symbol}** is trading at **${current_price:.2f}**\n\n"
                #         f"• Opened at **${open_price:.2f}** and is now sitting at **${current_price:.2f}**\n"
                #         f"• F% Change: **{f_percent_change:.0f} F%** (${price_change:.2f})\n"
                #         f"• Price is **{position_kijun}** & **{position_mid}**\n"
                #         f"• **Latest Signal**: {signal_text}\n"
                #     )

                #     return snapshot
 
              
                def _sparkline(series, bins="▁▂▃▄▅▆▇█"):
                    s = pd.Series(series).astype(float)
                    if len(s) == 0 or s.nunique() == 1:
                        return "—"
                    x = (s - s.min()) / (s.max() - s.min())
                    idx = np.clip((x * (len(bins)-1)).round().astype(int), 0, len(bins)-1)
                    return "".join(bins[i] for i in idx)
                
                def _fmt_delta(x, pos="▲", neg="▼", zero="→", unit=""):
                    if x > 0:  return f"{pos}{abs(x):.2f}{unit}"
                    if x < 0:  return f"{neg}{abs(x):.2f}{unit}"
                    return f"{zero}{abs(x):.2f}{unit}"
                
                def generate_market_snapshot(df, current_time, current_price, prev_close, symbol, last_n=26):
                    """
                    Vibrant, glanceable snapshot with:
                    - Time, price, arrows & F%
                    - Open & day range context
                    - Distance to Kijun & Mid (in $ and %)
                    - Latest signal (time ago)
                    - Mini sparkline of last N closes
                    """
                    # time
                    current_time_str = pd.to_datetime(current_time).strftime("%I:%M %p")
                
                    # core refs
                    open_price = float(df["Open"].iloc[0]) if "Open" in df else np.nan
                    day_high  = float(df["High"].max()) if "High" in df else np.nan
                    day_low   = float(df["Low"].min())  if "Low"  in df else np.nan
                
                    price_change = current_price - prev_close
                    f_percent_change = (price_change / prev_close) * 10000 if prev_close else np.nan
                
                    # structure
                    last_kijun = float(df["Kijun_sen"].iloc[-1]) if "Kijun_sen" in df else np.nan
                    last_mid   = float(df["F% MA"].iloc[-1])      if "F% MA" in df else np.nan
                
                    # distances
                    def pct_dist(a, b):
                        return ((a - b) / b * 100) if (b and not np.isnan(b)) else np.nan
                
                    d_kijun = current_price - last_kijun if not np.isnan(last_kijun) else np.nan
                    d_mid   = current_price - last_mid   if not np.isnan(last_mid)   else np.nan
                
                    d_kijun_pct = pct_dist(current_price, last_kijun) if not np.isnan(last_kijun) else np.nan
                    d_mid_pct   = pct_dist(current_price, last_mid)   if not np.isnan(last_mid)   else np.nan
                
                    pos_kijun = "🟢 above Kijun" if not np.isnan(last_kijun) and current_price > last_kijun else ("🔴 below Kijun" if not np.isnan(last_kijun) else "— Kijun n/a")
                    pos_mid   = "🟢 above Mid"   if not np.isnan(last_mid)   and current_price > last_mid   else ("🔴 below Mid"   if not np.isnan(last_mid)   else "— Mid n/a")
                
                    # signal + recency
                    signal_text, signal_age = "No Signal", ""
                    if "Wealth Signal" in df:
                        sig_rows = df.loc[df["Wealth Signal"].astype(str).str.len() > 0, ["Wealth Signal"]].tail(1)
                        if not sig_rows.empty:
                            signal_text = sig_rows["Wealth Signal"].values[0]
                            # if you have a time column, show staleness
                            time_col = None
                            for cand in ["Time", "Datetime", "DateTime", "Timestamp"]:
                                if cand in df.columns:
                                    time_col = cand
                                    break
                            if time_col:
                                last_sig_idx = df.loc[df["Wealth Signal"].astype(str).str.len() > 0].index[-1]
                                t_sig = pd.to_datetime(df.loc[last_sig_idx, time_col])
                                t_now = pd.to_datetime(current_time)
                                mins = int((t_now - t_sig).total_seconds() // 60)
                                if mins >= 0:
                                    signal_age = f" • {mins}m ago"
                
                    # sparkline of last closes
                    closes = df["Close"].tail(last_n) if "Close" in df else pd.Series([])
                    spark = _sparkline(closes)
                
                    # day context
                    range_line = "—"
                    if not np.isnan(day_high) and not np.isnan(day_low):
                        rng = day_high - day_low
                        from_low = current_price - day_low
                        pct_thru = (from_low / rng * 100) if rng != 0 else np.nan
                        if not np.isnan(pct_thru):
                            range_line = f"🧭 Day: {day_low:.2f} → {day_high:.2f} • {_fmt_delta(rng, pos='Δ', neg='Δ', zero='Δ', unit='$')} • {pct_thru:.0f}% thru"
                
                    # directional badges
                    arrow_price = _fmt_delta(price_change, unit="$")
                    arrow_f     = _fmt_delta(f_percent_change, unit=" F%") if not np.isnan(f_percent_change) else "—"
                    bias_badge  = "📈" if price_change > 0 else ("📉" if price_change < 0 else "⏸️")
                
                    # distance lines
                    def dist_line(label, d_abs, d_pct):
                        if np.isnan(d_abs) or np.isnan(d_pct):
                            return f"{label}: n/a"
                        return f"{label}: {_fmt_delta(d_abs, unit='$')} ({_fmt_delta(d_pct, unit='%', pos='▲', neg='▼', zero='→')})"
                
                    kijun_line = dist_line("📐 Kijun dist", d_kijun, d_kijun_pct)
                    mid_line   = dist_line("〰️ Mid dist",   d_mid,   d_mid_pct)
                
                    snapshot = (
                        f"📌 {current_time_str} — **{symbol}** {bias_badge} **${current_price:.2f}**  "
                        f"({arrow_price} | {arrow_f})\n"
                        # f"{range_line}\n"
                        # f"🧠 Position: {pos_kijun} \n"
                        f"{kijun_line} \n"
                        # f"🪙 Signal: **{signal_text}**{signal_age}\n"
                        # f"📈 {spark}\n"
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

    
                def calculate_f_theta_degrees(df, cot_scale=100):
                    """
                    Computes F% angle in degrees and scaled cotangent.
                    - Theta = arctangent of F% change, in degrees (bounded -90 to +90)
                    - Cotangent = 1 / tan(theta), scaled by `cot_scale`
                    """
                    if "F_numeric" in df.columns:
                        # First derivative (F% slope)
                        slope = df["F_numeric"].diff()
                
                        # Theta in degrees
                        df["F% Theta"] = np.degrees(np.arctan(slope))
                
                        # Cotangent in radians, then scale
                        df["F% Cotangent"] = np.where(
                            df["F% Theta"] != 0,
                            (1 / np.tan(np.radians(df["F% Theta"]))) * cot_scale,
                            0
                        )
                    else:
                        df["F% Theta"] = 0
                        df["F% Cotangent"] = 0
                
                    return df
                
                # Apply the function
                intraday = calculate_f_theta_degrees(intraday, cot_scale=100)  # or 10 if you prefer tighter scale
  
  



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
                                if f_now - f_start >= 33:
                                    intraday.loc[i, "King_Signal"] = "👑"
                                    break

                        # Red Kingdom 🔻👑
                        queen_sell_indices = intraday.index[intraday["Kijun_F_Cross"] == "Sell Kijun Cross"].tolist()
                        for q_idx in queen_sell_indices:
                            f_start = intraday.loc[q_idx, "F_numeric"]
                            for i in range(q_idx + 1, len(intraday)):
                                f_now = intraday.loc[i, "F_numeric"]
                                if f_now - f_start <= -33:
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

                    # Bullish Cross (🦅🐦‍⬛)
                    if prev_tenkan < prev_kijun and curr_tenkan >= curr_kijun:
                        intraday.loc[i, "Tenkan_Kijun_Cross"] = "🦅"

                    # Bearish Cross (🐦‍⬛)
                    elif prev_tenkan > prev_kijun and curr_tenkan <= curr_kijun:
                        intraday.loc[i, "Tenkan_Kijun_Cross"] = "🐦‍⬛"





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


                def assign_dmi_emojis(df):
                    """
                    Adds three emoji signals based on DMI and Kijun_F% cross logic:
                    🔦 scout_emoji — any +DI or -DI cross
                    🪽 wing_emoji — +DI cross within ±5 bars of Kijun_F up-cross
                    🦇 bat_emoji  — -DI cross within ±5 bars of Kijun_F down-cross
                    """
                    df = df.copy()
                    df["scout_emoji"] = ""
                    df["scout_position"] = np.nan  # <-- NEW: for visual offset
                    df["wing_emoji"] = ""
                    df["bat_emoji"] = ""
                
                    # Identify DMI crossovers
                    bullish_dmi_cross = (df["+DI_F%"] > df["-DI_F%"]) & (df["+DI_F%"].shift(1) <= df["-DI_F%"].shift(1))
                    bearish_dmi_cross = (df["-DI_F%"] > df["+DI_F%"]) & (df["-DI_F%"].shift(1) <= df["+DI_F%"].shift(1))
                
                    # Identify Kijun_F crosses
                    kijun_up_cross = (df["F_numeric"] > df["Kijun_F"]) & (df["F_numeric"].shift(1) <= df["Kijun_F"].shift(1))
                    kijun_down_cross = (df["F_numeric"] < df["Kijun_F"]) & (df["F_numeric"].shift(1) >= df["Kijun_F"].shift(1))
                
                    # Get indices of Kijun_F crosses
                    kijun_up_indices = set(df[kijun_up_cross].index)
                    kijun_down_indices = set(df[kijun_down_cross].index)
                
                    for i in range(1, len(df)):
                        if bullish_dmi_cross.iloc[i]:
                            df.at[i, "scout_emoji"] = "🔦"
                            df.at[i, "scout_position"] = df.at[i, "F_numeric"] + 2  # ABOVE for bullish
                
                            # 🪽 Wing check
                            if any((i - 5) <= j <= (i + 5) for j in kijun_up_indices):
                                df.at[i, "wing_emoji"] = "🪽"
                
                        elif bearish_dmi_cross.iloc[i]:
                            df.at[i, "scout_emoji"] = "🔦"
                            df.at[i, "scout_position"] = df.at[i, "F_numeric"] - 2  # BELOW for bearish
                
                            # 🪽 Bat check
                            if any((i - 5) <= j <= (i + 5) for j in kijun_down_indices):
                                df.at[i, "bat_emoji"] = "🪽"
                
                    return df

  
  
    
                intraday = assign_dmi_emojis(intraday)

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







                def add_mike_kijun_atr_emoji(df):
                  """
                  Adds 🚀 for upward cross and 🧨 for downward cross,
                  if ATR expansion occurs within ±3 bars of the cross.
                  """
                  crosses_up = (df["F_numeric"].shift(1) < df["Kijun_F"].shift(1)) & (df["F_numeric"] >= df["Kijun_F"])
                  crosses_down = (df["F_numeric"].shift(1) > df["Kijun_F"].shift(1)) & (df["F_numeric"] <= df["Kijun_F"])
              
                  emoji_flags = []
              
                  for i in range(len(df)):
                      if not (crosses_up.iloc[i] or crosses_down.iloc[i]):
                          emoji_flags.append("")
                          continue
              
                      start = max(0, i - 3)
                      end = min(len(df), i + 4)
                      atr_window = df.iloc[start:end]["ATR_Exp_Alert"]
              
                      if any(atr_window == "☄️"):
                          emoji_flags.append("🚀" if crosses_up.iloc[i] else "🧨")
                      else:
                          emoji_flags.append("")
              
                  df["Mike_Kijun_ATR_Emoji"] = emoji_flags
                  return df


                intraday = add_mike_kijun_atr_emoji(intraday)
        

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
                        end_idx = start_idx + 50

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
                    Mark a Pawn Down (♟️) or Pawn Up (♙) at any instance of Tenkan cross.
                    No buffer logic, no threshold—just clean cross detection.
                    """
                    f_prev = df["F_numeric"].shift(1)
                    t_prev = df["Tenkan_F"].shift(1)
                    f_curr = df["F_numeric"]
                    t_curr = df["Tenkan_F"]
                
                    cross_up   = (f_prev < t_prev) & (f_curr >= t_curr)
                    cross_down = (f_prev > t_prev) & (f_curr <= t_curr)
                
                    df["Tenkan_Pawn"] = ""
                    df.loc[cross_up,   "Tenkan_Pawn"] = "♙"
                    df.loc[cross_down, "Tenkan_Pawn"] = "♟️"
                
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


        
                def  detect_td_supply_cross_rooks(df):
                    """
                    Detects a cross of F_numeric through TD Supply/Demand,
                    and confirms it with the next candle's direction.
                
                    - ♖ (White Rook): Cross above Supply, then next close > current close
                    - ♜ (Black Rook): Cross below Demand, then next close < current close
                    """
                    df["TD_Supply_Rook"] = ""
                
                    for i in range(1, len(df) - 1):  # Avoid index error on last row
                        prev_f = df.loc[i - 1, "F_numeric"]
                        curr_f = df.loc[i, "F_numeric"]
                        next_f = df.loc[i + 1, "F_numeric"]
                
                        prev_supply = df.loc[i - 1, "TD Supply Line F"]
                        curr_supply = df.loc[i, "TD Supply Line F"]
                
                        prev_demand = df.loc[i - 1, "TD Demand Line F"]
                        curr_demand = df.loc[i, "TD Demand Line F"]
                
                        # 🏰 Cross above supply + next bar closes higher
                        if prev_f < prev_supply and curr_f >= curr_supply:
                            if next_f > curr_f:
                                df.loc[i + 1, "TD_Supply_Rook"] = "♖"
                
                        # 🏰 Cross below demand + next bar closes lower
                        if prev_f > prev_demand and curr_f <= curr_demand:
                            if next_f < curr_f:
                                df.loc[i + 1, "TD_Supply_Rook"] = "♜"
                
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


                def add_sharpe_column(intraday, window=10):
                    """
                    Adds a rolling Sharpe Ratio column based on F_numeric over a given window.
                    """
                    returns = intraday["F_numeric"] / 100  # Convert from percent to decimal
                    rolling_mean = returns.rolling(window=window).mean()
                    rolling_std = returns.rolling(window=window).std()
                
                    # Avoid division by zero
                    sharpe = rolling_mean / (rolling_std + 1e-9)
                
                    intraday["Sharpe_Ratio"] = sharpe.round(2)
                    return intraday

                intraday = add_sharpe_column(intraday, window=10)


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
                
                intraday["ATR_expansion"] = intraday["ATR_Exp_Alert"] == "☄️"
                # Optional — if you're using BBW too:
                intraday["BBW_expansion"] = intraday["BBW_Tight_Emoji"] == "🐝"








              
                def entryAlert(intraday, threshold=10.0, rvol_threshold=1.2, rvol_lookback=9):
                    """
                    Enhanced Entry Alert System:
                    - ✅: Structure + volume confirmation at entry bar.
                    - ☑️: Structure confirmed, but volume came later.
                    """
                
                    # Initialize columns
                    intraday["Entry_Alert_Long"] = False
                    intraday["Entry_Alert_Short"] = False
                    intraday["Entry_Emoji_Long"] = ""
                    intraday["Entry_Emoji_Short"] = ""
                
                    pending_long = []   # store indices awaiting volume
                    pending_short = []
                
                    for i in range(1, len(intraday) - 1):
                        prev_f = intraday.iloc[i-1]["F_numeric"]
                        prev_k = intraday.iloc[i-1]["Kijun_F"]
                        curr_f = intraday.iloc[i]["F_numeric"]
                        curr_k = intraday.iloc[i]["Kijun_F"]
                        next_f = intraday.iloc[i+1]["F_numeric"]
                
                        # ➡️ LONG CROSS
                        if (prev_f < prev_k - threshold) and (curr_f > curr_k + threshold):
                            if next_f >= curr_f:
                                # Check for recent RVOL
                                start_idx = max(0, i - rvol_lookback + 1)
                                rvol_window = intraday.iloc[start_idx:i+1]["RVOL_5"]
                                if (rvol_window > rvol_threshold).any():
                                    intraday.at[intraday.index[i], "Entry_Alert_Long"] = True
                                    intraday.at[intraday.index[i], "Entry_Emoji_Long"] = "✅"
                                else:
                                    pending_long.append(i)  # store for later ☑️ tagging
                
                        # ⬅️ SHORT CROSS
                        if (prev_f > prev_k + threshold) and (curr_f < curr_k - threshold):
                            if next_f <= curr_f:
                                start_idx = max(0, i - rvol_lookback + 1)
                                rvol_window = intraday.iloc[start_idx:i+1]["RVOL_5"]
                                if (rvol_window > rvol_threshold).any():
                                    intraday.at[intraday.index[i], "Entry_Alert_Short"] = True
                                    intraday.at[intraday.index[i], "Entry_Emoji_Short"] = "✅"
                                else:
                                    pending_short.append(i)
                
                    # 🔁 Second pass: look for volume spikes to validate pending entries
                    for i in range(len(intraday)):
                        rvol = intraday.iloc[i]["RVOL_5"]
                
                        if rvol > rvol_threshold:
                            if pending_long:
                                idx = pending_long.pop(0)
                                intraday.at[intraday.index[idx], "Entry_Alert_Long"] = True
                                intraday.at[intraday.index[idx], "Entry_Emoji_Long"] = "☑️"
                
                            if pending_short:
                                idx = pending_short.pop(0)
                                intraday.at[intraday.index[idx], "Entry_Alert_Short"] = True
                                intraday.at[intraday.index[idx], "Entry_Emoji_Short"] = "☑️"
            
                    return intraday


                intraday = entryAlert(intraday, threshold=0.1)
              
                # Vector candle filter: crosses Kijun AND tags a Bollinger band (upper or lower)
                def is_omen_candle(row):
                    kijun = row["Kijun_F"]
                    upper = row["F% Upper"]
                    lower = row["F% Lower"]
                    close = row["Close"]
                    open_ = row["Open"]
                    high = row["High"]
                    low = row["Low"]
                    
                    crossed_kijun = (open_ < kijun and close > kijun) or (open_ > kijun and close < kijun)
                    touched_band = (high >= upper) or (low <= lower)
                    
                    return crossed_kijun and touched_band
                
                intraday["Omen_Candle"] = intraday.apply(is_omen_candle, axis=1)

            
                
                def clamp(x, lo, hi): 
                    return np.minimum(np.maximum(x, lo), hi)
                
                def omen_score_row(r, prox_cap=0.5):
                    """
                    Expects columns:
                      F_numeric, F% Upper, F% Lower, Kijun_F, Open, High, Low, Close,
                      RVOL_5, z_vecE, z_jerk
                    Returns (strength, label)
                    """
                    # ---------------- basics ----------------
                    kijun   = r["Kijun_F"]
                    f_up    = r["F% Upper"]
                    f_lo    = r["F% Lower"]
                    f       = r["F_numeric"]
                    o, h, l, c = r["Open"], r["High"], r["Low"], r["Close"]
                    rvol    = float(r.get("RVOL_5", 0) or 0)
                    zE      = float(r.get("z_vecE", 0) or 0)
                    zJ      = float(r.get("z_jerk", 0) or 0)
                
                    # band geometry
                    half_bw = 0.5 * max((f_up - f_lo), 1e-9)   # avoid div/0
                    # proximity to nearest edge, normalized by half band width
                    dist_up = abs(f_up - f)
                    dist_lo = abs(f - f_lo)
                    dist    = min(dist_up, dist_lo)
                    prox    = clamp(dist / half_bw, 0.0, prox_cap)  # 0=at edge, 0.5=far
                    tagged  = (h >= f_up) or (l <= f_lo)
                    near    = (prox <= 0.25)  # within 25% of edge
                
                    # direction: cross & close beyond Kijun
                    crossed_up   = (o < kijun) and (c > kijun)
                    crossed_down = (o > kijun) and (c < kijun)
                
                    # body dominance toward touched side
                    rng  = max(h - l, 1e-9)
                    body = abs(c - o)
                    body_ok = (body / rng) >= 0.40
                    # "toward edge": for up-cross, close near upper edge; for down-cross, close near lower edge
                    toward_up   = dist_up <= dist_lo
                    toward_down = dist_lo <  dist_up
                    body_toward = (crossed_up and toward_up) or (crossed_down and toward_down)
                
                    # gates
                    fuel_ok    = (rvol >= 1.2)                         # "horse"
                    impulse_ok = (zE >= 1.5) or (zJ >= 2.0)
                
                    # score components (0–100)
                    score = 0.0
                    # 1) direction proof (close beyond kijun)
                    if crossed_up or crossed_down:
                        score += 25
                
                    # 2) band proximity (0 at 0.5, 25 at 0.0; tag = full 25)
                    if tagged:
                        score += 25
                    else:
                        # linear from prox_cap→0 maps to 0→25
                        score += (1.0 - prox / prox_cap) * 25.0
                
                    # 3) fuel (RVOL linearly from 1.0→1.8 → 0→20)
                    score += clamp((rvol - 1.0) / (1.8 - 1.0), 0.0, 1.0) * 20.0
                
                    # 4) impulse (max of zE or zJ, each clamped 0→3, scaled to 20 total)
                    imp_unit = max(clamp(zE, 0, 3), clamp(zJ, 0, 3)) / 3.0
                    score += imp_unit * 20.0
                
                    # 5) body dominance toward edge
                    if body_ok and body_toward:
                        score += 10.0
                
                    # label logic
                    label = ""
                    if (crossed_up or crossed_down) and fuel_ok and impulse_ok:
                        if tagged:
                            label = "🔥 Full Omen"
                        elif near:
                            label = "⚠️ Near Omen"
                
                    return round(score, 1), label
                
                def compute_omen(df):
                    # apply row-wise
                    out = df.apply(omen_score_row, axis=1, result_type="expand")
                    df["Omen_Strength"] = out[0]
                    df["Omen_Label"]    = out[1]
                
                    # convenience booleans
                    df["Omen_Full"] = df["Omen_Label"].eq("🔥 Full Omen")
                    df["Omen_Near"] = df["Omen_Label"].eq("⚠️ Near Omen")
                
                    # direction flags
                    df["Omen_Bull"] = (
                        (df["Open"] < df["Kijun_F"]) & (df["Close"] > df["Kijun_F"]) &
                        (df["Omen_Full"] | df["Omen_Near"])
                    )
                    df["Omen_Bear"] = (
                        (df["Open"] > df["Kijun_F"]) & (df["Close"] < df["Kijun_F"]) &
                        (df["Omen_Full"] | df["Omen_Near"])
                    )
                    return df
                
                # --- call it ---
                intraday = compute_omen(intraday)

# --- plotting hints (minimal) ---
# mask = intraday["Omen_Full"] | intraday["Omen_Near"]
# fig_displacement.add_trace(go.Scatter(
#     x=intraday.loc[mask, "Time"],
#     y=intraday.loc[mask, "Cumulative_Unit"],
#     mode="text",
#     text=intraday.loc[mask, "Omen_Label"].str.replace(" Omen","", regex=False),
#     textfont=dict(size=14),
#     hovertemplate="Omen %{text}<br>Strength=%{customdata[0]}<extra></extra>",
#     customdata=intraday.loc[mask, ["Omen_Strength"]].values,
#     showlegend=False,
#     name="Omen"
# ))


                #  def entryAlert(intraday, threshold=0.1, rvol_threshold=1.2, rvol_lookback=9):
                #     """
                #     Entry Alert System (Corrected):
                #     - Step 1: Detect clean cross of F% through Kijun_F with buffer threshold.
                #     - Step 2: Confirm with next bar continuation.
                #     - Step 3: Require at least one RVOL_5 > threshold in the last rvol_lookback bars.
                #     """

                #     intraday["Entry_Alert_Short"] = False
                #     intraday["Entry_Alert_Long"]  = False

                #     for i in range(1, len(intraday) - 1):
                #         prev_f = intraday.iloc[i-1]["F_numeric"]
                #         prev_k = intraday.iloc[i-1]["Kijun_F"]
                #         curr_f = intraday.iloc[i]["F_numeric"]
                #         curr_k = intraday.iloc[i]["Kijun_F"]
                #         next_f = intraday.iloc[i+1]["F_numeric"]

                #         # ➡️ LONG CROSS
                #         if (prev_f < prev_k - threshold) and (curr_f > curr_k + threshold):
                #             if next_f >= curr_f:
                #                 intraday.at[intraday.index[i], "Entry_Alert_Long"] = True

                #         # ⬅️ SHORT CROSS
                #         if (prev_f > prev_k + threshold) and (curr_f < curr_k - threshold):
                #             if next_f <= curr_f:
                #                 intraday.at[intraday.index[i], "Entry_Alert_Short"] = True

                #     # 🔍 Second pass: check if at least one high RVOL_5
                #     for i in range(1, len(intraday) - 1):
                #         if intraday.iloc[i]["Entry_Alert_Long"] or intraday.iloc[i]["Entry_Alert_Short"]:
                #             start_idx = max(0, i - rvol_lookback + 1)
                #             rvol_window = intraday.iloc[start_idx:i+1]["RVOL_5"]

                #             if (rvol_window > rvol_threshold).sum() == 0:
                #                 # 🛑 No bars with RVOL > threshold → kill alert
                #                 intraday.at[intraday.index[i], "Entry_Alert_Long"] = False
                #                 intraday.at[intraday.index[i], "Entry_Alert_Short"] = False

                #     return intraday



                # intraday = entryAlert(intraday, threshold=0.1)
                

                
                # 🔢 Normalize Call Option Value to match F% scale
                base_premium = intraday["Call_Option_Value"].iloc[0]
                f_base = intraday["F_numeric"].iloc[0]
                
                intraday["Call_Option_Scaled"] = (intraday["Call_Option_Value"] / base_premium) * f_base
                
                # 🌀 Optional: Smooth it
                intraday["Call_Option_Scaled_Smooth"] = intraday["Call_Option_Scaled"].rolling(3).mean()

                # Normalize and smooth Put Option Value
                base_premium_put = intraday["Put_Option_Value"].iloc[0]
                intraday["Put_Option_Scaled"] = (intraday["Put_Option_Value"] / base_premium_put) * f_base
                intraday["Put_Option_Scaled_Smooth"] = intraday["Put_Option_Scaled"].rolling(3).mean()
                
                 
                


 

                def get_kijun_streak_log_with_dollar(df):
                    """
                    Returns list of Kijun streaks with their dollar change.
                    Format: 'K+5 : $2.45', 'K-3 : $-1.20'
                    """
                    if "F_numeric" not in df.columns or "Kijun_F" not in df.columns or "Close" not in df.columns:
                        return []

                    streaks = []
                    current_streak = 0
                    current_state = None
                    start_price = None

                    for i in range(len(df)):
                        f_val = df["F_numeric"].iloc[i]
                        k_val = df["Kijun_F"].iloc[i]
                        close_price = df["Close"].iloc[i]

                        if pd.isna(f_val) or pd.isna(k_val) or pd.isna(close_price):
                            continue

                        is_above = f_val > k_val

                        if current_state is None:
                            current_state = is_above
                            current_streak = 1
                            start_price = close_price
                        elif is_above == current_state:
                            current_streak += 1
                        else:
                            end_price = df["Close"].iloc[i - 1]
                            dollar_return = end_price - start_price
                            label = f"K+{current_streak}" if current_state else f"K-{current_streak}"
                            streaks.append(f"{label} : ${dollar_return:.2f}")
                            current_state = is_above
                            current_streak = 1
                            start_price = close_price

                    if current_streak > 0 and start_price is not None:
                        end_price = df["Close"].iloc[-1]
                        dollar_return = end_price - start_price
                        label = f"K+{current_streak}" if current_state else f"K-{current_streak}"
                        streaks.append(f"{label} : ${dollar_return:.2f}")

                    return streaks



                log_with_returns = get_kijun_streak_log_with_dollar(intraday)

                # st.markdown("### 📘 Full Kijun Streak Log with $ Returns:")
                # for line in log_with_returns:
                #     st.markdown(f"<div style='font-size:20px'>{line}</div>", unsafe_allow_html=True)
             
  
             



                def add_mike_kijun_horse_emoji(df):
                    """
                    Adds 🏇🏽 emoji when Mike (F_numeric) crosses Kijun and relative volume (RVOL_5) > 1.5
                    in any of the 7 bars: 3 before, the cross itself, and 3 after.
                    """
                    crosses_up = (df["F_numeric"].shift(1) < df["Kijun_F"].shift(1)) & (df["F_numeric"] >= df["Kijun_F"])
                    crosses_down = (df["F_numeric"].shift(1) > df["Kijun_F"].shift(1)) & (df["F_numeric"] <= df["Kijun_F"])
                
                    emoji_flags = []
                
                    for i in range(len(df)):
                        if not (crosses_up.iloc[i] or crosses_down.iloc[i]):
                            emoji_flags.append("")
                            continue
                
                        start = max(0, i - 3)
                        end = min(len(df), i + 4)  # +4 because end index is exclusive
                        rvol_slice = df.iloc[start:end]["RVOL_5"]
                
                        if any(rvol_slice > 1.5):
                            emoji_flags.append("🏇🏽")
                        else:
                            emoji_flags.append("")
                
                    df["Mike_Kijun_Horse_Emoji"] = emoji_flags
                    return df
                
                # Apply to your intraday DataFrame
                intraday = add_mike_kijun_horse_emoji(intraday)

                def add_mike_kijun_bee_emoji(df):
                    """
                    Adds 🍯 emoji at the point Mike (F_numeric) crosses Kijun_F,
                    but only if a 🐝 (BBW tight) was seen within ±3 bars of the cross.
                    """
                    crosses_up = (df["F_numeric"].shift(1) < df["Kijun_F"].shift(1)) & (df["F_numeric"] >= df["Kijun_F"])
                    crosses_down = (df["F_numeric"].shift(1) > df["Kijun_F"].shift(1)) & (df["F_numeric"] <= df["Kijun_F"])
                
                    emoji_flags = []
                
                    for i in range(len(df)):
                        if not (crosses_up.iloc[i] or crosses_down.iloc[i]):
                            emoji_flags.append("")
                            continue
                
                        # Look ±3 bars for 🐝
                        start = max(0, i - 3)
                        end = min(len(df), i + 4)
                        bee_window = df.iloc[start:end]["BBW_Tight_Emoji"]
                
                        if "🐝" in bee_window.values:
                            emoji_flags.append("🍯")
                        else:
                            emoji_flags.append("")
                
                    df["Mike_Kijun_Bee_Emoji"] = emoji_flags
                    return df
                
                # Apply to your intraday DataFrame
                intraday = add_mike_kijun_bee_emoji(intraday)

           # Identify Top 3 Positive and Negative Velocity Vectors
                def extract_top_velocity_markers(df, col_name="Velocity", time_col="Time"):
                    # Convert to numeric (strip "%")
                    df = df.copy()
                    df["Velocity_Num"] = pd.to_numeric(df[col_name].str.replace("%", ""), errors="coerce")
                    
                    # Drop rows with NaNs
                    df_clean = df.dropna(subset=["Velocity_Num"])
                
                    # Get top 3 positive and negative
                    top_pos = df_clean.nlargest(3, "Velocity_Num")
                    top_neg = df_clean.nsmallest(3, "Velocity_Num")
                
                    return top_pos, top_neg
                
                # Get markers
                top_pos_vel, top_neg_vel = extract_top_velocity_markers(intraday)

     
             


              
                def add_mike_kijun_atr_emoji(df, atr_col="ATR"):
                    """
                    Adds 🌋 emoji at the point Mike (F_numeric) crosses Kijun_F,
                    but only if an ATR expansion (1.5x previous 10 bars max) is detected
                    within ±3 bars of the cross.
                    """
                
                    crosses_up = (df["F_numeric"].shift(1) < df["Kijun_F"].shift(1)) & (df["F_numeric"] >= df["Kijun_F"])
                    crosses_down = (df["F_numeric"].shift(1) > df["Kijun_F"].shift(1)) & (df["F_numeric"] <= df["Kijun_F"])
                    
                    emoji_flags = []
                
                    # Precompute ATR expansions
                    atr_expansion = []
                    for i in range(len(df)):
                        if i < 10:
                            atr_expansion.append(False)
                            continue
                        prior_max = df[atr_col].iloc[i-10:i].max()
                        atr_expansion.append(df[atr_col].iloc[i] > 1.5 * prior_max)
                    df["ATR_Expansion_Flag"] = atr_expansion
                
                    for i in range(len(df)):
                        if not (crosses_up.iloc[i] or crosses_down.iloc[i]):
                            emoji_flags.append("")
                            continue
                
                        # Look ±3 bars around this index for ATR expansion
                        start = max(0, i - 3)
                        end = min(len(df), i + 4)
                        window = df["ATR_Expansion_Flag"].iloc[start:end]
                
                        if window.any():
                            emoji_flags.append("🌋")
                        else:
                            emoji_flags.append("")
                
                    df["Mike_Kijun_ATR_Emoji"] = emoji_flags
                    return df
                intraday = add_mike_kijun_atr_emoji(intraday)
                
                def add_sharpe_column(intraday, window=26):
                    """
                    Adds a rolling Sharpe Ratio column based on F_numeric over a given window.
                    Designed for intraday (e.g. 5-min) data.
                
                    - Uses F_numeric (% change) converted to decimal.
                    - Applies rolling mean/std with min_periods=1 to avoid NaNs early on.
                    - Adds 'Sharpe_Ratio' column rounded to 2 decimals.
                    """
                    # Convert % change to decimal form (e.g., 34 becomes 0.34)
                    returns = intraday["F_numeric"] / 100
                
                    # Rolling mean and std with early minimum support
                    rolling_mean = returns.rolling(window=window, min_periods=1).mean()
                    rolling_std = returns.rolling(window=window, min_periods=1).std()
                
                    # Avoid division by zero (early std can be 0)
                    sharpe = rolling_mean / (rolling_std + 1e-9)
                
                    # Add result to DataFrame
                    intraday["Sharpe_Ratio"] = sharpe.round(2)
                
                    return intraday
                intraday = add_sharpe_column(intraday)

                             # --- CROSS CONDITIONS ---
                tenkan_above_kijun = (
                    (intraday["Tenkan"].shift(1) < intraday["Kijun"].shift(1)) &
                    (intraday["Tenkan"] >= intraday["Kijun"])
                )
                
                tenkan_below_kijun = (
                    (intraday["Tenkan"].shift(1) > intraday["Kijun"].shift(1)) &
                    (intraday["Tenkan"] <= intraday["Kijun"])
                )
                
                # --- CLOUD POSITION ---
                price_above_cloud = (intraday["Close"] > intraday["SpanA"]) & (intraday["Close"] > intraday["SpanB"])
                price_below_cloud = (intraday["Close"] < intraday["SpanA"]) & (intraday["Close"] < intraday["SpanB"])
                
                # --- CHIKOU POSITION (shifted back into present) ---
                chikou_above_price = (intraday["Chikou"].shift(26) > intraday["Close"].shift(26))
                chikou_below_price = (intraday["Chikou"].shift(26) < intraday["Close"].shift(26))
                
                # === FINAL COMBINED SIGNALS ===
                
                # 🟩 Sanyaku Kouten
                intraday["Sanyaku_Kouten"] = np.where(
                    tenkan_above_kijun & price_above_cloud & chikou_above_price,
                    "🟩", "")
                
                # 🟥 Sanyaku Gyakuten
                intraday["Sanyaku_Gyakuten"] = np.where(
                    tenkan_below_kijun & price_below_cloud & chikou_below_price,
                    "🟥", "")


           

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



            
                
             
                    # Quick check:
                    # intraday.loc[intraday["Omen_On"], ["Time","Omen_Label","Omen_Direction","Omen_Strength"]].head()

# --- plotting hints (minimal) ---
# mask = intraday["Omen_Full"] | intraday["Omen_Near"]
# fig_displacement.add_trace(go.Scatter(
#     x=intraday.loc[mask, "Time"],
#     y=intraday.loc[mask, "Cumulative_Unit"],
#     mode="text",
#     text=intraday.loc[mask, "Omen_Label"].str.replace(" Omen","", regex=False),
#     textfont=dict(size=14),
#     hovertemplate="Omen %{text}<br>Strength=%{customdata[0]}<extra></extra>",
#     customdata=intraday.loc[mask, ["Omen_Strength"]].values,
#     showlegend=False,
#     name="Omen"
# ))

               
                
                

                    
                    






                if gap_alert:
                    st.warning(gap_alert)








                    
                with st.expander("Show/Hide Data Table",  expanded=False):
                                # Show data table, including new columns
                    cols_to_show = [
                                    "RVOL_5","Range","Time","Volume","ADX_F%","+DM","-DM","Sharpe_Ratio","Call_BBW_Tight_Emoji","Put_BBW_Tight_Emoji","Compliance","Distensibility","Distensibility Alert","Volatility_Composite","Gravity_Break_Alert","F_numeric","Kijun_Cumulative","Unit%","Vector%","Unit Velocity","Velocity","Voltage","Vector_Charge","Vector_Capacitance","Charge_Polarity","Field_Intensity","Electric_Force","Unit Acceleration","Acceleration","Accel_Spike","Acceleration_Alert","Jerk_Unit","Jerk_Vector","Snap","Unit Momentum","Vector Momentum","Unit Force","Vector Force","Power","Intensity","Unit Energy","Vector Energy","Force_per_Range","Force_per_3bar_Range","Unit_Energy_per_Range","Vector_Energy_per_3bar_Range"]

                    st.dataframe(intraday[cols_to_show])

                ticker_tabs = st.tabs(["Mike Plot"])



                

                with st.expander("Market Profile (F% Letters View)", expanded=False):
    
                  # Detect Mike column — fallback to F_numeric if 'Mike' isn't present
                  mike_col = None
                  if "Mike" in intraday.columns:
                      mike_col = "Mike"
                  elif "F_numeric" in intraday.columns:
                      mike_col = "F_numeric"
                  else:
                      st.warning("Mike or F_numeric column not found.")
                      st.stop()
              
                  # Bin F% values — label as strings to prevent type issues
                  f_bins = np.arange(-400, 401, 20)
                  intraday['F_Bin'] = pd.cut(intraday[mike_col], bins=f_bins, labels=[str(x) for x in f_bins[:-1]])
              # Remove rows with missing or invalid Time
                  intraday = intraday[intraday['Time'].notna()]
                  st.write("Bad Time rows:", intraday[intraday['Time'].isna()])

                  # Assign each row a letter based on 15-minute intervals
                  # Drop NA first to avoid parsing issues
                  intraday = intraday[intraday['Time'].notna()]
                  
                  # Optional: filter out malformed time strings if needed
                  # intraday = intraday[intraday['Time'].str.match(r"\d{1,2}:\d{2} [APap][Mm]")]
                  
                  # Parse Time column safely
                  intraday['TimeIndex'] = pd.to_datetime(intraday['Time'], format="%I:%M %p", errors='coerce')
                  
                  # Drop rows where conversion failed
                  intraday = intraday[intraday['TimeIndex'].notna()]
                  intraday['LetterIndex'] = ((intraday['TimeIndex'].dt.hour * 60 + intraday['TimeIndex'].dt.minute) // 15).astype(int)
                  intraday['LetterIndex'] -= intraday['LetterIndex'].min()  # Normalize to start at 0
              
                  # Convert index to letters (A–Z, AA–AZ…)
                  def letter_code(n: int) -> str:
                      n = int(n)
                      letters = string.ascii_uppercase
                      if n < 26:
                          return letters[n]
                      else:
                          first = letters[(n // 26) - 1]
                          second = letters[n % 26]
                          return first + second
              
                  intraday['Letter'] = intraday['LetterIndex'].apply(letter_code)
              
                  # Step 1: Filter Initial Balance (first 4 letters: A–D)
                  initial_letters = ['A', 'B', 'C', 'D']
                  ib_df = intraday[intraday['Letter'].isin(initial_letters)]
                  ib_option_df = intraday[intraday['Letter'].isin(initial_letters)]



                  
                  # Step 2: Get IB high/low F% range
                  ib_high = ib_df[mike_col].max()
                  ib_low = ib_df[mike_col].min()


                  # Call Option IB High/Low
                  call_ib_high = ib_option_df["Call_Option_Smooth"].max()
                  call_ib_low  = ib_option_df["Call_Option_Smooth"].min()
                  
                  # Put Option IB High/Low
                  put_ib_high = ib_option_df["Put_Option_Smooth"].max()
                  put_ib_low  = ib_option_df["Put_Option_Smooth"].min()


                  
                  # Build Market Profile dictionary
                  profile = {}
                  for f_bin in f_bins[:-1]:
                      f_bin_str = str(f_bin)
                      letters = intraday.loc[intraday['F_Bin'] == f_bin_str, 'Letter'].dropna().unique()
                      if len(letters) > 0:
                          profile[f_bin_str] = ''.join(sorted(letters))
              
                  # Convert to DataFrame
                  profile_df = pd.DataFrame(list(profile.items()), columns=['F% Level', 'Letters'])
                  profile_df['F% Level'] = profile_df['F% Level'].astype(int)  # Convert back for display
              
                  # Add Tail column: 🪶 if only one unique letter
                  profile_df["Tail"] = profile_df["Letters"].apply(
                      lambda x: "🪶" if isinstance(x, str) and len(set(x)) == 1 else ""
                  )
              
                  # Detect Range Extension: 💥 for activity beyond IB range
                  def is_range_extension(row):
                      if pd.isna(row["Letters"]):
                          return False
                      post_ib_letters = [l for l in str(row["Letters"]) if l not in initial_letters]
                      if row["F% Level"] > ib_high and post_ib_letters:
                          return True
                      if row["F% Level"] < ib_low and post_ib_letters:
                          return True
                      return False
              
                  profile_df["Range_Extension"] = profile_df.apply(is_range_extension, axis=1)
                  profile_df["💥"] = profile_df["Range_Extension"].apply(lambda x: "💥" if x else "")


                  # Count letters per F% level
                  profile_df["Letter_Count"] = profile_df["Letters"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                  
                  # Total letters (to calculate 70% cutoff)
                  total_letters = profile_df["Letter_Count"].sum()
                  target_count = total_letters * 0.7
                  
                  # Sort by most active rows (letter count), center at Point of Control
                  profile_sorted = profile_df.sort_values(by="Letter_Count", ascending=False).reset_index(drop=True)
                  
                  # Build Value Area by accumulating from POC outward
                  value_area_levels = []
                  cumulative = 0
                  for i, row in profile_sorted.iterrows():
                      cumulative += row["Letter_Count"]
                      value_area_levels.append(row["F% Level"])
                      if cumulative >= target_count:
                          break
                  
                  # Add Value Area marker
                  profile_df["✅ ValueArea"] = profile_df["F% Level"].apply(lambda x: "✅" if x in value_area_levels else "")


                  # Sum volume per F% bin
                  vol_per_bin = intraday.groupby("F_Bin")["Volume"].sum()
                  
                  # Total volume across all bins
                  total_vol = vol_per_bin.sum()
                  
                  # Calculate % volume per bin
                  vol_percent = (vol_per_bin / total_vol * 100).round(2)
                  
                  # Merge into profile_df (align on string F% Level)
                  profile_df["%Vol"] = profile_df["F% Level"].astype(str).map(vol_percent).fillna(0)


                  # Define most volume bin level (used in resistance)
                  max_vol_level = profile_df.loc[profile_df['%Vol'].idxmax(), 'F% Level']
                  max_letter_level = profile_df.loc[profile_df['Letter_Count'].idxmax(), 'F% Level']

                  
                                    # Add earliest Time seen in each F% bin
                  bin_times = intraday.groupby('F_Bin')['Time'].min().reset_index()
                  bin_times['F% Level'] = bin_times['F_Bin'].astype(int)
                  profile_df = profile_df.merge(bin_times[['F% Level', 'Time']], on='F% Level', how='left')

                  # Current Mike value (latest row)
                  current_mike = intraday[mike_col].iloc[-1]
                  
                  # Compute min/max of value area for better boundary
                  va_min = min(value_area_levels)
                  va_max = max(value_area_levels)
                  
                  # # === STEP 1: Create Resistance Reference Data ===
                  # resistance_lines = {
                  #     "IB_High": ib_high,
                  #     "IB_Low": ib_low,
                  #     "High_Vol_Bin": max_vol_level,
                  #     "High_Letter_Bin": max_letter_level,
                  #     "VA_High": va_max,
                  #     "VA_Low": va_min
                  # }
                  
                  # res_df = pd.DataFrame({
                  #     "Level": list(resistance_lines.values()),
                  #     "Label": list(resistance_lines.keys())
                  # }).sort_values(by="Level", ascending=False).reset_index(drop=True)
                  


                         # Step 1: Identify the volume-dominant F% level
                  max_vol_level = profile_df.loc[profile_df['%Vol'].idxmax(), 'F% Level']
                  
                  # Step 2: Bin the current Mike value to its F% level
                  current_mike_bin = f_bins[np.digitize(current_mike, f_bins) - 1]
                  
                  # Step 3: Add a 🦻🏼 to any profile row that used to be the dominant %Vol level but is no longer the current Mike bin
                  def ear_marker(row):
                      if row['F% Level'] == max_vol_level and current_mike_bin != max_vol_level:
                          return "🦻🏼"  # Permanently add
                      return row.get("🦻🏼", "")  # Preserve existing 🦻🏼 if already present
                  
                  # Apply the marker logic
                  if "🦻🏼" not in profile_df.columns:
                      profile_df["🦻🏼"] = ""
                  
                  profile_df["🦻🏼"] = profile_df.apply(ear_marker, axis=1)
                  
                  # Step 1: Identify F% level with most letters (most time spent)
                  max_letter_level = profile_df.loc[profile_df['Letter_Count'].idxmax(), 'F% Level']
                  
                  # Step 2: Find current Mike bin
                  current_mike_bin = f_bins[np.digitize(current_mike, f_bins) - 1]
                  
                  # Step 3: If current Mike ≠ that time-dominant level, mark 👃🏽
                  def nose_marker(row):
                      if row['F% Level'] == max_letter_level and current_mike_bin != max_letter_level:
                          return "👃🏽"
                      return ""
                  
                  profile_df["👃🏽"] = profile_df.apply(nose_marker, axis=1)

                                 
       
                   # Define Initial Balance from first 12 candles
                  ib_data = intraday.iloc[:12]  # First hour (12 x 5min bars)
                  
                  ib_high = ib_data["F_numeric"].max()
                  ib_low = ib_data["F_numeric"].min()
                  
                  # Add to intraday as constant columns
                  intraday["IB_High"] = ib_high
                  intraday["IB_Low"] = ib_low
                  
                                
                                    # Initialize IB breakout emojis
                  intraday["IB_High_Break"] = ""
                  intraday["IB_Low_Break"] = ""
                  
                  # Track prior state (inside/outside IB)
                  intraday["Inside_IB"] = (intraday["F_numeric"] >= intraday["IB_Low"]) & (intraday["F_numeric"] <= intraday["IB_High"])
                  intraday["Prior_Inside_IB"] = intraday["Inside_IB"].shift(1)
                  
                  # 💸 Breakout above IB High
                  ib_high_break = (
                      (intraday["F_numeric"] > intraday["IB_High"]) &  # now above
                      (intraday["Prior_Inside_IB"] == True)            # came from inside
                  )
                  intraday.loc[ib_high_break, "IB_High_Break"] = "💸"
                  
                  # 🧧 Breakdown below IB Low
                  ib_low_break = (
                      (intraday["F_numeric"] < intraday["IB_Low"]) &   # now below
                      (intraday["Prior_Inside_IB"] == True)            # came from inside
                  )
                  intraday.loc[ib_low_break, "IB_Low_Break"] = "🧧"

                            
                  def add_stamina_signal(intraday, profile_df, f_bins, rvol_gate=1.2):
                      """
                      Adds Stamina_Signal column:
                        ⚪ = no stamina
                        🪨 = stamina but blocked (at volume- or time-memory level)
                        💪 = stamina + clear track
                      """
                      intraday = intraday.copy()
                  
                      # Ensure numeric
                      for col in ["F_numeric", "RVOL_5"]:
                          intraday[col] = pd.to_numeric(intraday[col], errors="coerce")
                  
                      if intraday[["F_numeric", "RVOL_5"]].isnull().all().any():
                          # Required series are entirely NaN → skip
                          return intraday
                  
                      max_vol_level    = int(profile_df.loc[profile_df['%Vol'].idxmax(), 'F% Level'])
                      max_letter_level = int(profile_df.loc[profile_df['Letter_Count'].idxmax(), 'F% Level'])
                  
                      stamina = []
                      last_bin = f_bins[-2]  # second-to-last edge (uppermost real bin)
                  
                      for mike, rvol in zip(intraday["F_numeric"], intraday["RVOL_5"]):
                          if pd.isna(mike) or pd.isna(rvol):
                              stamina.append("⚪")
                              continue
                  
                          # Safe binning
                          idx = np.digitize(mike, f_bins) - 1
                          idx = max(0, min(idx, len(f_bins) - 2))
                          current_bin = f_bins[idx]
                  
                          if rvol <= rvol_gate:
                              stamina.append("⚪")
                          elif current_bin in (max_vol_level, max_letter_level):
                              stamina.append("🪨")
                          else:
                              stamina.append("💪")
                  
                      intraday["Stamina_Signal"] = stamina
                      return intraday
                  
                  intraday = add_stamina_signal(intraday, profile_df, f_bins)
                  
                  print(intraday.columns[-5:])          # ‘Stamina_Signal’ should be listed
                  print(intraday.tail()[["F_numeric", "RVOL_5", "Stamina_Signal"]])
                  
                  

                  
                  #                   # === Top Dot Logic by 15-Minute Block ===
                  top_dots = (
                      intraday.loc[intraday.groupby("LetterIndex")["F_numeric"].idxmax()]
                      .sort_values("LetterIndex")
                      .reset_index(drop=True)
                  )
                  top_dots = (
                      intraday.groupby("LetterIndex").apply(lambda g: g.loc[g["F_numeric"].idxmax()])
                      .reset_index(drop=True)
                  )
                  # top_dots["Time"] = intraday.groupby("LetterIndex")["Time"].max().values  # Force dot to close of bracket

                  # Step 1: Get row of max F% per 15-min block (actual auction moment)
                  top_dots = (
                      intraday.groupby("LetterIndex").apply(lambda g: g.loc[g["F_numeric"].idxmax()])
                      .reset_index(drop=True)
                  )
                  
                  # Save actual time of high for ghost dot
                  top_dots["Time_HighMoment"] = top_dots["Time"]
                  
                  # Replace main Time column with bracket end time for dot alignment
                  top_dots["Time"] = intraday.groupby("LetterIndex")["Time"].max().values


                  # Compare each top with the previous group to decide direction
                  top_dots["Prev_High"] = top_dots["F_numeric"].shift(1)
                  
                  def assign_dot_color(row):
                      if pd.isna(row["Prev_High"]):
                          return "gray"
                      elif row["F_numeric"] > row["Prev_High"]:
                          return "green"
                      elif row["F_numeric"] < row["Prev_High"]:
                          return "red"
                      else:
                          return "gray"
                  
                  top_dots["DotColor"] = top_dots.apply(assign_dot_color, axis=1)
                  
      


                  # # Show DataFrame
                  st.dataframe(profile_df[["F% Level","Time","Letters",  "%Vol","💥","Tail","✅ ValueArea","🦻🏼", "👃🏽"]])

                  
                  def compute_ib_volume_weights(intraday, ib_high, ib_low):
                        """
                        Split the Initial Balance range into 3 equal compartments: Cellar, Core, Loft.
                        For each, compute:
                          - Total volume
                          - Volume per area (pressure)
                          - Weight (assuming unit gravity, w = mg)
                        """
                        df = intraday.copy()
                        
                        # Define IB levels
                        ib_range = ib_high - ib_low
                        third = ib_range / 3
                    
                        # Define compartment boundaries
                        cellar_top = ib_low + third
                        core_top = ib_low + 2 * third
                    
                        # Tag zones
                        def tag_zone(row):
                            price = row["Close"]
                            if price < cellar_top:
                                return "Cellar"
                            elif price < core_top:
                                return "Core"
                            else:
                                return "Loft"
                    
                        df["IB_Zone"] = df.apply(tag_zone, axis=1)
                    
                        # Compute zone stats
                        zone_stats = df.groupby("IB_Zone")["Volume"].agg(
                            Total_Volume="sum",
                            Bar_Count="count"
                        ).reset_index()
                    
                        # Assume equal area for all zones
                        zone_stats["Area"] = 1  # normalized
                    
                        # Volume per area = pressure
                        zone_stats["Volume_Pressure"] = zone_stats["Total_Volume"] / zone_stats["Area"]
                    
                        # Weight (w = m * g); here mass ~ volume, and gravity g = 1
                        zone_stats["Weight"] = zone_stats["Total_Volume"]  # since g = 1
                    
                        return zone_stats
  
                  ib_stats = compute_ib_volume_weights(intraday, ib_high=ib_high, ib_low=ib_low)
                


                  

                  
                  def add_ib_field_force(df, resistance_col="IB_High"):
                      df = df.copy()
                  
                      # 1. Convert Velocity to numeric (Voltage)
                      df["V_numeric"] = pd.to_numeric(df["Velocity"].str.replace("%", ""), errors="coerce")
                  
                      # 2. Charge = RVOL_5
                      Q = df["RVOL_5"]
                  
                      # 3. Distance from IB High to current level
                      d = (df[resistance_col] - df["Cumulative_Unit"]).abs().replace(0, np.nan)  # avoid div-by-zero
                  
                      # 4. Field = Voltage / Distance
                      df["IB_Field_Intensity"] = df["V_numeric"] / d
                  
                      # 5. Electric Force = Q * Field
                      df["IB_Electric_Force"] = Q * df["IB_Field_Intensity"]
                  
                      return df
                  intraday = add_ib_field_force(intraday)
                                         # Initialize columns
                  intraday["🪘"] = ""
                  intraday["Drum_Y"] = np.nan
                  above = False
                  ear_level = profile_df.loc[profile_df['%Vol'].idxmax(), 'F% Level']
                  nose_level = profile_df.loc[profile_df['Letter_Count'].idxmax(), 'F% Level']

                  for i in range(1, len(intraday)):
                      now = intraday.iloc[i]
                      prev = intraday.iloc[i - 1]
                  
                      crossed_up = (
                          not above and (
                              (prev["F_numeric"] <= ear_level and now["F_numeric"] > ear_level) or
                              (prev["F_numeric"] <= nose_level and now["F_numeric"] > nose_level)
                          )
                      )
                  
                      reset_state = (
                          above and
                          (now["F_numeric"] < ear_level and now["F_numeric"] < nose_level)
                      )
                  
                      if crossed_up:
                          intraday.at[intraday.index[i], "🪘"] = "🪘"
                          intraday.at[intraday.index[i], "Drum_Y"] = now["F_numeric"] + 16
                          above = True
                  
                      elif reset_state:
                          intraday.at[intraday.index[i], "🪘"] = "🪘"
                          intraday.at[intraday.index[i], "Drum_Y"] = now["F_numeric"] - 16
                          above = False


                  intraday['Smoothed_Compliance'] = intraday['Compliance'].rolling(window=5, min_periods=1).mean()
 
                            
                 
                  
                             
                            
                  with st.expander("MIDAS Curves (Bull + Bear Anchors)", expanded=False):
                  
                      # Detect price column
                      if "Mike" in intraday.columns:
                          price_col = "Mike"
                      elif "F_numeric" in intraday.columns:
                          price_col = "F_numeric"
                      else:
                          st.warning("Mike or F_numeric column not found.")
                          st.stop()
                  
                      if "Volume" not in intraday.columns:
                          st.warning("Volume column not found.")
                          st.stop()
                  
                      # Defensive check for valid prices
                      if intraday[price_col].dropna().empty:
                          st.warning(f"No valid values in '{price_col}' for MIDAS anchor calculation.")
                          st.stop()
                  
                      # Convert time
                      intraday['TimeIndex'] = pd.to_datetime(intraday['Time'], format="%I:%M %p")
                  
                      ### 🐻 BEARISH MIDAS (anchor at max)
                      anchor_idx_bear = intraday[price_col].idxmax()
                      anchor_time_bear = intraday.loc[anchor_idx_bear, 'TimeIndex']
                      anchor_price_bear = intraday.loc[anchor_idx_bear, price_col]
                  
                      midas_curve_bear = []
                      for i in range(anchor_idx_bear, len(intraday)):
                          vol_window = intraday.loc[anchor_idx_bear:i, 'Volume']
                          price_window = intraday.loc[anchor_idx_bear:i, price_col]
                          weights = vol_window / vol_window.sum()
                          midas_price = (price_window * weights).sum()
                          midas_curve_bear.append(midas_price)
                  
                      intraday["MIDAS_Bear"] = [np.nan] * anchor_idx_bear + midas_curve_bear
                  
                      ### 🐂 BULLISH MIDAS (anchor at min)
                      anchor_idx_bull = intraday[price_col].idxmin()
                      anchor_time_bull = intraday.loc[anchor_idx_bull, 'TimeIndex']
                      anchor_price_bull = intraday.loc[anchor_idx_bull, price_col]
                  
                      midas_curve_bull = []
                      for i in range(anchor_idx_bull, len(intraday)):
                          vol_window = intraday.loc[anchor_idx_bull:i, 'Volume']
                          price_window = intraday.loc[anchor_idx_bull:i, price_col]
                          weights = vol_window / vol_window.sum()
                          midas_price = (price_window * weights).sum()
                          midas_curve_bull.append(midas_price)
                  
                      intraday["MIDAS_Bull"] = [np.nan] * anchor_idx_bull + midas_curve_bull
                      intraday["Bear_Displacement"] = intraday["MIDAS_Bear"] - intraday["F_numeric"]
                      intraday["Bull_Displacement"] = intraday["F_numeric"] - intraday["MIDAS_Bull"]

                      intraday["Bear_Displacement_Change"] = intraday["Bear_Displacement"].diff()
                      intraday["Bull_Displacement_Change"] = intraday["Bull_Displacement"].diff()

                      intraday["Hold_Put"] = (
                          (intraday["Bear_Displacement_Change"] > 0) |
                          (intraday["Bear_Displacement"].rolling(3).min() > 20)  # stays deep
                      )
                      
                      intraday["Hold_Call"] = (
                          (intraday["Bull_Displacement_Change"] > 0) |
                          (intraday["Bull_Displacement"].rolling(3).min() > 20)
                      )


                      intraday["Bear_Displacement_Double"] = ""
                      
                      for i in range(1, len(intraday)):
                          prev = intraday["Bear_Displacement"].iloc[i - 1]
                          curr = intraday["Bear_Displacement"].iloc[i]
                          
                          if pd.notna(prev) and prev > 0 and pd.notna(curr):
                              if curr >= 2 * prev:
                                  intraday.at[intraday.index[i], "Bear_Displacement_Double"] = "💀"

                      intraday["Bull_Displacement_Double"] = ""
  
                      for i in range(1, len(intraday)):
                          prev = intraday["Bull_Displacement"].iloc[i - 1]
                          curr = intraday["Bull_Displacement"].iloc[i]
                          
                          if pd.notna(prev) and prev > 0 and pd.notna(curr):
                              if curr >= 2 * prev:
                                  intraday.at[intraday.index[i], "Bull_Displacement_Double"] = "👑"


                      intraday["Bear_Lethal_Accel"] = ""
                      
                      for i in range(2, len(intraday)):
                          prev = intraday["Bear_Displacement"].iloc[i - 1]
                          curr = intraday["Bear_Displacement"].iloc[i]
                          delta = curr - prev
                          recent_min = intraday["Bear_Displacement"].iloc[i-3:i].min()
                          
                          # Lethal if all 3 conditions met
                          if (
                              pd.notna(curr)
                              and curr > 1.5 * recent_min  # explosive growth in displacement
                              and delta > 5               # sharp jump in a single bar
                              and curr > 20               # absolute distance confirms real separation
                          ):
                              intraday.at[intraday.index[i], "Bear_Lethal_Accel"] = "🥊"
                      
                      intraday["Bull_Lethal_Accel"] = ""

                      for i in range(2, len(intraday)):
                          prev = intraday["Bull_Displacement"].iloc[i - 1]
                          curr = intraday["Bull_Displacement"].iloc[i]
                          delta = curr - prev
                          recent_min = intraday["Bull_Displacement"].iloc[i-3:i].min()
                          
                          if (
                              pd.notna(curr)
                              and curr > 1.5 * recent_min
                              and delta > 5
                              and curr > 20
                          ):
                              intraday.at[intraday.index[i], "Bull_Lethal_Accel"] = "🚀"






                              
                  
                      def check_midas_3delta_trigger(df):
                                    """
                                    Detects conditions to trigger a 3-delta buy alert:
                                    - A Midas Bull or Bear anchor appears
                                    - Price crosses a TD Supply or Demand line
                                    - The cross is in the same direction as Midas slope
                                    Returns the dataframe with Bull3DeltaTrigger and Bear3DeltaTrigger columns
                                    """
                                
                                    # 1. Detect anchor appearance
                                    df["BullAnchor"] = df["MIDAS_Bull"].notna() & df["MIDAS_Bull"].shift(1).isna()
                                    df["BearAnchor"] = df["MIDAS_Bear"].notna() & df["MIDAS_Bear"].shift(1).isna()
                                
                                    # 2. Detect price crossing TD lines
                                    df["Crossed_TD_Demand"] = (df["F_numeric"].shift(1) < df['TD Demand Line F']) & (df["F_numeric"] >= df['TD Demand Line F'])
                                    df["Crossed_TD_Supply"] = (df["F_numeric"].shift(1) > df['TD Supply Line F']) & (df["F_numeric"] <= df['TD Supply Line F'])
                                
                                    # 3. Calculate slope direction of Midas lines
                                    df["BullSlope"] = df["MIDAS_Bull"].diff()
                                    df["BearSlope"] = df["MIDAS_Bear"].diff()
                                
                                    df["BullSlopeUp"] = df["BullSlope"] > 0
                                    df["BearSlopeDown"] = df["BearSlope"] < 0
                                
                                    # 4. Final trigger logic
                                    df["Bull3DeltaTrigger"] = (
                                        df["BullAnchor"] &
                                        df["Crossed_TD_Demand"] &
                                        df["BullSlopeUp"]
                                    )
                                
                                    df["Bear3DeltaTrigger"] = (
                                        df["BearAnchor"] &
                                        df["Crossed_TD_Supply"] &
                                        df["BearSlopeDown"]
                                    )
                                
                                    return df
                
                      intraday = check_midas_3delta_trigger(intraday)





                    
                      def add_mike_midas_cross_emojis(df, price_col):
                          if not all(col in df.columns for col in [price_col, "MIDAS_Bull", "MIDAS_Bear"]):
                              return df, None, None
                  
                          price = df[price_col]
                          bull = df["MIDAS_Bull"]
                          bear = df["MIDAS_Bear"]
                          close_next = price.shift(-1)
                  
                          # 👋🏽 Bull breakout = price crosses above MIDAS_Bear from below
                          bull_cross = (price.shift(1) < bear.shift(1)) & (price >= bear)
                          # 🧤 Bear breakdown = price crosses below MIDAS_Bull from above
                          bear_cross = (price.shift(1) > bull.shift(1)) & (price <= bull)
                  
                          df["MIDAS_Bull_Hand"] = np.where(bull_cross, "👋🏽", "")
                          df["MIDAS_Bear_Glove"] = np.where(bear_cross, "🧤", "")

                                                  # 🎷 MIDAS Bull crosses above IB High
                          if "IB_High" in df.columns and "MIDAS_Bull" in df.columns:
                              bull_ib_cross = (df["MIDAS_Bull"].shift(1) < df["IB_High"].shift(1)) & (df["MIDAS_Bull"] >= df["IB_High"])
                              df["Midas_Cross_IB_High"] = np.where(bull_ib_cross, "🎷", "")
                          else:
                              df["Midas_Cross_IB_High"] = ""

                        
                    
                          if "IB_Low" in df.columns and "MIDAS_Bear" in df.columns:
                              bear_ib_cross = (df["MIDAS_Bear"].shift(1) > df["IB_Low"].shift(1)) & (df["MIDAS_Bear"] <= df["IB_Low"])
                              df["Midas_Bear_Cross_IB_Low"] = np.where(bear_ib_cross, "🎻", "")
                          else:
                              df["Midas_Bear_Cross_IB_Low"] = ""
                    
                      

                  # Compute option premium displacement from MIDAS anchors
                          intraday["Call_vs_Bull"] = intraday["Call_Option_Smooth"] - intraday["MIDAS_Bull"]
                          intraday["Put_vs_Bear"] = intraday["Put_Option_Smooth"] - intraday["MIDAS_Bear"]

                                                          # 🦵🏼 Leg Detection (rise of 12 after lowest point post-anchor)
                        
                                     # Option displacement
                          if "Call_Option_Smooth" in df.columns:
                              df["Call_vs_Bull"] = df["Call_Option_Smooth"] - df["MIDAS_Bull"]
                      
                              # 🦵🏼 Leg Detection (rise of 12 after lowest point post-anchor)
                              anchor_idx = df[price_col].idxmin()
                              post_anchor = df.loc[anchor_idx:].copy()
                      
                              if not post_anchor["Call_vs_Bull"].dropna().empty:
                                  low_idx = post_anchor["Call_vs_Bull"].idxmin()
                                  low_val = df.loc[low_idx, "Call_vs_Bull"]
                      
                                  df["Bull_Midas_Wake"] = ""
                                  for i in range(low_idx + 1, len(df)):
                                      if df.loc[i, "Call_vs_Bull"] >= low_val + 12:
                                          df.loc[i, "Bull_Midas_Wake"] = "🦵🏼"
                                          break  # Only mark the first wake
                              else:
                                  df["Bull_Midas_Wake"] = ""
                          else:
                              df["Call_vs_Bull"] = np.nan
                              df["Bull_Midas_Wake"] = ""
                      
                                                    
                                # 💥 Bear MIDAS Wake-Up Detection (Put vs MIDAS Bear)

                          # Step 1: Find the first index with a valid Put_vs_Bear after anchor
                          put_series = intraday["Put_vs_Bear"].copy()
                          start_idx = anchor_idx_bear  # From the Bear MIDAS anchor point
                          
                          min_val = float('inf')
                          wake_flags = []
                          wake_triggered = False
                          first_bear_midas_idx = None
                          
                          for i in range(start_idx, len(put_series)):
                              current = put_series.iloc[i]
                              
                              if pd.isna(current):
                                  wake_flags.append("")
                                  continue
                              
                              min_val = min(min_val, current)
                              
                              if not wake_triggered and current - min_val >= 12:
                                  wake_triggered = True
                                  first_bear_midas_idx = put_series.index[i]
                                  wake_flags.append("💥")
                              elif wake_triggered and current - min_val >= 12:
                                  wake_flags.append("💥")
                              else:
                                  wake_flags.append("")
                          
                          # Fill the column in the dataframe
                          intraday["Bear_Midas_Wake"] = [""] * start_idx + wake_flags
                                 # Detect first Bull MIDAS Wake-Up (🦵🏼)
                   

                          return df, bull_cross, bear_cross
                  
                      intraday, bull_cross, bear_cross = add_mike_midas_cross_emojis(intraday, price_col=price_col)
                      bull_wake_matches = intraday.index[intraday["Bull_Midas_Wake"] == "🦵🏼"]
                      first_bull_midas_idx = bull_wake_matches.min()  # 🔥 this will fail if the series is empty

                      
                      # Detect first Bear MIDAS Wake-Up (💥)
                      bear_wake_matches = intraday.index[intraday["Bear_Midas_Wake"] == "💥"]
                      first_bear_midas_idx = bear_wake_matches.min() if not bear_wake_matches.empty else None

                
 
                # # Display anchor info
                # st.write(f"🐻 **Bearish Anchor:** {anchor_time_bear.strftime('%I:%M %p')} — Price: {round(anchor_price_bear, 2)}")
                # st.write(f"🐂 **Bullish Anchor:** {anchor_time_bull.strftime('%I:%M %p')} — Price: {round(anchor_price_bull, 2)}")
  
  
                  # Ensure 👃🏽 and 🦻🏼 columns exist
                if "👃🏽" not in intraday.columns:
                    intraday["👃🏽"] = ""
                
                if "🦻🏼" not in intraday.columns:
                    intraday["🦻🏼"] = ""
                
                  
                # Initialize the new column for the 🎯 entry signal
                intraday["Put_FirstEntry_Emoji"] = ""
                
                # Step 1: Find the index of the MIDAS Bear anchor (first non-null value)
                anchor_idx_bear = intraday["MIDAS_Bear"].first_valid_index()
                
                # Defensive check
                if anchor_idx_bear is not None:
                    # Step 2: Start scanning from the anchor forward
                    drizzle_found = False
                    for i in range(intraday.index.get_loc(anchor_idx_bear), len(intraday)):
                        if intraday.iloc[i]["Drizzle_Emoji"] == "🌧️":
                            # Step 3: Mark the first drizzle bar after anchor
                            intraday.at[intraday.index[i], "Put_FirstEntry_Emoji"] = "🎯"
                            drizzle_found = True
                            break  # Only mark the first one
                
                intraday["Call_FirstEntry_Emoji"] = ""

                # Get Bull MIDAS anchor index
                anchor_idx_bull = intraday["MIDAS_Bull"].first_valid_index()
                
                if anchor_idx_bull is not None:
                    for i in range(intraday.index.get_loc(anchor_idx_bull), len(intraday)):
                        if intraday.iloc[i]["Heaven_Cloud"] == "☁️":
                            intraday.at[intraday.index[i], "Call_FirstEntry_Emoji"] = "🎯"
                            break


              
                            
                                # Initialize column
                intraday["Put_SecondEntry_Emoji"] = ""
                
                # Step 1: Find index of first 🎯
                first_entry_idx = intraday.index[intraday["Put_FirstEntry_Emoji"] == "🎯"]
                
                if not first_entry_idx.empty:
                    start_idx = first_entry_idx[0]  # Get first 🎯
                    i_start = intraday.index.get_loc(start_idx)
                
                    # Step 2: Loop from that index forward
                    for i in range(i_start + 1, len(intraday) - 1):  # Leave space for lookahead
                        prev_f = intraday["F_numeric"].iloc[i - 1]
                        curr_f = intraday["F_numeric"].iloc[i]
                        prev_kijun = intraday["Kijun_F"].iloc[i - 1]
                        curr_kijun = intraday["Kijun_F"].iloc[i]
                
                        # Crossed down KijunF
                        if pd.notna(prev_f) and pd.notna(prev_kijun) and pd.notna(curr_f) and pd.notna(curr_kijun):
                            if prev_f > prev_kijun and curr_f <= curr_kijun:
                                next_close = intraday["F_numeric"].iloc[i + 1]
                                if next_close < curr_f:
                                    intraday.at[intraday.index[i + 1], "Put_SecondEntry_Emoji"] = "🎯2"
                                    break  # Only mark first valid second entry
                
                intraday["Call_SecondEntry_Emoji"] = ""

                # Find 🎯 call entry
                first_call_idx = intraday.index[intraday["Call_FirstEntry_Emoji"] == "🎯"]
                
                if not first_call_idx.empty:
                    i_start = intraday.index.get_loc(first_call_idx[0])
                
                    for i in range(i_start + 1, len(intraday) - 1):
                        prev_f = intraday["F_numeric"].iloc[i - 1]
                        curr_f = intraday["F_numeric"].iloc[i]
                        prev_kijun = intraday["Kijun_F"].iloc[i - 1]
                        curr_kijun = intraday["Kijun_F"].iloc[i]
                
                        if prev_f < prev_kijun and curr_f >= curr_kijun:
                            next_close = intraday["F_numeric"].iloc[i + 1]
                            if next_close > curr_f:
                                intraday.at[intraday.index[i + 1], "Call_SecondEntry_Emoji"] = "🎯2"
                                break




             
                # Ensure F_numeric is numeric
                intraday["F_numeric"] = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                
                # Create container lists
                nose_aid_times_2 = []
                nose_aid_prices_2 = []
                ear_aid_times_2 = []
                ear_aid_prices_2 = []
                
                # Combine both second entries
                second_entry_mask = (
                    (intraday["Put_SecondEntry_Emoji"] == "🎯2") |
                    (intraday["Call_SecondEntry_Emoji"] == "🎯2")
                )
                
                for i in range(len(intraday)):
                    if second_entry_mask.iloc[i]:
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        aid_range = intraday.iloc[lower:upper]
                
                        # 👃🏽 Nose aid check
                        if (aid_range["👃🏽"] == "👃🏽").any():
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                nose_aid_times_2.append(intraday["Time"].iloc[i])
                                nose_aid_prices_2.append(f_val + 100)
                
                        # 🦻🏼 Ear aid check
                        if (aid_range["🦻🏼"] == "🦻🏼").any():
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                ear_aid_times_2.append(intraday["Time"].iloc[i])
                                ear_aid_prices_2.append(f_val - 100)
                



                 # Initialize third entry column
                intraday["Put_ThirdEntry_Emoji"] = ""
                
                # Step 1: Find 🎯 and 🎯2
                first_entry_idx = intraday.index[intraday["Put_FirstEntry_Emoji"] == "🎯"]
                second_entry_idx = intraday.index[intraday["Put_SecondEntry_Emoji"] == "🎯2"]
                
                if not first_entry_idx.empty and not second_entry_idx.empty:
                    first_i = intraday.index.get_loc(first_entry_idx[0])
                    second_i = intraday.index.get_loc(second_entry_idx[0])
                
                    # Step 2: Check if price has crossed below IB_Low by second entry
                    ib_low_crossed_by_second = False
                    for i in range(first_i, second_i + 1):
                        f = intraday["F_numeric"].iloc[i]
                        ib_low = intraday["IB_Low"].iloc[i]
                        if pd.notna(f) and pd.notna(ib_low) and f < ib_low:
                            ib_low_crossed_by_second = True
                            break
                
                    # Step 3: If not yet crossed, search forward for first cross below IB_Low
                    if not ib_low_crossed_by_second:
                        for i in range(second_i + 1, len(intraday) - 1):  # Leave space for lookahead
                            f_prev = intraday["F_numeric"].iloc[i - 1]
                            f_curr = intraday["F_numeric"].iloc[i]
                            ib_low_prev = intraday["IB_Low"].iloc[i - 1]
                            ib_low_curr = intraday["IB_Low"].iloc[i]
                
                            if pd.notna(f_prev) and pd.notna(f_curr) and pd.notna(ib_low_prev) and pd.notna(ib_low_curr):
                                if f_prev > ib_low_prev and f_curr <= ib_low_curr:
                                    # Crossed below IB_Low
                                    f_next = intraday["F_numeric"].iloc[i + 1]
                                    if pd.notna(f_next) and f_next < f_curr:
                                        intraday.at[intraday.index[i + 1], "Put_ThirdEntry_Emoji"] = "🎯3"
                                        break  # Only mark one
                intraday["Call_ThirdEntry_Emoji"] = ""
  
                first_call_idx = intraday.index[intraday["Call_FirstEntry_Emoji"] == "🎯"]
                second_call_idx = intraday.index[intraday["Call_SecondEntry_Emoji"] == "🎯2"]
                
                if not first_call_idx.empty and not second_call_idx.empty:
                    i_first = intraday.index.get_loc(first_call_idx[0])
                    i_second = intraday.index.get_loc(second_call_idx[0])
                
                    # Check if IB_High was already crossed
                    crossed_by_second = False
                    for i in range(i_first, i_second + 1):
                        f = intraday["F_numeric"].iloc[i]
                        ib_high = intraday["IB_High"].iloc[i]
                        if pd.notna(f) and pd.notna(ib_high) and f > ib_high:
                            crossed_by_second = True
                            break
                
                    if not crossed_by_second:
                        for i in range(i_second + 1, len(intraday) - 1):
                            f_prev = intraday["F_numeric"].iloc[i - 1]
                            f_curr = intraday["F_numeric"].iloc[i]
                            ib_high_prev = intraday["IB_High"].iloc[i - 1]
                            ib_high_curr = intraday["IB_High"].iloc[i]
                
                            if pd.notna(f_prev) and pd.notna(f_curr) and pd.notna(ib_high_prev) and pd.notna(ib_high_curr):
                                if f_prev < ib_high_prev and f_curr >= ib_high_curr:
                                    f_next = intraday["F_numeric"].iloc[i + 1]
                                    if pd.notna(f_next) and f_next > f_curr:
                                        intraday.at[intraday.index[i + 1], "Call_ThirdEntry_Emoji"] = "🎯3"
                                        break
                intraday["F_numeric"] = pd.to_numeric(intraday["F_numeric"], errors="coerce")

                # Collect indexes of entries (put or call) where 🐝 exists nearby
                            # Ensure numeric F_numeric
                intraday["F_numeric"] = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                
                bee_aid_times = []
                bee_aid_prices = []
                
                for i in range(len(intraday)):
                    if intraday["Put_FirstEntry_Emoji"].iloc[i] == "🎯" or intraday["Call_FirstEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        if (intraday["BBW_Tight_Emoji"].iloc[lower:upper] == "🐝").any():
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                bee_aid_times.append(intraday["Time"].iloc[i])
                                bee_aid_prices.append(f_val + 200)
                bee_aid_times_2 = []
                bee_aid_prices_2 = []
                
                for i in range(len(intraday)):
                    if intraday["Put_SecondEntry_Emoji"].iloc[i] == "🎯2" or intraday["Call_SecondEntry_Emoji"].iloc[i] == "🎯2":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        if (intraday["BBW_Tight_Emoji"].iloc[lower:upper] == "🐝").any():
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                bee_aid_times_2.append(intraday["Time"].iloc[i])
                                bee_aid_prices_2.append(f_val - 200)

                
                compliance_aid_times = []
                compliance_aid_prices = []
                
                for i in range(len(intraday)):
                    if intraday["Put_FirstEntry_Emoji"].iloc[i] == "🎯" or intraday["Call_FirstEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        if (intraday["Compliance Shift"].iloc[lower:upper] == "🫧").any():
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                compliance_aid_times.append(intraday["Time"].iloc[i])
                                compliance_aid_prices.append(f_val + 300)

        # Ensure F_numeric is numeric
                intraday["F_numeric"] = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                
                # Prepare profile bins for lookup
                bin_to_ear = set(profile_df.loc[profile_df["🦻🏼"] == "🦻🏼", "F% Level"])
                bin_to_nose = set(profile_df.loc[profile_df["👃🏽"] == "👃🏽", "F% Level"])
                
                # Compute bin for each row
                f_bins = np.arange(-400, 401, 20)
                intraday["F_Bin"] = pd.cut(intraday["F_numeric"], bins=f_bins, labels=f_bins[:-1])
                
                # Aid storage
                ear_aid_times = []
                ear_aid_prices = []
                
                nose_aid_times = []
                nose_aid_prices = []
                
                # Loop through each row with 🎯 entry
                for i in range(len(intraday)):
                    if intraday["Put_FirstEntry_Emoji"].iloc[i] == "🎯" or intraday["Call_FirstEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        sub = intraday.iloc[lower:upper]
                
                        if any(sub["F_Bin"].isin(bin_to_ear)):
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                ear_aid_times.append(intraday["Time"].iloc[i])
                                ear_aid_prices.append(f_val + 260)
                
                        if any(sub["F_Bin"].isin(bin_to_nose)):
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                nose_aid_times.append(intraday["Time"].iloc[i])
                                nose_aid_prices.append(f_val + 260)
  
                ember_aid_times = []
                ember_aid_prices = []
                
                # Loop through each bar to find 🎯 call entries
                for i in range(len(intraday)):
                    if intraday["Call_FirstEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))  # ±5 bars
                
                        phoenix_present = (intraday["STD_Alert"].iloc[lower:upper] == "🐦‍🔥").any()
                        fire_present = (intraday["BBW Alert"].iloc[lower:upper] == "🔥").any()
                
                        if phoenix_present and fire_present:
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                ember_aid_times.append(intraday["Time"].iloc[i])
                                ember_aid_prices.append(f_val + 288)  # Position above 🎯
                # Ensure F_numeric is numeric
                intraday["F_numeric"] = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                
                # Initialize aid lists
                comet_aid_times = []
                comet_aid_prices = []
                
                # Search around Entry 2 (Put or Call)
                for i in range(len(intraday)):
                    if intraday["Put_SecondEntry_Emoji"].iloc[i] == "🎯2" or intraday["Call_SecondEntry_Emoji"].iloc[i] == "🎯2":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        
                        if (intraday["ATR_Exp_Alert"].iloc[lower:upper] == "☄️").any():
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                comet_aid_times.append(intraday["Time"].iloc[i])
                                comet_aid_prices.append(f_val + 180)  # Offset for clean stacking
                # Step 1: Collect aid points with Volatility Composite values
                vol_aid_times = []
                vol_aid_prices = []
                vol_aid_values = []
                
                for i in range(len(intraday)):
                    if intraday["Call_FirstEntry_Emoji"].iloc[i] == "🎯" or intraday["Put_FirstEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        vol_window = intraday["Volatility_Composite"].iloc[lower:upper]
                        if (vol_window > 10).any():
                            f_val = intraday["F_numeric"].iloc[i]
                            if pd.notnull(f_val):
                                vol_aid_times.append(intraday["Time"].iloc[i])
                                vol_aid_prices.append(f_val + 20)
                                vol_aid_values.append(vol_window.max())  # highest value in the window
                
                # Plot it
                plt.scatter(vol_aid_times, vol_aid_prices, marker="o", color="black", label="💨 Volatility Spike")
                for t, p in zip(vol_aid_times, vol_aid_prices):
                    plt.text(t, p, "💨", fontsize=10, ha="center", va="bottom")


                force_aid_times = []
                force_aid_prices = []
                force_aid_vals = []
                
                # Ensure Force column is numeric
                intraday["Unit Force"] = pd.to_numeric(intraday["Unit Force"], errors="coerce")
                
                # Loop through the dataset
                for i in range(len(intraday)):
                    # Check for a 🎯 entry
                    if intraday["Call_FirstEntry_Emoji"].iloc[i] == "🎯":
                        # Look for **positive Force** peak in a ±5 bar window
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        force_window = intraday["Unit Force"].iloc[lower:upper]
                
                        # Only keep positive values (aiding bullish)
                        force_window = force_window[force_window > 0]
                
                        if not force_window.empty:
                            peak_idx = force_window.idxmax()
                            peak_time = intraday["Time"].loc[peak_idx]
                            peak_val = intraday["F_numeric"].loc[peak_idx] + 360  # offset above Momentum
                            force_val = force_window.loc[peak_idx]
                
                            force_aid_times.append(peak_time)
                            force_aid_prices.append(peak_val)
                            force_aid_vals.append(int(force_val))
                
                    elif intraday["Put_FirstEntry_Emoji"].iloc[i] == "🎯":
                        # Look for **negative Force** peak in a ±5 bar window
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        force_window = intraday["Unit Force"].iloc[lower:upper]
                
                        # Only keep negative values (aiding bearish)
                        force_window = force_window[force_window < 0]
                
                        if not force_window.empty:
                            trough_idx = force_window.idxmin()  # most negative = strongest
                            trough_time = intraday["Time"].loc[trough_idx]
                            trough_val = intraday["F_numeric"].loc[trough_idx] + 360
                            force_val = force_window.loc[trough_idx]
                
                            force_aid_times.append(trough_time)
                            force_aid_prices.append(trough_val)
                            force_aid_vals.append(int(force_val))
                cross_aid_times_call = []
                cross_aid_prices_call = []
                
                call_entry_idxs = intraday.index[intraday["Call_FirstEntry_Emoji"] == "🎯"]
                
                for idx in call_entry_idxs:
                    i = intraday.index.get_loc(idx)
                    lower = max(i - 3, 0)
                    upper = min(i + 4, len(intraday))
                
                    sub = intraday.iloc[lower:upper]
                    if (sub["PE_Cross_Bull"] == True).any():
                        cross_aid_times_call.append(intraday.at[idx, "Time"])
                        cross_aid_prices_call.append(intraday.at[idx, "F_numeric"] + 90)


                cross_aid_times_put = []
                cross_aid_prices_put = []
                
                put_entry_idxs = intraday.index[intraday["Put_FirstEntry_Emoji"] == "🎯"]
                
                for idx in put_entry_idxs:
                    i = intraday.index.get_loc(idx)
                    lower = max(i - 3, 0)
                    upper = min(i + 4, len(intraday))
                
                    sub = intraday.iloc[lower:upper]
                    if (sub["PE_Cross_Bear"] == True).any():
                        cross_aid_times_put.append(intraday.at[idx, "Time"])
                        cross_aid_prices_put.append(intraday.at[idx, "F_numeric"] - 90)

                momentum_aid_times = []
                momentum_aid_prices = []
                
                # Ensure Unit Momentum is numeric
                intraday["Unit Momentum"] = pd.to_numeric(intraday["Unit Momentum"], errors="coerce")
                
                # Iterate through the dataframe
                for i in range(len(intraday)):
                    if intraday["Call_FirstEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        window = intraday.iloc[lower:upper]
                
                        # Filter out NaNs
                        valid_window = window["Unit Momentum"].dropna()
                        if not valid_window.empty:
                            peak_idx = valid_window.idxmax()
                            peak_time = intraday["Time"].loc[peak_idx]
                            peak_value = intraday["F_numeric"].loc[peak_idx] + 300
                            peak_momentum = valid_window.loc[peak_idx]
                
                            momentum_aid_times.append(peak_time)
                            momentum_aid_prices.append(peak_value)
                            intraday.loc[peak_idx, "Momentum_Aid_Value"] = peak_momentum
                
                    elif intraday["Put_FirstEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        window = intraday.iloc[lower:upper]
                
                        # Filter out NaNs
                        valid_window = window["Unit Momentum"].dropna()
                        if not valid_window.empty:
                            trough_idx = valid_window.idxmin()
                            trough_time = intraday["Time"].loc[trough_idx]
                            trough_value = intraday["F_numeric"].loc[trough_idx] + 300
                            trough_momentum = valid_window.loc[trough_idx]
                
                            momentum_aid_times.append(trough_time)
                            momentum_aid_prices.append(trough_value)
                            intraday.loc[trough_idx, "Momentum_Aid_Value"] = trough_momentum


                
                vol_aid_times_call = []
                vol_aid_prices_call = []
                
                vol_aid_times_put = []
                vol_aid_prices_put = []

                  # Initialize containers
                energy_aid_times = []
                energy_aid_prices = []
                energy_aid_vals = []


                # # -- Cross-Check Independent Enhancers: Ear and Nose Crosses --
                # intraday["🦻🏼_Cross"] = ""
                # intraday["👃🏽_Cross"] = ""
                
                # # Use previously computed values from profile_df
                ear_line = max_vol_level
                nose_line = max_letter_level
                
                # # Loop from second row onward (to compare with previous)
                # for i in range(1, len(intraday)):
                #     current = intraday.iloc[i]
                #     previous = intraday.iloc[i - 1]
                
                #     # Ear Cross (🦻🏼): if price crosses above or below the volume-dominant level
                #     crossed_ear = (
                #         (previous["F_numeric"] < ear_line and current["F_numeric"] >= ear_line) or
                #         (previous["F_numeric"] > ear_line and current["F_numeric"] <= ear_line)
                #     )
                
                #     # Nose Cross (👃🏽): if price crosses above or below the time-dominant level
                #     crossed_nose = (
                #         (previous["F_numeric"] < nose_line and current["F_numeric"] >= nose_line) or
                #         (previous["F_numeric"] > nose_line and current["F_numeric"] <= nose_line)
                #     )
                
                #     # Set emojis in row `i` (not previous one)
                #     if crossed_ear:
                #         intraday.at[i, "🦻🏼_Cross"] = "🦻🏼"
                
                #     if crossed_nose:
                #         intraday.at[i, "👃🏽_Cross"] = "👃🏽"

                
                intraday["👃🏽_y"] = np.nan
                intraday["🦻🏼_y"] = np.nan
                
                for i in range(1, len(intraday)):
                    current = intraday.iloc[i]
                    previous = intraday.iloc[i - 1]
                
                    # Ear Cross (🦻🏼)
                    if previous["F_numeric"] < ear_line and current["F_numeric"] >= ear_line:
                        intraday.at[i, "🦻🏼_Cross"] = "🦻🏼"
                        intraday.at[i, "🦻🏼_y"] = current["F_numeric"] + 30  # above
                    elif previous["F_numeric"] > ear_line and current["F_numeric"] <= ear_line:
                        intraday.at[i, "🦻🏼_Cross"] = "🦻🏼"
                        intraday.at[i, "🦻🏼_y"] = current["F_numeric"] - 30  # below
                
                    # Nose Cross (👃🏽)
                    if previous["F_numeric"] < nose_line and current["F_numeric"] >= nose_line:
                        intraday.at[i, "👃🏽_Cross"] = "👃🏽"
                        intraday.at[i, "👃🏽_y"] = current["F_numeric"] + 30  # above
                    elif previous["F_numeric"] > nose_line and current["F_numeric"] <= nose_line:
                        intraday.at[i, "👃🏽_Cross"] = "👃🏽"
                        intraday.at[i, "👃🏽_y"] = current["F_numeric"] - 30  # below
                


                # --- CALL Entries ---
                call_entry_idxs = intraday.index[intraday["Call_FirstEntry_Emoji"] == "🎯"]
                
                for idx in call_entry_idxs:
                    i = intraday.index.get_loc(idx)
                    lower = max(i - 3, 0)
                    upper = min(i + 4, len(intraday))  # +4 because Python excludes upper
                
                    sub = intraday.iloc[lower:upper]
                
                    if (
                        (sub["BBW Alert"] == "🔥").any() or
                        (sub["BBW_Tight"] == "🐝").any() or

                        (sub["STD_Alert"] == "🐦‍🔥").any() or
                        (sub["ATR_Exp_Alert"] == "☄️").any() or
                        (sub["RVOL_5"] > 1.2).any()
                    ):
                        vol_aid_times_call.append(intraday.at[idx, "Time"])
                        vol_aid_prices_call.append(intraday.at[idx, "F_numeric"] + 120)
                
                
                # --- PUT Entries ---
                put_entry_idxs = intraday.index[intraday["Put_FirstEntry_Emoji"] == "🎯"]
                
                for idx in put_entry_idxs:
                    i = intraday.index.get_loc(idx)
                    lower = max(i - 3, 0)
                    upper = min(i + 4, len(intraday))
                
                    sub = intraday.iloc[lower:upper]
                
                    if (
                        (sub["BBW Alert"] == "🔥").any() or
                        (sub["BBW_Tight"] == "🐝").any() or

                        (sub["STD_Alert"] == "🐦‍🔥").any() or
                        (sub["ATR_Exp_Alert"] == "☄️").any() or
                        (sub["RVOL_5"] > 1.2).any()
                    ):
                        vol_aid_times_put.append(intraday.at[idx, "Time"])
                        vol_aid_prices_put.append(intraday.at[idx, "F_numeric"] - 120)

                # Ensure Vector Energy is numeric
                intraday["Vector Energy"] = pd.to_numeric(intraday["Vector Energy"], errors="coerce")
                
                for i in range(len(intraday)):
                    if intraday["Call_SecondEntry_Emoji"].iloc[i] == "🎯" or intraday["Put_SecondEntry_Emoji"].iloc[i] == "🎯":
                        lower = max(i - 5, 0)
                        upper = min(i + 6, len(intraday))
                        energy_window = intraday["Vector Energy"].iloc[lower:upper]
                
                        if energy_window.notna().any():
                            if intraday["Call_SecondEntry_Emoji"].iloc[i] == "🎯":
                                peak_idx = energy_window.idxmax()  # Most bullish
                            else:
                                peak_idx = energy_window.idxmin()  # Most bearish
                
                            peak_time = intraday["Time"].loc[peak_idx]
                            peak_value = intraday["F_numeric"].loc[peak_idx] + 100  # Offset for visibility
                            energy_val = energy_window.loc[peak_idx]
                
                            energy_aid_times.append(peak_time)
                            energy_aid_prices.append(peak_value)
                            energy_aid_vals.append(int(energy_val))
                
                            # Optional for hover
                            intraday.loc[peak_idx, "Energy_Aid_Value"] = energy_val




                 
                def calculate_compliance_midas(df, bbw_col="F% BBW", vol_col="RVOL_5"):
                    """
                    Calculate Compliance relative to MIDAS anchors.
                    - Bull: anchor at min
                    - Bear: anchor at max
                    Compliance = (BBW - BBW_at_anchor) / RVOL_5
                    """
                
                    # Initialize columns
                    df["Compliance_Bull"] = np.nan
                    df["Compliance_Bear"] = np.nan
                
                    # --- 🐂 Bull Midas ---
                    if "MIDAS_Bull" in df.columns:
                        # Anchor index
                        bull_anchor_idx = df["MIDAS_Bull"].first_valid_index()
                        if bull_anchor_idx is not None:
                            bbw_anchor = df.loc[bull_anchor_idx, bbw_col]
                            # Compliance from anchor forward
                            df.loc[bull_anchor_idx:, "Compliance_Bull"] = (
                                (df.loc[bull_anchor_idx:, bbw_col] - bbw_anchor) /
                                df.loc[bull_anchor_idx:, vol_col].replace(0, np.nan)
                            )
                
                    # --- 🐻 Bear Midas ---
                    if "MIDAS_Bear" in df.columns:
                        bear_anchor_idx = df["MIDAS_Bear"].first_valid_index()
                        if bear_anchor_idx is not None:
                            bbw_anchor = df.loc[bear_anchor_idx, bbw_col]
                            df.loc[bear_anchor_idx:, "Compliance_Bear"] = (
                                (df.loc[bear_anchor_idx:, bbw_col] - bbw_anchor) /
                                df.loc[bear_anchor_idx:, vol_col].replace(0, np.nan)
                            )
                
                    return df
                
                # Apply it
                intraday = calculate_compliance_midas(intraday)
 
                def mark_compliance_bull_flip(df):
                    """
                    Marks when Compliance Bull turns negative 
                    *after* a Bear MIDAS anchor is active.
                    """
                    df["Compliance_Bull_Flip"] = ""
                
                    if "Compliance_Bull" in df.columns and "MIDAS_Bear" in df.columns:
                        bear_anchor_idx = df["MIDAS_Bear"].first_valid_index()
                        if bear_anchor_idx is not None:
                            for i in range(bear_anchor_idx + 1, len(df)):
                                prev = df["Compliance_Bull"].iloc[i - 1]
                                curr = df["Compliance_Bull"].iloc[i]
                                if pd.notna(prev) and pd.notna(curr):
                                    if prev >= 0 and curr < 0:
                                        df.at[df.index[i], "Compliance_Bull_Flip"] = "💨"  # Aura collapse
                    return df
                
                # Apply
                intraday = mark_compliance_bull_flip(intraday)

                intraday["Headphone_Cross_Emoji"] = ""
                intraday["Headphone_Cross_Y"] = np.nan
                intraday["Ear_Level"] = max_vol_level  # 🦻🏼 Ear Line only
                
                for i in range(1, len(intraday)):
                    prev_f = intraday["F_numeric"].iloc[i - 1]
                    curr_f = intraday["F_numeric"].iloc[i]
                    ear = intraday["Ear_Level"].iloc[i]
                
                    # Crossed Up: below → above Ear line
                    if prev_f < ear and curr_f >= ear:
                        intraday.at[intraday.index[i], "Headphone_Cross_Emoji"] = "🎧"
                        intraday.at[intraday.index[i], "Headphone_Cross_Y"] = curr_f + 43  # Above Mike
                
                    # Crossed Down: above → below Ear line
                    elif prev_f > ear and curr_f <= ear:
                        intraday.at[intraday.index[i], "Headphone_Cross_Emoji"] = "🎧"
                        intraday.at[intraday.index[i], "Headphone_Cross_Y"] = curr_f - 43  # Below Mike

                
                            # 🐘 NOSE LINE CROSS (👃🏽)
                intraday["Elephant_Cross_Emoji"] = ""
                intraday["Elephant_Cross_Y"] = np.nan
                intraday["Nose_Level"] = max_letter_level
                
                for i in range(1, len(intraday)):
                    prev_f = intraday["F_numeric"].iloc[i - 1]
                    curr_f = intraday["F_numeric"].iloc[i]
                    nose = intraday["Nose_Level"].iloc[i]
                
                    # Crossed Up: below → above Nose line
                    if prev_f < nose and curr_f >= nose:
                        intraday.at[intraday.index[i], "Elephant_Cross_Emoji"] = "🐘"
                        intraday.at[intraday.index[i], "Elephant_Cross_Y"] = curr_f + 33  # Above Mike
                
                    # Crossed Down: above → below Nose line
                    elif prev_f > nose and curr_f <= nose:
                        intraday.at[intraday.index[i], "Elephant_Cross_Emoji"] = "🐘"
                        intraday.at[intraday.index[i], "Elephant_Cross_Y"] = curr_f - 33  # Below Mike


               
                             # Ensure required columns exist
               

                def mark_compliance_bear_flip(df):
                    """
                    Marks when Compliance Bear turns negative 
                    *after* a Bull MIDAS anchor is active.
                    """
                    df["Compliance_Bear_Flip"] = ""
                
                    if "Compliance_Bear" in df.columns and "MIDAS_Bull" in df.columns:
                        bull_anchor_idx = df["MIDAS_Bull"].first_valid_index()
                        if bull_anchor_idx is not None:
                            for i in range(bull_anchor_idx + 1, len(df)):
                                prev = df["Compliance_Bear"].iloc[i - 1]
                                curr = df["Compliance_Bear"].iloc[i]
                                if pd.notna(prev) and pd.notna(curr):
                                    if prev >= 0 and curr < 0:
                                        df.at[df.index[i], "Compliance_Bear_Flip"] = "🌑"  # Bear aura collapse
                    return df
                
                # Apply
                intraday = mark_compliance_bear_flip(intraday)
                def calculate_side_band_distance(df):
                    """
                    Adds Side_Dist_F (distance in F% units) from F_numeric to the
                    Bollinger band on the Kijun side:
                      - If F_numeric > Kijun_F → distance to Upper Band
                      - If F_numeric < Kijun_F → distance to Lower Band
                    Positive = inside band, 0 = at band, Negative = outside band
                    """
                    df = df.copy()
                
                    # Condition: bull (north kingdom) vs bear (south kingdom)
                    df["Side_Label"] = np.where(df["F_numeric"] > df["Kijun_F"], "upper", "lower")
                
                    # Distance in F% units
                    df["Side_Dist_F"] = np.where(
                        df["Side_Label"] == "upper",
                        df["F% Upper"] - df["F_numeric"],  # bull → distance to upper
                        df["F_numeric"] - df["F% Lower"],  # bear → distance to lower
                    )
                
                    return df
                
                # Apply
                intraday = calculate_side_band_distance(intraday)
                def add_vault_emoji(df):
                    """
                    Detects vault burst events between Bollinger bands:
                    🕊️ = bullish slingshot (lower → upper flip)
                    🐦‍⬛ = bearish slingshot (upper → lower flip)
                    """
                    df["Vault_Emoji"] = ""
                
                    if not all(col in df.columns for col in ["Side_Label", "Side_Dist_F"]):
                        return df
                
                    for i in range(1, len(df)):
                        prev_label = df.at[df.index[i - 1], "Side_Label"]
                        prev_dist  = df.at[df.index[i - 1], "Side_Dist_F"]
                        curr_label = df.at[df.index[i], "Side_Label"]
                        curr_dist  = df.at[df.index[i], "Side_Dist_F"]
                
                        # 🕊️ bullish burst: jumps from hugging lower → overshooting upper
                        if prev_label == "lower" and prev_dist > 0 and curr_label == "upper" and curr_dist < 0:
                            df.at[df.index[i], "Vault_Emoji"] = "🕊️"
                
                        # 🐦‍⬛ bearish burst: jumps from hugging upper → overshooting lower
                        elif prev_label == "upper" and prev_dist > 0 and curr_label == "lower" and curr_dist < 0:
                            df.at[df.index[i], "Vault_Emoji"] = "🐦‍⬛"
                
                    return df
                
                # Apply before plotting
                intraday = add_vault_emoji(intraday)
                def add_marengo_T0(intraday, tol=5):
                    """
                    Detects T0 (🚪) after Entry 1.
                    Shows only one 🚪 per Call/Put First Entry 🎯.
                    Uses Side_Dist_F as distance measure.
                    """
                    out = intraday.copy()
                
                    # Make sure required columns exist
                    need_cols = ["Side_Dist_F", "Call_FirstEntry_Emoji", "Put_FirstEntry_Emoji"]
                    for col in need_cols:
                        if col not in out.columns:
                            out["T0_Emoji"] = ""
                            return out
                
                    # Init column
                    out["T0_Emoji"] = ""
                
                    # Find all Entry 1 events (both Call & Put)
                    entry_idx = list(out.index[out["Call_FirstEntry_Emoji"] == "🎯"]) + \
                                list(out.index[out["Put_FirstEntry_Emoji"]  == "🎯"])
                    entry_idx = sorted(entry_idx)
                
                    # Loop through each Entry 1
                    for start in entry_idx:
                        found = False
                        for i in range(start + 1, len(out)):
                            dist = out.at[out.index[i], "Side_Dist_F"]
                            if pd.notna(dist) and dist <= tol:
                                out.at[out.index[i], "T0_Emoji"] = "🚪"
                                found = True
                                break
                        if found:
                            continue  # move on to next Entry 1
                
                    return out
                
                # Apply
                intraday = add_marengo_T0(intraday, tol=5)
                def add_marengo_T1(intraday, tol=-5):
                    """
                    Marks T1: first acceleration beyond Bollinger band after Entry 1.
                    tol = threshold in F% units (default = -5).
                    """
                    out = intraday.copy()
                
                    # Make sure Entry 1 exists
                    call_idx = out.index[out["Call_FirstEntry_Emoji"] == "🎯"]
                    put_idx  = out.index[out["Put_FirstEntry_Emoji"]  == "🎯"]
                
                    if len(call_idx) == 0 and len(put_idx) == 0:
                        out["T1_Emoji"] = ""
                        return out
                
                    # Get first entry index
                    first_call_i = out.index.get_loc(call_idx[0]) if len(call_idx) > 0 else None
                    first_put_i  = out.index.get_loc(put_idx[0])  if len(put_idx)  > 0 else None
                    start_i = min(i for i in [first_call_i, first_put_i] if i is not None)
                
                    # Init column
                    out["T1_Emoji"] = ""
                
                    # Loop forward after entry
                    for i in range(start_i + 1, len(out)):
                        if "Side_Dist_F" not in out.columns or pd.isna(out.at[out.index[i], "Side_Dist_F"]):
                            continue
                
                        if out.at[out.index[i], "Side_Dist_F"] <= tol:
                            out.at[out.index[i], "T1_Emoji"] = "🏇🏼"
                            break  # only first acceleration
                    return out
                
                # Apply
                intraday = add_marengo_T1(intraday)


   
   
                def add_marengo_T2(intraday):
                    """
                    Marks T2: momentum confirmation after T1.
                    ⚡ appears on the first bar after T1 where Close extends in the 
                    same direction as Entry 1 (up for calls, down for puts).
                    """
                    out = intraday.copy()
                
                    # Ensure dependencies exist
                    need_cols = ["Close", "T1_Emoji", "Call_FirstEntry_Emoji", "Put_FirstEntry_Emoji"]
                    for col in need_cols:
                        if col not in out.columns:
                            out["T2_Emoji"] = ""
                            return out
                
                    # Find Entry 1
                    call_idx = out.index[out["Call_FirstEntry_Emoji"] == "🎯"]
                    put_idx  = out.index[out["Put_FirstEntry_Emoji"]  == "🎯"]
                
                    if len(call_idx) == 0 and len(put_idx) == 0:
                        out["T2_Emoji"] = ""
                        return out
                
                    # Get first entry index
                    first_call_i = out.index.get_loc(call_idx[0]) if len(call_idx) > 0 else None
                    first_put_i  = out.index.get_loc(put_idx[0])  if len(put_idx)  > 0 else None
                    start_i = min(i for i in [first_call_i, first_put_i] if i is not None)
                
                    # Find T1
                    t1_idx = out.index[out["T1_Emoji"] == "🏇🏼"]
                    if len(t1_idx) == 0:
                        out["T2_Emoji"] = ""
                        return out
                    i_t1 = out.index.get_loc(t1_idx[0])
                
                    # Init column
                    out["T2_Emoji"] = ""
                
                    # Loop after T1
                    for i in range(i_t1 + 1, len(out)):
                        prev_close = out.at[out.index[i - 1], "Close"]
                        curr_close = out.at[out.index[i], "Close"]
                
                        if pd.isna(prev_close) or pd.isna(curr_close):
                            continue
                
                        # Call case (northbound)
                        if first_call_i is not None and first_call_i <= start_i:
                            if curr_close > prev_close:
                                out.at[out.index[i], "T2_Emoji"] = "⚡"
                                break
                
                        # Put case (southbound)
                        if first_put_i is not None and first_put_i <= start_i:
                            if curr_close < prev_close:
                                out.at[out.index[i], "T2_Emoji"] = "⚡"
                                break
                
                    return out
                
                # Apply
                intraday = add_marengo_T2(intraday)
                def add_parallel_phase(intraday):
                    """
                    Marks ⚡ Parallel phase:
                    After T1, as long as price stays on the correct side of Tenkan_F.
                    - For calls: F_numeric >= Tenkan_F
                    - For puts:  F_numeric <= Tenkan_F
                    """
                    out = intraday.copy()
                    out["Parallel_Emoji"] = ""
                
                    # Find T1
                    t1_idx = out.index[out["T1_Emoji"] == "🏇🏼"]
                    if len(t1_idx) == 0:
                        return out
                
                    start_i = out.index.get_loc(t1_idx[0])
                
                    # Loop forward bar by bar
                    side = None
                    if any(out["Call_FirstEntry_Emoji"] == "🎯"):
                        side = "call"
                    elif any(out["Put_FirstEntry_Emoji"] == "🎯"):
                        side = "put"
                
                    if side is None:
                        return out
                
                    for i in range(start_i + 1, len(out)):
                        mike = out.at[out.index[i], "F_numeric"]
                        tenkan = out.at[out.index[i], "Tenkan_F"]
                
                        if pd.isna(mike) or pd.isna(tenkan):
                            continue
                
                        # Check condition
                        if side == "call" and mike >= tenkan:
                            out.at[out.index[i], "Parallel_Emoji"] = "⚡"
                        elif side == "put" and mike <= tenkan:
                            out.at[out.index[i], "Parallel_Emoji"] = "⚡"
                        else:
                            break  # exits parallel phase
                
                    return out
                
                # Apply
                intraday = add_parallel_phase(intraday)
                def assign_label_simple(row, intraday):
                    if not any(tag in row["Type"] for tag in ["🎯1", "🎯2"]):
                        return ""
                
                    f_val = row["F%"]
                
                    ib_low = intraday["IB_Low"].iloc[0]
                    ib_high = intraday["IB_High"].iloc[0]
                
                    if ib_low <= f_val <= ib_high:
                        return "Endo"
                    elif f_val > ib_high:
                        return "Supra"
                    elif f_val < ib_low:
                        return "Infra"
                
                    return ""


                
                def compute_pae_1to2(entries_df, intraday):
                    # Initialize PAE column
                    entries_df["PAE_1to2"] = ""
                
                    for idx, row in entries_df.iterrows():
                        if "🎯1" not in row["Type"]:
                            continue  # only process Entry 1
                
                        entry_time = row["Time"]
                        entry_type = row["Type"]
                        entry_f = row["F%"]
                
                        # Locate entry index in intraday
                        entry_idx = intraday.index[
                            pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                        ][0]
                
                        # Find the next Entry 2 of the same side
                        if "Call" in entry_type:
                            next_entries = entries_df[
                                (entries_df["Type"] == "Call 🎯2") &
                                (entries_df.index > idx)
                            ]
                        else:
                            next_entries = entries_df[
                                (entries_df["Type"] == "Put 🎯2") &
                                (entries_df.index > idx)
                            ]
                
                        if next_entries.empty:
                            continue  # no Entry 2 found → leave blank
                
                        # Take the first matching Entry 2
                        exit_time = next_entries.iloc[0]["Time"]
                        exit_idx = intraday.index[
                            pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == exit_time
                        ][0]
                
                        # Subset intraday from entry to exit
                        window = intraday.loc[entry_idx:exit_idx]
                
                        if "Call" in entry_type:
                            min_f = window["F_numeric"].min()
                            pae = max(0, entry_f - min_f)
                        else:  # Put
                            max_f = window["F_numeric"].max()
                            pae = max(0, max_f - entry_f)
                
                        entries_df.at[idx, "PAE_1to2"] = pae
                
                    return entries_df
                
                def compute_pae_2to3(entries_df: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
                    """
                    Compute PAE (Pain After Entry) from Entry 2 ➝ Entry 3.
                    For each Call 🎯2 or Put 🎯2, measure the worst move against the trade
                    until the next same-side Entry 3.
                    """
                    pae_values = []
                
                    for i, row in entries_df.iterrows():
                        entry_type = row["Type"]
                
                        if "🎯2" not in entry_type:
                            pae_values.append(None)
                            continue
                
                        # Find the next Entry 3 of the same side (Call or Put)
                        side = "Call" if "Call" in entry_type else "Put"
                        later_rows = entries_df.iloc[i+1:]
                        next_entry3 = later_rows[later_rows["Type"] == f"{side} 🎯3"]
                
                        if next_entry3.empty:
                            pae_values.append(None)
                            continue
                
                        # Define entry and exit times
                        entry_time = row["Time"]
                        exit_time = next_entry3.iloc[0]["Time"]
                
                        # Map times to intraday indices
                        entry_idx = intraday.index[
                            pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                        ][0]
                        exit_idx = intraday.index[
                            pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == exit_time
                        ][0]
                
                        # Clip scan range
                        scan = intraday.loc[entry_idx:exit_idx, "F_numeric"]
                
                        if scan.empty:
                            pae_values.append(None)
                            continue
                
                        entry_f = scan.iloc[0]
                
                        if side == "Call":
                            # Pain = max dip below entry
                            pain = (scan.min() - entry_f)
                        else:  # Put
                            # Pain = max rise above entry
                            pain = (scan.max() - entry_f)
                
                        # Express pain as absolute F% units (positive number = painful move)
                        pae_values.append(abs(int(round(pain, 0))))
                
                    entries_df["PAE_2to3"] = pae_values
                    return entries_df
                def compute_pae_3to40F(entries_df: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
                    """Compute PAE from Entry 3 until reaching +40F% in favor of the trade."""
                    pae_values = []
                
                    for _, row in entries_df.iterrows():
                        if "🎯3" not in row["Type"]:
                            pae_values.append("")
                            continue
                
                        entry_time = row["Time"]
                        entry_type = row["Type"]
                        entry_f = row["F%"]
                
                        # Locate the entry index
                        entry_idx = intraday.index[
                            pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                        ][0]
                        entry_loc = intraday.index.get_loc(entry_idx)
                
                        # Scan forward from entry
                        forward = intraday.iloc[entry_loc + 1 :]
                        f_series = forward["F_numeric"]
                
                        # Track pain and stop once target is reached
                        worst_pain = 0
                        target_reached = False
                
                        for f in f_series:
                            if "Call" in entry_type:
                                # Pain is dip below entry
                                if f < entry_f:
                                    worst_pain = max(worst_pain, entry_f - f)
                
                                # Target = +40F
                                if f >= entry_f + 40:
                                    target_reached = True
                                    break
                
                            else:  # Put
                                if f > entry_f:
                                    worst_pain = max(worst_pain, f - entry_f)
                
                                if f <= entry_f - 40:
                                    target_reached = True
                                    break
                
                        # Assign result
                        pae_values.append(worst_pain if target_reached else "")
                
                    entries_df["PAE_3to40F"] = pae_values
                    return entries_df

             
                def assign_prototype(row):
                    if "Call 🎯1" in row["Type"]:
                        return "Ember"
                    elif "Put 🎯1" in row["Type"]:
                        return "Cliff"
                    return ""


             

             
                def add_exit_columns(entries_df: pd.DataFrame) -> pd.DataFrame:
                         """
                         For every Entry 1 (Call 🎯1 or Put 🎯1), find the first opposite Entry 1 after it.
                         Add Exit_Time and Exit_Price to the table.
                         If no opposite entry exists later, exit stays blank.
                         """
                     
                         # Make a copy so we don’t mutate original
                         df = entries_df.copy()
                     
                         # Ensure time is datetime for ordering
                         df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M")
                     
                         # Loop through Entry 1s
                         for idx, row in df.iterrows():
                             if "🎯1" not in row["Type"]:
                                 continue  # Only work on Entry 1s
                     
                             entry_type = row["Type"]
                             entry_time = row["Time_dt"]
                     
                             # Define what opposite means
                             if "Call" in entry_type:
                                 opposite_mask = df["Type"].str.contains("Put 🎯1")
                             else:
                                 opposite_mask = df["Type"].str.contains("Call 🎯1")
                     
                             # Find the first opposite entry *after* this entry
                             opposite_rows = df[opposite_mask & (df["Time_dt"] > entry_time)]
                             if not opposite_rows.empty:
                                 first_opposite = opposite_rows.iloc[0]
                                 df.at[idx, "Exit_Time"] = first_opposite["Time"]
                                 df.at[idx, "Exit_Price"] = first_opposite["Price ($)"]
                     
                         # Clean up helper column
                         df.drop(columns=["Time_dt"], inplace=True)
                     
                         return df
        

                def anchor_vol_confirm(intraday: pd.DataFrame, lookaround: int = 7) -> str:
                    """
                    One-liner result for the session:
                      'Confirmed' if ±lookaround bars around the first MIDAS_Bull anchor
                      contain 🔥 in 'BBW Alert'   OR
                               🐦‍🔥 in 'STD_Alert'.
                      Otherwise returns ''.
                    """
                    # 1️⃣ Locate first Bull anchor
                    anchor_idx = intraday["MIDAS_Bull"].first_valid_index()
                    if anchor_idx is None:                 # no anchor this session
                        return ""
                
                    anchor_loc = intraday.index.get_loc(anchor_idx)
                
                    # 2️⃣ Window around the anchor
                    lo = max(0, anchor_loc - lookaround)
                    hi = min(len(intraday) - 1, anchor_loc + lookaround)
                    win = intraday.iloc[lo : hi + 1]
                
                    # 3️⃣ Look for either alert (exact column names you showed)
                    fire_seen   = win["BBW Alert"].astype(str).str.contains("🔥").any()   if "BBW Alert" in win.columns else False
                    phoenix_seen = win["STD_Alert"].astype(str).str.contains("🐦‍🔥").any() if "STD_Alert" in win.columns else False
                
                    return "Confirmed" if (fire_seen or phoenix_seen) else ""
           
                def assign_suffix_simple(row, intraday, perimeter=7):
                    """
                    Detect suffix for Ember/Cliff entries:
                    - If 🔥 or 🐦‍🔥 appears → VolConfirmed
                    - If NO volatility but RVOL spike > 1.2 → Stampede
                    """
                
                    entry_time = row["Time"]
                    entry_type = row["Type"]
                
                    # Pick the right anchor
                    if "Call" in entry_type:
                        anchor_idx = intraday["MIDAS_Bull"].first_valid_index()
                    else:
                        anchor_idx = intraday["MIDAS_Bear"].first_valid_index()
                
                    if anchor_idx is None:
                        return ""
                
                    # Find entry index
                    entry_locs = intraday.index[pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time]
                    if len(entry_locs) == 0:
                        return ""
                    entry_idx = entry_locs[0]
                
                    anchor_loc = intraday.index.get_loc(anchor_idx)
                
                    # Window around anchor → [anchor-7 … anchor+7]
                    start_loc = max(0, anchor_loc - perimeter)
                    end_loc   = min(len(intraday), anchor_loc + perimeter + 1)
                    window = intraday.iloc[start_loc:end_loc]
                
                    # --- Check volatility signals
                    vol_confirm = (window["BBW Alert"] == "🔥").any() or (window["STD_Alert"] == "🐦‍🔥").any()
                
                    # --- Check RVOL spike
                    vol_spike = (window["RVOL_5"] > 1.2).any()
                
                    # Logic
                    if vol_confirm:
                        return "VolConfirmed"
                    elif vol_spike:
                        return "Stampede"
                
                    return ""

                def map_stall_after_entry(row, intraday: pd.DataFrame):
                    """For a given entry row, find the first 🚪 after that bar and return (emoji, time, price)."""
                    # locate the entry bar by HH:MM
                    entry_time = row["Time"]
                    locs = intraday.index[
                        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                    ]
                    if len(locs) == 0:
                        return pd.Series(["", "", ""])
                
                    entry_idx = locs[0]
                    entry_loc = intraday.index.get_loc(entry_idx)
                
                    # scan forward for the first door
                    fwd = intraday.iloc[entry_loc+1:]
                    hits = fwd[fwd.get("T0_Emoji", "") == "🚪"]
                    if hits.empty:
                        return pd.Series(["", "", ""])
                
                    r = hits.iloc[0]
                    # return emoji, time (HH:MM), and price
                    return pd.Series([
                        "🚪",
                        pd.to_datetime(r["Time"]).strftime("%H:%M"),
                        r["Close"]
                    ])
                def map_t1_after_entry(row, intraday):
                    """
                    For a given entry row, find the first 🏇🏼 after that bar and return (emoji, time, price).
                    """
                    # OPTIONAL: only attach T1 to Entry 1 rows. If you want it on all rows, delete this guard.
                    if "🎯1" not in row.get("Type", ""):
                        return pd.Series(["", "", ""])
                
                    # locate the entry bar by HH:MM
                    entry_time = row["Time"]
                    locs = intraday.index[
                        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                    ]
                    if len(locs) == 0:
                        return pd.Series(["", "", ""])
                
                    entry_idx = locs[0]
                    entry_loc = intraday.index.get_loc(entry_idx)
                
                    # scan forward for the first 🏇🏼
                    fwd = intraday.iloc[entry_loc + 1 :]
                    if "T1_Emoji" not in fwd.columns:
                        return pd.Series(["", "", ""])
                    hits = fwd[fwd["T1_Emoji"] == "🏇🏼"]
                    if hits.empty:
                        return pd.Series(["", "", ""])
                
                    r = hits.iloc[0]
                    return pd.Series([
                        "🏇🏼",
                        pd.to_datetime(r["Time"]).strftime("%H:%M"),
                        r["Close"],
                    ])


   
                def map_t2_after_entry(row, intraday):
                    """
                    For a given entry row, find the first T2 (⚡) AFTER that bar.
                    Returns (emoji, HH:MM time, price). Blank if none.
                    """
                    # Locate the entry bar by HH:MM
                    entry_time = row["Time"]
                    locs = intraday.index[
                        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                    ]
                    if len(locs) == 0:
                        return pd.Series(["", "", ""])
                
                    entry_idx = locs[0]
                    entry_loc = intraday.index.get_loc(entry_idx)
                
                    # Scan forward for the first ⚡
                    fwd = intraday.iloc[entry_loc + 1 :]
                    if "T2_Emoji" not in fwd.columns:
                        return pd.Series(["", "", ""])
                
                    hits = fwd[fwd["T2_Emoji"] == "⚡"]
                    if hits.empty:
                        return pd.Series(["", "", ""])
                
                    r = hits.iloc[0]
                    return pd.Series([
                        "⚡",
                        pd.to_datetime(r["Time"]).strftime("%H:%M"),
                        r["Close"],
                    ])
 
                def add_goldmine_from_e2(intraday, dist=120):
                    out = intraday.copy()
                    out["Goldmine_E2_Emoji"] = ""
                    # all E2 bars
                    e2_idx = list(out.index[out["Call_SecondEntry_Emoji"] == "🎯2"]) + \
                             list(out.index[out["Put_SecondEntry_Emoji"]  == "🎯2"])
                    e2_idx = sorted(e2_idx)
                    if not e2_idx:
                        return out
                
                    for idx in e2_idx:
                        is_call = (out.at[idx, "Call_SecondEntry_Emoji"] == "🎯2")
                        f_anchor = out.at[idx, "F_numeric"]
                        if pd.isna(f_anchor):
                            continue
                        start = out.index.get_loc(idx)
                        for j in range(start + 1, len(out)):
                            f = out.iloc[j]["F_numeric"]
                            if pd.isna(f):
                                continue
                            if is_call and f >= f_anchor + dist:
                                out.iat[j, out.columns.get_loc("Goldmine_E2_Emoji")] = "💰"
                                break
                            if (not is_call) and f <= f_anchor - dist:
                                out.iat[j, out.columns.get_loc("Goldmine_E2_Emoji")] = "💰"
                                break
                    return out

                
                def add_goldmine_from_t1(intraday, dist=120):
                    """
                    Marks 💰 Goldmine from T1 (🏇🏼):
                    After T1, when price (F_numeric) has moved at least `dist` F% away from Kijun_F
                    in the correct direction.
                    """
                    out = intraday.copy()
                    out["Goldmine_T1_Emoji"] = ""
                
                    # Find T1
                    t1_idx = out.index[out["T1_Emoji"] == "🏇🏼"]
                    if len(t1_idx) == 0:
                        return out
                
                    start = out.index.get_loc(t1_idx[0])
                
                    # Determine side (from Entry 1)
                    side = None
                    if any(out["Call_FirstEntry_Emoji"] == "🎯"):
                        side = "call"
                    elif any(out["Put_FirstEntry_Emoji"] == "🎯"):
                        side = "put"
                
                    if side is None:
                        return out
                
                    for i in range(start + 1, len(out)):
                        mike = out.at[out.index[i], "F_numeric"]
                        kijun = out.at[out.index[i], "Kijun_F"]
                
                        if pd.isna(mike) or pd.isna(kijun):
                            continue
                
                        if side == "call" and mike - kijun >= dist:
                            out.at[out.index[i], "Goldmine_T1_Emoji"] = "💰"
                            break
                        if side == "put" and kijun - mike >= dist:
                            out.at[out.index[i], "Goldmine_T1_Emoji"] = "💰"
                            break
                
                    return out
                
                
                # ✅ Apply both
                intraday = add_goldmine_from_e2(intraday, dist=120)
                intraday = add_goldmine_from_t1(intraday, dist=120)
                
                


                def map_goldmine_after_t1(row, intraday: pd.DataFrame, thresh=120):
                    """
                    From T1 forward, check if Mike reaches the Goldmine 💰.
                    Goldmine = ±120 F% beyond Kijun_F.
                    Works for both Call and Put entries.
                    """
                    # locate entry row
                    entry_time = row["Time"]
                    locs = intraday.index[
                        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                    ]
                    if len(locs) == 0:
                        return pd.Series(["", "", ""])
                
                    entry_idx = locs[0]
                    entry_loc = intraday.index.get_loc(entry_idx)
                
                    # find T1
                    fwd = intraday.iloc[entry_loc+1:]
                    t1_hits = fwd[fwd.get("T1_Emoji", "") == "🏇🏼"]
                    if t1_hits.empty:
                        return pd.Series(["", "", ""])
                    t1_idx = t1_hits.index[0]
                    t1_loc = intraday.index.get_loc(t1_idx)
                
                    # scan forward from T1
                    fwd_after_t1 = intraday.iloc[t1_loc+1:]
                    for i, r in fwd_after_t1.iterrows():
                        mike = r["F_numeric"]
                        kijun = r.get("Kijun_F", None)
                
                        if pd.isna(mike) or pd.isna(kijun):
                            continue
                
                        if "Call" in row["Type"] and mike >= kijun + thresh:
                            return pd.Series(["💰", pd.to_datetime(r["Time"]).strftime("%H:%M"), r["Close"]])
                        if "Put" in row["Type"] and mike <= kijun - thresh:
                            return pd.Series(["💰", pd.to_datetime(r["Time"]).strftime("%H:%M"), r["Close"]])
                
                    return pd.Series(["", "", ""])
                
                def map_goldmine_after_e2(row, intraday: pd.DataFrame, dist=120):
                    """
                    From the Entry 2 bar (the row's time), find the first 💰 hit measured
                    as ±dist F% from the *E2 bar's F_numeric* (fixed anchor).
                    Works when df['Type'] contains 'Call 🎯2' or 'Put 🎯2'.
                    Returns (emoji, time, price).
                    """
                    # 1) Locate the E2 bar by HH:MM
                    entry_time = row["Time"]
                    locs = intraday.index[
                        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                    ]
                    if len(locs) == 0:
                        return pd.Series(["", "", ""])
                
                    entry_idx = locs[0]
                    entry_loc = intraday.index.get_loc(entry_idx)
                
                    # 2) Infer side/level from df['Type'] (not from emoji columns)
                    t = str(row.get("Type", ""))
                    is_call_e2 = ("Call" in t) and ("🎯2" in t)
                    is_put_e2  = ("Put"  in t) and ("🎯2" in t)
                    if not (is_call_e2 or is_put_e2):
                        return pd.Series(["", "", ""])
                
                    # 3) Fixed anchor at the E2 bar's F%
                    f_anchor = intraday.at[entry_idx, "F_numeric"]
                    if pd.isna(f_anchor):
                        return pd.Series(["", "", ""])
                
                    # 4) Scan forward for +/− dist F% from that anchor
                    fwd = intraday.iloc[entry_loc + 1 :]
                    for _, r in fwd.iterrows():
                        f = r.get("F_numeric", None)
                        if pd.isna(f):
                            continue
                
                        if is_call_e2 and f >= f_anchor + dist:
                            return pd.Series(["💰", pd.to_datetime(r["Time"]).strftime("%H:%M"), r["Close"]])
                
                        if is_put_e2 and f <= f_anchor - dist:
                            return pd.Series(["💰", pd.to_datetime(r["Time"]).strftime("%H:%M"), r["Close"]])
                
                    return pd.Series(["", "", ""])


             
                def map_parallel_after_t2(row, intraday: pd.DataFrame):
                    """
                    For a given entry row, find T2 (⚡), then track the Parallel phase.
                    Returns (emoji, end_time, max_gain_F).
                    """
                    entry_time = row["Time"]
                
                    # locate the entry bar by HH:MM
                    locs = intraday.index[
                        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
                    ]
                    if len(locs) == 0:
                        return pd.Series(["", "", ""])
                
                    entry_idx = locs[0]
                    entry_loc = intraday.index.get_loc(entry_idx)
                
                    # scan forward for the first T2 ⚡
                    fwd = intraday.iloc[entry_loc+1:]
                    hits_t2 = fwd[fwd.get("T2_Emoji", "") == "⚡"]
                    if hits_t2.empty:
                        return pd.Series(["", "", ""])
                
                    # take first T2
                    t2 = hits_t2.iloc[0]
                    t2_loc = intraday.index.get_loc(t2.name)
                
                    # from T2 onward, track Parallel phase
                    fwd2 = intraday.iloc[t2_loc+1:]
                    parallels = fwd2[fwd2.get("Parallel_Emoji", "") == "⚡"]
                    if parallels.empty:
                        return pd.Series(["⚡", pd.to_datetime(t2["Time"]).strftime("%H:%M"), 0])
                
                    # compute max gain relative to T2
                    base_f = t2["F_numeric"]
                    max_f = parallels["F_numeric"].max() if "F_numeric" in parallels else base_f
                    gain_f = max_f - base_f
                
                    last = parallels.iloc[-1]
                    return pd.Series([
                        "⚡",
                        pd.to_datetime(last["Time"]).strftime("%H:%M"),
                        gain_f
                    ])

            

          

                def assign_prefix_tailbone(row, intraday, profile_df, f_bins, pre_anchor_buffer=3):
                     """
                     Prefix = 'Tailbone' if any 🪶 Tail exists from (anchor-3 bars) through the entry bar,
                     using profile *bins* (not raw F% equality).
                     Works for Call/Put; uses Bull/Bear MIDAS first-valid index as the anchor.
                     """
                 
                     # --- which anchor to use based on entry type
                     if "Call" in row["Type"]:
                         anchor_idx = intraday["MIDAS_Bull"].first_valid_index()
                     else:
                         anchor_idx = intraday["MIDAS_Bear"].first_valid_index()
                 
                     if anchor_idx is None:
                         return ""  # no anchor → no Tailbone
                 
                     # --- locate entry bar by time (HH:MM) — simple and robust enough on 5-min data
                     entry_time = row["Time"]
                     entry_locs = intraday.index[pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time]
                     if len(entry_locs) == 0:
                         return ""  # couldn't map entry to intraday
                     entry_idx = entry_locs[0]
                 
                     anchor_loc = intraday.index.get_loc(anchor_idx)
                     entry_loc  = intraday.index.get_loc(entry_idx)
                     if entry_loc <= anchor_loc - pre_anchor_buffer:
                         return ""  # entry occurs before our anchor window
                 
                     # --- window: [anchor-3 ... entry]
                     start_loc = max(0, anchor_loc - pre_anchor_buffer)
                     end_loc   = entry_loc
                 
                     segment_F = intraday.iloc[start_loc:end_loc+1]["F_numeric"].to_numpy()
                 
                     # map every bar's F% in the window to its profile bin
                     # note: np.digitize returns 1..N; subtract 1 to index into f_bins
                     bin_ix = np.clip(np.digitize(segment_F, f_bins) - 1, 0, len(f_bins)-1)
                     window_bins = pd.unique(f_bins[bin_ix])
                 
                     # all profile bins that have a Tail 🪶
                     tail_bins = set(profile_df.loc[profile_df["Tail"] == "🪶", "F% Level"].tolist())
                 
                     # Tailbone if any window bin is a tail bin (covers tails at/just before anchor, or soon after)
                     return "Tailbone" if any(b in tail_bins for b in window_bins) else ""
             

                     # --- volatility confirmation (±lookback bars around anchor)
                     anchor_window = intraday.iloc[
                         max(0, anchor_loc - lookback): min(len(intraday), anchor_loc + lookback + 1)
                     ]
                     has_fire = "🔥" in anchor_window.get("BBW_Alert", "").astype(str).sum()
                     has_std  = "🐦‍🔥" in anchor_window.get("STD_Alert", "").astype(str).sum()
                 
                     if has_fire or has_std:
                         return candidate
                     else:
                         return ""   # fallback to plain Ember


                def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
                    """
                    Normalize column names before CSV export:
                    - Lowercase
                    - Replace spaces with underscores
                    - Remove parentheses, $, and other special chars
                    """
                    df = df.copy()
                    df.columns = (
                        df.columns
                        .str.strip()
                        .str.lower()
                        .str.replace(r"[ $()]", "", regex=True)   # remove $, (, )
                        .str.replace(" ", "_", regex=True)        # replace spaces with underscores
                    )
                    return df



                     


             
                # ----------  Helpers (cached) ----------
                @st.cache_data(show_spinner=False)
                def build_entries_df(intraday: pd.DataFrame) -> pd.DataFrame:
                    """Build the tidy entries table (runs once unless `intraday` changes)."""
                    entries: List[dict] = []
            
                            # PUTS
                    for i in intraday.index[intraday["Put_FirstEntry_Emoji"] == "🎯"]:
                        entries.append({"Type": "Put 🎯1",
                                        "Time": pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
                                        "Price ($) ": intraday.at[i, "Close"],
                                        "F%": intraday.at[i, "F_numeric"],   # works for every row
})
                    for i in intraday.index[intraday["Put_SecondEntry_Emoji"] == "🎯2"]:
                        entries.append({"Type": "Put 🎯2",
                                        "Time": pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
                                        "Price ($) ": intraday.at[i, "Close"],
                                        "F%": intraday.at[i, "F_numeric"],   # works for every row
})
                    for i in intraday.index[intraday["Put_ThirdEntry_Emoji"] == "🎯3"]:
                        entries.append({"Type": "Put 🎯3",
                                        "Time": pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
                                        "Price ($) ": intraday.at[i, "Close"],
                                        "F%": intraday.at[i, "F_numeric"],   # works for every row

                                       })
                
                    # CALLS
                    for i in intraday.index[intraday["Call_FirstEntry_Emoji"] == "🎯"]:
                        entries.append({"Type": "Call 🎯1",
                                        "Time": pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
                                        "Price ($)": intraday.at[i, "Close"],
                                        "F%": intraday.at[i, "F_numeric"],   # works for every row
})
                    for i in intraday.index[intraday["Call_SecondEntry_Emoji"] == "🎯2"]:
                        entries.append({"Type": "Call 🎯2",
                                        "Time": pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
                                        "Price ($)": intraday.at[i, "Close"],
                                        "F%": intraday.at[i, "F_numeric"],   # works for every row

                                       
                                       })
                    for i in intraday.index[intraday["Call_ThirdEntry_Emoji"] == "🎯3"]:
                        entries.append({"Type": "Call 🎯3",
                                        "Time": pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
                                        "Price ($)": intraday.at[i, "Close"],
                                        "F%": intraday.at[i, "F_numeric"],   # works for every row

                                       }),
                    start_date = datetime.date.today().strftime("%Y-%m-%d")

          
                    df = (pd.DataFrame(entries)
                   .sort_values("Time")
                   .reset_index(drop=True))
                    df["Prototype"] = df.apply(assign_prototype, axis=1)
                    df["Label"] = df.apply(assign_label_simple, axis=1, args=(intraday,))
                    df["Suffix"] = df.apply(assign_suffix_simple, axis=1, args=(intraday,))
                    df["Date"] = start_date

                    df = add_exit_columns(df)   # ✅ just like the others, but cleaner
                    df = compute_pae_1to2(df, intraday)
                                    # 👉 map the next stall (🚪) *after* each entry row
                    df[["T0_Emoji", "T0_Time", "T0_Price"]] = df.apply(
                        map_stall_after_entry, axis=1, args=(intraday,), result_type="expand"
                    )
                    # 🏇🏼 map the first acceleration beyond band after each entry (typically after Entry 1)
                    df[["T1_Emoji", "T1_Time", "T1_Price)"]] = df.apply(
                        map_t1_after_entry, axis=1, args=(intraday,), result_type="expand"
                    )

                                     # 🔁 Map T2 (⚡) to each entry row
                    df[["T2_Emoji", "T2_Time", "T2_Price)"]] = df.apply(
                        map_t2_after_entry, axis=1, args=(intraday,), result_type="expand"
                    )

                    df[["Parallel_Emoji", "Parallel_Time", "Parallel_Gain"]] = df.apply(
                    map_parallel_after_t2, axis=1, args=(intraday,), result_type="expand"
                     )

                    # E2 route
                    df[["Goldmine_E2_Emoji", "Goldmine_E2_Time", "Goldmine_E2 Price"]] = df.apply(
                        map_goldmine_after_e2, axis=1, args=(intraday,), result_type="expand"
                    )
                    
                    # T1 route
                    df[["Goldmine_T1_Emoji", "Goldmine_T1_Time", "Goldmine_T1 Price"]] = df.apply(
                        map_goldmine_after_t1, axis=1, args=(intraday,), result_type="expand"
                    )
                 

                    
                    
                    

                    df =  compute_pae_2to3(df, intraday)
                    df = compute_pae_3to40F(df, intraday)
                
                    df["Prefix"] = df.apply(
                        assign_prefix_tailbone,
                        axis=1,
                        args=(intraday, profile_df, f_bins)  # pass your existing profile_df and f_bins
                    )
                  
              


                    return df
        

                @st.cache_data(show_spinner=False)
                def to_csv_bytes(df: pd.DataFrame) -> bytes:
                    """Create CSV bytes from df (cached)."""
                    df = clean_column_names(df)

                    return df.to_csv(index=False).encode("utf-8")
                
                
                # ----------  Build once, reuse always ----------
                entries_df = build_entries_df(intraday)
                csv_bytes  = to_csv_bytes(entries_df)             # cached by df content
                
                # keep these in session_state so other code can reuse without recompute
                # st.session_state.setdefault("entries_df", entries_df)
                # st.session_state.setdefault("entries_csv", csv_bytes)
                st.session_state["entries_df"] = entries_df
                st.session_state["entries_csv"] = csv_bytes
                # Optional: persist expander state across reruns
                st.session_state.setdefault("expand_entries", True)

               # # ----------  UI ----------
               #  with st.expander("Track Entry 1 · 2 · 3 🎯", expanded=True):
               #      st.dataframe(entries_df, use_container_width=True)
                
               #      # --- No-rerun download link ---
               #      csv_bytes = entries_df.to_csv(index=False).encode("utf-8")
               #      b64 = base64.b64encode(csv_bytes).decode("utf-8")
               #      file_name = "entries.csv"
                
               #      st.markdown(
               #          f'<a href="data:text/csv;base64,{b64}" download="{file_name}">⬇️ Download Entries (no rerun)</a>',
               #          unsafe_allow_html=True
               #      )
  # ----------  UI ----------
                with st.expander("Track Entry 1 · 2 · 3 🎯", expanded=True):
                    st.dataframe(entries_df, use_container_width=True)
                
                    # --- CSV Download ---
                    csv_bytes = entries_df.to_csv(index=False).encode("utf-8")
                    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")
                    st.markdown(
                        f'<a href="data:text/csv;base64,{csv_b64}" download="entries.csv">⬇️ Download Entries (CSV)</a>',
                        unsafe_allow_html=True
                    )
                
                  # --- JSON Download (preserve emojis) ---
                    json_str = entries_df.to_json(orient="records", indent=2, force_ascii=False)
                    json_b64 = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
                    st.markdown(
                        f'<a href="data:application/json;base64,{json_b64}" download="entries.json">⬇️ Download Entries (JSON)</a>',
                        unsafe_allow_html=True
                    )

              
                with ticker_tabs[0]:
                    # -- Create Subplots: Row1=F%, Row2=Momentum
                    fig = make_subplots(
                        rows=1,
                        cols=1,

                        vertical_spacing=0.03,
                         shared_xaxes=True,
                       
                         
                 
                    )

    
#**************************************************************************************************************************************************************************


  # (A) F% over time as lines+markers

                    max_abs_val = intraday["F_numeric"].abs().max()
                    scatter_f = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F_numeric"],
                        mode="lines+markers",
                        customdata=intraday["Close"],
                        line=dict(color="#57c7ff", width=1),  # Dodger Blue
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Close: $%{customdata:.2f}<extra></extra>",

                        name="F% (scaled)",

                    )
                    fig.add_trace(scatter_f, row=1, col=1)








                    # # (A.2) Dashed horizontal line at 0
                    # fig.add_hline(
                    #     y=0,
                    #     line_dash="dash",
                    #     row=1, col=1,
                    #     annotation_text="0%",
                    #     annotation_position="top left"
                    # )



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

                    # Detect Tenkan_F crossing up through MIDAS_Bull
                    tenkan_cross_up = (
                        (intraday["Tenkan_F"].shift(1) < intraday["MIDAS_Bull"].shift(1)) &
                        (intraday["Tenkan_F"] >= intraday["MIDAS_Bull"])
                    )
                    
                    # Add emoji marker 🫆 where cross happens
                    intraday["Tenkan_Midas_CrossUp"] = np.where(tenkan_cross_up, "🫆", "")



                         # Detect Tenkan_F crossing down through MIDAS_Bear
                   # Detect Tenkan_F crossing down through MIDAS_Bear
                    tenkan_cross_down = (
                        (intraday["Tenkan_F"].shift(1) > intraday["MIDAS_Bear"].shift(1)) &
                        (intraday["Tenkan_F"] <= intraday["MIDAS_Bear"])
                    )
                    
                    # Add emoji marker 🕸️ where cross happens
                    intraday["Tenkan_Midas_CrossDown"] = np.where(tenkan_cross_down, "🕸️", "")

                   # Calculate Chikou (lagging span) using Close price shifted BACKWARD
                    intraday["Chikou"] = intraday["Close"].shift(-26)

                    # Calculate Chikou_F using shifted price, keeping Time as-is
                    intraday["Chikou_F"] = ((intraday["Chikou"] - prev_close) / prev_close) * 10000

                    # # Drop rows where Chikou_F is NaN (due to shifting)
                    chikou_plot = intraday.dropna(subset=["Chikou_F"])

                    # Plot without shifting time
                    chikou_line = go.Scatter(
                        x=chikou_plot["Time"],
                        y=chikou_plot["Chikou_F"],
                        mode="lines",
                      
                        name="Chikou (F%)",
                        line=dict(color="purple", dash="dash", width=1)
                    )
                    fig.add_trace(chikou_line, row=1, col=1)

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

                


                    intraday["SpanA_F"] = ((intraday["SpanA"] - prev_close) / prev_close) * 10000
                    intraday["SpanB_F"] = ((intraday["SpanB"] - prev_close) / prev_close) * 10000


            




                                    # Calculate Chikou relation to current price
                    intraday["Chikou_Position"] = np.where(intraday["Chikou"] > intraday["Close"], "above",
                                                np.where(intraday["Chikou"] < intraday["Close"], "below", "equal"))

                    # Detect changes in Chikou relation
                    intraday["Chikou_Change"] = intraday["Chikou_Position"].ne(intraday["Chikou_Position"].shift())

                    # Filter first occurrence and changes
                    chikou_shift_mask = intraday["Chikou_Change"] & (intraday["Chikou_Position"] != "equal")

                    # # Assign emojis for only these changes
                    # intraday["Chikou_Emoji"] = np.where(intraday["Chikou_Position"] == "above", "👨🏻‍✈️",
                    #                             np.where(intraday["Chikou_Position"] == "below", "👮🏻‍♂️", ""))

                    # mask_chikou_above = chikou_shift_mask & (intraday["Chikou_Position"] == "above")


                    kijun_line = go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Kijun_F"],
                    mode="lines",
                    line=dict(color="#2ECC71", width=1.4),
                    name="Kijun (F% scale)"
                )
                    fig.add_trace(kijun_line, row=1, col=1)

                    tenkan_line = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["Tenkan_F"],
                        mode="lines",
                        line=dict(color="#E63946", width=0.6, dash="solid"),
                        name="Tenkan (F%)"
                    )
                    fig.add_trace(tenkan_line, row=1, col=1)


                                    # Mask for different RVOL thresholds
                    mask_rvol_extreme = intraday["RVOL_5"] > 1.8
                    mask_rvol_strong = (intraday["RVOL_5"] >= 1.5) & (intraday["RVOL_5"] < 1.8)
                    mask_rvol_moderate = (intraday["RVOL_5"] >= 1.2) & (intraday["RVOL_5"] < 1.5)

                    # Scatter plot for extreme volume spikes (red triangle)
                    scatter_rvol_extreme = go.Scatter(
                        x=intraday.loc[mask_rvol_extreme, "Time"],
                        y=intraday.loc[mask_rvol_extreme, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="red"),
                        name="RVOL > 1.8 (Extreme Surge)",
                        text="Extreme Volume",

                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for strong volume spikes (orange triangle)
                    scatter_rvol_strong = go.Scatter(
                        x=intraday.loc[mask_rvol_strong, "Time"],
                        y=intraday.loc[mask_rvol_strong, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="orange"),
                        name="RVOL 1.5-1.79 (Strong Surge)",
                        text="Strong Volume",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    )

                    # Scatter plot for moderate volume spikes (pink triangle)
                    scatter_rvol_moderate = go.Scatter(
                        x=intraday.loc[mask_rvol_moderate, "Time"],
                        y=intraday.loc[mask_rvol_moderate, "F_numeric"] + 3,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="pink"),
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
                        line=dict(dash="solid", color="#d3d3d3",width=1),
                        name="Upper Band"
                    )

                    # (C) Lower Band
                    lower_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Lower"],
                        mode="lines",
                        line=dict(dash="solid", color="#d3d3d3",width=1),
                        name="Lower Band"
                    )

                    # (D) Moving Average (Middle Band)
                    middle_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% MA"],
                        mode="lines",
                        line=dict(dash="dash",color="#d3d3d3",width=2),  # Set dash style
                        name="Middle Band (14-MA)"
                    )





                 
               
                    fig.add_trace(upper_band, row=1, col=1)
                    fig.add_trace(lower_band, row=1, col=1)
                    fig.add_trace(middle_band, row=1, col=1)
                    # Ensure 'Marengo' column has 🐎 or empty string
                    marengo_mask = intraday["Marengo"] == "🐎"
                    
                    # Vertical offset to place 🐎 just below the Upper Band
                    offset = 10  # adjust as needed for clarity
                    
                    # Marengo trace (plotted just below the upper band)
                    marengo_trace = go.Scatter(
                        x=intraday.loc[marengo_mask, "Time"],
                        y=intraday.loc[marengo_mask, "F% Upper"] + offset,
                        mode="text",
                        textfont=dict(size=28),
                        text=["🐎"] * marengo_mask.sum(),
                        textposition="top center",
                        name="Marengo",
                        showlegend=True
                    )
                    
                    # Add to your existing figure
                    fig.add_trace(marengo_trace, row=1, col=1)



                  # Mask for South Marengos
                    south_mask = intraday["South_Marengo"] == "🐎"
                    
                    # Offset downward from lower band
                    offset_south = 10
                    
                    south_marengo_trace = go.Scatter(
                        x=intraday.loc[south_mask, "Time"],
                        y=intraday.loc[south_mask, "F% Lower"] - offset_south,
                        mode="text",
                        text=["🐎"] * south_mask.sum(),
                        textfont=dict(size=28),
                        textposition="bottom center",
                        name="South Marengo",
                        showlegend=True
                    )
                    
                    fig.add_trace(south_marengo_trace, row=1, col=1)


        

# # ------------------------
                    # ✅ Yesterday's High - Blue Dashed Line (F% Scale)
                    y_high_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday Open F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="green", dash="dash",width=0.3),
                        name="Yesterday Open (F%)",
                        yaxis="y2",              # << ✅ this is key
                        showlegend=False,
                        hoverinfo='skip'
                    )

                    # ✅ Yesterday's High - Blue Dashed Line (F% Scale)
                    y_high_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday High F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="green", dash="dash",width=0.3),
                        name="Yesterday High (F%)",
                        yaxis="y2",              # << ✅ this is key
                        showlegend=False,
                        hoverinfo='skip'
                    )

                    # ✅ Yesterday's Low - Green Dashed Line (F% Scale)
                    y_low_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[intraday["Yesterday Low F%"].iloc[0]] * len(intraday),
                        mode="lines",
                        line=dict(color="red", dash="dash", width=0.3),
                        name="Yesterday Low (F%)",
                        yaxis="y2",              # << ✅ this is key
                        showlegend=False,
                        hoverinfo='skip'
                    )

                    # ✅ Yesterday's Close - Red Dashed Line (F% Scale) (Always at 0)
                    y_close_f_line = go.Scatter(
                        x=intraday["Time"],
                        y=[0] * len(intraday),
                        mode="lines",
                        line=dict(color="blue", dash="dash", width=0.3),
                        name="Yesterday Close (F%)",
                        yaxis="y2",              # << ✅ this is key
                        showlegend=False,
                        hoverinfo='skip'
                    )

               


                             # BBW Tight → Pink Bishops ♗
                    mask_bbw_tight = intraday["BBW_Tight_Emoji"] == "🐝"
                    
                    scatter_bishop_tight = go.Scatter(
                        x=intraday.loc[mask_bbw_tight, "Time"],
                        y=intraday.loc[mask_bbw_tight, "F_numeric"] + 12,  # Adjusted Y offset
                        mode="text",
                        text=["🐝"] * mask_bbw_tight.sum(),  # ♗ as symbol
                        textposition="top center",
                        textfont=dict(size=12, color="mediumvioletred"),  # 🎯 Pink / Purple shade
                        name="BBW Tight Bishop (♗🐝)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>BBW Tight Compression ♗🐝<extra></extra>"
                    )
                    
                    fig.add_trace(scatter_bishop_tight, row=1, col=1)





 # 🟢 BBW Expansion

                    mask_bbw_alert = intraday["BBW Alert"] != ""

                    scatter_bbw_alert = go.Scatter(
                        x=intraday.loc[mask_bbw_alert, "Time"],
                        y=intraday.loc[mask_bbw_alert, "F_numeric"] - 8,  # Offset above F%
                        mode="text",
                        text=intraday.loc[mask_bbw_alert, "BBW Alert"],
                        textposition="bottom center",
                        textfont=dict(size=12),
                        name="BBW Expansion Alert",
                        hovertemplate="Time: %{x}<br>BBW Ratio: %{customdata:.2f}<extra></extra>",
                        customdata=intraday.loc[mask_bbw_alert, "BBW_Ratio"]
                    )

                    fig.add_trace(scatter_bbw_alert, row=1, col=1)


  #🟢 ADX Expansion


                    mask_adx_alert = intraday["ADX_Alert"] != ""

                    scatter_adx_alert = go.Scatter(
                        x=intraday.loc[mask_adx_alert, "Time"],
                        y=intraday.loc[mask_adx_alert, "F_numeric"] + 10,  # Offset for visibility
                        mode="text",
                        text=intraday.loc[mask_adx_alert, "ADX_Alert"],
                        textposition="top center",
                        textfont=dict(size=11),
                        name="ADX Expansion Alert",
                        hovertemplate="Time: %{x}<br>ADX Ratio: %{customdata:.2f}<extra></extra>",
                        customdata=intraday.loc[mask_adx_alert, "ADX_Ratio"]
                    )

                    fig.add_trace(scatter_adx_alert, row=1, col=1)



# 🟢  STD Expansion  (🐦‍🔥)
                    mask_std_alert = intraday["STD_Alert"] != ""

                    scatter_std_alert = go.Scatter(
                        x=intraday.loc[mask_std_alert, "Time"],
                        y=intraday.loc[mask_std_alert, "F_numeric"] - 16,  # Offset above F%
                        mode="text",
                        text=intraday.loc[mask_std_alert, "STD_Alert"],
                        textposition="bottom center",
                        textfont=dict(size=11),
                        name="F% STD Expansion",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>STD Alert: %{text}<extra></extra>"
                    )

                    fig.add_trace(scatter_std_alert, row=1, col=1)

 #🟢   ATR Expansion
                    mask_atr_alert = intraday["ATR_Exp_Alert"] != ""

                    atr_alert_scatter = go.Scatter(
                        x=intraday.loc[mask_atr_alert, "Time"],
                        y=intraday.loc[mask_atr_alert, "F_numeric"]  - 8,  # place above F%
                        mode="text",
                        textposition="bottom center",

                        text=intraday.loc[mask_atr_alert, "ATR_Exp_Alert"],
                        textfont=dict(size=15),
                        name="ATR Expansion",
                        hoverinfo="text",
                        hovertext=intraday.loc[mask_atr_alert, "ATR_Exp_Alert"],
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>ATR_Exp_Alert: %{text}<extra></extra>"
                    )

                    fig.add_trace(atr_alert_scatter, row=1, col=1)


              




                    fig.add_trace(
                        go.Scatter(
                            x=intraday['Time'],
                            y=intraday['TD Supply Line F'],
                            mode='lines',
                            line=dict(width=0.5, color="#8A2BE2", dash='dot'),
                            name='TD Supply F%',
                            hovertemplate="Time: %{x}<br>Supply (F%): %{y:.2f}"
                        ),
                        row=1, col=1
                    )



 
                    fig.add_trace(
                        go.Scatter(
                            x=intraday['Time'],
                            y=intraday['TD Demand Line F'],
                            mode='lines',
                            line=dict(width=0.5, color="#5DADE2", dash='dot'),
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
                            textfont=dict(size=24, color="green"),
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
                            textfont=dict(size=24, color="red"),
                            name="Sell TDST",
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                        ),
                        row=1, col=1
                    )


                 
          
                                                  

                   


                    intraday["F_shift"] = intraday["F_numeric"].shift(1)

    


                   












                short_entry_trace = go.Scatter(
                    x=intraday.loc[intraday["Entry_Alert_Short"], "Time"],
                    y=intraday.loc[intraday["Entry_Alert_Short"], "F_numeric"] - 13,
                    mode="text",
                    text=[" ✅"] * intraday["Entry_Alert_Short"].sum(),
                    textposition="bottom left",
                    textfont=dict(size=13, color="lime"),
                    name="Short Entry (✅)"
                )
                fig.add_trace(short_entry_trace, row=1, col=1)






                long_entry_trace = go.Scatter(
                    x=intraday.loc[intraday["Entry_Alert_Long"], "Time"],
                    y=intraday.loc[intraday["Entry_Alert_Long"], "F_numeric"] + 13,
                    mode="text",
                    text=[" ✅"] * intraday["Entry_Alert_Long"].sum(),
                    textposition="top left",
                    textfont=dict(size=13, color="lime"),
                    name="Long Entry (✅)"
                )
                fig.add_trace(long_entry_trace, row=1, col=1)


                # 🔍 First Wake-Up Detection
                first_call_eye_idx = intraday.index[intraday["Call_Wake_Emoji"] == "👁️"]
                first_put_eye_idx  = intraday.index[intraday["Put_Wake_Emoji"]  == "🦉"]
                
                # ✅ Plot Call Wake 👁️ once
                if not first_call_eye_idx.empty:
                    first_idx = first_call_eye_idx[0]
                    fig.add_trace(go.Scatter(
                        x=[intraday.loc[first_idx, "Time"]],
                        y=[intraday.loc[first_idx, price_col] + 30],  # position above
                        mode="text",
                        text=["👁️"],
                        textposition="top center",
                        textfont=dict(size=28),
                        showlegend=True,
                        hoverinfo="text",
                        hovertemplate="<b>Call Wake-Up</b><br>Time: %{x}<br>F%%: %{y:.2f}<extra></extra>",

                        name="Call Wake-Up"
                    ), row=1, col=1)
                
                # ✅ Plot Put Wake 🦉 once
                if not first_put_eye_idx.empty:
                    first_idx = first_put_eye_idx[0]
                    fig.add_trace(go.Scatter(
                        x=[intraday.loc[first_idx, "Time"]],
                        y=[intraday.loc[first_idx, price_col] - 0],  # position below
                        mode="text",
                        text=["🦉"],
                        textposition="bottom right",
                        textfont=dict(size=21),
                        showlegend=True,
                        hoverinfo="text",
                        hovertemplate="<b>Put Wake-Up</b><br>Time: %{x}<br>F%%: %{y:.2f}<extra></extra>",

                        name="Put Wake-Up"
                    ), row=1, col=1)

                # ✅ Plot Call Solo Eye 👁️ (No Cross but strong rise)
                first_call_solo_eye_idx = intraday.index[intraday["Call_Eye_Solo"] == "👁️"]
                
                if not first_call_solo_eye_idx.empty:
                    first_idx = first_call_solo_eye_idx[0]
                    fig.add_trace(go.Scatter(
                        x=[intraday.loc[first_idx, "Time"]],
                        y=[intraday.loc[first_idx, price_col] + 15],  # Slightly below Wake-Up 👁️
                        mode="text",
                        text=["👁️"],
                        textposition="top center",
                        textfont=dict(size=21),
                        showlegend=True,
                        hoverinfo="text",
                        hovertemplate="<b>Call Rising (No Cross)</b><br>Time: %{x}<br>F%%: %{y:.2f}<extra></extra>",
                        name="Call Solo Eye"
                    ), row=1, col=1)



                  
                    # ✅ Plot Put Solo Eye 🦉 (No Cross but strong drop)
                    first_put_solo_eye_idx = intraday.index[intraday["Put_Eye_Solo"] == "🦉"]





                    if not first_put_solo_eye_idx.empty:
                        first_idx = first_put_solo_eye_idx[0]
                        fig.add_trace(go.Scatter(
                            x=[intraday.loc[first_idx, "Time"]],
                            y=[intraday.loc[first_idx, price_col] - 15],  # Slightly above Put Wake 🦉
                            mode="text",
                            text=["🦉"],
                            textposition="bottom center",
                            textfont=dict(size=24),
                            showlegend=True,
                            hoverinfo="text",
                            hovertemplate="<b>Put Falling (No Cross)</b><br>Time: %{x}<br>F%%: %{y:.2f}<extra></extra>",
                            name="Put Solo Eye"
                        ), row=1, col=1)

             
                intraday["Call_Option_Smooth"] = intraday["Call_Option_Value"].rolling(3).mean()
                intraday["Put_Option_Smooth"]  = intraday["Put_Option_Value"].rolling(3).mean()
                

                                       # Add IB High to MIDAS Option Plot
                               # Loop over the subplot rows: 1 = F%, 2 = Call/Put, 3 = Midas+Option
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=[ib_high] * len(intraday),
                    mode="lines",
                    line=dict(color="#FFD700", dash="dot", width=0.7),
                    name="IB High",
                    showlegend=True
                ), row=1, col=1)
                
                # 🟫 IB Low (subtle off-white line)
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=[ib_low] * len(intraday),
                    mode="lines",
                    line=dict(color="#FFD700", dash="dot", width=0.7),
                    name="IB Low",
                    showlegend=True
                ), row=1, col=1)
              
     
                ib_third = (ib_high - ib_low) / 3
                ib_upper_third = ib_low + 2 * ib_third
                ib_middle_third = ib_low + ib_third
  
  
                fig.add_hline(y=ib_middle_third, line=dict(color="gray", dash="dash", width=0.5), row=1, col=1)
                fig.add_hline(y=ib_upper_third, line=dict(color="gray", dash="dash", width=0.5), row=1, col=1)
                

                fig.add_hline(y=va_min,showlegend=True, line=dict(color="#0ff", dash="dot", width=0.5), row=1, col=1)
                fig.add_hline(y=va_max,showlegend=True, line=dict(color="#0ff", dash="dot", width=0.5), row=1, col=1)

            
       

               


                fig.add_trace(go.Scatter(
                    x=intraday['TimeIndex'],
                    y=intraday['MIDAS_Bear'],
                    mode="lines",
                    name="MIDAS Bear",
                    line=dict(color="#ff5ea8", dash="dot", width=2),
                    hovertemplate="Time: %{x}<br>MIDAS Bear: %{y:.2f}<extra></extra>"
                ))
                
                fig.add_trace(go.Scatter(
                    x=intraday['TimeIndex'],
                    y=intraday['MIDAS_Bull'],
                    mode="lines",
                    name="MIDAS Bull",
                    line=dict(color="#1ac997", dash="dash", width=2),
                    hovertemplate="Time: %{x}<br>MIDAS Bull: %{y:.2f}<extra></extra>"
                ))

                for _, row in profile_df.iterrows():
                    if row["Tail"] == "🪶":
                        # Safely get matching intraday rows for this F% Level
                        bin_str = str(row["F% Level"])
                        time_row = intraday[intraday["F_Bin"].astype(str) == bin_str]
                
                        if not time_row.empty:
                            time_at_level = time_row["TimeIndex"].iloc[0]  # earliest time at that level
                
                            fig.add_trace(go.Scatter(
                                x=[time_at_level],
                                y=[row["F% Level"]],
                                mode="text",
                                text=["🪶"],
                                textposition="middle right",
                                textfont=dict(size=22),
                                showlegend=True,
                                name="🪶 Tail",
                                hovertemplate=(
                                    "🪶 Tail<br>"
                                    f"F% Level: {row['F% Level']}<br>"
                                    f"Time: {time_at_level}<extra></extra>"
                                )
                            ), row=1, col=1)


                                # 👋🏽 Bull MIDAS Hand = price breaks **above** the Bear MIDAS line (resistance)
                bull_hand_rows = intraday[intraday["MIDAS_Bull_Hand"] == "👋🏽"]
                fig.add_trace(go.Scatter(
                    x=bull_hand_rows["TimeIndex"],
                    y=bull_hand_rows["MIDAS_Bear"] + 3,  # Adjust for spacing above line
                    mode="text",
                    text=["👋🏽"] * len(bull_hand_rows),
                    textposition="top right",
                    textfont=dict(size=22),
                    showlegend=False,
                    hovertemplate=(
                        "👋🏽 Bull MIDAS Breakout<br>"
                        "Time: %{x|%I:%M %p}<br>"
                        f"Bear MIDAS: {{y:.2f}}<extra></extra>"
                    )
                ), row=1, col=1)
                
                # 🧤 Bear MIDAS Glove = price breaks **below** the Bull MIDAS line (support)
                bear_glove_rows = intraday[intraday["MIDAS_Bear_Glove"] == "🧤"]
                fig.add_trace(go.Scatter(
                    x=bear_glove_rows["TimeIndex"],
                    y=bear_glove_rows["MIDAS_Bull"] - 3,  # Adjust for spacing below line
                    mode="text",
                    text=["🧤"] * len(bear_glove_rows),
                    textposition="bottom right",
                    textfont=dict(size=22),
                    showlegend=False,
                    hovertemplate=(
                        "🧤 Bear MIDAS Breakdown<br>"
                        "Time: %{x|%I:%M %p}<br>"
                        f"Bull MIDAS: {{y:.2f}}<extra></extra>"
                    )
                 ), row=1, col=1)

                     
                               # 🦵🏼 Bull MIDAS Wake
                if pd.notna(first_bull_midas_idx):
                    fig.add_trace(go.Scatter(
                        x=[intraday.loc[first_bull_midas_idx, "Time"]],
                        y=[intraday.loc[first_bull_midas_idx, price_col]],  # <- no quotes
                        mode="text",
                        text=["🦵🏼"],
                        textposition="top center",
                        textfont=dict(size=28),
                        showlegend=False,
                        hoverinfo="skip",
                        hovertemplate="🦵🏼 Bull MIDAS Wake<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>",

                        name="Bull MIDAS Wake (🦵🏼)"
                    ), row=1, col=1)
                
                # 🦶🏼 Bear MIDAS Wake
                if pd.notna(first_bear_midas_idx):
                    fig.add_trace(go.Scatter(
                        x=[intraday.loc[first_bear_midas_idx, "Time"]],
                        y=[intraday.loc[first_bear_midas_idx, price_col] + 20],  # <- no quotes
                        mode="text",
                        text=["💥"],
                        textfont=dict(size=30),
                        textposition="bottom center",
                        showlegend=False,
                        hovertemplate="💥 Bear MIDAS Wake<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>",

                        hoverinfo="skip",
                        name="Bear MIDAS Wake (💥)"
                    ), row=1, col=1)

                                
                                   # Plot 💸 for breakout above IB High
                high_break_df = intraday[intraday["IB_High_Break"] == "💸"]
                fig.add_trace(go.Scatter(
                    x=high_break_df["TimeIndex"],
                    y=high_break_df["F_numeric"] + 30,
                    mode="text",
                    text=high_break_df["IB_High_Break"],
                    textposition="top left",
                    textfont=dict(size=24),
                    name="Breakout Above IB 💸",
                    showlegend=True,
                    hovertemplate="Time: %{x}<br>💸 IB High Breakout"
                ), row=1, col=1)
                
                # Plot 🧧 for breakdown below IB Low
                low_break_df = intraday[intraday["IB_Low_Break"] == "🧧"]
                fig.add_trace(go.Scatter(
                    x=low_break_df["TimeIndex"],
                    y=low_break_df["F_numeric"] - 30,
                    mode="text",
                    text=low_break_df["IB_Low_Break"],
                    textposition="bottom left",
                    textfont=dict(size=24),
                    name="Breakdown Below IB 🧧",
                    showlegend=True,
                    hovertemplate="Time: %{x}<br>🧧 IB Low Breakdown"
                ), row=1, col=1)

                
                # 🦻🏼 Get the top 🦻🏼 ear level based on highest %Vol
                ear_row = (
                    profile_df[profile_df["🦻🏼"] == "🦻🏼"]
                    .sort_values(by="%Vol", ascending=False)
                    .head(1)  # only take the highest
                )
                
                if not ear_row.empty:
                    row = ear_row.iloc[0]
                    ear_level = row["F% Level"]
                    vol = row["%Vol"]
                    time = row["Time"]
                
                    # 🦻🏼 Add Ear memory line
                    fig.add_hline(
                        y=ear_level,
                        line=dict(color="darkgray", dash="dot", width=1.5),
                        row=1, col=1,
                        showlegend=False,
                        annotation_text="🦻🏼 Ear Shift",
                        annotation_position="top left",
                        annotation_font=dict(color="black"),
                    )
                
                    # 🦻🏼 Optional: Add emoji text near the memory level
                    fig.add_trace(go.Scatter(
                        x=[intraday["TimeIndex"].iloc[-1]],  # Use last bar's timestamp
                        y=[ear_level],
                        mode="text",
                        text=["🦻🏼"],
                        textposition="middle right",
                        textfont=dict(size=20),
                        showlegend=False,
                        hovertemplate=f"🦻🏼 Top Memory Line<br>%Vol: {vol:.2f}<br>Time: {time}<extra></extra>"
                    ), row=1, col=1)
                
              # 👃🏽 Get the F% level with the highest Letter_Count (most time spent)
                    nose_row = (
                        profile_df.sort_values(by="Letter_Count", ascending=False)
                        .head(1)  # Only the top one
                    )
                    
                    if not nose_row.empty:
                        poc_f_level = nose_row["F% Level"].values[0]
                        nose_time = nose_row["Time"].values[0]
                        
                        # Add pink dotted line for 👃🏽
                        fig.add_hline(
                            y=poc_f_level,
                            line=dict(color="#ff1493", dash="dot", width=0.3),
                            row=1, col=1,
                            showlegend=False
                        )
                    
                        # Plot 👃🏽 text marker at far right
                        fig.add_trace(go.Scatter(
                            x=[intraday["TimeIndex"].iloc[-1]],  # far right time
                            y=[poc_f_level],
                            mode="markers+text",
                            marker=dict(size=0, color="#ff1493"),
                            text=["👃🏽 Nose (Most Price Acceptance)"],
                            textposition="top right",
                            name="👃🏽 Nose Line",
                            showlegend=False,
                            hovertemplate=(
                                "👃🏽 Nose Line<br>"
                                f"F% Level: {poc_f_level}<br>"
                                f"Time: {nose_time}<extra></extra>"
                            )
                        ), row=1, col=1)

       

                


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



                 # Create separate masks for upward and downward crosses:
                    mask_kijun_up = intraday["Kijun_F_Cross_Emoji"] == "♕"
                    mask_kijun_down = intraday["Kijun_F_Cross_Emoji"] == "♛"

                    # Upward Cross Trace (♕)
                    up_cross_trace = go.Scatter(
                        x=intraday.loc[mask_kijun_up, "Time"],
                        y=intraday.loc[mask_kijun_up, "F_numeric"] + 89,  # Offset upward (adjust as needed)
                        mode="text",
                        text=intraday.loc[mask_kijun_up, "Kijun_F_Cross_Emoji"],
                        textposition="top center",  # Positioned above the point
                        textfont=dict(size=34, color="green"),
                        name="Kijun Cross Up (♕)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Upward Cross: %{text}<extra></extra>"
                    )

                    # Downward Cross Trace (♛)
                    down_cross_trace = go.Scatter(
                        x=intraday.loc[mask_kijun_down, "Time"],
                        y=intraday.loc[mask_kijun_down, "F_numeric"] - 89,  # Offset downward
                        mode="text",
                        text=intraday.loc[mask_kijun_down, "Kijun_F_Cross_Emoji"],
                        textposition="bottom center",  # Positioned below the point
                        textfont=dict(size=34, color="red"),
                        name="Kijun Cross Down (♛)",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Downward Cross: %{text}<extra></extra>"
                    )


                    fig.add_trace(up_cross_trace,   row=1, col=1)
                    fig.add_trace(down_cross_trace, row=1, col=1)


                mask_horse_buy = intraday["Kijun_Cross_Horse"] == "♘"
                mask_horse_sell = intraday["Kijun_Cross_Horse"] == "♞"

                # Buy Horse (♘) → normal above
                scatter_horse_buy = go.Scatter(
                    x=intraday.loc[mask_horse_buy, "Time"],
                    y=intraday.loc[mask_horse_buy, "F_numeric"] + 45,
                    mode="text",
                    text=["♘"] * mask_horse_buy.sum(),
                    textposition="top left",
                    textfont=dict(size=34, color="green"),  # You can make it white if you want
                    name="Horse After Buy Kijun Cross",
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>♘ Horse after Buy<extra></extra>"
                )

                # Sell Horse (♞) → below and red
                scatter_horse_sell = go.Scatter(
                    x=intraday.loc[mask_horse_sell, "Time"],
                    y=intraday.loc[mask_horse_sell, "F_numeric"] - 45,
                    mode="text",
                    text=["♞"] * mask_horse_sell.sum(),
                    textposition="bottom left",
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
         
                              
                mask_rook_up = intraday["TD_Supply_Rook"] == "♖"
                mask_rook_down = intraday["TD_Supply_Rook"] == "♜"

                # White rook (up cross)
                scatter_rook_up = go.Scatter(
                    x=intraday.loc[mask_rook_up, "Time"],
                    y=intraday.loc[mask_rook_up, "F_numeric"] + 13,  # Offset upward
                    mode="text",
                    text=intraday.loc[mask_rook_up, "TD_Supply_Rook"],
                    textposition="top left",
                    textfont=dict(size=21,  color="green"),
                    name="TD Supply Cross Up (♖)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>TD Supply Crossed Up ♖<extra></extra>"
                )

                # Black rook (down cross)
                scatter_rook_down = go.Scatter(
                    x=intraday.loc[mask_rook_down, "Time"],
                    y=intraday.loc[mask_rook_down, "F_numeric"] - 13,  # Offset downward
                    mode="text",
                    text=intraday.loc[mask_rook_down, "TD_Supply_Rook"],
                    textposition="bottom left",
                    textfont=dict(size=21,  color="red"),
                    name="TD Supply Cross Down (♜)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>TD Supply Crossed Down ♜<extra></extra>"
                )

             
 
                mask_pawn_up   = intraday["Tenkan_Pawn"] == "♙"
                mask_pawn_down = intraday["Tenkan_Pawn"] == "♟️"     # <-- changed ♙ → ♟️

                # ♙ Upward pawn
                pawn_up = go.Scatter(
                    x=intraday.loc[mask_pawn_up, "Time"],
                    y=intraday.loc[mask_pawn_up, "F_numeric"] + 8,
                    mode="text",
                    text=intraday.loc[mask_pawn_up, "Tenkan_Pawn"],
                    textposition="top center",
                    textfont=dict(size=16, color="green"),            # green for up
                    name="Pawn Up (Tenkan Cross)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>♙ Upward Tenkan Cross<extra></extra>"
                )

                # ♟️ Downward pawn
                pawn_down = go.Scatter(
                    x=intraday.loc[mask_pawn_down, "Time"],
                    y=intraday.loc[mask_pawn_down, "F_numeric"] - 8,
                    mode="text",
                    text=intraday.loc[mask_pawn_down, "Tenkan_Pawn"],
                    textposition="bottom center",
                    textfont=dict(size=14, color="red"),             # red for down
                    name="Pawn Down (Tenkan Cross)",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>♟️ Downward Tenkan Cross<extra></extra>"
                )

                fig.add_trace(pawn_up,   row=1, col=1)
                fig.add_trace(pawn_down, row=1, col=1)

                            # Calculate Chikou relation to current price
                intraday["Chikou_Position"] = np.where(intraday["Chikou"] > intraday["Close"], "above",
                                            np.where(intraday["Chikou"] < intraday["Close"], "below", "equal"))

                # Detect changes in Chikou relation
                intraday["Chikou_Change"] = intraday["Chikou_Position"].ne(intraday["Chikou_Position"].shift())

                # Filter first occurrence and changes
                chikou_shift_mask = intraday["Chikou_Change"] & (intraday["Chikou_Position"] != "equal")
                intraday["Chikou_Comparison_Price"] = intraday["Close"].shift(+26)
                intraday["Chikou_Comparison_Time"] = intraday["Time"].shift(+26)

                intraday["Chikou_Emoji"] = np.where(intraday["Chikou_Position"] == "above", "👨🏻‍✈️",
                                            np.where(intraday["Chikou_Position"] == "below", "👮🏻‍♂️", ""))


                cloud_mask = intraday["Heaven_Cloud"] == "☁️"
  
                fig.add_trace(go.Scatter(
                    x=intraday.loc[cloud_mask, "Time"],
                    y=intraday.loc[cloud_mask, "F_numeric"] + 100,
                    mode="text",
                    text=intraday.loc[cloud_mask, "Heaven_Cloud"],
                    textposition="top center",
                    textfont=dict(size=21),
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
                    y=intraday.loc[drizzle_mask, "F_numeric"] + 100,  # Position below the bar
                    mode="text",
                    text=intraday.loc[drizzle_mask, "Drizzle_Emoji"],
                    textposition="bottom center",
                    textfont=dict(size=21),
                    name="Price Dropped Below Demand 🌧️",
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>Crossed Below Demand<extra></extra>"
                ), row=1, col=1)
  
             

   
        

             

          


      
                                            
                                
                

 
              
                astronaut_points = intraday[intraday["Astronaut_Emoji"] == "👨🏽‍🚀"]

                scatter_astronaut = go.Scatter(
                    x=astronaut_points["Time"],
                    y=astronaut_points["F_numeric"] + 124,  # Higher offset
                    mode="text",
                    text=astronaut_points["Astronaut_Emoji"],
                    textposition="top center",
                    name="New Highs 👨🏽‍🚀",
                    textfont=dict(size=21),
                 )
 
#                     # Add to figure
                # fig.add_trace(up_high_trace, row=1, col=1)
 


                    # Filter where the Astronaut or Moon emoji exist
                astronaut_points = intraday[intraday["Astronaut_Emoji"] != ""]

                scatter_astronaut = go.Scatter(
                    x=astronaut_points["Time"],
                    y=astronaut_points["F_numeric"] + 124,  # Offset so it floats higher
                    mode="text",
                    text=astronaut_points["Astronaut_Emoji"],  # Either 👨🏽‍🚀 or 🌒
                    textposition="top center",
                    name="New Highs 🌒",
                    textfont=dict(size=21),
                   
                )

                fig.add_trace(scatter_astronaut, row=1, col=1)


                # Filter where Swimmer or Squid exist
                swimmer_points = intraday[intraday["Swimmer_Emoji"] != ""]

                scatter_swimmer = go.Scatter(
                    x=swimmer_points["Time"],
                    y=swimmer_points["F_numeric"] - 104,  # Offset downward so it floats below price
                    mode="text",
                    text=swimmer_points["Swimmer_Emoji"],  # Either 🏊🏽‍♂️ or 🦑
                    textposition="bottom center",
                    name="New Lows 🏊🏽‍♂️🦑",
                    textfont=dict(size=24),
                    showlegend=True
                )

                fig.add_trace(scatter_swimmer, row=1, col=1)


  
                first_entry_mask = intraday["Put_FirstEntry_Emoji"] == "🎯"
                
                fig.add_trace(go.Scatter(
                    x=intraday.loc[first_entry_mask, "Time"],
                    y=intraday.loc[first_entry_mask, "F_numeric"] - 34,
                    mode="text",
                    text=intraday.loc[first_entry_mask, "Put_FirstEntry_Emoji"],
                    textposition="top center",
                    textfont=dict(size=34),
                    name="🎯 Put Entry (Midas Bear + First Drizzle)",
                    hovertemplate="Time: %{x}<br>F%%: %{y} Put_FirstEntry_Emoji<extra></extra>"
                ), row=1, col=1)

                second_entry_mask = intraday["Put_SecondEntry_Emoji"] == "🎯2"
                
                fig.add_trace(go.Scatter(
                    x=intraday.loc[second_entry_mask, "Time"],
                    y=intraday.loc[second_entry_mask, "F_numeric"] - 34,
                    mode="text",
                    text=intraday.loc[second_entry_mask, "Put_SecondEntry_Emoji"],
                    textposition="top center",
                    textfont=dict(size=34),
                    name="🎯2 Put Second Entry",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<extra></extra>"
                ), row=1, col=1)
                
                third_entry_mask = intraday["Put_ThirdEntry_Emoji"] == "🎯3"
                
                fig.add_trace(go.Scatter(
                    x=intraday.loc[third_entry_mask, "Time"],
                    y=intraday.loc[third_entry_mask, "F_numeric"] - 34,
                    mode="text",
                    text=intraday.loc[third_entry_mask, "Put_ThirdEntry_Emoji"],
                    textposition="top center",
                    textfont=dict(size=34),
                    name="🎯3 Put Third Entry",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<extra></extra>"
                ), row=1, col=1)
                
                # 🎯 Call Entry 1
                call1_mask = intraday["Call_FirstEntry_Emoji"] == "🎯"
                fig.add_trace(go.Scatter(
                    x=intraday.loc[call1_mask, "Time"],
                    y=intraday.loc[call1_mask, "F_numeric"] + 34,
                    mode="text",
                    text=intraday.loc[call1_mask, "Call_FirstEntry_Emoji"],
                    textposition="top center",
                    textfont=dict(size=34),
                    name="🎯 Call Entry 1",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<extra></extra>"
                ), row=1, col=1)
                
                # 🎯2 Call Entry 2
                call2_mask = intraday["Call_SecondEntry_Emoji"] == "🎯2"
                fig.add_trace(go.Scatter(
                    x=intraday.loc[call2_mask, "Time"],
                    y=intraday.loc[call2_mask, "F_numeric"] + 34,
                    mode="text",
                    text=intraday.loc[call2_mask, "Call_SecondEntry_Emoji"],
                    textposition="top center",
                    textfont=dict(size=34),
                    name="🎯2 Call Entry 2",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<extra></extra>"
                ), row=1, col=1)
                
                # 🎯3 Call Entry 3
                call3_mask = intraday["Call_ThirdEntry_Emoji"] == "🎯3"
                fig.add_trace(go.Scatter(
                    x=intraday.loc[call3_mask, "Time"],
                    y=intraday.loc[call3_mask, "F_numeric"] + 34,
                    mode="text",
                    text=intraday.loc[call3_mask, "Call_ThirdEntry_Emoji"],
                    textposition="top center",
                    textfont=dict(size=34),
                    name="🎯3 Call Entry 3",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<extra></extra>"
                ), row=1, col=1)



                put_pe_mask = (intraday["Put_FirstEntry_Emoji"] == "🎯") & (intraday["Put_PE"] > intraday["Call_PE"])
                
                fig.add_trace(go.Scatter(
                    x=intraday.loc[put_pe_mask, "Time"],
                    y=intraday.loc[put_pe_mask, "F_numeric"] - 44,
                    mode="text",
                    text=["🧬"] * put_pe_mask.sum(),
                    textposition="top center",
                    textfont=dict(size=22),
                    name="🧬 PE Enhancer (Put)",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<br>Elasticity confirms Put<extra></extra>"
                ), row=1, col=1)

                
                
                call_pe_mask = (intraday["Call_FirstEntry_Emoji"] == "🎯") & (intraday["Call_PE"] > intraday["Put_PE"])
                
                fig.add_trace(go.Scatter(
                    x=intraday.loc[call_pe_mask, "Time"],
                    y=intraday.loc[call_pe_mask, "F_numeric"] + 44,
                    mode="text",
                    text=["🧬"] * call_pe_mask.sum(),
                    textposition="top center",
                    textfont=dict(size=22),
                    name="🧬 PE Enhancer (Call)",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<br>Elasticity confirms Call<extra></extra>"
                ), row=1, col=1)



                # 🏹 Call Enhancer
                fig.add_trace(go.Scatter(
                    x=cross_aid_times_call,
                    y=cross_aid_prices_call ,
                    mode="text",
                    text=["🏹"] * len(cross_aid_times_call),
                    textposition="top center",
                    textfont=dict(size=20),
                    name="🏹 PE Cross Enhancer (Call)",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<br>PE_Cross_Bull within ±3 bars<extra></extra>"
                ), row=1, col=1)
                
                # 🏹 Put Enhancer
                fig.add_trace(go.Scatter(
                    x=cross_aid_times_put,
                    y=cross_aid_prices_put,
                    mode="text",
                    text=["🏹"] * len(cross_aid_times_put),
                    textposition="top center",
                    textfont=dict(size=20),
                    name="🏹 PE Cross Enhancer (Put)",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<br>PE_Cross_Bear within ±3 bars<extra></extra>"
                ), row=1, col=1)



                # 💥 Volatility Enhancer (Call)
                fig.add_trace(go.Scatter(
                    x=vol_aid_times_call,
                    y=vol_aid_prices_call,
                    mode="text",
                    text=["💥"] * len(vol_aid_times_call),
                    textposition="top center",
                    textfont=dict(size=20),
                    name="💥 Volatility Enhancer (Call)",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<br>Volatility surge detected<extra></extra>"
                ), row=1, col=1)
                
                # 💥 Volatility Enhancer (Put)
                fig.add_trace(go.Scatter(
                    x=vol_aid_times_put,
                    y=vol_aid_prices_put,
                    mode="text",
                    text=["💥"] * len(vol_aid_times_put),
                    textposition="top center",
                    textfont=dict(size=20),
                    name="💥 Volatility Enhancer (Put)",
                    hovertemplate="Time: %{x}<br>F%%: %{y}<br>Volatility surge detected<extra></extra>"
                ), row=1, col=1)
                vault_mask_dove = intraday["Vault_Emoji"] == "🕊️"
                vault_mask_crow = intraday["Vault_Emoji"] == "🐦‍⬛"
                
                # 🕊️ Dove above Kijun
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[vault_mask_dove, "Time"],
                        y=intraday.loc[vault_mask_dove, "Kijun_F"] + 60,  # offset +20 F% (tune if needed)
                        mode="text",
                        text=intraday.loc[vault_mask_dove, "Vault_Emoji"],
                        textposition="bottom center",
                        textfont=dict(size=32, color="purple"),
                        name="Bull Vault 🕊️",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    ),
                    row=1, col=1
                )
                
                # 🐦‍⬛ Crow below Kijun
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[vault_mask_crow, "Time"],
                        y=intraday.loc[vault_mask_crow, "Kijun_F"] - 60,  # offset -20 F% (tune if needed)
                        mode="text",
                        text=intraday.loc[vault_mask_crow, "Vault_Emoji"],
                        textposition="top center",
                        textfont=dict(size=32, color="black"),
                        name="Bear Vault 🐦‍⬛",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    ),
                    row=1, col=1
)
                # 🚪 T0 markers
                t0_mask = intraday["T0_Emoji"] == "🚪"
                
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[t0_mask, "Time"],
                        y=intraday.loc[t0_mask, "F_numeric"] - 3,
                        mode="text",
                        text=intraday.loc[t0_mask, "T0_Emoji"],
                        textposition="middle right",   # shift a bit for readability
                        textfont=dict(size=20, color="orange"),
                        name="T0 Gate",
                        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Phase: T0 🚪<extra></extra>"
                    ),
                    row=1, col=1
                )

                 # 🏇🏼 Marengo T1 Marker (acceleration after Entry 1)
                t1_mask = intraday["T1_Emoji"] == "⏩"
                
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[t1_mask, "Time"],
                        y=intraday.loc[t1_mask, "F_numeric"],
                        mode="text",
                        text=["🏇🏼"] * t1_mask.sum(),  # you can swap 🏇🏼 for ⏩ if you prefer
                        textposition="top center",
                        textfont=dict(size=32, color="black"),
                        name="Marengo T1",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>🏇🏼"
                    ),
                    row=1, col=1
                )


             # ⚡ Marengo T2 marker
                t2_mask = intraday["T2_Emoji"] == "⚡"
                
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[t2_mask, "Time"],
                        y=intraday.loc[t2_mask, "F_numeric"] - 10,
                        mode="text",
                        text=["⚡"] * t2_mask.sum(),
                        textposition="middle center",
                        textfont=dict(size=26, color="orange"),
                        name="Marengo T2",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                    ),
                    row=1, col=1
                )
            # ⚡ Parallel markers
                parallel_mask = intraday["Parallel_Emoji"] == "⚡"
                
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[parallel_mask, "Time"],
                        y=intraday.loc[parallel_mask, "F_numeric"] + 33,
                        mode="text",
                        text=["⚡"] * parallel_mask.sum(),   # ❌ numpy.int64 can break this

                        textposition="top center",
                        textfont=dict(size=21, color="orange"),
                        name="Parallel ⚡",
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>Crossed Ear/Nose Line 🎧<extra></extra>"                    ),
                    row=1, col=1
                )

                    # 💰 Goldmine markers
                # goldmine_mask = intraday["Goldmine_Emoji"] == "💰"
                
                # fig.add_trace(
                #     go.Scatter(
                #         x=intraday.loc[goldmine_mask, "Time"],
                #         y=intraday.loc[goldmine_mask, "F_numeric"] + 33,   # offset so emoji floats above
                #         mode="text",
                #         text=["💰"] * goldmine_mask.sum(),
                #         textposition="top center",
                #         textfont=dict(size=40, color="gold"),
                #         name="Goldmine 💰",
                #         hovertemplate="Time: %{x}<br>F%: %{y}<br>Reached the Goldmine 💰<extra></extra>"
                #     ),
                #     row=1, col=1
                # )


                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Headphone_Cross_Y"],
                    text=intraday["Headphone_Cross_Emoji"],
                    mode="text",
                    textfont=dict(size=24),
                    name="🎧 Ear Cross",
                    hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                ), row=1, col=1)

 
                # 🐘 Nose Line Cross
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Elephant_Cross_Y"],
                    text=intraday["Elephant_Cross_Emoji"],
                    mode="text",
                    textfont=dict(size=24),
                    name="🐘 Nose Cross",
                    hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                ), row=1, col=1)


                gm_e2_mask = intraday["Goldmine_E2_Emoji"] == "💰"
                gm_t1_mask = intraday["Goldmine_T1_Emoji"] == "💰"
   
                fig.add_trace(go.Scatter(
                    x=intraday.loc[gm_e2_mask, "Time"],
                    y=intraday.loc[gm_e2_mask, "F_numeric"] + 33,   # float above price
                    mode="text",
                    text=["💰"] * int(gm_e2_mask.sum()),
                    textposition="top center",
                    textfont=dict(size=21, color="gold"),
                    name="Goldmine (E2)",
                    hovertemplate="E2 Goldmine<br>Time: %{x}<br>F%: %{y}<extra></extra>",
                ),
                   row=1, col=1
            )
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[gm_t1_mask, "Time"],
                        y=intraday.loc[gm_t1_mask, "F_numeric"] + 45,   # a bit higher so icons don’t overlap
                        mode="text",
                        text=["💰"] * int(gm_t1_mask.sum()),
                        textposition="top center",
                        textfont=dict(size=21, color="limegreen"),      # distinct hue
                        name="Goldmine (T1)",
                        hovertemplate="T1 Goldmine<br>Time: %{x}<br>F%: %{y}<extra></extra>",
                    ),
                    row=1, col=1
                )

            
                if yva_min is not None and yva_max is not None:
                    st.markdown(f"**📘 Yesterday’s Value Area**: {yva_min} → {yva_max}")
                if prev_close:
                    range_f_pct = round((prev_high - prev_low) / prev_close * 100, 1)
                    st.markdown(f"📏 Yesterday’s Range: **{prev_low:.2f} → {prev_high:.2f}** ({yesterday_range_str} pts | {range_f_pct}%)")


                # Force datetime index for safety
                intraday.index = pd.to_datetime(intraday.index)
                
                if yva_min is not None and yva_max is not None and prev_high is not None and prev_low is not None:
                    opening_price = intraday["Close"].iloc[0]
                    first_timestamp = intraday.index[0]
                    cutoff = first_timestamp + pd.Timedelta(minutes=30)
                    first_window = intraday[intraday.index < cutoff]
                
                    last_price_window = first_window["Close"].iloc[-1]
                
               

                fig.update_layout(
                    title=f"VOLMIKE.COM  - {start_date.strftime('%Y-%m-%d')}",
                    margin=dict(l=30, r=30, t=50, b=30),
                    height=1080,  # Increase overall figure height (default ~450-600)
                    showlegend=False,
                

                     
                )


            
                st.plotly_chart(fig, use_container_width=True)


     

            except Exception as e:
                st.error(f"Error fetching data for {t}: {e}")
            

  
            
       
