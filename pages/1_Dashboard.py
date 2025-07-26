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

    # (optional) flag collapse
    if va_min == va_max:
        st.warning("⚠️ Value area collapsed to one level – "
                   "range too narrow, even after adaptive binning.")

    return va_min, va_max, profile_df

 
# =================
# Page Config
# =================
st.set_page_config(
    page_title="Volmike.com",
    layout="wide"
)


st.title("VOLMIKE.COM")

# ======================================
# Sidebar - User Inputs & Advanced Options
# ======================================
st.sidebar.header("Input Options")

default_tickers = ["QQQ","SPY","NVDA","MU", "AVGO","MSFT","PLTR","AMD","MRVL","uber","AMZN","AAPL","googl","META","SMCI","nke","GM","c","wfc","hood","coin","bac","jpm","HIMS","TXM","QCOM","MU","INTC","CRDO","RMBS","ON","ORCL", "CRWD","PANW","APP","MSTR","IBM","AMAT","DELL","WDC","CRM","CHWY","ETSY","CART","W"]
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
                    df["Dollar_Move_From_F"] = (df["F%_Move"] / 10_000) * spot_open
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
                    df["Call_BBW_Tight_Emoji"] = df["Call_BBW_Is_Tight"].rolling(5).apply(lambda x: x.sum() >= 3).fillna(0).astype(bool).map({True: "🐝", False: ""})
                    df["Put_BBW_Tight_Emoji"]  = df["Put_BBW_Is_Tight"] .rolling(5).apply(lambda x: x.sum() >= 3).fillna(0).astype(bool).map({True: "🐝", False: ""})

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


                
               











                if gap_alert:
                    st.warning(gap_alert)








                    
                with st.expander("Show/Hide Data Table",  expanded=False):
                                # Show data table, including new columns
                    cols_to_show = [
                                    "RVOL_5","Range","Time","Volume","F_numeric","Unit%","Vector%","Unit Velocity","Velocity","Unit Acceleration","Acceleration","Unit Momentum","Vector Momentum","Unit Force","Vector Force","Unit Energy","Vector Energy","Force_per_Range","Force_per_3bar_Range","Unit_Energy_per_Range","Vector_Energy_per_3bar_Range" ]

                    st.dataframe(intraday[cols_to_show])

                ticker_tabs = st.tabs(["Mike Plot", "Mike Table"])



                

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



                  
                                    # Add earliest Time seen in each F% bin
                  bin_times = intraday.groupby('F_Bin')['Time'].min().reset_index()
                  bin_times['F% Level'] = bin_times['F_Bin'].astype(int)
                  profile_df = profile_df.merge(bin_times[['F% Level', 'Time']], on='F% Level', how='left')

                  # Current Mike value (latest row)
                  current_mike = intraday[mike_col].iloc[-1]
                  
                  # Compute min/max of value area for better boundary
                  va_min = min(value_area_levels)
                  va_max = max(value_area_levels)
                  



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
                  st.dataframe(profile_df[["F% Level","Time", "Letters",  "%Vol","💥","Tail","✅ ValueArea","🦻🏼", "👃🏽"]])

                
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

                
 
                # Display anchor info
                st.write(f"🐻 **Bearish Anchor:** {anchor_time_bear.strftime('%I:%M %p')} — Price: {round(anchor_price_bear, 2)}")
                st.write(f"🐂 **Bullish Anchor:** {anchor_time_bull.strftime('%I:%M %p')} — Price: {round(anchor_price_bull, 2)}")
  
  
  
  



                # with st.expander("🕯️ Hidden Candlestick + Ichimoku View", expanded=True):
                #               fig_ichimoku = go.Figure()
              
                #               fig_ichimoku.add_trace(go.Candlestick(
                #                   x=intraday['Time'],
                #                   open=intraday['Open'],
                #                   high=intraday['High'],
                #                   low=intraday['Low'],
                #                   close=intraday['Close'],
                #                   name='Candles'
                #               ))
              
                #               fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Tenkan'], line=dict(color='red'), name='Tenkan-sen'))
                #               fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Kijun'], line=dict(color='green'), name='Kijun-sen'))
                #               fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanA'], line=dict(color='yellow'), name='Span A'))
                #               fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['SpanB'], line=dict(color='blue'), name='Span B'))
                #               fig_ichimoku.add_trace(go.Scatter(x=intraday['Time'], y=intraday['Chikou'], line=dict(color='purple'), name='Chikou'))
              
                #               fig_ichimoku.add_trace(go.Scatter(
                #                   x=intraday['Time'],
                #                   y=intraday['SpanA'],
                #                   line=dict(width=0),
                #                   showlegend=False
                #               ))
              
                #               fig_ichimoku.add_trace(go.Scatter(
                #                   x=intraday['Time'],
                #                   y=intraday['SpanB'],
                #                   fill='tonexty',
                #                   fillcolor='rgba(128, 128, 128, 0.2)',
                #                   line=dict(width=0),
                #                   showlegend=False
                #               ))
              
                #               fig_ichimoku.update_layout(
                #                   title="Ichimoku Candlestick Chart",
                #                   height=450,
                #                   width=450,
                #                   xaxis_rangeslider_visible=False,
                #                   margin=dict(l=30, r=30, t=40, b=20)
                #               )
              
                #               st.plotly_chart(fig_ichimoku, use_container_width=True)

                with st.expander("🪞 MIDAS Anchor Table", expanded=False):
                                    st.dataframe(
                                        intraday[[
                                            'Time', price_col, 'Volume',
                                            'MIDAS_Bear', 'MIDAS_Bull',"Bear_Displacement","Bull_Displacement", "Bull_Lethal_Accel", "Bear_Lethal_Accel","Bear_Displacement_Double","Bull_Displacement_Change","Bear_Displacement_Change",
                                            'MIDAS_Bull_Hand', 'MIDAS_Bear_Glove',"Hold_Call","Hold_Put",
                                            'Bull_Midas_Wake', 'Bear_Midas_Wake'
                                        ]]
                                        .dropna(subset=['MIDAS_Bear', 'MIDAS_Bull'], how='all')
                                        .reset_index(drop=True)
                                    )
                
                
                                   
                               # Convert Unit% to numeric
                intraday["Unit%_Numeric"] = (
                    intraday["Unit%"].str.replace("%", "", regex=False).replace("", "0").astype(float)
                )
                
                # Calculate cumulative sum
                intraday["Cumulative_Unit"] = intraday["Unit%_Numeric"].cumsum()



             
                with st.expander("🧠 Mike's Physics Engine – Displacement Plot", expanded=False):
                   
                    fig_displacement = go.Figure()

                    # === Cumulative Unit% Line ===
                    fig_displacement.add_trace(go.Scatter(
                        x=intraday["Time"],
                        y=intraday["Cumulative_Unit"],
                        mode="lines",
                        name="Cumulative Unit%",
                        line=dict(color="dodgerblue", width=2),
                        hovertemplate="Time: %{x}<br>Cumulative Unit%: %{y:.1f}<extra></extra>"
                    ))
                
                    # === Zero Reference Line ===
                    fig_displacement.add_trace(go.Scatter(
                        x=intraday["Time"],
                        y=[0] * len(intraday),
                        mode="lines",
                        name="Zero Line",
                        line=dict(color="gray", width=1, dash="dot"),
                        showlegend=False
                    ))

                   # === ADD: Vector% Markers ===
                    vector_rows = intraday[intraday["Vector%"].str.endswith("%")].copy()
                    vector_rows["Vector%_numeric"] = vector_rows["Vector%"].str.replace("%", "").astype(int)
                    
                    fig_displacement.add_trace(go.Scatter(
                        x=vector_rows["Time"],
                        y=vector_rows["Cumulative_Unit"],
                        mode="markers+text",
                        name="Vector%",
                        marker=dict(
                            size=9,
                            color=["limegreen" if val > 0 else "orangered" for val in vector_rows["Vector%_numeric"]],
                            symbol="circle"
                        ),
                        text=[f"{val:+}%" for val in vector_rows["Vector%_numeric"]],
                        textposition="top center",
                        textfont=dict(size=9),
                        hovertemplate="Vector%: %{text}<br>Time: %{x}<extra></extra>"
                    ))

                    
                                      
                    # === Layout ===
                    fig_displacement.update_layout(
                        height=350,
                        plot_bgcolor="black",
                        paper_bgcolor="black",
                        font=dict(color="white"),
                        title="🧠 Mike's Physics Engine – Flight Path via Cumulative Unit%",
                        xaxis=dict(title="Time"),
                        yaxis=dict(title="Cumulative Unit (×10⁻⁴)", zeroline=True),
                        margin=dict(t=40, b=40),
                        legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
                    )
                              
                st.plotly_chart(fig_displacement, use_container_width=True)

             




                           



                with ticker_tabs[0]:
                    # -- Create Subplots: Row1=F%, Row2=Momentum
                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        vertical_spacing=0.03,
                        row_heights=[0.60, 0.20, 0.20],  
                        subplot_titles=("F% Structure", "Option Flow (Call/Put)","Option vs MIDAS"),
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

                    # # === Overlay Colored Top Dots ===
                    # colored_dots = go.Scatter(
                    #     x=top_dots["Time"],
                    #     y=top_dots["F_numeric"],
                    #     mode="markers",
                    #     marker=dict(
                    #         size=8,
                    #         color=top_dots["DotColor"],
                    #         symbol="circle"
                    #     ),
                    #     name="15-min Directional Dot",
                    #     hovertemplate="Time: %{x}<br>Top F%: %{y:.2f}<extra></extra>",
                    # )
                    # fig.add_trace(colored_dots, row=1, col=1)
                                        
                    #             # === Define Dots ===
                    # main_dot = go.Scatter(
                    #     x=top_dots["Time"],
                    #     y=top_dots["F_numeric"],
                    #     mode="markers",
                    #     marker=dict(
                    #         color=top_dots["DotColor"],
                    #         size=8,
                    #         symbol="circle"
                    #     ),
                    #     name="15-min Top Dot",
                    #     hovertemplate="Bracket End: %{x}<br>F%: %{y}<extra></extra>"
                    # )
                    
                    # ghost_dot = go.Scatter(
                    #     x=top_dots["Time_HighMoment"],
                    #     y=top_dots["F_numeric"],
                    #     mode="markers",
                    #     marker=dict(
                    #         color=top_dots["DotColor"],
                    #         size=20,
                    #         symbol="circle-open",
                    #         opacity=0.4
                    #     ),
                    #     name="Ghost Dot 🫥",
                    #     hovertemplate="Auction Push: %{x}<br>F%: %{y}<extra></extra>"
                    # )
                    
                    # === Now Add to Plot ===
                    # fig.add_trace(main_dot, row=1, col=1)
                    # fig.add_trace(ghost_dot, row=1, col=1)
                    #**************************************************************************************************************************************************************************


                                                        # 🟢 40% RETRACEMENT




#**************************************************************************************************************************************************************************

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

                    # # Plot without shifting time
                    # chikou_line = go.Scatter(
                    #     x=chikou_plot["Time"],
                    #     y=chikou_plot["Chikou_F"],
                    #     mode="lines",
                      
                    #     name="Chikou (F%)",
                    #     line=dict(color="purple", dash="dash", width=1)
                    # )
                    # fig.add_trace(chikou_line, row=1, col=1)

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


                    # intraday["SpanA_F"] = ((intraday["SpanA"] - prev_close) / prev_close) * 10000
                    # intraday["SpanB_F"] = ((intraday["SpanB"] - prev_close) / prev_close) * 10000



                    #                     # Span A – Yellow Line
                    # span_a_line = go.Scatter(
                    #     x=intraday["Time"],
                    #     y=intraday["SpanA_F"],
                    #     mode="lines",
                    #     line=dict(color="yellow", width=0.4),
                    #     name="Span A (F%)"
                    # )
                    # fig.add_trace(span_a_line, row=1, col=1)

                    # # Span B – Blue Line
                    # span_b_line = go.Scatter(
                    #     x=intraday["Time"],
                    #     y=intraday["SpanB_F"],
                    #     mode="lines",
                    #     line=dict(color="blue", width=0.4),
                    #     name="Span B (F%)"
                    # )
                    # fig.add_trace(span_b_line, row=1, col=1)

                    # # Invisible SpanA for cloud base
                    # fig.add_trace(go.Scatter(
                    #     x=intraday["Time"],
                    #     y=intraday["SpanA_F"],
                    #     line=dict(width=0),
                    #     mode='lines',
                    #     showlegend=False
                    # ), row=1, col=1)

                    # # SpanB with fill → grey Kumo
                    # fig.add_trace(go.Scatter(
                    #     x=intraday["Time"],
                    #     y=intraday["SpanB_F"],
                    #     fill='tonexty',
                    #     fillcolor='rgba(128, 128, 128, 0.25)',  # transparent grey
                    #     line=dict(width=0),
                    #     mode='lines',
                    #     name='Kumo Cloud'
                    # ), row=1, col=1)


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
                        line=dict(dash="solid", color="#bebebe",width=0.4),
                        name="Upper Band"
                    )

                    # (C) Lower Band
                    lower_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% Lower"],
                        mode="lines",
                        line=dict(dash="solid", color="#bebebe",width=0.4),
                        name="Lower Band"
                    )

                    # (D) Moving Average (Middle Band)
                    middle_band = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F% MA"],
                        mode="lines",
                        line=dict(dash="dash",color="#d3d3d3",width=0.4),  # Set dash style
                        name="Middle Band (14-MA)"
                    )


                    # #                   # (E) Compliance Shift Bubbles on Main Plot
                    # shift_bubbles = go.Scatter(
                    #     x=intraday["Time"],
                    #     showlegend=False,

                    #     y=intraday["F%"].where(intraday["Compliance Shift"] == "🫧") + 50,
                    #     mode="markers",
                    #     marker=dict(size=30, symbol="circle", color="#00ccff", line=dict(color="white", width=1)),
                    #     name="🫧 Compliance Shift",
                    # )
                    
                   # (E) Compliance Shift Bubbles on Main Plot
                    shift_bubbles = go.Scatter(
                        x=intraday["Time"],
                        y=intraday["F%"].where(intraday["Compliance Shift"] == "🫧"),
                        mode="markers",
                        marker=dict(size=14, symbol="circle", color="#00ccff", line=dict(color="white", width=1)),
                        name="🫧 Compliance Shift",
                        hovertemplate="Time: %{x|%H:%M}<br>F%%: %{y:.2f}<extra></extra>"
                    )
                    
                    fig.add_trace(shift_bubbles, row=1, col=1)

                   
                    # # Create a Boolean mask for rows with surge emojis
                    # mask_compliance_surge = intraday["Compliance Surge"] != ""
                    
                    # # Plot Surge Emojis using same structure as BBW Alert
                    # scatter_compliance_surge = go.Scatter(
                    #     x=intraday.loc[mask_compliance_surge, "Time"],
                    #     y=intraday.loc[mask_compliance_surge, "F_numeric"] - 24,  # Offset slightly below F%
                    #     mode="text",
                    #     text=intraday.loc[mask_compliance_surge, "Compliance Surge"],
                    #     textposition="top center",
                    #     textfont=dict(size=14),
                    #     name="Compliance Surge",
                    #     hovertemplate="Time: %{x}<br>Surge: %{text}<extra></extra>"
                    # )
                    
                    # fig.add_trace(scatter_compliance_surge, row=1, col=1)


                    # (G) Distensibility Alert on Main Plot
                    
                    # # Mask bars that triggered the 🪟 emoji
                    # mask_distensibility = intraday["Distensibility Alert"] != ""
                    
                    # # Plot emoji above price (or F_numeric)but rangebut range
                    # scatter_distensibility = go.Scatter(
                    #     x=intraday.loc[mask_distensibility, "Time"],
                    #     y=intraday.loc[mask_distensibility, "F_numeric"] + 70,  # Slight offset upward
                    #     mode="text",
                    #     text=intraday.loc[mask_distensibility, "Distensibility Alert"],
                    #     textposition="top center",
                    #     textfont=dict(size=24),
                    #     name="Distensibility 🪟",
                    #     hovertemplate="Time: %{x|%H:%M}<br>Distensibility: %{customdata:.2f}<extra></extra>",
                    #     customdata=intraday.loc[mask_distensibility, "Distensibility"]
                    # )
                    
                    # fig.add_trace(scatter_distensibility, row=1, col=1)


                    # Create a Boolean mask for rows with Stroke Growth ⭐ emojis
                    # mask_stroke_growth = intraday["Stroke Growth ⭐"] != ""
                    
                    # # Plot Stroke Growth ⭐ emojis just below the F_numeric line
                    # scatter_stroke_growth = go.Scatter(
                    #     x=intraday.loc[mask_stroke_growth, "Time"],
                    #     y=intraday.loc[mask_stroke_growth, "F_numeric"] + 38,  # Adjust vertical offset if needed
                    #     mode="text",
                    #     text=intraday.loc[mask_stroke_growth, "Stroke Growth ⭐"],
                    #     textposition="top center",
                    #     textfont=dict(size=16),
                    #     name="Stroke Growth ⭐",
                    #     hovertemplate="Time: %{x}<br>⭐ Stroke Volume Growth<extra></extra>"
                    # )
                    
                    # fig.add_trace(scatter_stroke_growth, row=1, col=1)

                  

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
                        textfont=dict(size=24),
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
                        textfont=dict(size=24),
                        textposition="bottom center",
                        name="South Marengo",
                        showlegend=True
                    )
                    
                    fig.add_trace(south_marengo_trace, row=1, col=1)


        

# # ------------------------

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

                     # 🎯 Add all lines to the F% plot
                    # fig.add_trace(y_open_f_line, row=1, col=1)
                    fig.add_trace(y_high_f_line, row=1, col=1)
                    fig.add_trace(y_low_f_line, row=1, col=1)
                    fig.add_trace(y_close_f_line, row=1, col=1)

                    # scatter_drum = go.Scatter(
                    # x=intraday.loc[intraday["🪘"] != "", "Time"],
                    # y=intraday.loc[intraday["🪘"] != "", "Drum_Y"],
                    # mode="text",
                    # text=intraday.loc[intraday["🪘"] != "", "🪘"],
                    # textposition="middle center",
                    # textfont=dict(size=20),
                    # name="Drum Cross",
                    # hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>"
                    # )
                
                    # fig.add_trace(scatter_drum, row=1, col=1)



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


                    # mask_adx_alert = intraday["ADX_Alert"] != ""

                    # scatter_adx_alert = go.Scatter(
                    #     x=intraday.loc[mask_adx_alert, "Time"],
                    #     y=intraday.loc[mask_adx_alert, "F_numeric"] + 10,  # Offset for visibility
                    #     mode="text",
                    #     text=intraday.loc[mask_adx_alert, "ADX_Alert"],
                    #     textposition="top center",
                    #     textfont=dict(size=11),
                    #     name="ADX Expansion Alert",
                    #     hovertemplate="Time: %{x}<br>ADX Ratio: %{customdata:.2f}<extra></extra>",
                    #     customdata=intraday.loc[mask_adx_alert, "ADX_Ratio"]
                    # )

                    # fig.add_trace(scatter_adx_alert, row=1, col=1)



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
                        y=intraday.loc[mask_atr_alert, "F_numeric"]  - 3,  # place above F%
                        mode="text",
                        textposition="bottom center",

                        text=intraday.loc[mask_atr_alert, "ATR_Exp_Alert"],
                        textfont=dict(size=8),
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

                    tdst_buy_mask = intraday["TDST"].str.contains("Buy TDST", na=False)
                    tdst_sell_mask = intraday["TDST"].str.contains("Sell TDST", na=False)

                                        # Step 1: For each Buy TDST bar, get the F% level
                    buy_tdst_levels = intraday.loc[tdst_buy_mask, "F_numeric"]

                    # Step 2: Loop through each Buy TDST and track from that point forward
                    for buy_idx, tdst_level in buy_tdst_levels.items():
                        # Get index location of the TDST signal
                        i = intraday.index.get_loc(buy_idx)

                        # Look at all bars forward from the TDST bar
                        future = intraday.iloc[i+1:].copy()

                        # Find where F% crosses and stays above the TDST level for 2 bars
                        above = future["F_numeric"] > tdst_level
                        two_bar_hold = above & above.shift(-1)

                        # Find the first time this happens
                        if two_bar_hold.any():
                            ghost_idx = two_bar_hold[two_bar_hold].index[0]  # first valid bar

                            # Plot 👻 emoji on the first bar
                            fig.add_trace(
                                go.Scatter(
                                    x=[intraday.at[ghost_idx, "Time"]],
                                    y=[intraday.at[ghost_idx, "F_numeric"] + 8],
                                    mode="text",
                                    text=["🔑"],
                                    textposition="middle center",
                                    textfont=dict(size=20, color="purple"),
                                    name="Confirmed Buy TDST Breakout",
                                    hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                                ),
                                row=1, col=1
                            )


                    # Step 1: Get all Sell TDST points (each defines its own world)
                    sell_tdst_levels = intraday.loc[tdst_sell_mask, "F_numeric"]
                    sell_tdst_indices = list(sell_tdst_levels.index) + [intraday.index[-1]]  # add end of session as last boundary

                    # Step 2: Loop through each Sell TDST "world"
                    for i in range(len(sell_tdst_levels)):
                        tdst_idx = sell_tdst_levels.index[i]
                        tdst_level = sell_tdst_levels.iloc[i]

                        # Define the domain: from this Sell TDST until the next one (or end of day)
                        domain_start = intraday.index.get_loc(tdst_idx) + 1
                        domain_end = intraday.index.get_loc(sell_tdst_indices[i+1])  # next TDST or end

                        domain = intraday.iloc[domain_start:domain_end]

                        # Condition: F% crosses below and stays below for 2 bars
                        below = domain["F_numeric"] < tdst_level
                        confirmed = below & below.shift(-1)

                        if confirmed.any():
                            ghost_idx = confirmed[confirmed].index[0]
                            fig.add_trace(
                                go.Scatter(
                                    x=[intraday.at[ghost_idx, "Time"]],
                                    y=[intraday.at[ghost_idx, "F_numeric"] - 8],
                                    mode="text",
                                    text=["👻"],
                                    textposition="middle center",
                                    textfont=dict(size=20, color="purple"),
                                    name="Confirmed Sell TDST Breakdown",
                                    hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                                ),
                                row=1, col=1
                            )



                   












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
                        textfont=dict(size=28),
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
                        textfont=dict(size=24),
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

             
                # Smooth first if needed
                intraday["Call_Option_Smooth"] = intraday["Call_Option_Value"].rolling(3).mean()
                intraday["Put_Option_Smooth"]  = intraday["Put_Option_Value"].rolling(3).mean()
                
                # 🎯 Call Flow
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Call_Option_Smooth"],
                    mode="lines",
                    name="Call Option Value",
                    line=dict(color="darkviolet", width=1.5),
                    showlegend=True,
                    hovertemplate=
                    "<b>Time:</b> %{x}<br>" +
                    "<b>Call Option:</b> %{y:.2f}<br>" +
                    "<b>%{text}</b><extra></extra>"
              

                ), row=2, col=1)
                
                # 🎯 Put Flow
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Put_Option_Smooth"],
                    mode="lines",
                    name="Put Option Value",
                    line=dict(color="darkcyan", width=1.5),
                    showlegend=True,
                    hovertemplate=
                    "<b>Time:</b> %{x}<br>" +
                    "<b>Put Option:</b> %{y:.2f}<br>" +
                    "<b>%{text}</b><extra></extra>"
            

                ), row=2, col=1)
                
                # Compute displacement from MIDAS curves
                intraday["Call_vs_Bull"] = intraday["Call_Option_Smooth"] - intraday["MIDAS_Bull"]
                intraday["Put_vs_Bear"] = intraday["Put_Option_Smooth"] - intraday["MIDAS_Bear"]
                
                # Plot them in Row 3
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Call_vs_Bull"],
                    mode="lines",
                    name="Call vs Midas Bull",
                    line=dict(color="darkviolet", width=1.5, dash="dot"),
                    showlegend=True
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Put_vs_Bear"],
                    mode="lines",
                    name="Put vs Midas Bear",
                    line=dict(color="darkcyan", width=1.5, dash="dot"),
                    showlegend=True
                ), row=3, col=1)



                




                #       # 🐅 Tiger markers on top of Call Option Value
                # fig.add_trace(go.Scatter(
                #     x=intraday["Time"],
                #     y=intraday["tiger],
                #     mode="text",
                #     text=intraday["Tiger"],
                #     textposition="top center",
                #     showlegend=False,
                #     name="Tiger"
                # ), row=2, col=1)
                
                # horse_df = intraday[intraday["Mike_Kijun_Horse_Emoji"] == "🏇🏽"]
                
                # # Add trace to your figure
                # fig.add_trace(go.Scatter(
                #     x=horse_df["TimeIndex"],
                #     y=horse_df["F_numeric"] + 10,
                #     mode="text",
                #     text=horse_df["Mike_Kijun_Horse_Emoji"],
                #     textposition="bottom left",
                #     textfont=dict(size=30),
                #     name="Mike x Kijun + Horse",
                #     showlegend=True
                # ))

                 


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
                # Plot 🚀 emoji markers when Mike crosses Kijun and ATR expansion occurred recently
             # Bullish crosses (🚀)

              
         



#                 # 🦻🏼 Add Ear line if it exists
#                 fig.add_hline(y=call_ib_high, showlegend=True,     
# line=dict(color="gold", dash="dot", width=0.6), row=2, col=1)
                
            # 🟡 Call IB High (hoverable)
                fig.add_trace(go.Scatter(
                    x=[intraday['TimeIndex'].min(), intraday['TimeIndex'].max()],
                    y=[call_ib_high, call_ib_high],
                    mode='lines',
                    line=dict(color="gold", dash="dot", width=0.6),
                    name="Call IB High",
                    hovertemplate="Call IB High: %{y:.2f}<extra></extra>"
                ), row=2, col=1)
                
                # 🟡 Call IB Low (hoverable)
                fig.add_trace(go.Scatter(
                    x=[intraday['TimeIndex'].min(), intraday['TimeIndex'].max()],
                    y=[call_ib_low, call_ib_low],
                    mode='lines',
                    line=dict(color="gold", dash="dot", width=0.6),
                    name="Call IB Low",
                    hovertemplate="Call IB Low: %{y:.2f}<extra></extra>"
                ), row=2, col=1)
                
                # # 🔷 Value Area Min (hoverable)
                # fig.add_trace(go.Scatter(
                #     x=[intraday['TimeIndex'].min(), intraday['TimeIndex'].max()],
                #     y=[va_min, va_min],
                #     mode='lines',
                #     line=dict(color="#0ff", dash="dot", width=0.5),
                #     name="VA Min",
                #     hovertemplate="Value Area Min: %{y:.2f}<extra></extra>"
                # ), row=1, col=1)
                
                # # 🔷 Value Area Max (hoverable)
                # fig.add_trace(go.Scatter(
                #     x=[intraday['TimeIndex'].min(), intraday['TimeIndex'].max()],
                #     y=[va_max, va_max],
                #     mode='lines',
                #     line=dict(color="#0ff", dash="dot", width=0.5),
                #     name="VA Max",
                #     hovertemplate="Value Area Max: %{y:.2f}<extra></extra>"
                # ), row=1, col=1)


     
                ib_third = (ib_high - ib_low) / 3
                ib_upper_third = ib_low + 2 * ib_third
                ib_middle_third = ib_low + ib_third
  
  
                fig.add_hline(y=ib_middle_third, line=dict(color="gray", dash="dash", width=0.5), row=1, col=1)
                fig.add_hline(y=ib_upper_third, line=dict(color="gray", dash="dash", width=0.5), row=1, col=1)
                

                fig.add_hline(y=va_min,showlegend=True, line=dict(color="#0ff", dash="dot", width=0.5), row=1, col=1)
                fig.add_hline(y=va_max,showlegend=True, line=dict(color="#0ff", dash="dot", width=0.5), row=1, col=1)

            
                # # Filter only rows where oxygen quality is rich
                # lungs_only = intraday[intraday["O2 Quality"] == "🫁"]


                # scatter_lungs = go.Scatter(
                # x=lungs_only["Time"],
                # y=lungs_only["F_numeric"] - 8,
                # mode="text",
                # text=lungs_only["O2 Quality"],  # Will just be 🫁
                # textposition="middle center",
                # textfont=dict(size=9),
                # name="Lungs Only",
                # hovertemplate="Time: %{x}<br>O₂: %{text}<extra></extra>"
                #   )

                # fig.add_trace(scatter_lungs, row=1, col=1)

             

                                
                # fig.add_trace(go.Scatter(x=intraday['TimeIndex'],showlegend=True, mode="lines", y=intraday['MIDAS_Bear'], name="MIDAS Bear", line=dict(color="pink", dash="solid", width=0.9)))
                # fig.add_trace(go.Scatter(x=intraday['TimeIndex'],showlegend=True, mode="lines", y=intraday['MIDAS_Bull'], name="MIDAS Bull",line=dict(color="pink", dash="solid", width=0.9)))



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
                # 🦻🏼 Add Ear line if it exists
                ear_row = profile_df[profile_df["🦻🏼"] == "🦻🏼"]
                
                if not ear_row.empty:
                    ear_level = ear_row["F% Level"].values[0]  # take the first (most recent) ear
                    fig.add_hline(
                        y=ear_level,
                        line=dict(color="darkgray", dash="dot", width=0.3),
                        row=1, col=1,
                        showlegend=True,
                        annotation_text="🦻🏼 Ear Shift",
                        annotation_position="top left",
                        annotation_font=dict(color="black")
                    )

                
   

                    # Step 1: Get the F% Level marked with 🦻🏼
                    ear_row = profile_df[profile_df["🦻🏼"] == "🦻🏼"]
                    
                    # if not ear_row.empty:
                    #     ear_level = ear_row["F% Level"].values[0]  # numeric
                    #     # Step 2: Find a matching row in intraday that hit that F% bin and came after the ear shift
                    #     # Convert F% bin to string to match 'F_Bin'
                    #     ear_bin_str = str(ear_level)
                    #     matching_rows = intraday[intraday["F_Bin"] == ear_bin_str]
                    
                    #     if not matching_rows.empty:
                    #         # Use the last known time this level was touched
                    #         last_touch = matching_rows.iloc[-1]
                    #         fig.add_trace(go.Scatter(
                    #             x=[last_touch["TimeIndex"]],
                    #             y=[last_touch["F_numeric"] + 10],  # small vertical offset
                    #             mode="text",
                    #             text=["🦻🏼"],
                    #             showlegend=True,
                    #             textposition="bottom center",
                    #             textfont=dict(size=24),
                    #             name="Ear Shift",
                    #             hovertemplate="Time: %{x}<br>🦻🏼: %{y}<br>%{text}"

                             
                    #         ))

                
                # Step: Add 👃🏽 marker into intraday at the bar where breakout happened
                # Get the F% level with the most letters
                max_letter_level = profile_df.loc[profile_df['Letter_Count'].idxmax(), 'F% Level']
                
                # Get the first row where current Mike broke away from that level
                breakout_row = intraday[np.digitize(intraday[mike_col], f_bins) - 1 != max_letter_level]
                if not breakout_row.empty:
                    first_break = breakout_row.iloc[0].name
                    intraday.loc[first_break, "Mike_Nose_Emoji"] = "👃🏽"


                # # Plot 👃🏽 emoji on the intraday plot
                # nose_df = intraday[intraday["Mike_Nose_Emoji"] == "👃🏽"]
                
                # fig.add_trace(go.Scatter(
                #     x=nose_df["TimeIndex"],
                #     y=nose_df["F_numeric"] + 10,  # Adjust position above Mike
                #     mode="text",
                #     text=nose_df["Mike_Nose_Emoji"],
                #     textposition="top center",
                #     textfont=dict(size=18),
                #     name="Mike breaks from Letter POC 👃🏽",
                #     showlegend=True
                # ))



                # Get the F% level with the most letters (temporal Point of Control)
                poc_f_level = profile_df.loc[profile_df['Letter_Count'].idxmax(), 'F% Level']
                # Find first row where Mike exits the POC level
                current_bins = np.digitize(intraday[mike_col], f_bins) - 1
                intraday["Current_F_Bin"] = f_bins[current_bins]
                
                breakout_row_nose = intraday[intraday["Current_F_Bin"] != poc_f_level]
                
                # Place 👃🏽 emoji on the first breakout
                if not breakout_row_nose.empty:
                    first_nose_index = breakout_row_nose.iloc[0].name
                    intraday.loc[first_nose_index, "Mike_Nose_Emoji"] = "👃🏽"
                
                nose_df = intraday[intraday["Mike_Nose_Emoji"] == "👃🏽"]
                
                fig.add_trace(go.Scatter(
                    x=nose_df["TimeIndex"],
                    y=nose_df["F_numeric"] + 10,
                    mode="text",
                    text=nose_df["Mike_Nose_Emoji"],
                    textposition="top center",
                    textfont=dict(size=18),
                    hovertemplate="👃🏽 Nose Line: %{y}<extra></extra>",

                    name="Mike breaks from Letter POC 👃🏽",
                    showlegend=True
                ))
                       


                # Get F% level (already stored in `poc_f_level`) and its earliest time
                nose_row = profile_df[profile_df["F% Level"] == poc_f_level]
                nose_time = nose_row["Time"].values[0] if not nose_row.empty else "N/A"
                

# 1. Add the pink dotted line as a shape (visual line)
                fig.add_hline(
                    y=poc_f_level,
                    showlegend=True,

                    line=dict(color="#ff1493", dash="dot", width=0.3),
                    row=1, col=1
                )
                
                
                fig.add_trace(go.Scatter(
                    x=[intraday["TimeIndex"].iloc[-1]],  # Just use the latest time or any valid x
                    y=[poc_f_level],
                    mode="markers+text",
                    marker=dict(size=0, color="#ff1493"),
                    text=["👃🏽 Nose (Most Price Acceptance)"],
                    textposition="top right",
                    name="👃🏽 Nose Line",
                    showlegend=True,
                    hovertemplate=(
                          "👃🏽 Nose Line<br>"
                          "F% Level: %{y}<br>"
                          f"Time: {nose_time}<extra></extra>"
                      )
                ), row=1, col=1)
                for _, row in profile_df.iterrows():
                    if row["Tail"] == "🪶":
                          # Get actual TimeIndex from intraday at this F% Level
                        time_row = intraday[intraday["F_Bin"] == str(row["F% Level"])]
                        if not time_row.empty:
                            time_at_level = time_row["TimeIndex"].iloc[0]  # earliest bar at this F% level
                
                            fig.add_trace(go.Scatter(
                                x=[time_at_level],
                                y=[row["F% Level"]],
                                mode="text",
                                text=["🪶"],
                                textposition="middle right",
                                textfont=dict(size=20),
                                showlegend=True,
                                name="🪶 Tail",  # <-- Add this line

                                hovertemplate=(
                                    "🪶 Tail<br>"
                                    f"F% Level: {row['F% Level']}<br>"
                                    f"Time: {row['Time']}<extra></extra>"
                                )
                            ), row=1, col=1)
  



                # # 🚀 Bullish cross (Mike crosses above Kijun with ATR expansion)
                # bullish_df = intraday[intraday["Mike_Kijun_ATR_Emoji"] == "🚀"]
                # fig.add_trace(go.Scatter(
                #     x=bullish_df["TimeIndex"] ,
                #     y=bullish_df["F_numeric"] + 14,
                #     mode="text",
                #     text=bullish_df["Mike_Kijun_ATR_Emoji"],
                #     textposition="top right",
                #     textfont=dict(size=20),
                #     name="Bullish Mike x Kijun + ATR 🚀",
                #     showlegend=True
                # ))


           


                # 🧨 Bearish cross (Mike crosses below Kijun with ATR expansion)
                bearish_df = intraday[intraday["Mike_Kijun_ATR_Emoji"] == "⚓️"]
                fig.add_trace(go.Scatter(
                    x=bearish_df["TimeIndex"],
                    y=bearish_df["F_numeric"] - 14,
                    mode="text",
                    text=bearish_df["Mike_Kijun_ATR_Emoji"],
                    textposition="bottom right",
                    textfont=dict(size=24),
                    name="Bearish Mike x Kijun + ATR ⚓️",
                    showlegend=True
                ))

                emoji_df = intraday[intraday["Mike_Kijun_Bee_Emoji"] == "🍯"]
  
                fig.add_trace(go.Scatter(
                    x=emoji_df["TimeIndex"],
                    y=emoji_df["F_numeric"] - 24,
                    mode="text",
                    text=emoji_df["Mike_Kijun_Bee_Emoji"],
                    textposition="top center",
                    textfont=dict(size=18),
                    name="Mike x Kijun + Bees",
                    showlegend=True
                ))
          
                # 🌋 Magma Pass Plot (Mike x Kijun + ATR Expansion)

                volcano_df = intraday[intraday["Mike_Kijun_ATR_Emoji"] == "🌋"]
                
                fig.add_trace(go.Scatter(
                    x=volcano_df["TimeIndex"],
                    y=volcano_df["F_numeric"] - 32,  # Slightly lower to avoid overlap with 🍯
                    mode="text",
                    text=volcano_df["Mike_Kijun_ATR_Emoji"],
                    textposition="top center",
                    textfont=dict(size=18),
                    name="Mike x Kijun + ATR Expansion",
                    showlegend=True
                ))

                #                 # 👋🏽 Bull MIDAS Hand = price breaks **above** the Bear MIDAS line (resistance)
                # bull_hand_rows = intraday[intraday["MIDAS_Bull_Hand"] == "👋🏽"]
                # fig.add_trace(go.Scatter(
                #     x=bull_hand_rows["TimeIndex"],
                #     y=bull_hand_rows["MIDAS_Bear"] + 3,  # Adjust for spacing above line
                #     mode="text",
                #     text=["👋🏽"] * len(bull_hand_rows),
                #     textposition="top right",
                #     textfont=dict(size=22),
                #     showlegend=False,
                #     hovertemplate=(
                #         "👋🏽 Bull MIDAS Breakout<br>"
                #         "Time: %{x|%I:%M %p}<br>"
                #         f"Bear MIDAS: {{y:.2f}}<extra></extra>"
                #     )
                # ), row=1, col=1)
                
                # # 🧤 Bear MIDAS Glove = price breaks **below** the Bull MIDAS line (support)
                # bear_glove_rows = intraday[intraday["MIDAS_Bear_Glove"] == "🧤"]
                # fig.add_trace(go.Scatter(
                #     x=bear_glove_rows["TimeIndex"],
                #     y=bear_glove_rows["MIDAS_Bull"] - 3,  # Adjust for spacing below line
                #     mode="text",
                #     text=["🧤"] * len(bear_glove_rows),
                #     textposition="bottom right",
                #     textfont=dict(size=22),
                #     showlegend=False,
                #     hovertemplate=(
                #         "🧤 Bear MIDAS Breakdown<br>"
                #         "Time: %{x|%I:%M %p}<br>"
                #         f"Bull MIDAS: {{y:.2f}}<extra></extra>"
                #     )
                # ), row=1, col=1)

                # # 🥊 Bear Lethal Acceleration = strong downward force after MIDAS Bear breach
                # bear_lethal_rows = intraday[intraday["Bear_Lethal_Accel"] == "🥊"]
                # fig.add_trace(go.Scatter(
                #     x=bear_lethal_rows["TimeIndex"],
                #     y=bear_lethal_rows["F_numeric"] - 20,  # Offset below Mike for clarity
                #     mode="text",
                #     text=["🥊"] * len(bear_lethal_rows),
                #     textposition="bottom right",
                #     textfont=dict(size=20),
                #     showlegend=False,
                #     hovertemplate=(
                #         "🥊 Bear Lethal Acceleration<br>"
                #         "Time: %{x|%I:%M %p}<br>"
                #         "F%: %{y:.2f}<extra></extra>"
                #     )
                # ), row=1, col=1)

                # # 🚀 Bull Lethal Acceleration = strong breakout upward beyond MIDAS Bull
                # bull_lethal_rows = intraday[intraday["Bull_Lethal_Accel"] == "🚀"]
                # fig.add_trace(go.Scatter(
                #     x=bull_lethal_rows["TimeIndex"],
                #     y=bull_lethal_rows["F_numeric"] + 20,  # Offset above Mike for clarity
                #     mode="text",
                #     text=["🚀"] * len(bull_lethal_rows),
                #     textposition="top right",
                #     textfont=dict(size=20),
                #     showlegend=False,
                #     hovertemplate=(
                #         "🚀 Bull Lethal Acceleration<br>"
                #         "Time: %{x|%I:%M %p}<br>"
                #         "F%: %{y:.2f}<extra></extra>"
                #     )
                # ), row=1, col=1)


                # Wake-up Emojis 📈
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Call_Option_Smooth"] ,
                    mode="text",
                    text=intraday["Call_Wake_Emoji"],
                    textposition="top center",
                    showlegend=False
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Put_Option_Smooth"],
                    mode="text",
                    text=intraday["Put_Wake_Emoji"] ,
                    textposition="bottom center",
                    showlegend=False
                ), row=2, col=1)

           
                fig.add_trace(go.Scatter(
                   x=intraday["Time"],
                   y=intraday["Call_vs_Bull"],
                   mode="text",
                   text=intraday["Bull_Midas_Wake"],
                   textposition="top center",
                   showlegend=False,
                   hoverinfo="skip"
                ), row=3, col=1)
        
                fig.add_trace(go.Scatter(
                    x=intraday["Time"],
                    y=intraday["Put_vs_Bear"],
                    mode="text",
                    textfont=dict(size=20),
                    text=intraday["Bear_Midas_Wake"],
                    textposition="bottom center",
                    showlegend=False,
                    hovertemplate="Put vs Bear MIDAS: %{y:.2f}<extra></extra>"
                ), row=3, col=1)

                               
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

                                
                #                    # Plot 💸 for breakout above IB High
                # high_break_df = intraday[intraday["IB_High_Break"] == "💸"]
                # fig.add_trace(go.Scatter(
                #     x=high_break_df["TimeIndex"],
                #     y=high_break_df["F_numeric"] + 30,
                #     mode="text",
                #     text=high_break_df["IB_High_Break"],
                #     textposition="top left",
                #     textfont=dict(size=24),
                #     name="Breakout Above IB 💸",
                #     showlegend=True,
                #     hovertemplate="Time: %{x}<br>💸 IB High Breakout"
                # ), row=1, col=1)
                
                # # Plot 🧧 for breakdown below IB Low
                # low_break_df = intraday[intraday["IB_Low_Break"] == "🧧"]
                # fig.add_trace(go.Scatter(
                #     x=low_break_df["TimeIndex"],
                #     y=low_break_df["F_numeric"] - 30,
                #     mode="text",
                #     text=low_break_df["IB_Low_Break"],
                #     textposition="bottom left",
                #     textfont=dict(size=24),
                #     name="Breakdown Below IB 🧧",
                #     showlegend=True,
                #     hovertemplate="Time: %{x}<br>🧧 IB Low Breakdown"
                # ), row=1, col=1)



              
                               # 🦻🏼 Top 3 Ear Lines based on %Vol
               
                #              # Step 1: Filter Ear-marked rows
                #                # 🦻🏼 Top 3 Ear Lines based on %Vol
                # top_ears = profile_df[profile_df["🦻🏼"] == "🦻🏼"].nlargest(3, "%Vol")
                
                # for _, row in top_ears.iterrows():
                #     ear_level = row["F% Level"]
                #     vol = row["%Vol"]
                #     time = row["Time"]  # Or row["TimeIndex"] if Time is not a string
                
                #     fig.add_hline(
                #         y=ear_level,
                #         line=dict(color="darkgray", dash="dot", width=1.5),
                #         row=1, col=1,
                #         showlegend=False,
                #         annotation_text="🦻🏼",
                #         annotation_position="top left",
                #         annotation_font=dict(color="black"),
              
                #     )


             
               # 🦻🏼 Add Ear line if it exists
                ear_row = profile_df[profile_df["🦻🏼"] == "🦻🏼"]
                
                if not ear_row.empty:
                    ear_level = ear_row["F% Level"].values[0]  # take the first (most recent) ear
                    fig.add_hline(
                        y=ear_level,
                        line=dict(color="darkgray", dash="dot", width=0.7),
                        row=1, col=1,
                        showlegend=True,
                        annotation_text="🦻🏼 Ear Shift",
                        annotation_position="top left",
                        annotation_font=dict(color="black")
                    )

                top_ears = profile_df.nlargest(3, "%Vol")
                
                x_hover = intraday["TimeIndex"].iloc[-1]  # Use last bar time for clean placement
                
                for _, row in top_ears.iterrows():
                    ear_level = row["F% Level"]
                    vol = row["%Vol"]
                    time = row["Time"]
                
                    fig.add_trace(go.Scatter(
                        x=[x_hover],
                        y=[ear_level],
                        mode="text",
                        text=["🥁"],
                        textposition="middle right",
                        hovertemplate=f"🥁 Top %Vol<br>%Vol: {vol:.2f}<br>Time: {time}<extra></extra>",
                        showlegend=False
                    ), row=1, col=1)
              
                                        
                top_price_levels = profile_df.nlargest(3, "Letter_Count")
                
                for _, row in top_price_levels.iterrows():
                    fig.add_annotation(
                        x=intraday["TimeIndex"].min(),  # far left
                        y=row["F% Level"],
                        text="💲",
                        showarrow=False,
                        font=dict(size=20),
                        xanchor="right",
                        yanchor="middle",
                        hovertext=f"💲 Time-Zone<br>Letters: {row['Letter_Count']}<br>First Seen: {row['Time']}",
                   
                        ax=0, ay=0
                    )

                

                    # mask_green_king = intraday["King_Signal"] == "👑"
                    # scatter_green_king = go.Scatter(
                    #     x=intraday.loc[mask_green_king, "Time"],
                    #     y=intraday.loc[mask_green_king, "F_numeric"] + 48,
                    #     mode="text",
                    #     text=["♔"] * mask_green_king.sum(),
                    #     textfont=dict(size=55, color="green"),
                    #     name="Green King Signal (♔)",
                    #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>👑 Green Kingdom Crowned ♔<extra></extra>"
                    # )


                    # mask_red_king = intraday["King_Signal"] == "🔻👑"
                    # scatter_red_king = go.Scatter(
                    #     x=intraday.loc[mask_red_king, "Time"],
                    #     y=intraday.loc[mask_red_king, "F_numeric"] - 48,
                    #     mode="text",
                    #     text=["♔"] * mask_red_king.sum(),
                    #     textfont=dict(size=55, color="red"),
                    #     name="Red King Signal (♔)",
                    #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>🔻👑 Red Kingdom Crowned ♔<extra></extra>"
                    # )


                    # fig.add_trace(scatter_green_king, row=1, col=1)
                    # fig.add_trace(scatter_red_king, row=1, col=1)


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
                    y=intraday.loc[mask_horse_buy, "F_numeric"] + 55,
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
                    y=intraday.loc[mask_horse_sell, "F_numeric"] - 55,
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

                # Add both to figure
                fig.add_trace(scatter_rook_up, row=1, col=1)
                fig.add_trace(scatter_rook_down, row=1, col=1)

                mask_tenkan_cross_up = (
                    (intraday["Tenkan_F"].shift(1) < intraday["MIDAS_Bull"].shift(1)) &
                    (intraday["Tenkan_F"] >= intraday["MIDAS_Bull"])
                )
                
                # # Create a new column with the emoji (optional but clean)
                # intraday["Tenkan_Midas_CrossUp"] = np.where(mask_tenkan_cross_up, "🧲", "")
                
                # # Scatter plot for 🫆 (slightly above F_numeric)
                # scatter_tenkan_cross_up = go.Scatter(
                #     x=intraday.loc[mask_tenkan_cross_up, "Time"],
                #     y=intraday.loc[mask_tenkan_cross_up, "F_numeric"] + 4,
                #     mode="text",
                #     text=intraday.loc[mask_tenkan_cross_up, "Tenkan_Midas_CrossUp"],
                #     textposition="top right",
                #     textfont=dict(size=24, color="orange"),
                #     name="Tenkan Cross MIDAS Bull (🧲)",
                #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Tenkan ↗ MIDAS Bull 🧲<extra></extra>"
                # )
                
                # # Add to figure
                # fig.add_trace(scatter_tenkan_cross_up, row=1, col=1)


 
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

                #             # Calculate Chikou relation to current price
                # intraday["Chikou_Position"] = np.where(intraday["Chikou"] > intraday["Close"], "above",
                #                             np.where(intraday["Chikou"] < intraday["Close"], "below", "equal"))

                # # Detect changes in Chikou relation
                # intraday["Chikou_Change"] = intraday["Chikou_Position"].ne(intraday["Chikou_Position"].shift())

                # # Filter first occurrence and changes
                # chikou_shift_mask = intraday["Chikou_Change"] & (intraday["Chikou_Position"] != "equal")

                # intraday["Chikou_Emoji"] = np.where(intraday["Chikou_Position"] == "above", "👨🏻‍✈️",
                #                             np.where(intraday["Chikou_Position"] == "below", "👮🏻‍♂️", ""))

                # mask_chikou_above = chikou_shift_mask & (intraday["Chikou_Position"] == "above")

                # fig.add_trace(go.Scatter(
                #     x=intraday.loc[mask_chikou_above, "Time"],
                #     y=intraday.loc[mask_chikou_above, "F_numeric"] + 64,
                #     mode="text",
                #     text=["👨🏻‍✈️"] * mask_chikou_above.sum(),
                #     textposition="top center",
                #     textfont=dict(size=34),
                #     name="Chikou Above Price",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Chikou moved above<extra></extra>"
                # ), row=1, col=1)

                # mask_chikou_below = chikou_shift_mask & (intraday["Chikou_Position"] == "below")

                # fig.add_trace(go.Scatter(
                #     x=intraday.loc[mask_chikou_below, "Time"],
                #     y=intraday.loc[mask_chikou_below, "F_numeric"] - 64,
                #     mode="text",
                #     text=["👮🏿‍♂️"] * mask_chikou_below.sum(),
                #     textposition="bottom center",
                #     textfont=dict(size=34),
                #     name="Chikou Below Price",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>Chikou moved below<extra></extra>"
                # ), row=1, col=1)





                cloud_mask = intraday["Heaven_Cloud"] == "☁️"
  
                fig.add_trace(go.Scatter(
                    x=intraday.loc[cloud_mask, "Time"],
                    y=intraday.loc[cloud_mask, "F_numeric"] + 89,
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
                    y=intraday.loc[drizzle_mask, "F_numeric"] + 89,  # Position below the bar
                    mode="text",
                    text=intraday.loc[drizzle_mask, "Drizzle_Emoji"],
                    textposition="bottom center",
                    textfont=dict(size=21),
                    name="Price Dropped Below Demand 🌧️",
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>Crossed Below Demand<extra></extra>"
                ), row=1, col=1)
  
                #          # Mask for Tenkan_F crossing down through MIDAS_Bear
                # mask_tenkan_cross_down = (
                #     (intraday["Tenkan_F"].shift(1) > intraday["MIDAS_Bear"].shift(1)) &
                #     (intraday["Tenkan_F"] <= intraday["MIDAS_Bear"])
                # )
                
                # # Create a new column with the emoji (optional but clean)
                # intraday["Tenkan_Midas_CrossDown"] = np.where(mask_tenkan_cross_down, "🕸️", "")
                
                # # Scatter plot for 🕸️ (slightly below F_numeric)
                # scatter_tenkan_cross_down = go.Scatter(
                #     x=intraday.loc[mask_tenkan_cross_down, "Time"],
                #     y=intraday.loc[mask_tenkan_cross_down, "F_numeric"] - 20,
                #     mode="text",
                #     text=intraday.loc[mask_tenkan_cross_down, "Tenkan_Midas_CrossDown"],
                #     textposition="bottom right",
                #     textfont=dict(size=28, color="black"),
                #     name="Tenkan Cross MIDAS Bear (🕸️)",
                #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Tenkan ↘ MIDAS Bear 🕸️<extra></extra>"
                # )
                
                # # Add to figure
                # fig.add_trace(scatter_tenkan_cross_down, row=1, col=1)


   
                # bullish_scout_mask = (intraday["scout_emoji"] == "🔦") & (intraday["scout_position"] > intraday["F_numeric"])
                
                # fig.add_trace(go.Scatter(
                #     x=intraday.loc[bullish_scout_mask, "TimeIndex"],
                #     y=intraday.loc[bullish_scout_mask, "scout_position"] + 5,
                #     mode="text",
                #     text=intraday.loc[bullish_scout_mask, "scout_emoji"],
                #     textposition="top center",
                #     textfont=dict(size=16, color="black"),
                #     name="Scout 🔦 (Bullish)",
                #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<extra>DMI Bullish Scout</extra>"
                # ), row=1, col=1)




                # bearish_scout_mask = (intraday["scout_emoji"] == "🔦") & (intraday["scout_position"] < intraday["F_numeric"])
                
                # fig.add_trace(go.Scatter(
                #     x=intraday.loc[bearish_scout_mask, "TimeIndex"],
                #     y=intraday.loc[bearish_scout_mask, "scout_position"] - 5,
                #     mode="text",
                #     text=intraday.loc[bearish_scout_mask, "scout_emoji"],
                #     textposition="bottom center",
                #     textfont=dict(size=16, color="black"),
                #     name="Scout 🔦 (Bearish)",
                #     hovertemplate="Time: %{x}<br>F%: %{y:.2f}<extra>DMI Bearish Scout</extra>"
                # ), row=1, col=1)



              

                            # ➤ 🪽 Wing Emoji (+DI near Kijun up-cross)
                wing_mask = intraday["wing_emoji"] == "🪽"
                fig.add_trace(go.Scatter(
                    x=intraday.loc[wing_mask, "TimeIndex"],
                    y=intraday.loc[wing_mask, "F_numeric"] + 35,
                    mode="text",
                    text=intraday.loc[wing_mask, "wing_emoji"],
                    textposition="top center",
                    textfont=dict(size=26, color="green"),
                    name="Wing 🪽",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<extra>+DI & Kijun Up</extra>"
                ), row=1, col=1)
              # ➤ 🐦‍⬛ Bat Emoji (-DI near Kijun down-cross)



                bat_mask = intraday["bat_emoji"] == "🪽"
                fig.add_trace(go.Scatter(
                    x=intraday.loc[bat_mask, "TimeIndex"],
                    y=intraday.loc[bat_mask, "F_numeric"] - 35,
                    mode="text",
                    text=intraday.loc[bat_mask, "bat_emoji"],
                    textposition="bottom center",
                    textfont=dict(size=26, color="red"),
                    name="Bat 🪽",
                    hovertemplate="Time: %{x}<br>F%: %{y:.2f}<extra>-DI & Kijun Down</extra>"
                ), row=1, col=1)

  
        
                mask_tk_sun = intraday["Tenkan_Kijun_Cross"] == "🦅"
                mask_tk_moon = intraday["Tenkan_Kijun_Cross"] == "🐦‍⬛"

                # 🌞 Bullish Tenkan-Kijun Cross (Sun Emoji)
                scatter_tk_sun = go.Scatter(
                    x=intraday.loc[mask_tk_sun, "Time"],
                    y=intraday.loc[mask_tk_sun, "F_numeric"] + 99,  # Offset for visibility
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
                    y=intraday.loc[mask_tk_moon, "F_numeric"] - 99,  # Offset for visibility
                    mode="text",
                    text="🌙",
                    textposition="bottom center",
                    textfont=dict(size=34),
                    name="Tenkan-Kijun Bearish Cross",
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>Tenkan Crossed Below Kijun<extra></extra>"
                )


   # # Add to the F% Plot
                fig.add_trace(scatter_tk_sun, row=1, col=1)
                fig.add_trace(scatter_tk_moon, row=1, col=1)
                # # 👼🏻 Bullish Sanyaku Kouten
                # mask_sanyaku_kouten = intraday["Sanyaku_Kouten"] == "🟩"
                
                # # 👺 Bearish Sanyaku Gyakuten
                # mask_sanyaku_gyakuten = intraday["Sanyaku_Gyakuten"] == "🟥"
                

                # # 👼🏻 Sanyaku Kouten marker (Bullish)
                # scatter_sanyaku_kouten = go.Scatter(
                #     x=intraday.loc[mask_sanyaku_kouten, "Time"],
                #     y=intraday.loc[mask_sanyaku_kouten, "F_numeric"] + 233,  # Lower offset
                #     mode="text",
                #     text="👼🏻",
                #     textposition="bottom center",
                #     textfont=dict(size=55),
                #     name="Sanyaku Kouten",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>👼🏻 Sanyaku Kouten (Bullish Reversal)<extra></extra>"
                # )
                
                # # 👺 Sanyaku Gyakuten marker (Bearish)
                # scatter_sanyaku_gyakuten = go.Scatter(
                #     x=intraday.loc[mask_sanyaku_gyakuten, "Time"],
                #     y=intraday.loc[mask_sanyaku_gyakuten, "F_numeric"] - 233,  # Lower offset
                #     mode="text",
                #     text="👺",
                #     textposition="top center",
                #     textfont=dict(size=55),
                #     name="Sanyaku Gyakuten",
                #     hovertemplate="Time: %{x}<br>F%: %{y}<br>👺 Sanyaku Gyakuten (Bearish Reversal)<extra></extra>"
                # )
                
                # # Add to figure
                # fig.add_trace(scatter_sanyaku_kouten, row=1, col=1)
                # fig.add_trace(scatter_sanyaku_gyakuten, row=1, col=1)





             

                # cross_points = intraday[intraday["Midas_Cross_IB_High"] == "🎷"]
                # fig.add_trace(go.Scatter(
                #     x=cross_points["Time"],
                #     y=cross_points[price_col] + 20,
                #     textfont=dict(size=34),
                #     mode="text",
                #     text=cross_points["Midas_Cross_IB_High"],
                #     textposition="top center",
                #     showlegend=True
                # ))
                
                # # 🎻 plot for Bear MIDAS crossing IB Low
                # bear_cross_points = intraday[intraday["Midas_Bear_Cross_IB_Low"] == "🎻"]
                # fig.add_trace(go.Scatter(
                #     x=bear_cross_points["Time"],
                #     y=bear_cross_points[price_col] - 20,
                #     textfont=dict(size=34),
                #     mode="text",
                #     text=bear_cross_points["Midas_Bear_Cross_IB_Low"],
                #     textposition="bottom center",
                #     showlegend=True
                # ))


          


                # # Get y-range from F_numeric (or your price_col)
                # y_min = intraday[price_col].min()
                # y_max = intraday[price_col].max()
                # margin = (y_max - y_min) * 0.05  # 15% buffer
                
                # fig.update_yaxes(range=[y_min - margin, y_max + margin], row=1, col=1)
                
                #Columns: price_col = 'F_numeric', BB_upper = 'BB_upper'

                # #Step 1: Get data range including Bollinger Bands
                # y_min = min(intraday[[price_col, 'F% Lower']].min())
                # y_max_data = max(intraday[[price_col, 'F% Upper']].max())
                
                # # Step 2: Account for emoji buffer (emojis plotted at y + 3)
                # emoji_buffer = 4
                # y_max = y_max_data + emoji_buffer
                
                # # Step 3: Margin
                # margin = (y_max - y_min) * 0.05
                
                # # Final update (only apply to first subplot)
                # fig.update_yaxes(range=[y_min - margin, y_max + margin], row=1, col=1)

                                            
                                
                


                # # 💀 plot for Bear Displacement Doubling
                # bear_double_points = intraday[intraday["Bear_Displacement_Double"] == "💀"]
                # fig.add_trace(go.Scatter(
                #     x=bear_double_points["Time"],
                #     y=bear_double_points[price_col] - 12,  # Lower than 🎻 to avoid overlap
                #     text=bear_double_points["Bear_Displacement_Double"],
                #     mode="text",
                #     textfont=dict(size=18),
                #     textposition="bottom center",
                #     showlegend=True,
                #     hovertemplate="Time: %{x}<br>Bear_Displacement_Double: %{y}<br>💀 Bear_Displacement_Double<extra></extra>"
                # ))
                # # 👑 plot for Bull Displacement Doubling
                # bull_double_points = intraday[intraday["Bull_Displacement_Double"] == "👑"]
                # fig.add_trace(go.Scatter(
                #     x=bull_double_points["Time"],
                #     y=bull_double_points[price_col] + 12,  # Lower than 💀 and 🎻 to avoid clutter
                #     text=bull_double_points["Bull_Displacement_Double"],
                #     mode="text",
                #     textfont=dict(size=10),
                #     textposition="bottom center",
                #     showlegend=False
                # ))


   # (A.1) 40ish Reversal (star markers)
                mask_40ish = intraday["40ish"] != ""
                scatter_40ish = go.Scatter(
                    x=intraday.loc[mask_40ish, "Time"],
                    y=intraday.loc[mask_40ish, "F_numeric"] + 89,
                    mode="markers",
                    marker_symbol="star",
                    marker_size=18,
                    marker_color="gold",
                    name="40ish Reversal",
                    text=intraday.loc[mask_40ish, "40ish"],

                    hovertemplate="Time: %{x}<br>F%: %{y}<br>%{text}"
                )
                fig.add_trace(scatter_40ish, row=1, col=1)

                up_high_mask = intraday["Y_High_Cross"] == "✈️"
                up_high_trace = go.Scatter(
                    x=intraday.loc[up_high_mask, "Time"],
                    y=intraday.loc[up_high_mask, "F_numeric"] + 40,
                    mode="text",
                    text=intraday.loc[up_high_mask, "Y_High_Cross"],
                    textposition="top center",
                    textfont=dict(size=28),
                    name="Cross Above Y-High (✈️)"
                )

                
                breach_mask = intraday["Y_Low_Cross"] == "🛟"
                breach_trace = go.Scatter(
                    x=intraday.loc[breach_mask, "Time"],
                    y=intraday.loc[breach_mask, "F_numeric"] - 40,  # Offset downward for clarity
                    mode="text",
                    text=intraday.loc[breach_mask, "Y_Low_Cross"],
                    textposition="bottom center",
                    textfont=dict(size=28),
                    name="Cross Below Y-Low (🛟)"
                )
                
                                
                recovery_mask = intraday["Y_Low_Cross"] == "🚣🏽"
                recovery_trace = go.Scatter(
                    x=intraday.loc[recovery_mask, "Time"],
                    y=intraday.loc[recovery_mask, "F_numeric"] + 40,  # Offset for visibility
                    mode="text",
                    text=intraday.loc[recovery_mask, "Y_Low_Cross"],
                    textposition="top center",
                    textfont=dict(size=28),
                    name="Cross Above Y-Low (🚣🏽)"
                )

                astronaut_points = intraday[intraday["Astronaut_Emoji"] == "👨🏽‍🚀"]

                scatter_astronaut = go.Scatter(
                    x=astronaut_points["Time"],
                    y=astronaut_points["F_numeric"] + 144,  # Higher offset
                    mode="text",
                    text=astronaut_points["Astronaut_Emoji"],
                    textposition="top center",
                    name="New Highs 👨🏽‍🚀",
                    textfont=dict(size=21),
                 )
 
#                     # Add to figure
                fig.add_trace(up_high_trace, row=1, col=1)
 


                    # Filter where the Astronaut or Moon emoji exist
                astronaut_points = intraday[intraday["Astronaut_Emoji"] != ""]

                scatter_astronaut = go.Scatter(
                    x=astronaut_points["Time"],
                    y=astronaut_points["F_numeric"] + 144,  # Offset so it floats higher
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
                    y=swimmer_points["F_numeric"] - -38,  # Offset downward so it floats below price
                    mode="text",
                    text=swimmer_points["Swimmer_Emoji"],  # Either 🏊🏽‍♂️ or 🦑
                    textposition="bottom center",
                    name="New Lows 🏊🏽‍♂️🦑",
                    textfont=dict(size=24),
                    showlegend=True
                )

                fig.add_trace(scatter_swimmer, row=1, col=1)


  
                # # intraday["SpanA_F"] = ((intraday["SpanA"] - prev_close) / prev_close) * 10000
                # # intraday["SpanB_F"] = ((intraday["SpanB"] - prev_close) / prev_close) * 10000
  
  
  
                #                     # Span A – Yellow Line
                # span_a_line = go.Scatter(
                #     x=intraday["Time"],
                #     y=intraday["SpanA_F"],
                #     mode="lines",
                #     line=dict(color="yellow", width=0.4),
                #     name="Span A (F%)"
                # )
                # fig.add_trace(span_a_line, row=1, col=1)
  
                # # Span B – Blue Line
                # span_b_line = go.Scatter(
                #     x=intraday["Time"],
                #     y=intraday["SpanB_F"],
                #     mode="lines",
                #     line=dict(color="rgba(0, 150, 255, 0.4)", width=0.5),                    
                #     name="Span B (F%)"
                # )
                # fig.add_trace(span_b_line, row=1, col=1)
  
                # # Invisible SpanA for cloud base
                # fig.add_trace(go.Scatter(
                #     x=intraday["Time"],
                #     y=intraday["SpanA_F"],
                #     line=dict(width=0),
                #     mode='lines',
                #     showlegend=False
                # ), row=1, col=1)
  
                # # SpanB with fill → grey Kumo
                # fig.add_trace(go.Scatter(
                #     x=intraday["Time"],
                #     y=intraday["SpanB_F"],
                #     fill='tonexty',
                #     fillcolor='rgba(20, 20, 30, 0.10)',  # transparent grey
                #     line=dict(width=0),
                #     mode='lines',
                #     name='Kumo Cloud'
                # ), row=1, col=1)

                # # if yva_min is not None and yva_max is not None:
                #     # Show in text
                #     st.markdown(f"**📘 Yesterday’s Value Area**: {yva_min} → {yva_max}")
                # if prev_close:
                #     range_f_pct = round((prev_high - prev_low) / prev_close * 100, 1)
                #     st.markdown(f"📏 Yesterday’s Range: **{prev_low:.2f} → {prev_high:.2f}** ({yesterday_range_str} pts | {range_f_pct}%)")
                       
                      # Show YVA and Yesterday Range
                # if yva_min is not None and yva_max is not None:
                #     st.markdown(f"**📘 Yesterday’s Value Area**: {yva_min:.2f} → {yva_max:.2f}")
                # if prev_close:
                #     range_f_pct = round((prev_high - prev_low) / prev_close * 100, 1)
                #     st.markdown(f"📏 Yesterday’s Range: **{prev_low:.2f} → {prev_high:.2f}** ({yesterday_range_str} pts | {range_f_pct}%)")
                
                # # 🧭 Opening Position vs YVA
                # if yva_min is not None and yva_max is not None:
                #     opening_price = intraday["Close"].iloc[0]
                
                #     if yva_min < opening_price < yva_max:
                #         yva_position_msg = "✅ Opened **within** Yesterday's Value Area"
                #     elif opening_price >= yva_max:
                #         yva_position_msg = "⬆️ Opened **above** Yesterday's Value Area"
                #     elif opening_price <= yva_min:
                #         yva_position_msg = "⬇️ Opened **below** Yesterday's Value Area"
                #     else:
                #         yva_position_msg = "⚠️ Could not determine opening position relative to YVA"
                
                #     st.markdown(f"### {yva_position_msg}")


                    #   # ✅ Detect Initiative Breakout from Yesterday’s Value Area
                    # if yva_min is not None and yva_max is not None and not intraday.empty:
                    #     opening_price = intraday["Close"].iloc[0]
                    #     opened_inside_yva = yva_min < opening_price < yva_max
                    
                    #     # First 30 min = first 6 bars on 5-min timeframe
                    #     first_6 = intraday.iloc[:6]
                    #     broke_above_yva = first_6["Close"].max() > yva_max
                    #     broke_below_yva = first_6["Close"].min() < yva_min
                    
                    #     if opened_inside_yva:
                    #         if broke_above_yva:
                    #             st.markdown("🚀 **Breakout Alert: Opened *inside* YVA → Broke *above* within 30 min**")
                    #         elif broke_below_yva:
                    #             st.markdown("🔻 **Breakout Alert: Opened *inside* YVA → Broke *below* within 30 min**")
                    #         else:
                    #             st.markdown("🟨 Opened inside YVA – No early breakout")
                    
                    #     else:
                    #         st.markdown("🟦 Market did *not* open inside YVA")
  
                        # ✅ Acceptance Outside of Previous Day's Range
                        # When price opens above yesterday's high OR below yesterday's low
                        # AND remains there throughout the first 30 minutes
                    
                    # opened_above_yh = opening_price > prev_high
                    # opened_below_yl = opening_price < prev_low
                    
                    # first_6 = intraday.iloc[:6]
                    # stayed_above_yh = (first_6["Close"] > prev_high).all()
                    # stayed_below_yl = (first_6["Close"] < prev_low).all()
                    
                    # if opened_above_yh and stayed_above_yh:
                    #     st.markdown("🟢 **ACCEPTANCE ABOVE Yesterday’s High: Breakout confirmed**")
                    
                    # elif opened_below_yl and stayed_below_yl:
                    #     st.markdown("🔴 **ACCEPTANCE BELOW Yesterday’s Low: Breakdown confirmed**")
                    
                    # elif opened_above_yh or opened_below_yl:
                    #     st.markdown("🟠 **Open Outside Range but NOT Accepted (possible fade or retest)**")
                        
                        


                fig.update_yaxes(title_text="Option Value", row=2, col=1)

   
 

                fig.update_layout(
                    title=f"{t} – VOLMIKE.COM – {start_date.strftime('%Y-%m-%d')}",
                    margin=dict(l=30, r=30, t=50, b=30),
                    height=1800,  # Increase overall figure height (default ~450-600)

                     
                )
 



                st.plotly_chart(fig, use_container_width=True)




            except Exception as e:
                st.error(f"Error fetching data for {t}: {e}")

           
                


          


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
        #             fillcolor='rgba(128, 128, 128, 0.05)'  # 5% opacity (very faint)
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
