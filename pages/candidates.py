import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Candidates Snapshot", layout="wide")

st.title("📊 Multi-Timeframe Snapshot")

# Define your tickers
selected_tickers = st.multiselect(
    "Select Tickers",
    ["SPY", "QQQ","TSLA","NVDA", "AVGO","AMD","PLTR","MRVL","uber","mu","crwd","AMZN","AAPL","googl","MSFT","META","tsla","sbux","nke","chwy","DKNG","GM","cmg","c","wfc","hood","coin","bac","jpm","PYPL","tgt","wmt","elf"],
    default=["MSFT", "NVDA", "TSLA"]
)

# Define your timeframes (cleaned)
intervals = ["2m","5m","15m"]

# Date range selection like original
today = datetime.today()
one_year_ago = today - timedelta(days=365)
date_range = st.date_input("Select Date Range:", value=[today - timedelta(days=5), today])

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a start and end date.")
    st.stop()

# Timezone conversion helper
def format_time_ny(utc_timestamp):
    eastern = pytz.timezone('US/Eastern')
    if utc_timestamp.tzinfo is None:
        utc_timestamp = utc_timestamp.tz_localize('UTC')
    else:
        utc_timestamp = utc_timestamp.tz_convert('UTC')
    local_time = utc_timestamp.astimezone(eastern)
    return local_time.strftime('%H:%M')

def calculate_f(intraday_df, prev_close):
    if prev_close is not None and not intraday_df.empty:
        intraday_df["F%"] = ((intraday_df["Close"] - prev_close) / prev_close) * 10000
    else:
        intraday_df["F%"] = 0
    return intraday_df



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

# Run Analysis Button
if st.button("Run Analysis"):
    if selected_tickers:
        for interval in intervals:
            st.subheader(f"⏱ Interval: {interval}")

            combined_data = []

            for ticker in selected_tickers:
                try:
                    intraday_start_date = start_date - timedelta(days=1)

                    df = yf.download(ticker, interval=interval, start=start_date, end=end_date, progress=False)

                    hist_daily = yf.download(ticker, interval="1d", period="7d", progress=False)
                    if (
                        isinstance(hist_daily, pd.DataFrame)
                        and 'Close' in hist_daily.columns
                        and len(hist_daily) >= 2
                    ):
                        prev_close = hist_daily['Close'].iloc[-2].item()
                    else:
                        prev_close = None

                    if df.empty:
                        continue

                    df['RVOL_5'] = df['Volume'] / df['Volume'].rolling(window=5).mean()





                    # Ichimoku calculations
                    high_9 = df['High'].rolling(window=9).max()
                    low_9 = df['Low'].rolling(window=9).min()
                    df['Tenkan_sen'] = (high_9 + low_9) / 2

                    high_26 = df['High'].rolling(window=26).max()
                    low_26 = df['Low'].rolling(window=26).min()
                    df['Kijun_sen'] = (high_26 + low_26) / 2

                    df['Chikou_Span'] = df['Close'].shift(26)

                    # Get latest row
                    latest = df.iloc[-1]

                    open_price = float(latest['Open'])
                    close_price = float(latest['Close'])
                    dollar_change = close_price - open_price
                    percent_change = (dollar_change / open_price) * 100
                    f_open = round(((close_price - open_price) / open_price) * 10000, 0)

                    # Tenkan logic
                    tenkan_value = df['Tenkan_sen'].iloc[-1]
                    tenkan_value = round(float(tenkan_value), 2) if pd.notna(tenkan_value) else None
                    if tenkan_value is None:
                        tenkan_check = '❓'
                    elif close_price >= tenkan_value:
                        tenkan_check = '✅'
                    else:
                        tenkan_check = '❌'

                    # Kijun logic
                    kijun_value = df['Kijun_sen'].iloc[-1]
                    kijun_value = round(float(kijun_value), 2) if pd.notna(kijun_value) else None

                    if kijun_value is None:
                        kijun_check = '❓'
                    elif close_price >= kijun_value:
                        kijun_check = '✅'
                    else:
                        kijun_check = '❌'

                    def calculate_f_percentage(intraday_df, prev_close_val):
                        if prev_close_val is not None and not intraday_df.empty:
                            intraday_df["F%"] = (
                                (intraday_df["Close"] - prev_close_val) / prev_close_val
                            ) * 10000
                            intraday_df["F%"] = (
                                intraday_df["F%"].round(0).astype(int).astype(str) + "%"
                            )
                        else:
                            intraday_df["F%"] = "N/A"
                        return intraday_df

                    # Relative F% lines to prev_close
                    if prev_close:
                        df["Tenkan_F"] = ((df["Tenkan_sen"] - prev_close) / prev_close) * 10000
                        df["Kijun_F"] = ((df["Kijun_sen"] - prev_close) / prev_close) * 10000
                    else:
                        df["Tenkan_F"] = None
                        df["Kijun_F"] = None



                    df["F_numeric"] = ((df["Close"] - prev_close) / prev_close) * 10000

                    df = calculate_td_demand_supply_lines_fpercent(df)

                    tenkan_f = (
                        round(((tenkan_value - prev_close) / prev_close) * 10000, 0)
                        if prev_close and tenkan_value
                        else None
                    )
                    kijun_f = (
                        round(((kijun_value - prev_close) / prev_close) * 10000, 0)
                        if prev_close and kijun_value
                        else None
                    )

                    if pd.isna(df['Tenkan_F'].iloc[-1]) or pd.isna(df['F_numeric'].iloc[-1]):
                        tenkan_check = '❓'
                    elif df['F_numeric'].iloc[-1] >= df['Tenkan_F'].iloc[-1]:
                        tenkan_check = '✅'
                    else:
                        tenkan_check = '❌'

                    if kijun_f is None:
                        kijun_f_status = "❓"
                    elif close_price >= kijun_value:
                        kijun_f_status = f"✅ {int(kijun_f)}%"
                    else:
                        kijun_f_status = f"❌ {int(kijun_f)}%"

                    # Chikou logic
                    chikou_value = df['Chikou_Span'].iloc[-1]
                    chikou_value = round(float(chikou_value), 2) if pd.notna(chikou_value) else None
                    if chikou_value is None:
                        chikou_check = '❓'
                    elif close_price > chikou_value:
                        chikou_check = '✅'
                    else:
                        chikou_check = '❌'

                    # Simulate Chikou if price dropped 1%
                    simulated_price = close_price * 0.99  # 1% drop
                    if len(df) >= 27:
                        past_high = float(df['High'].iloc[-27])
                        past_low = float(df['Low'].iloc[-27])
                        if simulated_price > past_high or simulated_price < past_low:
                            chikou_touch_pred = "✅ No Touch (Clear)"
                        else:
                            chikou_touch_pred = "❌ Touch (Wick/Body)"
                    else:
                        chikou_touch_pred = "❓ Not enough data"

                                    #Tekan > Kijun
                    if pd.notna(df['Tenkan_F'].iloc[-1]) and pd.notna(df['Kijun_F'].iloc[-1]):
                        tenkan_above_kijun = '✅' if df['Tenkan_F'].iloc[-1] > df['Kijun_F'].iloc[-1] else '❌'
                    else:
                        tenkan_above_kijun = '❓'


                    if pd.notna(df['Tenkan_F'].iloc[-1]) and pd.notna(df['F_numeric'].iloc[-1]):
                        distance_from_tenkan = abs(df['F_numeric'].iloc[-1] - df['Tenkan_F'].iloc[-1])
                        distance_flag = "✅ Close" if distance_from_tenkan <= 20 else "❌ Far"
                    else:
                        distance_from_tenkan = None
                        distance_flag = "❓"




                                    # --- New F%-based Distance from Kijun ---
                    if pd.notna(df['Kijun_F'].iloc[-1]) and pd.notna(df['F_numeric'].iloc[-1]):
                        distance_from_kijun = abs(df['F_numeric'].iloc[-1] - df['Kijun_F'].iloc[-1])
                        kijun_distance_flag = "✅ Close" if distance_from_kijun <= 20 else "❌ Far"
                    else:
                        distance_from_kijun = None
                        kijun_distance_flag = "❓"

                    # Tenkan trend
                    tenkan_trend = df['Tenkan_sen'].iloc[-1] - df['Tenkan_sen'].iloc[-3]
                    tenkan_trend_flag = '✅ Up' if tenkan_trend > 0.1 else '❌ Down'

                    # Kijun trend
                    kijun_trend = df['Kijun_sen'].iloc[-1] - df['Kijun_sen'].iloc[-3]
                    kijun_trend_flag = '✅ Up' if kijun_trend > 0 else '❌ Flat or Down'

                    crossed_kijun_up = (
                        pd.notna(df['F_numeric'].iloc[-1]) and
                        pd.notna(df['F_numeric'].iloc[-2]) and
                        pd.notna(df['Kijun_F'].iloc[-1]) and
                        df['F_numeric'].iloc[-2] < df['Kijun_F'].iloc[-1] and
                        df['F_numeric'].iloc[-1] > df['Kijun_F'].iloc[-1]
                    )
                    crossed_kijun_down = (
                        pd.notna(df['F_numeric'].iloc[-1]) and
                        pd.notna(df['F_numeric'].iloc[-2]) and
                        pd.notna(df['Kijun_F'].iloc[-1]) and
                        df['F_numeric'].iloc[-2] > df['Kijun_F'].iloc[-1] and
                        df['F_numeric'].iloc[-1] < df['Kijun_F'].iloc[-1]
                    )
                    rvol_val = float(latest['RVOL_5'])

                    if crossed_kijun_up and rvol_val > 1.2:
                        wealth_signal = "✅"
                    elif crossed_kijun_down:
                        if rvol_val > 2:
                            wealth_signal = "🟩❌"
                        elif rvol_val > 1.5:
                            wealth_signal = "🟨❌"
                        elif rvol_val > 1.2:
                            wealth_signal = "🟥❌"
                        else:
                            wealth_signal = ""
                    else:
                        wealth_signal = ""

                    tenkan_f_now = df['Tenkan_F'].iloc[-1]
                    tenkan_f_prev = df['Tenkan_F'].iloc[-2]
                    kijun_f_now = df['Kijun_F'].iloc[-1]
                    kijun_f_prev = df['Kijun_F'].iloc[-2]

                    if pd.notna(tenkan_f_now) and pd.notna(kijun_f_now):
                        if tenkan_f_prev < kijun_f_prev and tenkan_f_now > kijun_f_now:
                            tenkan_kijun_cross = "✅ Bullish Cross"
                        elif tenkan_f_prev > kijun_f_prev and tenkan_f_now < kijun_f_now:
                            tenkan_kijun_cross = "❌ Bearish Cross"
                        else:
                            tenkan_kijun_cross = "– No Cross"
                    else:
                        tenkan_kijun_cross = "❓"


                    if prev_close:
                        f_percent = round(((close_price - prev_close) / prev_close) * 10000, 0)
                    else:
                        f_percent = None

                                # --- Core Pre‑Order Signal (1‑bar cross) ---


                    if pd.isna(df['Tenkan_F'].iloc[-1]) or pd.isna(df['F_numeric'].iloc[-1]):
                        tenkan_f_status = '❓'
                    elif df['F_numeric'].iloc[-1] >= df['Tenkan_F'].iloc[-1]:
                        tenkan_f_status = f"✅ {int(df['Tenkan_F'].iloc[-1])}%"
                    else:
                        tenkan_f_status = f"❌ {int(df['Tenkan_F'].iloc[-1])}%"

                    # Step 1: Detect F% Kijun cross
                    df['Crossed_Kijun'] = (df['F_numeric'].shift(1) < df['Kijun_F'].shift(1)) & (df['F_numeric'] > df['Kijun_F'])

                    # Step 2: Initialize columns
                    df['Pre_Cross_RVOL_Spike'] = ""
                    df['Post_Cross_RVOL_Spike'] = ""

                    # Step 3: Check pre and post
                    for idx in df.index[df['Crossed_Kijun']]:
                        idx_pos = df.index.get_loc(idx)

                        # --- Pre-cross window (look back 9 bars)
                        pre_window = df.iloc[max(0, idx_pos-9):idx_pos]
                        if (pre_window['RVOL_5'] > 1.2).any():
                            df.at[idx, 'Pre_Cross_RVOL_Spike'] = "✅"

                        # --- Post-cross window (look forward 9 bars)
                        post_window = df.iloc[idx_pos+1:idx_pos+10]
                        if (post_window['RVOL_5'] > 1.2).any():
                            df.at[idx, 'Post_Cross_RVOL_Spike'] = "✅"
                        # Step 1: Detect F%-based Kijun cross
                        df['Crossed_Kijun'] = (df['F_numeric'].shift(1) < df['Kijun_F'].shift(1)) & (df['F_numeric'] > df['Kijun_F'])

                        # Step 2: Initialize new columns
                        df['Pre_Cross_RVOL_Spike'] = ""
                        df['Post_Cross_RVOL_Spike'] = ""

                        # Step 3: Loop over each detected cross
                        for idx in df.index[df['Crossed_Kijun']]:
                            idx_pos = df.index.get_loc(idx)

                            # Pre-cross window (9 bars back)
                            pre_window = df.iloc[max(0, idx_pos-9):idx_pos]
                            if (pre_window['RVOL_5'] > 1.2).any():
                                df.at[idx, 'Pre_Cross_RVOL_Spike'] = "✅"

                            # Post-cross window (9 bars forward)
                            post_window = df.iloc[idx_pos+1:idx_pos+10]
                            if (post_window['RVOL_5'] > 1.2).any():
                                df.at[idx, 'Post_Cross_RVOL_Spike'] = "✅"
                    pre_cross_rvol = df['Pre_Cross_RVOL_Spike'].iloc[-1]
                    post_cross_rvol = df['Post_Cross_RVOL_Spike'].iloc[-1]






                    combined_data.append({
                        'Ticker': ticker,
                        'Time (NY)': format_time_ny(latest.name),
                        'Open': round(latest['Open'].item(), 2),
                        'High': round(latest['High'].item(), 2),
                        'Low': round(latest['Low'].item(), 2),
                        'Close': round(latest['Close'].item(), 2),
                        'Volume': int(latest['Volume'].item()),
                        'RVOL_5': round(latest['RVOL_5'].item(), 2),
                        '$ Change': round(dollar_change, 2),
                        '% Change': round(percent_change, 2),
                        'Tenkan': tenkan_value,
                        '✓ Above Tenkan': tenkan_check,
                        'Kijun': kijun_value,
                        '✓ Above Kijun': kijun_check,
                        'F%': f"{int(f_percent)}%" if f_percent is not None else "N/A",
                        'F% (Last Candle)': f"{int(f_open)}%",
                        'Tenkan F%': f"{int(df['Tenkan_F'].iloc[-1])}%"
                            if pd.notna(df['Tenkan_F'].iloc[-1]) else "N/A",
                        'Kijun F%': f"{int(df['Kijun_F'].iloc[-1])}%"
                            if pd.notna(df['Kijun_F'].iloc[-1]) else "N/A",
                        'Above Tenkan F%': tenkan_f_status,
                        'Above Kijun F%': kijun_f_status,
                        'Chikou': chikou_value,
                        '✓ Above Chikou': chikou_check,
                        'Tenkan > Kijun': tenkan_above_kijun,
                        'Distance from Tenkan': round(distance_from_tenkan, 2)
                            if distance_from_tenkan is not None else "N/A",
                        '✓ Distance OK?': distance_flag,
                        'Distance from Kijun': round(distance_from_kijun, 2)
                            if distance_from_kijun is not None else "N/A",
                        '✓ Distance OK (Kijun)?': kijun_distance_flag,
                        'Tenkan Trend': tenkan_trend_flag,
                        'Kijun Trend': kijun_trend_flag,
                        'Chikou Touch w/ 1% Drop': chikou_touch_pred,
                        'Tenkan-Kijun Cross': tenkan_kijun_cross,
                        'Wealth Signal': wealth_signal,
                        # "Dominance Signal": dominance_signal,
                        'TD Supply F%': f"{int(df['TD Supply Line F'].iloc[-1])}%" if pd.notna(df['TD Supply Line F'].iloc[-1]) else "N/A",
                        'TD Demand F%': f"{int(df['TD Demand Line F'].iloc[-1])}%" if pd.notna(df['TD Demand Line F'].iloc[-1]) else "N/A",
                         'Distance from Tenkan': round(distance_from_tenkan, 2) if distance_from_tenkan is not None else "N/A",
                        'Distance from Kijun': round(distance_from_kijun, 2) if distance_from_kijun is not None else "N/A",
                        'Pre-Cross RVOL Spike': pre_cross_rvol,
                        'Post-Cross RVOL Spike': post_cross_rvol,

                            # --- Core Pre-Order Signal ---
                        # --- Core Pre-Order Signal ---



                    })

                except Exception as e:
                    st.warning(f"{ticker} ({interval}) failed to load: {e}")

            if combined_data:
                df_display = pd.DataFrame(combined_data)
                columns_to_keep = [
                    'Ticker','Time (NY)',

                    'F% (Last Candle)','$ Change',   'RVOL_5',
                    'Pre-Cross RVOL Spike', '✓ Above Tenkan',  '✓ Above Kijun','Post-Cross RVOL Spike', 'Wealth Signal','Tenkan > Kijun',
                    'Distance from Tenkan','✓ Distance OK?','Distance from Kijun',
                    '✓ Distance OK (Kijun)?','Tenkan Trend','Kijun Trend','Chikou',
                    '✓ Above Chikou','Tenkan-Kijun Cross',
                ]
                df_display = df_display[columns_to_keep]
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info(f"No data available for interval {interval}.")
    else:
        st.info("Please select at least one ticker to continue.")



st.subheader("🔍 View Full Intraday Table")

selected_full_view_ticker = st.selectbox("Pick a ticker to view full intraday table:", selected_tickers)
selected_interval = st.radio("Interval", ["5m", "15m"], horizontal=True)

if st.button("Show Full Table"):
    df_full = yf.download(
        selected_full_view_ticker,
        interval=selected_interval,
        start=start_date,
        end=end_date,
        progress=False
    )

    if not df_full.empty:
        # Convert index to US/Eastern time
        df_full.index = df_full.index.tz_convert('US/Eastern')

        # Compute technical indicators
        df_full['RVOL_5'] = df_full['Volume'] / df_full['Volume'].rolling(window=5).mean()
        df_full['Tenkan_sen'] = (
            df_full['High'].rolling(window=9, min_periods=1).max() +
            df_full['Low'].rolling(window=9, min_periods=1).min()
        ) / 2
        df_full['Kijun_sen'] = (
            df_full['High'].rolling(window=26, min_periods=1).max() +
            df_full['Low'].rolling(window=26, min_periods=1).min()
        ) / 2

        # ——— Copy your “prev_close” logic ———
        hist_daily = yf.download(selected_full_view_ticker, interval="1d", period="7d", progress=False)
        prev_close = hist_daily['Close'].iloc[-2] if len(hist_daily)>=2 else np.nan

        # 3️⃣ Calculate F%-based lines
        df_full['F_numeric'] = (df_full['Close'] - prev_close) / prev_close * 10000
        df_full['Tenkan_F'] = (df_full['Tenkan_sen'] - prev_close) / prev_close * 10000
        df_full['Kijun_F'] = (df_full['Kijun_sen'] - prev_close) / prev_close * 10000

        # 4️⃣ Detect One-bar Kijun Cross
        df_full['Crossed_Kijun'] = (df_full['F_numeric'].shift(1) < df_full['Kijun_F'].shift(1)) & (df_full['F_numeric'] > df_full['Kijun_F'])




        # — after you compute df_full['Crossed_Kijun'] … —

        # 1️⃣ Initialize columns
        df_full['Pre_Cross_RVOL_Spike']  = ""
        df_full['Post_Cross_RVOL_Spike'] = ""

        # 2️⃣ Loop over each detected cross
        for idx in df_full.index[df_full['Crossed_Kijun']]:
            pos = df_full.index.get_loc(idx)
            pre  = df_full['RVOL_5'].iloc[max(0, pos-9):pos]
            post = df_full['RVOL_5'].iloc[pos+1:pos+10]

            if (pre > 1.2).any():
                df_full.at[idx, 'Pre_Cross_RVOL_Spike'] = "✅"
            if (post > 1.2).any():
                df_full.at[idx, 'Post_Cross_RVOL_Spike'] = "✅"


        signals = []
        # Iterate through each row (starting at index 1)
        for i in range(1, len(df_full)):
            try:
                row = df_full.iloc[i]
                open_val = float(row['Open'])
                close_val = float(row['Close'])
                rvol_val = float(row['RVOL_5'])
                tenkan_val = float(row['Tenkan_sen'])
                kijun_val = float(row['Kijun_sen'])
                f_now = float(row['F_numeric'])
                tenkan_f_now = float(row['Tenkan_F'])
                kijun_f_now = float(row['Kijun_F'])


        # --- Above Tenkan / Kijun ---
                above_tenkan = "✅" if close_val >= tenkan_val else "❌"
                above_kijun = "✅" if close_val >= kijun_val else "❌"

                    # 2) Tenkan > Kijun (F%-lines)
                if pd.isna(tenkan_f_now) or pd.isna(kijun_f_now):
                    tenkan_above_kijun = ""           # blank if no data
                elif tenkan_f_now > kijun_f_now:
                    tenkan_above_kijun = "✅"
                else:
                    tenkan_above_kijun = "❌"









                # --- Wealth Signal Logic ---
                crossed_kijun_up = open_val < kijun_val and close_val > kijun_val
                crossed_kijun_down = open_val > kijun_val and close_val < kijun_val

                if crossed_kijun_up and rvol_val > 1.2:
                    wealth_signal = "✅"
                elif crossed_kijun_down:
                    if rvol_val > 2:
                        wealth_signal = "🟩❌"
                    elif rvol_val > 1.5:
                        wealth_signal = "🟨❌"
                    elif rvol_val > 1.1:
                        wealth_signal = "🟥❌"
                    else:
                        wealth_signal = ""
                else:
                    wealth_signal = ""

                signals.append({
                    "Time (NY)": row.name.strftime("%Y-%m-%d %H:%M"),
                    "Open": round(open_val, 2),
                    "Close": round(close_val, 2),
                    "RVOL_5": round(rvol_val, 2),
                    "Tenkan": round(tenkan_val, 2),
                    "Kijun": round(kijun_val, 2),
                    "✓ Above Tenkan": above_tenkan,
                    "✓ Above Kijun": above_kijun,

                    "Wealth Signal": wealth_signal
                })

            except Exception as e:
                # Skip rows where an error occurs
                continue

        df_signals = pd.DataFrame(signals)
        st.dataframe(df_signals, use_container_width=True)
    else:
        st.warning("No data available.")





st.subheader("📉 F%-Based Mini Ichimoku Plot")

# Let user select a ticker to visualize F%/Tenkan/Kijun
selected_plot_ticker = st.selectbox("Pick a stock to plot F% lines:", selected_tickers)

if selected_plot_ticker:
    df_plot = yf.download(
        selected_plot_ticker,
        interval="5m",
        start=start_date,
        end=end_date,
        progress=False
    )

    # Convert index to US/Eastern
    if not df_plot.empty:
        df_plot.index = df_plot.index.tz_convert('US/Eastern')

        # Get prev_close from the last daily candle
        hist_daily = yf.download(
            selected_plot_ticker,
            interval="1d",
            period="7d",
            progress=False
        )
        if (
            isinstance(hist_daily, pd.DataFrame)
            and 'Close' in hist_daily.columns
            and len(hist_daily) >= 2
        ):
            prev_close = hist_daily['Close'].iloc[-2].item()
        else:
            prev_close = None

        # We'll define chikou_fill_value to ensure it's a float scalar
        chikou_fill_value = float(df_plot["Close"].iloc[0])


        # --- Upgraded Ichimoku: start calculations right at 9:30 using min_periods=1
        df_plot['Tenkan_sen'] = (
            df_plot['High'].rolling(window=9, min_periods=1).max()
            + df_plot['Low'].rolling(window=9, min_periods=1).min()
        ) / 2
        df_plot['Kijun_sen'] = (
            df_plot['High'].rolling(window=26, min_periods=1).max()
            + df_plot['Low'].rolling(window=26, min_periods=1).min()
        ) / 2

        df_plot["Chikou_Span"] = df_plot["Close"].shift(
            26,
            fill_value=chikou_fill_value  # <--- ensure it's a scalar float
        )
        # If we have a valid prev_close, calculate the F%-based lines
        if prev_close is not None:
            df_plot["F"] = ((df_plot["Close"] - prev_close) / prev_close) * 10000
            df_plot["F_numeric"] = df_plot["F"]  # for ring logic

            df_plot = calculate_td_demand_supply_lines_fpercent(df_plot)

            df_plot["Tenkan_F"] = (
                (df_plot["Tenkan_sen"] - prev_close) / prev_close
            ) * 10000
            df_plot["Kijun_F"] = (
                (df_plot["Kijun_sen"] - prev_close) / prev_close
            ) * 10000
            df_plot["Chikou_F"] = (
                (df_plot["Chikou_Span"] - prev_close) / prev_close
            ) * 10000

            # Filter & rename columns for plotting
            df_plot_clean = df_plot[["F", "Tenkan_F", "Kijun_F", "Chikou_F"]].dropna()
            df_plot_clean.columns = ["F%", "Tenkan F%", "Kijun F%", "Chikou F%"]
        else:
            st.warning("Could not calculate prev_close.")
            st.stop()
    else:
        st.warning("No data available for selected ticker.")
        st.stop()

    # --- Create Plotly Figure ---
    fig = go.Figure()

    # Main F% (close-based)
    fig.add_trace(go.Scatter(
        x=df_plot_clean.index,
        y=df_plot_clean["F%"],
        mode='lines+markers',
        name='F%',
        line=dict(width=2)
    ))

    # Tenkan line
    fig.add_trace(go.Scatter(
        x=df_plot_clean.index,
        y=df_plot_clean["Tenkan F%"],
        mode='lines',
        name='Tenkan F%',
        line=dict(width=2, dash='dot', color='red')
    ))

    # Kijun line
    fig.add_trace(go.Scatter(
        x=df_plot_clean.index,
        y=df_plot_clean["Kijun F%"],
        mode='lines',
        name='Kijun F%',
        line=dict(width=2, dash='solid', color='green')
    ))


    # TD Demand Line
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["TD Demand Line F"],
        mode='lines',
        name='TD Demand F%',
        line=dict(width=1.5, dash='dash', color='lightpink')
    ))

    # TD Supply Line
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["TD Supply Line F"],
        mode='lines',
        name='TD Supply F%',
        line=dict(width=1.5, dash='dash', color='gray')
    ))

    # # Chikou line, shifted 26 bars behind in TIME, so it visually trails
    # chikou_data = df_plot_clean["Chikou F%"].dropna()
    # chikou_x = chikou_data.index - pd.Timedelta(minutes=5 * 26)  # 26 bars * 5min
    # fig.add_trace(go.Scatter(
    #     x=chikou_x,
    #     y=chikou_data,
    #     mode='lines',
    #     name='Chikou F%',
    #     line=dict(width=2, color='purple')
    # ))

    fig.update_layout(
        title=f"F% Ichimoku View - {selected_plot_ticker}",
        xaxis_title="Time",
        yaxis_title="F%",
        legend_title="Line",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
