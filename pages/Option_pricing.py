import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

# Set page config first thing
st.set_page_config(page_title="Options Wealth - Option Pricing", layout="wide")






st.title("💰 Option Pricing Module")

# --- Ticker Input ---
ticker = st.text_input("Enter stock symbol", value="AAPL", key="stock_input")
st.write("You entered:", ticker)

# --- STEP 1: Define All Needed Functions (Greeks + Delta-Neutral) ---

def estimate_otm_option_price(atm_price, spot_price, otm_strike, delta_pct=0.5):
    """
    Estimate OTM option price using delta-weighted adjustment from ATM price.
    """
    intrinsic_diff = abs(otm_strike - spot_price)
    adjustment = delta_pct * intrinsic_diff

    if otm_strike > spot_price:  # OTM Call
        return round(atm_price - adjustment, 2)
    else:  # OTM Put
        return round(atm_price + adjustment, 2)


def calculate_annual_volatility(df, window=20):
    """
    Calculate rolling annualized volatility (%), based on daily return std dev.
    df: DataFrame with 'Close' column
    window: Rolling window size (default 20 days)
    """
    df = df.copy()
    df["Daily Return"] = df["Close"].pct_change()
    df["Rolling StdDev (Daily Return)"] = df["Daily Return"].rolling(window=window).std()
    df["Annual Volatility (%)"] = df["Rolling StdDev (Daily Return)"] * np.sqrt(252) * 100
    df["Annual Volatility (%)"] = df["Annual Volatility (%)"].round(2)
    return df

def calculate_greeks(option_type, S, K, T, r, sigma):
    try:
        if T <= 0 or sigma <= 0:
            return None, None, None, None, None
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type.lower() == "call":
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
            charm = -norm.pdf(d1) * (2 * r * T - sigma**2) / (2 * sigma * sqrt(T)) - r * norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
            charm = -norm.pdf(d1) * (2 * r * T - sigma**2) / (2 * sigma * sqrt(T)) + r * norm.cdf(-d1)

        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T)

        return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta / 365, 4), round(charm / 365, 4)

    except:
        return None, None, None, None, None

# 1) Define a helper function to calculate *historical* annualized volatility
def get_annual_volatility(ticker):
    # Download 1 year of daily price data
    df = yf.download(ticker, period='1y', interval='1d')
    df["returns"] = df["Close"].pct_change()
    daily_std = df["returns"].std()
    # Annualize the daily standard deviation (~252 trading days/yr)
    annual_vol = daily_std * np.sqrt(252)
    return annual_vol

def estimate_atm_premium(spot, iv, T):
    """
    Estimate ATM option premium as a % of spot using 0.4 * IV * sqrt(T)
    """
    return round(spot * 0.4 * iv * np.sqrt(T), 2)


def get_iv_rank_and_percentile(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y", interval="1d")
    current_price = hist["Close"].iloc[-1]

    # Pick 1y of option chains to approximate daily IV
    iv_list = []

    for date in pd.date_range(end=datetime.today(), periods=252, freq='B'):
        try:
            options = stock.option_chain(date.strftime('%Y-%m-%d'))
            calls = options.calls
            atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
            iv = atm_call["impliedVolatility"].values[0]
            iv_list.append(iv)
        except:
            continue  # skip dates that fail

    if len(iv_list) >= 30:
        iv_series = pd.Series(iv_list)
        iv_rank = round((iv_series.iloc[-1] - iv_series.min()) / (iv_series.max() - iv_series.min()) * 100, 2)
        iv_percentile = round((iv_series < iv_series.iloc[-1]).sum() / len(iv_series) * 100, 2)
        return iv_rank, iv_percentile
    else:
        return None, None



def find_delta_neutral_matches(option_type, strike, merged_df, tolerance=0.05):
    """
    Given an option_type ('call' or 'put') and a strike from 'merged_df',
    find combos on the opposite side that offset the selected option's delta
    within 'tolerance'.
    """
    opposite = "Call" if option_type == "put" else "Put"  # Opposite side
    selected_delta_col = f"{option_type.capitalize()} Delta"
    hedge_delta_col = f"{opposite} Delta"
    price_col = f"{opposite} Price"

    try:
        # This is the chosen option row (call or put) by its strike
        selected_row = merged_df[merged_df["strike"] == strike].iloc[0]
        selected_delta = selected_row[selected_delta_col]
        st.write(f"📐 Selected {option_type.capitalize()} Delta: {selected_delta}")
        selected_premium = selected_row[f"{option_type.capitalize()} Price"]
        st.write(f"💵 Selected {option_type.capitalize()} Premium: ${selected_premium}")



                # Subset with the hedge side
        iv_col = f"{opposite} IV"
        hedge_df = merged_df[["strike", hedge_delta_col, price_col, iv_col]].copy()

        hedge_df.rename(columns={
            hedge_delta_col: "Hedge Delta",
            price_col: "Hedge Price",
            iv_col: "IV"

        }, inplace=True)

        combo_list = []
        # We'll try 1x to 4x to see if that gets us close to neutral
        for i in range(1, 5):
            hedge_df["Legs"] = i
            hedge_df["Total Delta"] = hedge_df["Hedge Delta"] * i
            hedge_df["Net Position Delta"] = selected_delta + hedge_df["Total Delta"]
            hedge_df["Delta Diff"] = hedge_df["Net Position Delta"].abs()
            hedge_df["Total Cost"] = hedge_df["Hedge Price"] * i
            hedge_df["Premium"] = hedge_df["Hedge Price"]  # single leg price
            hedge_df["Gamma"] = merged[f"{opposite} Gamma"]  # get gamma per leg
            hedge_df["Total Gamma"] = hedge_df["Gamma"] * hedge_df["Legs"]
            hedge_df["Gamma Alert"] = hedge_df["Gamma"].apply(lambda g: "⚠️ High Gamma" if g > 0.07 else "")
            hedge_df["Full Combo Cost"] = hedge_df["Total Cost"] + selected_row[f"{option_type.capitalize()} Price"]
            main_delta = selected_row[f"{option_type.capitalize()} Delta"]
            main_gamma = selected_row[f"{option_type.capitalize()} Gamma"]
            main_price = selected_row[f"{option_type.capitalize()} Price"]
            # Assuming daily percent move (S%) → e.g., 0.02 for a 2% move
            expected_move_pct = 0.02
            hedge_df["Delta-Hedged P&L"] = ((expected_move_pct ** 2) * hedge_df["Total Gamma"]) / 2


            a = 0.5 * main_gamma
            b = abs(main_delta)

            c = -hedge_df["Full Combo Cost"]  # full strangle cost

            discriminant = (b ** 2 + 4 * a * (-c)).clip(lower=0)
            move_to_cover = (-b + np.sqrt(discriminant)) / (2 * a)

            if option_type == "put":
                move_to_cover *= -1

            hedge_df["Move to Cover"] = move_to_cover









            a_h = 0.5 * hedge_df["Total Gamma"]
            b_h = abs(hedge_df["Total Delta"])
            c_h = -hedge_df["Full Combo Cost"]  # same full cost!

            discriminant_h = (b_h**2 + 4 * a_h * (-c_h)).clip(lower=0)
            hedge_move_to_cover = (-b_h + np.sqrt(discriminant_h)) / (2 * a_h)

            if option_type == "call":
                hedge_move_to_cover *= -1

            hedge_df["Hedge Move to Cover"] = hedge_move_to_cover




            hedge_move_to_cover = (-b_h + np.sqrt(discriminant_h)) / (2 * a_h)

            # Direction depends on hedge type (opposite of main leg)
            if option_type == "call":
                hedge_move_to_cover *= -1  # hedge is a put → needs stock drop
            else:
                hedge_move_to_cover *= 1   # hedge is a call → needs stock rise

            hedge_df["Hedge Move to Cover"] = hedge_move_to_cover


            # Inside find_delta_neutral_matches()
            hedge_df["Variance"] = hedge_df["IV"] ** 2
            hedge_df["Variance Edge"] = hedge_df["Gamma"] * hedge_df["Variance"]
            hedge_df["Total Variance Edge"] = hedge_df["Variance Edge"] * hedge_df["Legs"]




            close_to_neutral = hedge_df[hedge_df["Delta Diff"] <= tolerance].copy()
            combo_list.append(close_to_neutral)

        if len(combo_list) > 0:
            all_matches = pd.concat(combo_list).sort_values(by="Delta Diff")
            return all_matches
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"⚠️ Failed to calculate delta-neutral match: {e}")
        return pd.DataFrame()





# Initialize session_state keys if they don't exist
if "merged" not in st.session_state:
    st.session_state.merged = None
if "stock_price" not in st.session_state:
    st.session_state.stock_price = None
if "expirations" not in st.session_state:
    st.session_state.expirations = []
if "selected_exp" not in st.session_state:
    st.session_state.selected_exp = None


# --- STEP 2: Let User Load the Data ---

if st.button("Load Options Data"):
    # If user provided a ticker, fetch its expiration dates
    if ticker:
        stock = yf.Ticker(ticker)
        exps = stock.options
        if exps:
            # Store expiration dates in session state
            st.session_state.expirations = exps
            # By default, pick the first expiration
            st.session_state.selected_exp = exps[0]
        else:
            st.error("No expiration dates found for this ticker.")
    else:
        st.warning("Please enter a ticker symbol.")



# If we have expiration dates, let user pick from them
if st.session_state.expirations:
    st.session_state.selected_exp = st.selectbox(
        "Select Expiration Date",
        st.session_state.expirations,
        index=0
    )

# Once we have an expiration selected, we can load the chain
if st.session_state.selected_exp:
    # Try to fetch & process the chain
    try:
        stock = yf.Ticker(ticker)
        option_chain = stock.option_chain(st.session_state.selected_exp)
        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()
        calls["Open Interest"] = option_chain.calls["openInterest"]
        puts["Open Interest"] = option_chain.puts["openInterest"]

        # Get current stock price
        stock_price = stock.history(period="1d")["Close"].iloc[-1]
        st.session_state.stock_price = stock_price

        # Calculate Put/Call Ratio
        total_call_volume = calls["volume"].sum()
        total_put_volume = puts["volume"].sum()

        # Option classification
        calls["Type"] = "Call"
        puts["Type"] = "Put"
        calls["Classification"] = calls["strike"].apply(
            lambda x: "ATM" if abs(x - stock_price) < 0.5 else ("ITM" if x < stock_price else "OTM")
        )
        puts["Classification"] = puts["strike"].apply(
            lambda x: "ATM" if abs(x - stock_price) < 0.5 else ("ITM" if x > stock_price else "OTM")
        )

        # Merge calls and puts
        merged = pd.merge(
            calls[["strike", "lastPrice", "impliedVolatility", "volume","Open Interest"]].rename(columns={
                "lastPrice": "Call Price",
                "impliedVolatility": "Call IV",
                "volume": "Call Volume",
                 "Open Interest": "Call OI"
            }),
            puts[["strike", "lastPrice", "impliedVolatility", "volume", "Open Interest"]].rename(columns={
                "lastPrice": "Put Price",
                "impliedVolatility": "Put IV",
                "volume": "Put Volume",
                 "Open Interest": "Put OI"
            }),
            on="strike"
        )
        merged["Straddle Cost (1SD)"] = merged["Call Price"] + merged["Put Price"]
        r = 0.05  # 5% risk-free rate

        # Convert expiration string (e.g., '2025-04-19') to datetime
        exp_date = datetime.strptime(st.session_state.selected_exp, "%Y-%m-%d")
        today = datetime.today()
        days_to_exp = (exp_date - today).days

        # If it's a valid future date, compute T in years
        T = max(days_to_exp, 1) / 365
        # Add Expected Move based on Call IV
        merged["Call Expected Move ($)"] = round(stock_price * merged["Call IV"] * np.sqrt(T * 252), 2)
        merged["Put Expected Move ($)"] = round(stock_price * merged["Put IV"] * np.sqrt(T * 252), 2)



        iv_rank, iv_percentile = get_iv_rank_and_percentile(ticker)
        merged["IV Rank (%)"] = iv_rank
        merged["IV Percentile (%)"] = iv_percentile


        # Calculate call/put greeks
        call_greeks = merged.apply(
            lambda row: calculate_greeks("call", stock_price, row["strike"], T, r, row["Call IV"]),
            axis=1
        )
        put_greeks = merged.apply(
            lambda row: calculate_greeks("put", stock_price, row["strike"], T, r, row["Put IV"]),
            axis=1
        )
        annual_vol = get_annual_volatility(ticker)
        merged["Annual Volatility (%)"] = round(annual_vol * 100, 2)
        # Add AV - IV difference for both Call and Put
        merged["Call AV - IV"] = round((annual_vol * 100) - (merged["Call IV"] * 100), 2)
        merged["Put AV - IV"] = round((annual_vol * 100) - (merged["Put IV"] * 100), 2)
        # Add IV / AV ratio (using decimal values, then rounded to 2 decimals)
        merged["Call IV / AV"] = round(merged["Call IV"] / annual_vol, 2)
        merged["Put IV / AV"] = round(merged["Put IV"] / annual_vol, 2)

        merged["Call Delta"], merged["Call Gamma"], merged["Call Vega"], merged["Call Theta"], merged["Call Charm"] = zip(*call_greeks)

        merged["Put Delta"], merged["Put Gamma"], merged["Put Vega"], merged["Put Theta"], merged["Put Charm"] = zip(*put_greeks)

             # Limit to strikes near spot
        price = st.session_state.stock_price
        min_strike = price * 0.9
        max_strike = price * 1.1
        
        filtered = merged[(merged["strike"] >= min_strike) & (merged["strike"] <= max_strike)]
        
        avg_call_voi = (filtered["Call Volume"] / filtered["Call OI"]).replace([np.inf, -np.inf], np.nan).mean().round(2)
        avg_put_voi = (filtered["Put Volume"] / filtered["Put OI"]).replace([np.inf, -np.inf], np.nan).mean().round(2)

        
        # st.markdown(f"""
        # ### 📊 Volume/Open Interest Insight
        
        # - **Average Call VOI Ratio:** {avg_call_voi}  
        # - **Average Put VOI Ratio:** {avg_put_voi}  
        
        # > VOI = Volume / Open Interest  
        # > - **VOI > 1.0** = Fresh interest (possibly new positioning)  
        # > - **VOI < 0.5** = Mostly churn or stale liquidity  
        # """)

 


        # Store 'merged' in session_state so it persists across reruns
        st.session_state.merged = merged

        # Show current price
        st.success(f"Current Price: ${stock_price:.2f}")

        # Show PCR
        if total_call_volume > 0:
            pcr = round(total_put_volume / total_call_volume, 2)
            st.info(f"Put-Call Ratio (PCR): {pcr}")
        else:
            st.warning("Not enough call volume to calculate PCR.")
    
    # # ✅ BEGIN: VOI STRIKE SELECTOR LOGIC
    #     if st.session_state.merged is not None:
    #         merged = st.session_state.merged  # local reference
            
    #         st.subheader("🎯 Select a Strike to Analyze VOI Ratio")
            
    #         selected_strike = st.selectbox(
    #             "Choose a Strike:",
    #             merged["strike"].sort_values().unique()
    #         )
            
    #         strike_row = merged[merged["strike"] == selected_strike].iloc[0]
            
    #         call_vol = strike_row["Call Volume"]
    #         call_oi = strike_row["Call OI"]
    #         put_vol = strike_row["Put Volume"]
    #         put_oi = strike_row["Put OI"]
            
    #         call_voi = round(call_vol / call_oi, 2) if call_oi > 0 else np.nan
    #         put_voi = round(put_vol / put_oi, 2) if put_oi > 0 else np.nan
            
    #         st.markdown(f"""
    #         #### 🔍 VOI Breakdown for Strike **{selected_strike}**
            
    #         | Metric         | Call       | Put        |
    #         |----------------|------------|------------|
    #         | Volume         | {call_vol} | {put_vol}  |
    #         | Open Interest  | {call_oi}  | {put_oi}   |
    #         | **VOI Ratio**  | {call_voi} | {put_voi}  |
    #         """, unsafe_allow_html=True)
            
    #         def voi_comment(voi):
    #             if voi > 2:
    #                 return "🔥 Very High (Aggressive Activity)"
    #             elif voi > 1:
    #                 return "📈 High (Likely New Positioning)"
    #             elif voi > 0.5:
    #                 return "😐 Moderate (Churn)"
    #             else:
    #                 return "🧊 Low (Stale or Passive)"
            
    #         st.info(f"📣 Interpretation:\n\n- **Call VOI:** {voi_comment(call_voi)}\n- **Put VOI:** {voi_comment(put_voi)}")
    #     # ✅ END



    except Exception as e:
        st.error(f"Error loading options data: {e}")

        
        # --- STEP 3: If we have 'merged' in session_state, show the Option Chain & Vol Plot & Delta-Neutral Helper ---
        
        if st.session_state.merged is not None:
            merged = st.session_state.merged  # local reference
        
       

    
    stock_price = st.session_state.stock_price
    # --- Liquidity Filter: Show only strikes with high OI ---
    merged = merged[(merged["Call OI"] >= 500) | (merged["Put OI"] >= 500)]


    # 🔽 🔽 🔽 INSERT BELOW THIS 🔽 🔽 🔽
    merged["Call VOL/OI"] = (merged["Call Volume"] / merged["Call OI"]).replace([np.inf, -np.inf], np.nan).round(2)
    merged["Put VOL/OI"] = (merged["Put Volume"] / merged["Put OI"]).replace([np.inf, -np.inf], np.nan).round(2)
    merged["Total VOL/OI"] = merged["Call VOL/OI"] + merged["Put VOL/OI"].round(2)

    
    # 🧠 Calculate Gamma Exposure (GEX)
    S = st.session_state.stock_price  # current spot price
    

    merged["Call GEX"] = merged["Call Gamma"] * merged["Call OI"] * 100 * (S ** 2)
    merged["Put GEX"] = merged["Put Gamma"] * merged["Put OI"] * 100 * (S ** 2)
    merged["Total GEX"] = merged["Call GEX"] + merged["Put GEX"]
    merged["Net GEX"] = merged["Call GEX"] - merged["Put GEX"]

    
        # 🧮 Scale GEX values to millions for readability
    merged["Call GEX ($M)"] = (merged["Call GEX"] / 1_000_000).round(2)
    merged["Put GEX ($M)"] = (merged["Put GEX"] / 1_000_000).round(2)
    merged["Total GEX ($M)"] = (merged["Total GEX"] / 1_000_000).round(2)
    merged["Net GEX ($M)"] = (merged["Net GEX"] / 1_000_000).round(2)

  
    
    st.subheader("📋 Option Chain Summary with Greeks")
    st.dataframe(
        merged.sort_values(by="Straddle Cost (1SD)", ascending=False)[[
            "strike", "Call Price", "Put Price",
            "Call Volume", "Put Volume", "Call OI", "Put OI",
            "Call IV", "Put IV",
            "Call Delta", "Put Delta",
            "Call Gamma", "Put Gamma",   
 
        ]],
        use_container_width=True
    )

    st.subheader("💣 Spot Gamma Exposure (GEX) Table")
    
    st.dataframe(
        merged[[
            "strike",
            "Call GEX ($M)",
            "Put GEX ($M)",
            "Total GEX ($M)",
            "Net GEX ($M)"
        ]].sort_values(by="Total GEX ($M)", ascending=False),
        use_container_width=True
    )
        
     # Gamma Flip Logic — Revised for clarity
    merged_sorted = merged.sort_values(by="strike").copy()
    merged_sorted["Sign"] = np.sign(merged_sorted["Net GEX ($M)"])
    
    # Only keep rows where sign is not NaN
    merged_sorted = merged_sorted.dropna(subset=["Sign"])
    
    # Shift and find sign change
    merged_sorted["Sign_Change"] = merged_sorted["Sign"].diff().fillna(0)
    
    flip_rows = merged_sorted[merged_sorted["Sign_Change"] != 0]
    
    # if not flip_rows.empty:
    #     flip_strike = flip_rows["strike"].iloc[0]
    #     st.success(f"🌀 Gamma Flip Zone Detected at Strike: **{flip_strike}**")
    # else:
    #     st.info("ℹ️ No Gamma Flip detected — Net GEX remains all positive or all negative.")



    
  # 📅 Calculate Daily Gamma Exposure (all strikes, all expiries)
    daily_call_gex = merged["Call GEX"].sum()
    daily_put_gex = merged["Put GEX"].sum()
    daily_net_gex = daily_call_gex + daily_put_gex  # not minus; both add to total gamma load
    
    # Scale to millions for readability
    daily_call_gex_m = round(daily_call_gex / 1_000_000, 2)
    daily_put_gex_m = round(daily_put_gex / 1_000_000, 2)
    daily_net_gex_m = round(daily_net_gex / 1_000_000, 2)


    # Compute Ceiling and Floor based on GEX
    ceiling_strike = merged.loc[merged["Call GEX"].idxmax(), "strike"]
    floor_strike = merged.loc[merged["Put GEX"].idxmax(), "strike"]

    
        
     # Calculate Net GEX if not already present
    merged["Net GEX"] = merged["Call GEX"] - merged["Put GEX"]
    
    # Sort by strike price for clean visualization
    merged_sorted = merged.sort_values("strike")

    
    # 📍  put this right after you finish computing Call GEX & Put GEX
    # ----------------------------------------------------------------
    # 1)  Build a mini‑table that the bias logic will use
    gex_df = merged[["strike", "Call GEX", "Put GEX"]].rename(
        columns={"Call GEX": "call_gex", "Put GEX": "put_gex"}
    )
    
    # 2)  Spot price for reference
    spot = st.session_state.stock_price
    
    # 3)  Separate strikes above and below spot
    above = gex_df[gex_df["strike"] > spot]
    below = gex_df[gex_df["strike"] < spot]
    
    # 4)  Net gamma pressure on each side
    above_gex = above["call_gex"].sum() - above["put_gex"].sum()
    below_gex = below["put_gex"].sum() - below["call_gex"].sum()
    
    net_bias = above_gex - below_gex       # positive ⇒ upward pull
    threshold = 1e5                        # tune for noise‑filtering
    
    

    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_sorted["strike"],
        y=merged_sorted["Net GEX"],
        mode="lines+markers",
        name="Net GEX",
        line=dict(width=2)
    ))
    
    # Add horizontal line at y=0 to mark Flip Zone
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="Flip Zone",
        annotation_position="bottom right"
    )
    
    # Customize layout
    fig.update_layout(
        title="Gamma Exposure by Strike (Net GEX)",
        xaxis_title="Strike Price",
        yaxis_title="Net Gamma Exposure ($)",
        template="plotly_white",
        height=500
    )
    # Add vertical lines to your plot
    fig.add_vline(x=ceiling_strike, line=dict(color="blue", dash="dot"),
                  annotation_text=f"🔵 Ceiling @ {ceiling_strike}", 
                  annotation_position="top right")
    
    fig.add_vline(x=floor_strike, line=dict(color="red", dash="dot"), 
                  annotation_text=f"🔴 Floor @ {floor_strike}", 
                  annotation_position="bottom right")




    # 5)  Display directional bias
    if net_bias > threshold:
        st.success("🟢  GEX Bias: Upward pressure – calls dominate below spot")
    elif net_bias < -threshold:
        st.error("🔴  GEX Bias: Downward pressure – puts dominate above spot")
    else:
        st.warning("⚖️  GEX Bias: Neutral – balanced gamma")
    # ----------------------------------------------------------------
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)




    
 # Calcular Max Pain (strike con mayor Total OI)
    merged["Total OI"] = merged["Call OI"] + merged["Put OI"]
    max_pain_strike = merged.loc[merged["Total OI"].idxmax(), "strike"]
    
    # Calculate gaps
    cushion_gap = st.session_state.stock_price - floor_strike
    ceiling_gap = ceiling_strike - st.session_state.stock_price
    gravity_gap = st.session_state.stock_price - max_pain_strike
    
    # Display Gravity Gap
    st.metric("🎯 Gravity Gap", f"{gravity_gap:.2f}")
    
    # Display Cushion and Ceiling Gaps side by side
    col1, col2 = st.columns(2)
    col1.metric("🛡️ Cushion Gap", f"{cushion_gap:.2f}")
    col2.metric("🚧 Ceiling Gap", f"{ceiling_gap:.2f}")

    if gravity_gap < 0 and cushion_gap < ceiling_gap:
        st.success("Bias: 🟢 Call buyers in control – upward pull")
    elif gravity_gap > 0 and ceiling_gap < cushion_gap:
        st.error("Bias: 🔴 Put pressure – gravity pulling lower")







    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["strike"],
        y=merged["Call IV"],
        mode='lines+markers',
        name='Call IV',
        customdata=np.stack((merged["Call Volume"], merged["Call IV"] * 100), axis=-1),

        hovertemplate=
        "IV: %{customdata[1]:.2f}%<br>" +
        "Volume: %{customdata[0]}<extra></extra>" +
        "Open Interest: %{customdata[1]}<extra></extra>"

    ))
 
    fig.add_trace(go.Scatter(
        x=merged["strike"],
        y=merged["Put IV"],
        mode='lines+markers',
        name='Put IV',
        line=dict(dash='dash')
    ))


    


     




    fig.update_layout(
        title="Volatility Smile",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- OI Chart ---
    st.subheader("📊 Open Interest by Strike")
    
    # Calcular Max Pain (strike con mayor Total OI)
    merged["Total OI"] = merged["Call OI"] + merged["Put OI"]
    max_pain_strike = merged.loc[merged["Total OI"].idxmax(), "strike"]
    
    # Mostrar texto arriba del gráfico
    st.info(f"🧨 Max Pain Strike: **{max_pain_strike}** (Highest Total OI)")
    
    # Crear gráfico
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(
        x=merged["strike"],
        y=merged["Call OI"],
        name="Call OI"
    ))
    fig_oi.add_trace(go.Bar(
        x=merged["strike"],
        y=merged["Put OI"],
        name="Put OI"
    ))
    
    # Línea vertical en Max Pain
    fig_oi.add_vline(
        x=max_pain_strike,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max Pain: {max_pain_strike}",
        annotation_position="top"
    )
    
    fig_oi.update_layout(
        title="Open Interest by Strike",
        barmode="group",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_oi, use_container_width=True)
    
    st.subheader("💥 Intraday Action Table (Sorted by Aggression)")
    
    action_table = merged[[
        "strike", 
        "Call Price", "Put Price",
        "Call Volume", "Call OI", "Call VOL/OI",
        "Put Volume", "Put OI", "Put VOL/OI",
        "Total VOL/OI"
    ]].sort_values(by="Total VOL/OI", ascending=False).head(20)
    
    st.dataframe(action_table.style.background_gradient(
        subset=["Call VOL/OI", "Put VOL/OI", "Total VOL/OI"], cmap="OrRd"
    ), use_container_width=True)


    st.subheader("💥 Intraday Aggression Bar Chart")
    
    # Only keep strikes within 10% of spot
    price = st.session_state.stock_price
    min_strike = price * 0.9
    max_strike = price * 1.1
    
    filtered = merged[(merged["strike"] >= min_strike) & (merged["strike"] <= max_strike)]
    top_aggression = filtered.sort_values(by="Total VOL/OI", ascending=False).head(20)
        
    fig_aggression = go.Figure()
    
    fig_aggression.add_trace(go.Bar(
        x=top_aggression["strike"].astype(str),
        y=top_aggression["Call VOL/OI"],
        name="Call VOL/OI",
        marker_color="green"
    ))
    
    fig_aggression.add_trace(go.Bar(
        x=top_aggression["strike"].astype(str),
        y=top_aggression["Put VOL/OI"],
        name="Put VOL/OI",
        marker_color="red"
    ))
    
    fig_aggression.update_layout(
        barmode="group",
        xaxis_title="Strike",
        yaxis_title="VOL / OI Ratio",
        title="Top 20 Aggressive Strikes (Call vs Put)",
        template="plotly_white",
        height=450
    )
    
    st.plotly_chart(fig_aggression, use_container_width=True)

   
