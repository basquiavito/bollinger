import streamlit as st
import pandas as pd
import os
import yfinance as yf
from datetime import datetime

GEX_FILE = "gex_levels.csv"

LEDGER_FILE = "trading_ledger.csv"
# --- Inventory (your tradeable tickers) ---
TICKERS = [
    "NVDA", "AMD", "AVGO", "MRVL", "MU", "SMCI", "QCOM",
    "MSFT", "AMZN", "AAPL", "GOOGL", "UBER", "PLTR", "META", "TSLA",
    "HOOD", "COIN", "C", "WFC", "JPM",
    "SPY", "QQQ"
]

st.title("📒 Trading Ledger")

# --- Load ledger if exists ---
if os.path.exists(LEDGER_FILE):
    ledger = pd.read_csv(LEDGER_FILE)
else:
    ledger = pd.DataFrame(columns=["Date", "Ticker", "Entry","Time", "Delta","Entry_Level","Enhancer",  "Ear", "Nose","Type", "StopLoss", "PnL", "Notes"])


# --- Upload to restore ---
uploaded = st.file_uploader("📤 Upload existing ledger (CSV)", type="csv")
if uploaded is not None:
    ledger = pd.read_csv(uploaded)
    ledger.to_csv(LEDGER_FILE, index=False)
    st.success("Ledger restored from uploaded file.")

# # --- Trade entry form ---
# with st.form("trade_entry"):
#     col1, col2 = st.columns(2)
#     with col1:
#         date = st.date_input("Date")
#         ticker = st.text_input("Ticker (e.g. TSLA)")
#         trade_type = st.selectbox("Type", ["Call", "Put"])
#     with col2:
#         entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01)
#         delta = st.number_input("Delta", min_value=0.0, step=0.01)
#         stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)

#     notes = st.text_area("Notes (optional)")
#     submitted = st.form_submit_button("➕ Add Trade")

# if submitted:
#     new_row = {
#         "Date": date,
#         "Ticker": ticker.upper(),
#         "Entry": entry_price,
#         "Delta": delta,
#         "Type": trade_type,
#         "StopLoss": stop_loss,
#         "Notes": notes
#     }
#     ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
#     ledger.to_csv(LEDGER_FILE, index=False)
#     st.success(f"Added {ticker} trade.")


if os.path.exists(GEX_FILE):
    gex_df = pd.read_csv(GEX_FILE)
else:
    gex_df = pd.DataFrame(columns=["Ticker", "GEX Ceiling", "GEX Floor"])

with st.expander("📡 GEX Radar"):
    st.write("Premarket GEX Levels")

    selected_tickers = st.multiselect(
        "Select Tickers",
        TICKERS,
        default=gex_df["Ticker"].tolist()  # ✅ remembers last saved tickers
    )

    radar_data = []
    for ticker in selected_tickers:
        # Get saved values if they exist
        saved_ceiling = gex_df[gex_df["Ticker"] == ticker]["GEX Ceiling"].values
        saved_floor = gex_df[gex_df["Ticker"] == ticker]["GEX Floor"].values

        gex_ceiling = st.number_input(
            f"{ticker} GEX Ceiling", 
            step=0.1, 
            value=float(saved_ceiling[0]) if len(saved_ceiling) else 0.0, 
            key=f"{ticker}_ceiling"
        )
        gex_floor = st.number_input(
            f"{ticker} GEX Floor", 
            step=0.1, 
            value=float(saved_floor[0]) if len(saved_floor) else 0.0, 
            key=f"{ticker}_floor"
        )

        # --- Update GEX table ---
        new_levels = pd.DataFrame(
            [[ticker, gex_ceiling, gex_floor]],
            columns=["Ticker", "GEX Ceiling", "GEX Floor"]
        )
        gex_df = gex_df[gex_df["Ticker"] != ticker]  # remove old row if exists
        gex_df = pd.concat([gex_df, new_levels], ignore_index=True)
        gex_df.to_csv(GEX_FILE, index=False)

        # Get last price
        try:
            last_price = yf.Ticker(ticker).history(period="1d", interval="1m")["Close"].iloc[-1]
        except:
            last_price = None

        # Condition highlight
        status = "⚪ Between"
        if last_price and gex_ceiling and last_price > gex_ceiling:
            status = "🟢 Above Ceiling"
        elif last_price and gex_floor and last_price < gex_floor:
            status = "🔴 Below Floor"

        radar_data.append([ticker, gex_ceiling, gex_floor, last_price, status])

    # Build table
    if radar_data:
        df = pd.DataFrame(radar_data, columns=["Ticker", "GEX Ceiling", "GEX Floor", "Last Price", "Status"])
        st.dataframe(df, hide_index=True)
 # ✅ Add timestamp
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# --- Trade entry form ---
with st.form("trade_entry"):
    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Date")
        entry_time = st.time_input("Entry Time")   # 👈 New field

        ticker = st.selectbox("Ticker", TICKERS)
        enhancer = st.checkbox("Enhancer present?")
        ear = st.checkbox("👂 Ear (Volume Memory crossed?)")
        nose = st.checkbox("👃 Nose (Time Memory crossed?)")

        trade_type = st.selectbox("Type", ["Call", "Put"])
        entry_level = st.selectbox(
        "Entry Level",
        ["Entry 1", "Entry 2", "Entry 3", "Mirror 1", "Mirror 2", "Mirror 3"]
    )

    with col2:
        entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01)
        delta = st.number_input("Delta", min_value=0.0, step=0.01)
        stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)
    with col3:
        exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01, format="%.2f")
        notes = st.text_area("Notes (optional)", height=50)
 
    submitted = st.form_submit_button("➕ Add Trade")

if submitted:
    pnl = exit_price - entry_price if exit_price > 0 else 0
    new_row = {
        "Date": date,
        "Ticker": ticker.upper(),
        "Time": entry_time.strftime("%H:%M:%S"),
        "Entry": entry_price,
        "Exit": exit_price,
        "Delta": delta,
        "Type": trade_type,
        "Entry_Level": entry_level,   # 👈 new column
        "StopLoss": stop_loss,
        "Enhancer": enhancer,   # 👈 boolean
        "Ear": ear,
        "Nose": nose,

        "PnL": pnl,
        "Notes": notes
    }
    ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
    ledger.to_csv(LEDGER_FILE, index=False)
    st.success(f"Added {ticker} trade.")



# --- Show ledger ---
st.subheader("📊 Current Ledger")
st.dataframe(ledger, use_container_width=True)

# --- Reset Ledger button ---
if st.button("🗑 Reset Ledger (Clear Trades)"):
    ledger = pd.DataFrame(columns=["Date", "Ticker", "Entry","Time", "Delta","Entry_Level",
                                   "Enhancer", "Ear", "Nose","Type", "StopLoss", "PnL", "Notes"])
    ledger.to_csv(LEDGER_FILE, index=False)
    st.warning("Ledger reset. GEX levels remain untouched ✅")


import altair as alt
import numpy as np

def make_gex_position_df(df):
    # expects columns: Ticker, GEX Floor, GEX Ceiling, Last Price
    d = df.copy()
    d["range"] = d["GEX Ceiling"] - d["GEX Floor"]
    d["pos"] = (d["Last Price"] - d["GEX Floor"]) / d["range"]  # 0=floor, 1=ceiling
    # classify
    def status_row(r):
        if r["Last Price"] is None or r["range"] <= 0: return "Unset"
        if r["pos"] < 0: return "Below"
        if r["pos"] > 1: return "Above"
        return "Between"
    d["State"] = d.apply(status_row, axis=1)

    # Distance to nearest boundary; negative if outside (stronger signal)
    def edge_pressure(r):
        if r["Last Price"] is None or r["range"] <= 0: return np.nan
        if r["pos"] < 0: return r["pos"]  # negative
        if r["pos"] > 1: return 1 - r["pos"]  # negative
        return -min(r["pos"], 1 - r["pos"])  # closer to edge => more negative
    d["EdgePressure"] = d.apply(edge_pressure, axis=1)
    return d

# inside your GEX expander, after building df:
viz_df = make_gex_position_df(df).dropna(subset=["EdgePressure"]).sort_values("EdgePressure")

color_scale = alt.Scale(
    domain=["Below","Between","Above","Unset"],
    range=["#e74c3c","#bdc3c7","#2ecc71","#7f8c8d"]
)

bars = alt.Chart(viz_df).mark_bar().encode(
    x=alt.X("pos:Q", title="Position (0=floor, 1=ceiling)", scale=alt.Scale(domain=[-0.2,1.2])),
    y=alt.Y("Ticker:N", sort=viz_df["Ticker"].tolist()),
    color=alt.Color("State:N", scale=color_scale, legend=alt.Legend(title="GEX State")),
    tooltip=[
        "Ticker:N",
        alt.Tooltip("Last Price:Q", format=".2f"),
        alt.Tooltip("GEX Floor:Q", format=".2f"),
        alt.Tooltip("GEX Ceiling:Q", format=".2f"),
        alt.Tooltip("pos:Q", title="Dial", format=".2f"),
        "State:N"
    ]
)

rules = alt.Chart(viz_df).mark_rule(strokeDash=[2,2], opacity=0.5).encode(x="value:Q")
rule_data = pd.DataFrame({"value":[0,1]})



# ============================
# ⚡ GEX Cross Feed (drop-in)
# ============================
import pandas as pd
from datetime import datetime

EVENTS_FILE = "gex_events.csv"
STATE_FILE  = "gex_state.csv"

# optional: auto-refresh the section (milliseconds)
st.autorefresh(interval=15_000, key="gex_cross_autorefresh")  # 15s; change as you like

# ---- helpers
def classify_state(last_price, floor, ceiling):
    if pd.isna(last_price) or pd.isna(floor) or pd.isna(ceiling):
        return "Unset"
    rng = ceiling - floor
    if rng <= 0:
        return "Unset"
    if last_price < floor:
        return "Below"
    if last_price > ceiling:
        return "Above"
    return "Between"

def event_label(prev, curr):
    if prev == curr or prev == "Unset" and curr == "Unset":
        return None
    # simple map of transitions -> labels
    mapping = {
        ("Between","Below"):  "Hit Floor",
        ("Between","Above"):  "Break Ceiling",
        ("Below","Between"):  "Re-enter (from Below)",
        ("Above","Between"):  "Re-enter (from Above)",
        ("Below","Above"):    "Through Range Up",
        ("Above","Below"):    "Through Range Down",
        ("Unset","Between"):  "Init (Between)",
        ("Unset","Below"):    "Init (Below)",
        ("Unset","Above"):    "Init (Above)",
        ("Between","Unset"):  "Levels Unset",
        ("Below","Unset"):    "Levels Unset",
        ("Above","Unset"):    "Levels Unset",
    }
    return mapping.get((prev, curr), f"{prev} → {curr}")

def nearest_distance(last_price, floor, ceiling):
    if pd.isna(last_price) or pd.isna(floor) or pd.isna(ceiling):
        return pd.NA
    if ceiling - floor <= 0:
        return pd.NA
    if last_price < floor:
        return last_price - floor  # negative below
    if last_price > ceiling:
        return last_price - ceiling  # positive above (still distance)
    # inside band: positive min distance to boundary
    return min(last_price - floor, ceiling - last_price)

# ---- load prior state & events
if os.path.exists(STATE_FILE):
    prev_state_df = pd.read_csv(STATE_FILE)
else:
    prev_state_df = pd.DataFrame(columns=["Ticker","State"])

if os.path.exists(EVENTS_FILE):
    events = pd.read_csv(EVENTS_FILE)
else:
    events = pd.DataFrame(columns=[
        "Time","Ticker","Event","From","To",
        "Last Price","GEX Floor","GEX Ceiling","Dist to Boundary"
    ])

# ensure df has necessary columns
needed_cols = {"Ticker","GEX Floor","GEX Ceiling","Last Price"}
missing = needed_cols - set(df.columns)
if missing:
    st.error(f"GEX Cross Feed needs columns missing from df: {missing}")
else:
    # compute current states
    curr_rows = []
    for _, r in df.iterrows():
        tkr = r["Ticker"]
        floor = r["GEX Floor"]
        ceil  = r["GEX Ceiling"]
        lp    = r["Last Price"]
        curr  = classify_state(lp, floor, ceil)
        curr_rows.append((tkr, curr))
    curr_state_df = pd.DataFrame(curr_rows, columns=["Ticker","State"])

    # join with previous to detect transitions
    merged = curr_state_df.merge(prev_state_df, on="Ticker", how="left", suffixes=("", "_prev"))
    merged["State_prev"] = merged["State_prev"].fillna("Unset")

    # loop & collect new events
    new_events = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_idx = df.set_index("Ticker")

    for _, row in merged.iterrows():
        tkr = row["Ticker"]
        prev = row["State_prev"]
        curr = row["State"]
        if prev == curr:
            continue  # no change

        lbl = event_label(prev, curr)
        if lbl is None:
            continue

        # grab metrics for context
        floor = df_idx.loc[tkr, "GEX Floor"]
        ceil  = df_idx.loc[tkr, "GEX Ceiling"]
        lp    = df_idx.loc[tkr, "Last Price"]
        dist  = nearest_distance(lp, floor, ceil)

        new_events.append({
            "Time": stamp,
            "Ticker": tkr,
            "Event": lbl,
            "From": prev,
            "To": curr,
            "Last Price": lp,
            "GEX Floor": floor,
            "GEX Ceiling": ceil,
            "Dist to Boundary": dist
        })

    # persist: append new events & update state
    if new_events:
        events = pd.concat([events, pd.DataFrame(new_events)], ignore_index=True)
        events.to_csv(EVENTS_FILE, index=False)

    curr_state_df.to_csv(STATE_FILE, index=False)

    # ---- UI: feed + filters
    st.subheader("⚡ GEX Cross Feed")

    # Basic filters
    colf1, colf2, colf3 = st.columns([2,1,1])
    with colf1:
        only_breaches = st.checkbox("Only Breaches (Hit/Break/Through)", value=False)
    with colf2:
        show_reentries = st.checkbox("Include Re-entries", value=True)
    with colf3:
        max_rows = st.number_input("Rows", min_value=10, max_value=500, value=100, step=10)

    feed = events.copy()
    # filter logic
    breaches = ["Hit Floor","Break Ceiling","Through Range Up","Through Range Down"]
    reenters = ["Re-enter (from Below)","Re-enter (from Above)"]
    if only_breaches:
        feed = feed[feed["Event"].isin(breaches)]
    elif not show_reentries:
        feed = feed[~feed["Event"].isin(reenters)]

    # newest on top
    if not feed.empty:
        feed = feed.sort_values("Time", ascending=False).head(int(max_rows))

        # friendly emojis
        emoji_map = {
            "Hit Floor":"🔴",
            "Break Ceiling":"🟢",
            "Re-enter (from Below)":"⚪",
            "Re-enter (from Above)":"⚪",
            "Through Range Up":"🟢⤴️",
            "Through Range Down":"🔴⤵️",
            "Init (Between)":"⚪",
            "Init (Below)":"🔴",
            "Init (Above)":"🟢",
            "Levels Unset":"⚫",
        }
        feed["Event"] = feed["Event"].map(lambda x: f"{emoji_map.get(x,'')} {x}".strip())

        # distance formatting
        def fmt_dist(x):
            if pd.isna(x): return ""
            # negative = below floor; positive outside above; inside = positive distance to nearest boundary
            return f"{x:+.2f}"
        feed["Dist to Boundary"] = feed["Dist to Boundary"].map(fmt_dist)

        st.dataframe(
            feed[["Time","Ticker","Event","Last Price","GEX Floor","GEX Ceiling","Dist to Boundary"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No cross events yet.")

    # maintenance
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🧹 Clear Feed (events)"):
            pd.DataFrame(columns=events.columns).to_csv(EVENTS_FILE, index=False)
            st.warning("Event feed cleared.")
    with c2:
        if st.button("🔄 Reset State (force init on next run)"):
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
            st.warning("State file removed. Next run will log Init events.")




st.subheader("🎯 GEX Dial Leaderboard")
st.altair_chart((bars + rules.transform_calculate(value="0") + rules.transform_calculate(value="1")), use_container_width=True)


# --- Download ledger ---
csv = ledger.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Ledger (CSV)",
    data=csv,
    file_name="trading_ledger.csv",
    mime="text/csv",
)
