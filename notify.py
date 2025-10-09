from twilio.rest import Client
import streamlit as st

def send_text(symbol, direction, price, time_str):
    sid   = st.secrets["TWILIO_SID"]
    token = st.secrets["TWILIO_AUTH"]
    from_ = st.secrets["TWILIO_FROM"]
    to_   = st.secrets["TWILIO_TO"]

    body = f"🎯 Entry 1 {symbol} {direction} @ {price}  ({time_str})"
    Client(sid, token).messages.create(to=to_, from_=from_, body=body)
