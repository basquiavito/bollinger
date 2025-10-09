import streamlit as st
from notify import send_text

# 🔥 QUICK TEST — remove these 2 lines after you see the text
send_text("AMD", "CALL", "220.85", "10:30")

st.set_page_config(page_title="Options Wealth", layout="wide")

# Sidebar
st.sidebar.header("🔍 Navigation")
st.sidebar.markdown("Select a page from the sidebar menu.")
st.sidebar.markdown("---")
st.sidebar.markdown("Created by [Your Name]")

# Main content
st.title("📊 Welcome to Options Wealth")
st.markdown("- 📈 bollinger")
st.markdown("- 💰 Option Pricing")
st.markdown("-  Volatility Screener")
st.markdown("-  Analysis")
st.markdown(" Candidates")

st.markdown("term structure ")

