for t in selected_tickers:
    try:
        # --- filter per ticker ---
        df = intraday.loc[intraday["Ticker"] == t].sort_values("Time").copy()
        if df.empty:
            st.warning(f"No data for {t}")
            continue

        # --- indicators per ticker ---
        win = 20
        df["BB_MA"]    = df["Close"].rolling(win, min_periods=1).mean()
        df["BB_STD"]   = df["Close"].rolling(win, min_periods=1).std()
        df["BB_Upper"] = df["BB_MA"] + 2 * df["BB_STD"]
        df["BB_Lower"] = df["BB_MA"] - 2 * df["BB_STD"]
        df["RVOL_5"]   = df["Volume"] / df["Volume"].rolling(5, min_periods=1).mean()

        import numpy as np
        vol_colors = np.where(df["Close"] >= df["Open"],
                              "rgba(0,204,150,0.55)",
                              "rgba(239,85,59,0.55)")

        with st.expander(f"🕯️ {t} — Candles + Ichimoku + BB + Volume", expanded=True):
            fig = go.Figure()

            # candles
            fig.add_trace(go.Candlestick(
                x=df["Time"], open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Candles",
                hovertemplate="Open: %{open:.2f}<br>Close: %{close:.2f}<extra></extra>"
            ))

            # ichimoku
            fig.add_trace(go.Scatter(x=df["Time"], y=df["Tenkan"], line=dict(color="red"),   name="Tenkan-sen", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["Kijun"],  line=dict(color="green"), name="Kijun-sen",  hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["SpanA"],  line=dict(color="yellow"),name="Span A",     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["SpanB"],  line=dict(color="blue"),  name="Span B",     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["Chikou"], line=dict(color="purple"),name="Chikou",     hoverinfo="skip"))

            # BB cloud + lines
            fig.add_trace(go.Scatter(x=df["Time"], y=df["BB_Upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["BB_Lower"], fill="tonexty",
                                     fillcolor="rgba(128,128,128,0.15)", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["BB_Upper"], line=dict(color="darkgray", width=1.5), name="BB Upper",  hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["BB_MA"],    line=dict(color="white",    width=1, dash="dash"), name="BB Middle", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["BB_Lower"], line=dict(color="darkgray", width=1.5), name="BB Lower",  hoverinfo="skip"))

            # volume (with RVOL in hover)
            fig.add_trace(go.Bar(
                x=df["Time"], y=df["Volume"], name="Volume",
                marker_color=vol_colors, yaxis="y2",
                customdata=np.column_stack([df["RVOL_5"]]),
                hovertemplate="Volume: %{y:,}<br>RVOL(5): %{customdata[0]:.2f}×<extra></extra>"
            ))

            # layout
            fig.update_layout(
                title=f"{t} — Ichimoku + Bollinger + Volume",
                height=540, xaxis_rangeslider_visible=False, hovermode="x unified",
                margin=dict(l=30, r=30, t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Price",  domain=[0.32, 1.0]),
                yaxis2=dict(title="Volume", domain=[0.00, 0.27], showgrid=False),
                bargap=0.0, bargroupgap=0.0,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )

            st.plotly_chart(fig, use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True, "doubleClick": "reset", "responsive": True}
            )

    except Exception as e:
        st.error(f"Error rendering {t}: {e}")
