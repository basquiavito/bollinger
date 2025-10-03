
;                 @st.cache_data(show_spinner=False)
;                 def to_csv_bytes(df: pd.DataFrame) -> bytes:
;                     """Create CSV bytes from df (cached)."""
;                     df = clean_column_names(df)

;                     return df.to_csv(index=False).encode("utf-8")
                
                
;                 # ----------  Build once, reuse always ----------
;                 entries_df = build_entries_df(intraday).round(2)
;                 csv_bytes  = to_csv_bytes(entries_df)             # cached by df content
;                 # Force a ticker column so downstream JSON always has it
;                 # if "Ticker" not in entries_df.columns:
;                 #           entries_df["Ticker"] = tickers[0] if isinstance(tickers, list) and tickers else "UNKNOWN"
;                 entries_df["Ticker"] = entries_df.get("Ticker", entries_df.get("ticker", entries_df.get("name", "UNKNOWN")))
;                       # ✅ Ensure no NaN values so Mongo won’t choke
;                 # entries_df = entries_df.where(pd.notnull(entries_df), "")
                      
;                 #       # ✅ Force a Ticker column if missing or empty
;             # ✅ Ensure no NaN values so Mongo won’t choke
 
           
;                 #       # ✅ Fix ticker column intelligently
;                 # if "Ticker" not in entries_df.columns or entries_df["Ticker"].isnull().all() or (entries_df["Ticker"] == "").all():
;                 #     if "ticker" in entries_df.columns:
;                 #         entries_df["Ticker"] = entries_df["ticker"].astype(str).str.upper()
;                 #     elif "name" in entries_df.columns:
;                 #         entries_df["Ticker"] = entries_df["name"].astype(str).str.upper()
;                 #     else:
;                 #               # only fallback if truly nothing else available
;                 #         entries_df["Ticker"] = tickers[0] if isinstance(tickers, list) and tickers else "UNKNOWN"


;                 # keep these in session_state so other code can reuse without recompute
;                 # st.session_state.setdefault("entries_df", entries_df)
;                 # st.session_state.setdefault("entries_csv", csv_bytes)
;                 st.session_state["entries_df"] = entries_df
;                 st.session_state["entries_csv"] = csv_bytes
;                 # Optional: persist expander state across reruns
;                 st.session_state.setdefault("expand_entries", True)



;                 with st.expander("Track Entry 1 · 2 · 3 🎯", expanded=True):
;                     st.dataframe(entries_df, use_container_width=True)
                
;                     # ---------- CSV (unchanged) ----------
;                     csv_bytes = entries_df.to_csv(index=False).encode("utf-8")
;                     csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")
;                     st.markdown(
;                         f'<a href="data:text/csv;base64,{csv_b64}" download="entries.csv">⬇️ Download Entries (CSV)</a>',
;                         unsafe_allow_html=True
;                     )
;                 # ✅ CLEANUP: replace NaN with None (Mongo safe), ensure Ticker exists
;                     entries_df = entries_df.where(pd.notnull(entries_df), None)
;                     if "Ticker" not in entries_df.columns:
;                         entries_df["Ticker"] = tickers[0] if isinstance(tickers, list) and tickers else "UNKNOWN"
;                     # ---------- JSON (grouped) ----------
;                     grouped_docs = {}
                
;                     for row in entries_df.to_dict(orient="records"):
;                         # identify ticker + date key  (adjust the column names if yours differ)
;                         # ticker = row.get("name") or row.get("Ticker") or "UNKNOWN"
;                         ticker = (row.get("Ticker") or row.get("ticker") or row.get("name") or "UNKNOWN").strip().upper()

;                         date   = row["Date"]
;                         key = f"{ticker}_{date}"
                
;                         # 🎯 number extracted from the Type string, e.g. "Call 🎯2"
;                         # entry_num = row["Type"].split("🎯")[-1].strip() if "🎯" in row["Type"] else "1"
;                 # Detect Call vs Put
;                         entry_type = row.get("Type", "")
;                         side = "callPath" if "Call" in entry_type else "putPath"

;                         # create shell doc if first time
;                         if key not in grouped_docs:
;                             # ticker = row.get("Ticker") or row.get("ticker") or row.get("name")
;                             slug = f"{ticker}-{date}-{row.get('Prefix','')}-{row.get('Prototype','')}"
;                             slug = slug.lower().replace(" ", "-")
                      


;                             grouped_docs[key] = {
                              
;                                  "name": str(ticker or "UNKNOWN").lower(),

;                                 "date"      : date,
;                                  "slug": slug,   # 👈 NEW
;                                  "archive": True,   # 👈 always included by default
;                                  "cardPng":"",
;                                  "value":"",
;                                  "opus":"",
;                                  "note":"",
;                                 "Prototype" : row.get("Prototype", ""),
;                                 "label"     : row.get("Label", ""),
;                                 "suffix"    : row.get("Suffix", ""),
;                                 "prefix"    : row.get("Prefix", ""),
;                                 # "entry1"    : {                      # full data for the first entry
;                                 #     "Type" : row["Type"],
                                 

;                                 #     "Time" : row["Time"],
;                                 #     "Price ($)": row["Price ($)"],
;                                 #     "F%"   : row.get("F%", ""),
;                                 #     "Exit_Time":row.get("Exit_Time", ""),
;                                 #     "Exit_Price":row.get("Exit_Price", ""),
;                                 #     "PAE_1to2":row.get("PAE_1to2", ""),
;                                 #     "PAE_2to3":row.get("PAE_2to32", ""),
;                                 #     "PAE_3to40F":row.get("PAE_3to40F", ""),
                                  

;                                 #     "T0"   : {
;                                 #         "emoji" : row.get("T0_Emoji", ""),
;                                 #         "time"  : row.get("T0_Time",  ""),
;                                 #         "price" : row.get("T0_Price", "")
;                                 #     },

                                 
;                                 #     "T1"   : {
;                                 #         "emoji" : row.get("T1_Emoji", ""),
;                                 #         "time"  : row.get("T1_Time",  ""),
;                                 #         "price" : row.get("T1_Price", "")
;                                 #     },
                                 
;                                 #     "T2"   : {
;                                 #           "emoji" : row.get("T2_Emoji", ""),
;                                 #           "time"  : row.get("T2_Time",  ""),
;                                 #           "price" : row.get("T2_Price", "")
;                                 #       },
                            
                                                             
;                                 #     # 🔽 Add Parallel
;                                 #     "Parallel" : {
;                                 #         "emoji" : row.get("Parallel_Emoji", ""),
;                                 #         "time"  : row.get("Parallel_Time", ""),
;                                 #         "gain"  : row.get("Parallel_Gain", "")
;                                 #     },
                            
;                                 #     # 🔽 Add Goldmine E2
;                                 #     "Goldmine_E2" : {
;                                 #         "emoji" : row.get("Goldmine_E2_Emoji", ""),
;                                 #         "time"  : row.get("Goldmine_E2_Time", ""),
;                                 #         "price" : row.get("Goldmine_E2 Price", "")
;                                 #     },
                            
;                                 #     # 🔽 Add Goldmine T1
;                                 #     "Goldmine_T1" : {
;                                 #         "emoji" : row.get("Goldmine_T1_Emoji", ""),
;                                 #         "time"  : row.get("Goldmine_T1_Time", ""),
;                                 #         "price" : row.get("Goldmine_T1 Price", "")
;                                 #     }
                                  
;                                 #     # keep any other milestone fields you like...
;                                 # },

;                                 "callPath": {"entries": [], "milestones": {}},
;                                 "putPath": {"entries": [], "milestones": {}},
;                                 "extraEntries": []                   # will hold 🎯2, 🎯3 …
;                             }
;                         doc = grouped_docs[key]
;                         entry_obj = {
;                               "Type" : entry_type,
;                               "Time" : row.get("Time", ""),
;                               "Price": row.get("Price ($)", "")
;                           }

;                         if "🎯1" in entry_type:
;                             milestones = {
;                                 "T0": {"emoji": row.get("T0_Emoji", ""), "time": row.get("T0_Time", ""), "price": row.get("T0_Price", "")},
;                                 "T1": {"emoji": row.get("T1_Emoji", ""), "time": row.get("T1_Time", ""), "price": row.get("T1_Price", "")},
;                                 "T2": {"emoji": row.get("T2_Emoji", ""), "time": row.get("T2_Time", ""), "price": row.get("T2_Price", "")},
;                                 "Parallel": {"emoji": row.get("Parallel_Emoji", ""), "time": row.get("Parallel_Time", ""), "gain": row.get("Parallel_Gain", "")},
;                                 "Goldmine_E2": {"emoji": row.get("Goldmine_E2_Emoji", ""), "time": row.get("Goldmine_E2_Time", ""), "price": row.get("Goldmine_E2 Price", "")},
;                                 "Goldmine_T1": {"emoji": row.get("Goldmine_T1_Emoji", ""), "time": row.get("Goldmine_T1_Time", ""), "price": row.get("Goldmine_T1 Price", "")}
;                             }
;                             doc[side]["milestones"] = milestones
; # Always append the entry
;                         doc[side]["entries"].append(entry_obj)
;                         # else:
;                         #     # if this row is NOT 🎯1, add the minimalist checkpoint
;                         #     if entry_num != "1":
;                         #         grouped_docs[key]["extraEntries"].append({
;                         #             "Type" : row["Type"],
;                         #             "Time" : row["Time"],
;                         #             "Price": row["Price ($)"]
;                         #         })
                
;                     # final list to export
;                     json_ready = list(grouped_docs.values())
                
;                     json_str  = json.dumps(json_ready, indent=2, ensure_ascii=False)
;                     json_b64  = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
;                     st.markdown(
;                         f'<a href="data:application/json;base64,{json_b64}" download="entries.json">⬇️ Download Entries (JSON)</a>',
;                         unsafe_allow_html=True
;                     )
