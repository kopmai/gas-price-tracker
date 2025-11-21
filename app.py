import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from openai import OpenAI 
from datetime import datetime, timedelta

# --- 0. CONFIG & SECRETS ---
st.set_page_config(page_title="Energy Tracker", layout="wide", page_icon="⚡", initial_sidebar_state="collapsed")

API_KEY = st.secrets.get("GROQ_API_KEY", "")

# --- 1. CSS (OPTIMIZED & MINIFIED) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif; color: #333; overflow: hidden; }
    .stApp { background-color: #ffffff; }
    
    /* Top Bar */
    .gemini-bar {
        position: fixed; top: 0; left: 0; width: 100%; height: 60px;
        background: white; border-bottom: 1px solid #dadce0; z-index: 99999;
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 200px 0 80px; color: #1f1f1f; font-weight: 600; font-size: 20px;
    }
    .date-badge { font-size: 14px; color: #5f6368; background: #f1f3f4; padding: 4px 12px; border-radius: 20px; font-weight: 400; }

    /* Mobile Responsive */
    @media (max-width: 600px) {
        .gemini-bar { padding: 5px 10px 5px 60px; flex-direction: column; align-items: flex-start; justify-content: center; height: auto; min-height: 60px; }
        .gemini-bar span:first-child { font-size: 18px; }
        .date-badge { font-size: 11px; margin-top: 2px; }
        .main .block-container { padding-top: 85px !important; }
    }

    /* Layout & Scroll Locking */
    .main .block-container { padding: 70px 1rem 0 1rem !important; max-width: 100% !important; }
    div[data-testid="column"]:nth-of-type(1) { height: calc(100vh - 80px); overflow: hidden; padding-right: 15px; border-right: 1px solid #f0f0f0; }
    div[data-testid="column"]:nth-of-type(2) { height: calc(100vh - 80px); overflow-y: auto; padding-left: 15px; display: flex; flex-direction: column; justify-content: flex-end; }
    
    /* Components */
    div[data-testid="stMetric"] { background: #f8f9fa; border: 1px solid #eee; padding: 8px; border-radius: 6px; text-align: center; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; }
    div[data-testid="stMetricValue"] { font-size: 16px !important; font-weight: 600; }
    .stChatInput { padding-bottom: 10px; z-index: 100; }
    header[data-testid="stHeader"] { background: transparent; z-index: 100000; }
    header .decoration { display: none; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINE (CACHED & OPTIMIZED) ---
@st.cache_data(ttl=3600) # Cache 1 hour
def fetch_market_data(ticker):
    try:
        df = yf.download(ticker, period="max", progress=False)
        if df.empty: return None
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

@st.cache_data(ttl=3600)
def get_ft_data_static():
    # ข้อมูล Ft อัปเดตล่าสุด (Hardcoded for speed)
    data = [
        ("2018-01-01", -0.1590), ("2018-05-01", -0.1590), ("2018-09-01", -0.1590),
        ("2019-01-01", -0.1160), ("2019-05-01", -0.1160), ("2019-09-01", -0.1160),
        ("2020-01-01", -0.1160), ("2020-05-01", -0.1160), ("2020-09-01", -0.1243),
        ("2021-01-01", -0.1532), ("2021-05-01", -0.1532), ("2021-09-01", -0.1532),
        ("2022-01-01", 0.0139),  ("2022-05-01", 0.2477),  ("2022-09-01", 0.9343),
        ("2023-01-01", 0.9343),  ("2023-05-01", 0.9119),  ("2023-09-01", 0.2048),
        ("2024-01-01", 0.3972),  ("2024-05-01", 0.3972),  ("2024-09-01", 0.3972),
        ("2025-01-01", 0.3672),  ("2025-05-01", 0.1972),  ("2025-09-01", 0.1572)
    ]
    df = pd.DataFrame(data, columns=["Date", "Close"])
    df['Date'] = pd.to_datetime(df['Date'])
    # Upsample to daily
    idx = pd.date_range(start=df.Date.min(), end=datetime.now())
    df = df.set_index('Date').reindex(idx, method='bfill').reset_index().rename(columns={'index': 'Date'})
    return df

def get_data_point(df, target_date):
    """Find closest price on or before target_date"""
    mask = df['Date'] <= pd.Timestamp(target_date)
    if not mask.any(): return None, None
    row = df.loc[mask].iloc[-1]
    return row['Close'], row['Date']

# --- 3. CONFIGURATION ---
ASSETS = {
    "USD/THB":   {"type": "api", "ticker": "THB=X", "unit": "Baht", "curr": "THB"},
    "Ft (Thai)": {"type": "manual", "ticker": "FT",    "unit": "Baht", "curr": "THB"},
    "JKM (LNG)": {"type": "api", "ticker": "JKM=F", "unit": "$/MMBtu", "curr": "USD"}, 
    "Henry Hub": {"type": "api", "ticker": "NG=F",  "unit": "$/MMBtu", "curr": "USD"},
}

PERIODS = {"1mo":30, "3mo":90, "6mo":180, "1y":365, "5y":1825, "Max":3650}

# --- 4. SIDEBAR ---
st.sidebar.title("⚙️ Control")
with st.sidebar:
    target_date = st.date_input("📅 Select Date", value=datetime.now(), max_value=datetime.now())
    st.divider()
    sel_period = st.selectbox("⏳ Timeframe", list(PERIODS.keys()), index=4)
    st.divider()
    is_thb = st.toggle("🇹🇭 THB Convert", False)
    is_norm = st.toggle("📏 Normalize (Max=1)", False)
    st.divider()
    sel_assets = st.multiselect("Compare:", list(ASSETS.keys()), default=["Ft (Thai)", "JKM (LNG)"])

# --- 5. MAIN LOGIC ---
display_date = target_date.strftime("%d/%m/%Y")
st.markdown(f'<div class="gemini-bar"><span>⚡ Energy Price Tracker</span><span class="date-badge">📅 As of: {display_date}</span></div>', unsafe_allow_html=True)

col_dash, col_chat = st.columns([7, 3])

# === LEFT: DASHBOARD ===
with col_dash:
    # Pre-fetch baseline currencies
    thb_df = fetch_market_data("THB=X")
    
    # 5.1 METRICS ROW
    cols = st.columns(len(ASSETS))
    summary_text = f"Market Data ({display_date}):\n"
    
    for idx, (name, conf) in enumerate(ASSETS.items()):
        # Get Data
        df = get_ft_data_static() if conf["type"] == "manual" else fetch_market_data(conf["ticker"])
        
        with cols[idx]:
            if df is not None:
                price, p_date = get_data_point(df, target_date)
                
                if price is not None:
                    # Calc 1D Change
                    try:
                        prev_idx = df[df['Date'] == p_date].index[0] - 1
                        prev = df.iloc[prev_idx]['Close'] if prev_idx >= 0 else price
                        pct = ((price - prev)/prev)*100 if prev!=0 else 0
                    except: pct = 0
                    
                    # Convert Currency
                    unit = conf['unit']
                    if is_thb and conf['curr'] == 'USD' and thb_df is not None:
                        rate, _ = get_data_point(thb_df, p_date)
                        if rate: 
                            price *= rate
                            unit = "฿"
                    
                    # Display Metric with (1D) context
                    st.metric(name, f"{price:,.2f} {unit}", f"{pct:+.2f}% (1D)")
                    summary_text += f"- {name}: {price:.2f} {unit}\n"
                else: st.metric(name, "No Data", "-")
            else: st.metric(name, "Error", "-")

    # 5.2 GRAPH ROW
    if sel_assets:
        chart_data = []
        start_dt = datetime.now() - timedelta(days=PERIODS[sel_period])
        
        for name in sel_assets:
            conf = ASSETS[name]
            df = get_ft_data_static() if conf["type"] == "manual" else fetch_market_data(conf["ticker"])
            
            if df is not None:
                # Filter Timeframe
                sub = df[df['Date'] >= start_dt].copy()
                
                # Convert Logic
                if is_thb and conf['curr'] == 'USD' and thb_df is not None:
                    merged = pd.merge(sub, thb_df[['Date', 'Close']], on='Date', how='inner', suffixes=('', '_R'))
                    sub['Close'] *= merged['Close_R']
                
                # Normalize Logic
                label = name
                if is_norm:
                    mx = sub['Close'].max()
                    if mx != 0: 
                        sub['Close'] /= mx
                        label = f"{name} (Norm)"
                
                sub['Asset'] = label
                chart_data.append(sub[['Date', 'Close', 'Asset']])
        
        if chart_data:
            final_df = pd.concat(chart_data)
            fig = px.line(final_df, x='Date', y='Close', color='Asset', template="plotly_white")
            
            # Add selected date line
            fig.add_vline(x=datetime.combine(target_date, datetime.min.time()).timestamp() * 1000, 
                          line_dash="dash", line_color="red")
            
            # Freeze Layout for Mobile
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0), height=600, hovermode="x unified",
                xaxis_title=None, yaxis_title="Normalized" if is_norm else "Price",
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                dragmode=False
            )
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False, 'showTips': False})
    else:
        st.info("Select assets to view graph")

# === RIGHT: CHAT ===
with col_chat:
    st.markdown("##### 💬 AI Analyst")
    if "msgs" not in st.session_state: st.session_state.msgs = []
    
    # Auto Analysis (First Run)
    if not st.session_state.msgs and API_KEY:
        try:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Briefly analyze energy market on {display_date} based on: {summary_text} (in Thai)"}]
            )
            st.session_state.msgs.append({"role": "assistant", "content": f"**Analysis ({display_date}):**\n{res.choices[0].message.content}"})
        except: pass

    # Chat UI
    for m in st.session_state.msgs:
        st.chat_message(m["role"], avatar="👤" if m["role"]=="user" else "🤖").write(m["content"])

    if prompt := st.chat_input("Ask AI..."):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        st.rerun()

    # Response Logic
    if st.session_state.msgs and st.session_state.msgs[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🤖"):
            try:
                client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)
                stream = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": "Energy analyst. Thai language."}, 
                              *[{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs]],
                    stream=True
                )
                st.session_state.msgs.append({"role": "assistant", "content": st.write_stream(stream)})
            except Exception as e: st.error(str(e))
