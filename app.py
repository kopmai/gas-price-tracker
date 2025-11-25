import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from openai import OpenAI 
from datetime import datetime, timedelta

# --- 0. CONFIG & SECRETS ---
st.set_page_config(page_title="Energy Tracker", layout="wide", page_icon="⚡", initial_sidebar_state="collapsed")
API_KEY = st.secrets.get("GROQ_API_KEY", "")

# --- 1. CSS ---
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

    /* Sidebar Button */
    [data-testid="stSidebarCollapsedControl"] {
        z-index: 100000 !important; background-color: white; border-radius: 50%; width: 40px; height: 40px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15); border: 1px solid #eee; top: 10px !important; left: 15px !important;
        display: flex; align-items: center; justify-content: center;
    }
    [data-testid="stSidebarCollapsedControl"] svg { display: none !important; }
    [data-testid="stSidebarCollapsedControl"]::after { content: "⚙️"; font-size: 22px; margin-bottom: 3px; }
    [data-testid="stSidebarCollapsedControl"]:hover { transform: rotate(45deg); transition: transform 0.3s ease; background-color: #f1f3f4; }

    /* Responsive */
    @media (max-width: 600px) {
        .gemini-bar { padding: 5px 10px 5px 65px; flex-direction: column; align-items: flex-start; justify-content: center; height: auto; min-height: 60px; }
        .gemini-bar span:first-child { font-size: 18px; }
        .date-badge { font-size: 11px; margin-top: 2px; }
        .main .block-container { padding-top: 85px !important; }
        [data-testid="stSidebarCollapsedControl"] { top: 10px !important; left: 10px !important; width: 35px; height: 35px; }
    }

    .main .block-container { padding: 70px 1rem 0 1rem !important; max-width: 100% !important; }
    div[data-testid="column"]:nth-of-type(1) { height: calc(100vh - 80px); overflow: hidden; padding-right: 15px; border-right: 1px solid #f0f0f0; }
    div[data-testid="column"]:nth-of-type(2) { height: calc(100vh - 80px); overflow-y: auto; padding-left: 15px; display: flex; flex-direction: column; justify-content: flex-end; }
    
    div[data-testid="stMetric"] { background: #f8f9fa; border: 1px solid #eee; padding: 8px; border-radius: 6px; text-align: center; }
    div[data-testid="stMetricLabel"] { font-size: 14px !important; font-weight: 500; }
    div[data-testid="stMetricValue"] { font-size: 18px !important; font-weight: 600; }
    .stChatInput { padding-bottom: 10px; z-index: 100; }
    header[data-testid="stHeader"] { background: transparent; z-index: 100000; }
    header .decoration { display: none; }
    button[kind="secondary"] { width: 100%; border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_market_data(ticker):
    try:
        df = yf.download(ticker, period="max", progress=False)
        if df.empty: return None
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
        return df.sort_values('Date')
    except: return None

@st.cache_data(ttl=3600)
def get_manual_data(data_type):
    # 1.1 FT Data (บาท/หน่วย) - ย้อนหลังถึง 2021
    ft_data = [
        ("2021-01-01", -0.1532), ("2021-05-01", -0.1532), ("2021-09-01", -0.1532),
        ("2022-01-01", 0.0139),  ("2022-05-01", 0.2477),  ("2022-09-01", 0.9343),
        ("2023-01-01", 0.9343),  ("2023-05-01", 0.9119),  ("2023-09-01", 0.2048),
        ("2024-01-01", 0.3972),  ("2024-05-01", 0.3972),  ("2024-09-01", 0.3972),
        ("2025-01-01", 0.3672),  ("2025-05-01", 0.1972),  ("2025-09-01", 0.1572)
    ]
    
    # 1.2 Pool Gas Data (บาท/MMBtu) - ย้อนหลังถึง 2021 (ช่วงวิกฤต)
    pool_gas_data = [
        # 2021: เริ่มขยับขึ้นปลายปี
        ("2021-01-01", 216.00), ("2021-05-01", 225.50), ("2021-09-01", 280.00),
        # 2022: ปีแห่งวิกฤตสงครามรัสเซีย-ยูเครน (พุ่งสูงสุด)
        ("2022-01-01", 350.00), ("2022-05-01", 400.50), ("2022-09-01", 560.00), # Peak Crisis
        # 2023: ราคาสูงค้างฟ้า แล้วค่อยๆ ลง
        ("2023-01-01", 438.28), ("2023-05-01", 389.00), ("2023-09-01", 304.79),
        # 2024: เริ่มทรงตัว
        ("2024-01-01", 318.25), ("2024-05-01", 309.00), ("2024-09-01", 297.00),
        # 2025: แนวโน้ม
        ("2025-01-01", 301.00), ("2025-05-01", 300.29), ("2025-10-01", 270.10)
    ]

    source = ft_data if data_type == "ft" else pool_gas_data
    df = pd.DataFrame(source, columns=["Date", "Close"])
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    
    today = pd.Timestamp.now().normalize()
    if df['Date'].max() < today:
        new_row = pd.DataFrame({"Date": [today], "Close": [None]})
        df = pd.concat([df, new_row], ignore_index=True)
        
    idx = pd.date_range(start=df.Date.min(), end=today)
    # ใช้ ffill เพื่อลากเส้นค่าคงที่ (Step Line)
    df = df.set_index('Date').reindex(idx).ffill().reset_index().rename(columns={'index': 'Date'})
    return df

def get_data_point(df, target_date):
    mask = df['Date'] <= pd.Timestamp(target_date).normalize()
    if not mask.any(): return None, None
    row = df.loc[mask].iloc[-1]
    return row['Close'], row['Date']

# --- 3. CONFIG ---
ASSETS = {
    "ราคา Pool Gas (Thai)": {"type": "manual_pool", "ticker": "POOL", "unit": "Baht/MMBtu", "curr": "THB"},
    "ราคาตลาด JKM": {"type": "api", "ticker": "JKM=F", "unit": "$/MMBtu", "curr": "USD"},
    "อัตราแลกเปลี่ยน (USD/THB)": {"type": "api", "ticker": "THB=X", "unit": "Baht", "curr": "THB"},
    "ค่าไฟฟ้าผันแปร (Ft)": {"type": "manual_ft", "ticker": "FT",    "unit": "Baht/Unit", "curr": "THB"},
}
PERIODS = {"1mo":30, "3mo":90, "6mo":180, "1y":365, "5y":1825, "Max":3650}

# --- 4. SIDEBAR & TOGGLES ---
st.sidebar.title("⚙️ Control Panel")
with st.sidebar:
    is_dark = st.toggle("🌗 Dark Mode", value=False)
    if st.button("🔄 Reset to Today"):
        st.session_state.date_selector = datetime.now()
        st.rerun()

    target_date = st.date_input("Pick a date", value=datetime.now(), max_value=datetime.now(), key="date_selector", label_visibility="collapsed")
    st.divider()
    sel_period = st.selectbox("⏳ Timeframe", list(PERIODS.keys()), index=4)
    st.divider()
    is_thb = st.toggle("🇹🇭 THB Convert", False)
    is_norm = st.toggle("📏 Normalize (Max=1)", False)
    st.divider()
    sel_assets = st.multiselect("Compare:", list(ASSETS.keys()), default=["ราคา Pool Gas (Thai)", "ราคาตลาด JKM"])

# --- 5. DYNAMIC CSS ---
bg_color = "#0e1117" if is_dark else "#ffffff"
text_color = "#fafafa" if is_dark else "#333333"
card_bg = "#262730" if is_dark else "#f8f9fa"
topbar_bg = "#1e1e1e" if is_dark else "#ffffff"
border_color = "#444" if is_dark else "#dadce0"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Prompt', sans-serif; color: {text_color}; overflow: hidden; }}
    .stApp {{ background-color: {bg_color}; }}
    
    .gemini-bar {{
        position: fixed; top: 0; left: 0; width: 100%; height: 60px;
        background: {topbar_bg}; border-bottom: 1px solid {border_color}; z-index: 99999;
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 200px 0 80px; color: {text_color}; font-weight: 600; font-size: 20px;
    }}
    .date-badge {{ 
        font-size: 14px; color: {text_color}; 
        background: {'#333' if is_dark else '#f1f3f4'}; 
        padding: 4px 12px; border-radius: 20px; font-weight: 400; border: 1px solid {border_color};
    }}

    [data-testid="stSidebarCollapsedControl"] {{
        z-index: 100000 !important; background-color: {topbar_bg}; 
        border-radius: 50%; width: 40px; height: 40px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15); border: 1px solid {border_color}; 
        top: 10px !important; left: 15px !important;
        display: flex; align-items: center; justify-content: center;
        color: {text_color};
    }}
    [data-testid="stSidebarCollapsedControl"] svg {{ display: none !important; }}
    [data-testid="stSidebarCollapsedControl"]::after {{ content: "⚙️"; font-size: 22px; margin-bottom: 3px; }}
    [data-testid="stSidebarCollapsedControl"]:hover {{ transform: rotate(45deg); transition: transform 0.3s ease; opacity: 0.8; }}

    @media (max-width: 600px) {{
        .gemini-bar {{ padding: 5px 10px 5px 65px; flex-direction: column; align-items: flex-start; justify-content: center; height: auto; min-height: 60px; }}
        .gemini-bar span:first-child {{ font-size: 18px; }}
        .date-badge {{ font-size: 11px; margin-top: 2px; }}
        .main .block-container {{ padding-top: 85px !important; }}
        [data-testid="stSidebarCollapsedControl"] {{ top: 10px !important; left: 10px !important; width: 35px; height: 35px; }}
    }}

    .main .block-container {{ padding: 70px 1rem 0 1rem !important; max-width: 100% !important; }}
    div[data-testid="column"]:nth-of-type(1) {{ height: calc(100vh - 80px); overflow: hidden; padding-right: 15px; border-right: 1px solid {border_color}; }}
    div[data-testid="column"]:nth-of-type(2) {{ height: calc(100vh - 80px); overflow-y: auto; padding-left: 15px; display: flex; flex-direction: column; justify-content: flex-end; }}
    
    div[data-testid="stMetric"] {{ background: {card_bg}; border: 1px solid {border_color}; padding: 8px; border-radius: 6px; text-align: center; }}
    div[data-testid="stMetricLabel"] {{ font-size: 14px !important; font-weight: 500; color: {text_color}; opacity: 0.8; }}
    div[data-testid="stMetricValue"] {{ font-size: 18px !important; font-weight: 600; color: {text_color}; }}
    
    .stChatInput {{ padding-bottom: 10px; z-index: 100; }}
    header[data-testid="stHeader"] {{ background: transparent; z-index: 100000; }}
    header .decoration {{ display: none; }}
    button[kind="secondary"] {{ width: 100%; border: 1px solid {border_color}; color: {text_color}; }}
</style>
""", unsafe_allow_html=True)

# --- 6. MAIN LOGIC ---
display_date = target_date.strftime("%d/%m/%Y")
st.markdown(f'<div class="gemini-bar"><span>⚡ Energy Price Tracker</span><span class="date-badge">📅 As of: {display_date}</span></div>', unsafe_allow_html=True)
col_dash, col_chat = st.columns([7, 3])

# === LEFT ===
with col_dash:
    st.subheader("สรุปภาพรวมตลาดวันนี้")
    thb_df = fetch_market_data("THB=X")
    
    # 6.1 Metrics Loop
    cols = st.columns(len(ASSETS))
    summary_text = f"Market Data ({display_date}):\n"
    for idx, (name, conf) in enumerate(ASSETS.items()):
        if conf["type"] == "manual_ft": df = get_manual_data("ft")
        elif conf["type"] == "manual_pool": df = get_manual_data("pool")
        else: df = fetch_market_data(conf["ticker"])
        
        with cols[idx]:
            if df is not None:
                price, p_date = get_data_point(df, target_date)
                if price is not None and not pd.isna(price):
                    try:
                        curr_idx = df[df['Date'] == p_date].index[0]
                        prev_idx = curr_idx - 1
                        prev = df.iloc[prev_idx]['Close'] if prev_idx >= 0 else price
                        pct = ((price - prev)/prev)*100 if prev!=0 else 0
                    except: pct = 0
                    
                    unit = conf['unit']
                    if is_thb and conf['curr'] == 'USD' and thb_df is not None:
                        rate, _ = get_data_point(thb_df, p_date)
                        if rate: price *= rate; unit = "Baht/MMBtu" if "MMBtu" in unit else "Baht"
                    
                    st.metric(name, f"{price:,.2f} {unit}", f"{pct:+.2f}% (1D)")
                    summary_text += f"- {name}: {price:.2f} {unit}\n"
                else: st.metric(name, "No Data", "-")
            else: st.metric(name, "Error", "-")

    # 6.2 Graph Logic
    if sel_assets:
        chart_data = []
        start_dt = (datetime.now() - timedelta(days=PERIODS[sel_period])).replace(hour=0, minute=0, second=0, microsecond=0)
        
        for name in sel_assets:
            conf = ASSETS[name]
            if conf["type"] == "manual_ft": df = get_manual_data("ft")
            elif conf["type"] == "manual_pool": df = get_manual_data("pool")
            else: df = fetch_market_data(conf["ticker"])
            
            if df is not None:
                sub = df[df['Date'] >= start_dt].copy()
                sub['Date'] = pd.to_datetime(sub['Date']).dt.tz_localize(None)
                
                if is_thb and conf['curr'] == 'USD' and thb_df is not None:
                    thb_clean = thb_df.copy()
                    thb_clean['Date'] = pd.to_datetime(thb_clean['Date']).dt.tz_localize(None)
                    sub = sub.set_index('Date')
                    thb_indexed = thb_clean.set_index('Date')
                    sub['Rate'] = thb_indexed['Close'].reindex(sub.index, method='ffill')
                    sub['Close'] = sub['Close'] * sub['Rate']
                    sub = sub.reset_index()
                
                label = name
                if is_norm:
                    mx = sub['Close'].max()
                    if mx != 0 and not pd.isna(mx): sub['Close'] /= mx; label = f"{name} (Norm)"
                
                sub['Asset'] = label
                sub = sub.dropna(subset=['Close'])
                chart_data.append(sub[['Date', 'Close', 'Asset']])
        
        if chart_data:
            final_df = pd.concat(chart_data)
            y_vals = final_df['Close']
            y_min, y_max = y_vals.min(), y_vals.max()
            if pd.isna(y_max): y_max = 1
            if pd.isna(y_min): y_min = 0
            padding = (y_max - y_min) * 0.1 if y_max != y_min else (y_max * 0.1 if y_max !=0 else 1.0)
            
            template = "plotly_dark" if is_dark else "plotly_white"
            fig = px.line(final_df, x='Date', y='Close', color='Asset', template=template)
            fig.update_traces(connectgaps=True)
            fig.add_vline(x=datetime.combine(target_date, datetime.min.time()).timestamp() * 1000, line_dash="dash", line_color="red")
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0), height=600, hovermode="x unified",
                xaxis_title=None, yaxis_title="Normalized" if is_norm else "Price",
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                dragmode=False,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(fixedrange=True, range=[start_dt, datetime.now()])
            fig.update_yaxes(fixedrange=True, range=[y_min - padding, y_max + padding])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False, 'showTips': False})
    else: st.info("Select assets")

# === RIGHT ===
with col_chat:
    st.markdown("##### 💬 AI Analyst")
    if "msgs" not in st.session_state: st.session_state.msgs = []
    
    if not st.session_state.msgs and API_KEY:
        try:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Analyze energy market data on {display_date}: {summary_text}. Provide a concise executive summary in English."}]
            )
            st.session_state.msgs.append({"role": "assistant", "content": f"**Analysis ({display_date}):**\n{res.choices[0].message.content}"})
        except: pass

    for m in st.session_state.msgs:
        st.chat_message(m["role"], avatar="👤" if m["role"]=="user" else "🤖").write(m["content"])

    if prompt := st.chat_input("Ask AI (English/Thai)..."):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.msgs and st.session_state.msgs[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🤖"):
            try:
                client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)
                stream = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": "You are a professional energy analyst. Answer in English. Be concise and data-driven."}, 
                              *[{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs]], stream=True
                )
                st.session_state.msgs.append({"role": "assistant", "content": st.write_stream(stream)})
            except Exception as e: st.error(str(e))
