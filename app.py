import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from openai import OpenAI 
from datetime import datetime, timedelta

# --- 0. CONFIG & SECRETS ---
st.set_page_config(page_title="Energy Tracker", layout="wide", page_icon="⚡", initial_sidebar_state="collapsed")
API_KEY = st.secrets.get("GROQ_API_KEY", "")

# --- 1. DATA ENGINE ---
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
    ft_data = [
        ("2021-01-01", -0.1532), ("2021-05-01", -0.1532), ("2021-09-01", -0.1532),
        ("2022-01-01", 0.0139),  ("2022-05-01", 0.2477),  ("2022-09-01", 0.9343),
        ("2023-01-01", 0.9343),  ("2023-05-01", 0.9119),  ("2023-09-01", 0.2048),
        ("2024-01-01", 0.3972),  ("2024-05-01", 0.3972),  ("2024-09-01", 0.3972),
        ("2025-01-01", 0.3672),  ("2025-05-01", 0.1972),  ("2025-09-01", 0.1572)
    ]
    pool_gas_data = [
        ("2021-01-01", 216.00), ("2021-05-01", 225.50), ("2021-09-01", 280.00),
        ("2022-01-01", 350.00), ("2022-05-01", 400.50), ("2022-09-01", 560.00),
        ("2023-01-01", 438.28), ("2023-05-01", 389.00), ("2023-09-01", 304.79),
        ("2024-01-01", 318.25), ("2024-05-01", 309.00), ("2024-09-01", 297.00),
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
    "ราคาตลาด Henry Hub": {"type": "api", "ticker": "NG=F",  "unit": "$/MMBtu", "curr": "USD"},
    "อัตราแลกเปลี่ยน (USD/THB)": {"type": "api", "ticker": "THB=X", "unit": "Baht", "curr": "THB", "is_ref": True},
    "ค่าไฟฟ้าผันแปร (Ft)": {"type": "manual_ft", "ticker": "FT",    "unit": "Baht/Unit", "curr": "THB"},
}
PERIODS = {"1mo":30, "3mo":90, "6mo":180, "1y":365, "5y":1825, "Max":3650}

# --- 4. SIDEBAR ---
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
    is_thb = st.toggle("🇺🇸 Show in USD", value=False)
    is_norm = st.toggle("📏 Normalize (Max=1)", value=False) 
    st.divider()
    sel_assets = st.multiselect("Compare:", list(ASSETS.keys()), default=["ราคา Pool Gas (Thai)", "ราคาตลาด JKM", "อัตราแลกเปลี่ยน (USD/THB)", "ค่าไฟฟ้าผันแปร (Ft)"])

# --- 5. CSS (THE ULTIMATE FONT FIX) ---
bg_color = "#0e1117" if is_dark else "#ffffff"
text_color = "#ffffff" if is_dark else "#333333" 
card_bg = "#1e1e1e" if is_dark else "#f8f9fa" 
border_color = "#444" if is_dark else "#e9ecef"
topbar_bg = "#161b22" if is_dark else "#ffffff"
input_bg = "#262730" if is_dark else "#ffffff" 

st.markdown(f"""
<style>
    /* --- [NEW] Import 'Kanit' Font --- */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600&display=swap');
    
    /* --- [NUCLEAR FONT FIX] Apply to EVERYTHING --- */
    html, body, [class*="css"], .stApp, 
    h1, h2, h3, h4, h5, h6, p, div, span, a, button, input, select, textarea, label, li {{ 
        font-family: 'Kanit', sans-serif !important; 
        color: {text_color} !important;
    }}
    /* --------------------------------------------- */
    
    .stApp {{ background-color: {bg_color}; overflow: hidden; }}
    
    /* --- Dropdown & Popover Fixes --- */
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"] {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color};
    }}
    
    ul[data-baseweb="menu"] li,
    li[role="option"] {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
    }}
    
    ul[data-baseweb="menu"] li[aria-selected="true"] {{
        background-color: #ff4b4b !important;
        color: white !important;
    }}
    
    /* Calendar Fix */
    div[data-baseweb="calendar"] {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
    }}
    div[data-baseweb="calendar"] div[aria-label]:hover {{
        background-color: #ff4b4b !important;
        color: white !important;
        cursor: pointer;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {bg_color} !important;
        border-right: 1px solid {border_color};
    }}

    /* Reset Button */
    [data-testid="stSidebar"] button {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        width: 100%;
    }}
    [data-testid="stSidebar"] button:hover {{
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
    }}

    /* Inputs */
    .stSelectbox div[data-baseweb="select"] > div,
    .stDateInput div[data-baseweb="input"],
    .stMultiSelect div[data-baseweb="select"] > div {{
        color: {text_color} !important;
        background-color: {input_bg} !important;
        border-color: {border_color} !important;
    }}
    .stDateInput input {{ color: {text_color} !important; }}
    
    /* Toggle */
    [data-testid="stCheckbox"] label {{ opacity: 1 !important; font-weight: 500; }}

    /* Top Bar */
    .gemini-bar {{
        position: fixed; top: 0; left: 0; width: 100%; height: 60px;
        background: {topbar_bg}; border-bottom: 1px solid {border_color}; z-index: 99999;
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 200px 0 80px;
    }}
    .gemini-bar span {{ font-weight: 600; font-size: 20px; font-family: 'Kanit', sans-serif !important; }}
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
        color: {text_color} !important;
    }}
    [data-testid="stSidebarCollapsedControl"] svg {{ display: none !important; }}
    [data-testid="stSidebarCollapsedControl"]::after {{ content: "⚙️"; font-size: 22px; margin-bottom: 3px; }}

    .main .block-container {{ 
        padding-top: 65px !important; 
        padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important; 
    }}
    .main .block-container h3 {{ margin-top: 0 !important; padding-top: 10px !important; }}
    
    div[data-testid="column"]:nth-of-type(1) {{ height: calc(100vh - 80px); overflow: hidden; padding-right: 15px; border-right: 1px solid {border_color}; }}
    div[data-testid="column"]:nth-of-type(2) {{ height: calc(100vh - 80px); overflow-y: auto; padding-left: 15px; display: flex; flex-direction: column; justify-content: flex-end; }}
    
    .custom-card {{
        background-color: {card_bg}; border: 1px solid {border_color}; border-radius: 10px; padding: 15px 10px; text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column; justify-content: center; gap: 2px;
    }}
    .card-title {{ font-size: 12px; opacity: 0.7; margin-bottom: 2px; font-family: 'Kanit', sans-serif !important; }}
    .card-price {{ font-size: 22px; font-weight: 600; line-height: 1.2; font-family: 'Kanit', sans-serif !important; }}
    .card-unit  {{ font-size: 13px; opacity: 0.9; margin-bottom: 5px; font-family: 'Kanit', sans-serif !important; }}
    .card-delta {{ font-size: 13px; font-weight: 500; padding: 2px 8px; border-radius: 12px; display: inline-block; font-family: 'Kanit', sans-serif !important; }}
    
    .delta-pos {{ color: #00cc66 !important; background: rgba(0,204,102,0.15); }}
    .delta-neg {{ color: #ff4d4d !important; background: rgba(255,77,77,0.15); }}
    .delta-neu {{ color: #aaa !important; background: rgba(128,128,128,0.15); }}

    .stChatInput {{ padding-bottom: 10px; z-index: 100; }}
    header[data-testid="stHeader"] {{ background: transparent; z-index: 100000; }}
    header .decoration {{ display: none; }}
    
    @media (max-width: 600px) {{
        .gemini-bar {{ padding: 5px 10px 5px 65px; flex-direction: column; align-items: flex-start; justify-content: center; height: auto; min-height: 60px; }}
        .date-badge {{ font-size: 11px; margin-top: 2px; }}
        .main .block-container {{ padding-top: 85px !important; }}
        [data-testid="stSidebarCollapsedControl"] {{ top: 10px !important; left: 10px !important; width: 35px; height: 35px; }}
    }}
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
    
    cols = st.columns(len(ASSETS))
    summary_text = f"Market Data ({display_date}):\n"
    for idx, (name, conf) in enumerate(ASSETS.items()):
        if conf["type"] == "manual_ft": df = get_manual_data("ft")
        elif conf["type"] == "manual_pool": df = get_manual_data("pool")
        else: df = fetch_market_data(conf["ticker"])
        
        with cols[idx]:
            card_html = ""
            if df is not None:
                price, p_date = get_data_point(df, target_date)
                if price is not None and not pd.isna(price):
                    try:
                        curr_idx = df[df['Date'] == p_date].index[0]
                        prev_idx = curr_idx - 1
                        prev = df.iloc[prev_idx]['Close'] if prev_idx >= 0 else price
                        pct = ((price - prev)/prev)*100 if prev!=0 else 0
                        
                        idx_1y = max(0, curr_idx - 252)
                        price_1y = df.iloc[idx_1y]['Close']
                        pct_yoy = ((price - price_1y)/price_1y)*100 if price_1y!=0 else 0
                    except: 
                        pct = 0
                        pct_yoy = 0
                    
                    unit = conf['unit']
                    if not conf.get("is_ref"):
                        rate, _ = get_data_point(thb_df, p_date)
                        if rate:
                            if is_thb: 
                                if conf['curr'] == "THB":
                                    price /= rate
                                    unit = unit.replace("Baht", "$").replace("บาท", "$")
                            else: 
                                if conf['curr'] == "USD":
                                    price *= rate
                                    unit = unit.replace("$", "Baht")

                    delta_class = "delta-pos" if pct > 0 else ("delta-neg" if pct < 0 else "delta-neu")
                    arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "•")
                    
                    card_html = f"""
                    <div class="custom-card">
                        <div class="card-title">{name}</div>
                        <div class="card-price">{price:,.2f}</div>
                        <div class="card-unit">{unit}</div>
                        <div><span class="card-delta {delta_class}">{arrow} {pct:+.2f}% (1D)</span></div>
                    </div>
                    """
                    summary_text += f"- {name}: {price:.2f} {unit} (Change 1D: {pct:+.2f}%, YoY: {pct_yoy:+.2f}%)\n"
                else:
                    card_html = f"""<div class="custom-card"><div class="card-title">{name}</div><div class="card-price">No Data</div></div>"""
            else:
                card_html = f"""<div class="custom-card"><div class="card-title">{name}</div><div class="card-price">Error</div></div>"""
            st.markdown(card_html, unsafe_allow_html=True)

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
                
                if not conf.get("is_ref") and thb_df is not None:
                    thb_clean = thb_df.copy()
                    thb_clean['Date'] = pd.to_datetime(thb_clean['Date']).dt.tz_localize(None)
                    thb_lookup = thb_clean.set_index('Date')['Close'].sort_index().ffill()
                    rates = thb_lookup.asof(sub['Date'])
                    
                    if is_thb: 
                        if conf['curr'] == "THB": sub['Close'] = sub['Close'] / rates.values
                    else:
                        if conf['curr'] == "USD": sub['Close'] = sub['Close'] * rates.values
                
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
                font=dict(color=text_color, family="Kanit"),
                margin=dict(l=0, r=0, t=30, b=0), height=600, hovermode="x unified",
                xaxis_title=None, yaxis_title="Normalized" if is_norm else ("Price (USD)" if is_thb else "Price (THB)"),
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                dragmode=False,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(fixedrange=True, range=[start_dt, datetime.now()], showgrid=False)
            fig.update_yaxes(fixedrange=True, range=[y_min - padding, y_max + padding], showgrid=True, gridcolor=border_color)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False, 'showTips': False})
    else: st.info("Select assets")

# === RIGHT ===
with col_chat:
    st.markdown("##### 💬 AI Analyst")
    if "msgs" not in st.session_state: st.session_state.msgs = []
    
    if not st.session_state.msgs and API_KEY:
        try:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)
            prompt_text = f"""
            วิเคราะห์สถานการณ์ตลาดพลังงาน ณ วันที่ {display_date}:
            {summary_text}
            
            โจทย์การวิเคราะห์ (ภาษาไทย):
            1. เปรียบเทียบราคาวันนี้กับ "ปีที่แล้ว (YoY)" ว่าเปลี่ยนแปลงอย่างไร สูงขึ้นหรือต่ำลง
            2. วิเคราะห์ผลกระทบของ Geopolitics (ภูมิรัฐศาสตร์) ในช่วงเวลานั้น ที่มีผลต่อราคา (เช่น สงคราม, ฤดูหนาว, Supply Chain)
            3. สรุปผลกระทบต่อต้นทุนค่าไฟฟ้า (Ft) ของไทย
            
            ตอบสั้นๆ กระชับ แบบผู้บริหาร
            """
            
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_text}]
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
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": "You are an Energy & Geopolitics Strategist. Answer in Thai. Be professional and concise."}, 
                              *[{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs]], stream=True
                )
                st.session_state.msgs.append({"role": "assistant", "content": st.write_stream(stream)})
            except Exception as e: st.error(str(e))
