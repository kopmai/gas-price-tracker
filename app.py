import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from openai import OpenAI 
from datetime import datetime, timedelta

# --- 0. API KEY ---
# ลองดึงจาก Secrets ของระบบก่อน (สำหรับตอน Deploy)
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = "" # ถ้าไม่มี ให้ไปรอรับจาก Sidebar หรือปล่อยว่าง

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Gas Price Tracking", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="collapsed" 
)

# --- CSS ปรับแต่ง (Dashboard Mode: Lock Screen) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
        color: #333; 
        overflow: hidden; /* 🔒 ล็อกหน้าจอหลักห้ามเลื่อน */
    }
    .stApp { background-color: #ffffff; }

    /* Top Bar */
    .gemini-bar {
        position: fixed; top: 0; left: 0; width: 100%; height: 60px;
        background-color: #ffffff; border-bottom: 1px solid #dadce0;
        z-index: 99999; display: flex; align-items: center;
        padding-left: 80px; font-size: 20px; font-weight: 600; color: #1f1f1f;
    }
    
    .main .block-container { 
        padding-top: 70px !important; 
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* ซ้าย: Dashboard (Fixed) */
    div[data-testid="column"]:nth-of-type(1) {
        height: calc(100vh - 80px);
        overflow: hidden; 
        padding-right: 15px;
        border-right: 1px solid #f0f0f0;
    }

    /* ขวา: Chat (Scrollable) */
    div[data-testid="column"]:nth-of-type(2) {
        height: calc(100vh - 80px);
        overflow-y: auto;
        padding-left: 15px;
        display: flex; flex-direction: column; justify-content: flex-end;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa; border: 1px solid #eee;
        padding: 8px; border-radius: 6px; text-align: center;
    }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; }
    div[data-testid="stMetricValue"] { font-size: 16px !important; font-weight: 600; }

    /* Chat Input */
    .stChatInput { padding-bottom: 10px; z-index: 100; }

    header[data-testid="stHeader"] { background: transparent; z-index: 100000; }
    header .decoration { display: none; }
</style>
""", unsafe_allow_html=True)

# --- Top Bar ---
st.markdown('<div class="gemini-bar"><span>⚡ Energy Price Tracker</span></div>', unsafe_allow_html=True)

# --- Functions ---
@st.cache_data(ttl=300)
def get_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period="max", progress=False)
        if data.empty: return None, "No Data"
        data.reset_index(inplace=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if period != "max":
            days_map = {"1mo":30, "3mo":90, "6mo":180, "1y":365, "5y":1825, "10y":3650}
            cutoff = datetime.now() - timedelta(days=days_map.get(period, 3650))
            data = data[data['Date'] >= cutoff]
        return data, None
    except Exception: return None, "Error"

def get_ft_data(period_days=365):
    ft_history = [
        {"Date": "2018-01-01", "Ft": -0.1590}, {"Date": "2018-05-01", "Ft": -0.1590}, {"Date": "2018-09-01", "Ft": -0.1590},
        {"Date": "2019-01-01", "Ft": -0.1160}, {"Date": "2019-05-01", "Ft": -0.1160}, {"Date": "2019-09-01", "Ft": -0.1160},
        {"Date": "2020-01-01", "Ft": -0.1160}, {"Date": "2020-05-01", "Ft": -0.1160}, {"Date": "2020-09-01", "Ft": -0.1243},
        {"Date": "2021-01-01", "Ft": -0.1532}, {"Date": "2021-05-01", "Ft": -0.1532}, {"Date": "2021-09-01", "Ft": -0.1532},
        {"Date": "2022-01-01", "Ft": 0.0139},  {"Date": "2022-05-01", "Ft": 0.2477},  {"Date": "2022-09-01", "Ft": 0.9343},
        {"Date": "2023-01-01", "Ft": 0.9343},  {"Date": "2023-05-01", "Ft": 0.9119},  {"Date": "2023-09-01", "Ft": 0.2048},
        {"Date": "2024-01-01", "Ft": 0.3972},  {"Date": "2024-05-01", "Ft": 0.3972},  {"Date": "2024-09-01", "Ft": 0.3972},
        {"Date": "2025-01-01", "Ft": 0.3672},  {"Date": "2025-05-01", "Ft": 0.1972},  {"Date": "2025-09-01", "Ft": 0.1572},
    ]
    df_ft = pd.DataFrame(ft_history)
    df_ft['Date'] = pd.to_datetime(df_ft['Date'])
    date_range = pd.date_range(start=df_ft['Date'].min(), end=datetime.now())
    df_daily = pd.DataFrame(date_range, columns=['Date'])
    df_merged = pd.merge_asof(df_daily, df_ft, on='Date', direction='backward')
    
    today = datetime.now()
    start_date = today - timedelta(days=period_days)
    df_final = df_merged[df_merged['Date'] >= start_date].copy()
    df_final.rename(columns={'Ft': 'Close'}, inplace=True)
    return df_final

assets_config = {
    "USD/THB":   {"type": "yahoo", "ticker": "THB=X", "unit": "Baht", "currency": "THB"},
    "Ft (Thai)": {"type": "manual", "ticker": "FT",    "unit": "Baht", "currency": "THB"},
    "JKM (LNG)": {"type": "yahoo", "ticker": "JKM=F", "unit": "$/MMBtu", "currency": "USD"}, 
    "Henry Hub": {"type": "yahoo", "ticker": "NG=F",  "unit": "$/MMBtu", "currency": "USD"},
}

# --- Sidebar ---
st.sidebar.title("⚙️ Control")
if not GROQ_API_KEY: api_key = st.sidebar.text_input("API Key", type="password")
else: api_key = GROQ_API_KEY

period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825, "10y (Max)": 3650}
selected_period_str = st.sidebar.selectbox("⏳ Timeframe", list(period_map.keys()), index=4)
selected_days = period_map[selected_period_str]
st.sidebar.divider()
convert_to_thb = st.sidebar.toggle("🇹🇭 THB Convert", value=False)
normalize_mode = st.sidebar.toggle("📏 Normalize (Max=1)", value=False)
st.sidebar.divider()
selected_assets = st.sidebar.multiselect("Compare:", list(assets_config.keys()), default=["Ft (Thai)", "JKM (LNG)"])

# --- Layout ---
col_dash, col_chat = st.columns([7, 3])

# === LEFT: Dashboard ===
with col_dash:
    # 1. Metrics & Data Calculation for AI
    thb_df, _ = get_data("THB=X", period=selected_period_str)
    cols_m = st.columns(len(assets_config))
    data_summary_text = "ข้อมูลตลาดพลังงานล่าสุด (คำนวณเทียบอดีตให้แล้ว):\n" 

    for idx, (name, config) in enumerate(assets_config.items()):
        # Fetch long history to calculate YoY/MoM
        if config["type"] == "manual": 
            df_full = get_ft_data(700) # 2 ปี
        else: 
            df_full, _ = get_data(config["ticker"], "2y")
        
        with cols_m[idx]:
            if df_full is not None and not df_full.empty:
                latest = df_full['Close'].iloc[-1]
                price, unit = latest, config['unit']
                
                # แปลงค่าเงิน (เฉพาะ Metric)
                if convert_to_thb and config['currency'] == 'USD' and thb_df is not None:
                    price = latest * thb_df['Close'].iloc[-1]
                    unit = "฿"
                
                # คำนวณเทียบเดือนก่อน/ปีก่อน
                try:
                    # Index -22 (ประมาณ 1 เดือน) และ -252 (ประมาณ 1 ปี)
                    idx_1m = max(0, len(df_full) - 22)
                    idx_1y = max(0, len(df_full) - 252)
                    
                    price_1m = df_full['Close'].iloc[idx_1m]
                    price_1y = df_full['Close'].iloc[idx_1y]
                    
                    pct_1d = ((latest - df_full['Close'].iloc[-2]) / df_full['Close'].iloc[-2]) * 100
                    mom = ((latest - price_1m) / price_1m) * 100
                    yoy = ((latest - price_1y) / price_1y) * 100
                    
                    st.metric(name, f"{price:,.2f} {unit}", f"{pct_1d:.2f}%")
                    
                    # เก็บข้อมูลส่ง AI
                    data_summary_text += f"- {name}: ราคาปัจจุบัน {latest:.2f} (MoM: {mom:+.1f}%, YoY: {yoy:+.1f}%)\n"
                except:
                    st.metric(name, f"{price:,.2f} {unit}", "-")
            else: st.metric(name, "-", "-")

    # 2. Graph
    if selected_assets:
        combined_df = pd.DataFrame()
        for asset_name in selected_assets:
            config = assets_config[asset_name]
            if config["type"] == "manual": df_trend = get_ft_data(selected_days)
            else: df_trend, _ = get_data(config["ticker"], selected_period_str)
            
            if df_trend is not None:
                temp_df = df_trend[['Date', 'Close']].copy()
                if convert_to_thb and config['currency'] == 'USD' and thb_df is not None:
                    merged = pd.merge(temp_df, thb_df[['Date', 'Close']], on='Date', how='inner', suffixes=('', '_Rate'))
                    temp_df['Close'] = merged['Close'] * merged['Close_Rate']
                
                asset_label = asset_name
                if normalize_mode:
                    max_val = temp_df['Close'].max()
                    if max_val != 0:
                        temp_df['Close'] = temp_df['Close'] / max_val
                        asset_label = f"{asset_name} (Norm)"
                
                temp_df['Asset'] = asset_label
                combined_df = pd.concat([combined_df, temp_df[['Date', 'Close', 'Asset']]])

        if not combined_df.empty:
            y_title = "Norm (Max=1)" if normalize_mode else "Price"
            fig = px.line(combined_df, x='Date', y='Close', color='Asset', template="plotly_white")
            fig.update_layout(
                xaxis_title=None, yaxis_title=y_title, legend_title=None,
                hovermode="x unified", height=600, 
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    else: st.info("Select assets")

# === RIGHT: Chat (Auto Analysis) ===
with col_chat:
    st.markdown("##### 💬 AI Analyst")
    
    if "messages" not in st.session_state: 
        st.session_state.messages = []
        # [AUTO ANALYSIS]
        if api_key:
            try:
                # Prompt สั่งให้เปรียบเทียบ MoM, YoY ชัดเจน
                initial_prompt = f"""
                คุณคือนักวิเคราะห์พลังงาน
                ข้อมูลล่าสุด (คำนวณมาให้แล้ว):
                {data_summary_text}
                
                ข้อกำหนด:
                1. Ft คือ 'ค่าไฟฟ้าผันแปร (Variable Electricity Tariff)' ของไทย
                2. วิเคราะห์สั้นๆ 3-4 บรรทัด โดยเน้นเปรียบเทียบกับเดือนที่แล้ว (MoM) และปีที่แล้ว (YoY)
                3. มองแนวโน้มผลกระทบต่อค่าไฟไทย
                """
                
                client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": initial_prompt}]
                )
                ai_summary = completion.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": f"**📊 Market Brief (MoM/YoY Analysis):**\n{ai_summary}"})
            except: pass

    for msg in st.session_state.messages:
        role = "👤" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=role).write(msg["content"])

    if prompt := st.chat_input("Ask AI..."):
        if not api_key: st.error("Key Missing")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant", avatar="🤖"):
            try:
                client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
                stream = client.chat.completions.create(
                    model="llama-3.1-8b-instant", 
                    messages=[
                        {"role": "system", "content": "You are an energy analyst. Note: Ft = Variable Electricity Tariff (ค่าไฟฟ้าผันแปร). Answer in Thai."},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    ], stream=True
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e: st.error(str(e))