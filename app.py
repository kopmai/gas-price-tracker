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

# --- CSS ปรับแต่ง ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
        color: #333; 
        overflow: hidden; 
    }
    .stApp { background-color: #ffffff; }

    /* Top Bar */
    .gemini-bar {
        position: fixed; top: 0; left: 0; width: 100%; height: 64px;
        background-color: #ffffff; border-bottom: 1px solid #dadce0;
        z-index: 99999; 
        display: flex; align-items: center; justify-content: space-between;
        padding-left: 80px; padding-right: 200px;
        font-size: 20px; font-weight: 600; color: #1f1f1f;
    }
    
    .date-badge {
        font-size: 14px; color: #5f6368; background-color: #f1f3f4;
        padding: 5px 12px; border-radius: 20px; font-weight: 400; white-space: nowrap;
    }

    /* Mobile Responsive */
    @media (max-width: 600px) {
        .gemini-bar {
            padding-left: 60px; padding-right: 10px;
            flex-direction: column; align-items: flex-start; justify-content: center;
            gap: 2px; height: auto; min-height: 60px; padding-top: 5px; padding-bottom: 5px;
        }
        .gemini-bar span:first-child { font-size: 18px; line-height: 1.2; }
        .date-badge { font-size: 11px; padding: 2px 8px; margin-top: 2px; }
        .main .block-container { padding-top: 85px !important; }
    }

    .main .block-container { 
        padding-top: 80px !important; padding-bottom: 0 !important;
        padding-left: 1rem !important; padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Layout Columns */
    div[data-testid="column"]:nth-of-type(1) {
        height: calc(100vh - 80px); overflow: hidden; 
        padding-right: 15px; border-right: 1px solid #f0f0f0;
    }
    div[data-testid="column"]:nth-of-type(2) {
        height: calc(100vh - 80px); overflow-y: auto;
        padding-left: 15px; display: flex; flex-direction: column; justify-content: flex-end;
    }

    div[data-testid="stMetric"] {
        background-color: #f8f9fa; border: 1px solid #eee;
        padding: 8px; border-radius: 6px; text-align: center;
    }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; }
    div[data-testid="stMetricValue"] { font-size: 16px !important; font-weight: 600; }

    .stChatInput { padding-bottom: 10px; z-index: 100; }
    header[data-testid="stHeader"] { background: transparent; z-index: 100000; }
    header .decoration { display: none; }
</style>
""", unsafe_allow_html=True)

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
            data_filtered = data[data['Date'] >= cutoff]
            return data_filtered, None, data
        return data, None, data
    except Exception: return None, "Error", None

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
    df_filtered = df_merged[df_merged['Date'] >= start_date].copy()
    df_filtered.rename(columns={'Ft': 'Close'}, inplace=True)
    df_full = df_merged.copy() 
    df_full.rename(columns={'Ft': 'Close'}, inplace=True)
    return df_filtered, df_full

def get_price_at_date(df, target_date):
    target_ts = pd.Timestamp(target_date)
    past_data = df[df['Date'] <= target_ts]
    if not past_data.empty:
        row = past_data.iloc[-1]
        return row['Close'], row['Date']
    return None, None

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

st.sidebar.divider()
st.sidebar.subheader("📅 Date Selection")
target_date = st.sidebar.date_input("Select Date", value=datetime.now(), max_value=datetime.now())

st.sidebar.divider()
period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825, "10y (Max)": 3650}
selected_period_str = st.sidebar.selectbox("⏳ Graph Timeframe", list(period_map.keys()), index=4)
selected_days = period_map[selected_period_str]

st.sidebar.divider()
convert_to_thb = st.sidebar.toggle("🇹🇭 THB Convert", value=False)
normalize_mode = st.sidebar.toggle("📏 Normalize (Max=1)", value=False)
st.sidebar.divider()
selected_assets = st.sidebar.multiselect("Compare:", list(assets_config.keys()), default=["Ft (Thai)", "JKM (LNG)"])

display_date_str = target_date.strftime("%d/%m/%Y")

# --- Top Bar ---
st.markdown(f"""
    <div class="gemini-bar">
        <span>⚡ Energy Price Tracker</span>
        <span class="date-badge">📅 Data as of: {display_date_str}</span>
    </div>
""", unsafe_allow_html=True)

# --- Layout ---
col_dash, col_chat = st.columns([7, 3])

# === LEFT: Dashboard ===
with col_dash:
    # 1. Metrics
    _, _, thb_full = get_data("THB=X", "max")
    cols_m = st.columns(len(assets_config))
    data_summary_text = f"ข้อมูลตลาดพลังงาน ณ วันที่ {display_date_str}:\n"
    
    for idx, (name, config) in enumerate(assets_config.items()):
        if config["type"] == "manual": _, df_full = get_ft_data()
        else: _, _, df_full = get_data(config["ticker"], "max")
        
        with cols_m[idx]:
            price_today = None
            if df_full is not None and not df_full.empty:
                price_today, date_today = get_price_at_date(df_full, target_date)
                if price_today is not None:
                    try:
                        idx_today = df_full[df_full['Date'] == date_today].index[0]
                        prev_price = df_full.iloc[idx_today - 1]['Close'] if idx_today > 0 else price_today
                        pct_change = ((price_today - prev_price) / prev_price) * 100
                    except: pct_change = 0
                    
                    unit = config['unit']
                    if convert_to_thb and config['currency'] == 'USD' and thb_full is not None:
                        rate, _ = get_price_at_date(thb_full, date_today)
                        if rate:
                            price_today = price_today * rate
                            unit = "฿"
                    
                    st.metric(name, f"{price_today:,.2f} {unit}", f"{pct_change:+.2f}%")
                    data_summary_text += f"- {name}: {price_today:,.2f} {unit}\n"
                else: st.metric(name, "No Data", "-")
            else: st.metric(name, "-", "-")
    
    # 2. Graph
    if selected_assets:
        combined_df = pd.DataFrame()
        for asset_name in selected_assets:
            config = assets_config[asset_name]
            if config["type"] == "manual": df_trend, _ = get_ft_data(selected_days)
            else: df_trend, _, _ = get_data(config["ticker"], selected_period_str)
            
            if df_trend is not None:
                temp_df = df_trend[['Date', 'Close']].copy()
                if convert_to_thb and config['currency'] == 'USD' and thb_full is not None:
                    merged = pd.merge(temp_df, thb_full[['Date', 'Close']], on='Date', how='inner', suffixes=('', '_Rate'))
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
            fig.add_vline(x=datetime.timestamp(datetime.combine(target_date, datetime.min.time())) * 1000, 
                          line_width=2, line_dash="dash", line_color="red")
            
            # [FEATURE UPDATED] ปิด ModeBar และ Zoom เพื่อประสบการณ์ที่ดีบนมือถือ
            fig.update_layout(
                xaxis_title=None, yaxis_title=y_title, legend_title=None,
                hovermode="x unified", height=600, 
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # ใส่ config ปิดเมนูซูม
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displayModeBar': False,  # ซ่อนแถบเครื่องมือด้านบน
                    'scrollZoom': False,      # ห้ามใช้เมาส์/นิ้วซูมเข้าออก (ป้องกันกราฟขยับมั่ว)
                    'showTips': False         # ปิด Tips ที่เด้งกวนใจ
                }
            )
    else: st.info("Select assets")

# === RIGHT: Chat ===
with col_chat:
    st.markdown("##### 💬 AI Analyst")
    
    if "messages" not in st.session_state: 
        st.session_state.messages = []
        if api_key:
            try:
                initial_prompt = f"วิเคราะห์ตลาดพลังงาน ณ วันที่ {target_date.strftime('%d/%m/%Y')} จากข้อมูลนี้: {data_summary_text} (ตอบสั้นๆ)"
                client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": initial_prompt}]
                )
                st.session_state.messages.append({"role": "assistant", "content": f"**Analysis ({target_date.strftime('%d/%m')}):**\n{completion.choices[0].message.content}"})
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
                        {"role": "system", "content": "Energy analyst. Answer in Thai."},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    ], stream=True
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e: st.error(str(e))
