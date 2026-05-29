import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# --- 網頁基礎設定 ---
st.set_page_config(
    page_title="咖啡銷量 AI 智慧決策系統", 
    page_icon="☕", 
    layout="wide", # 改為寬螢幕視覺，更像後台儀表板
    initial_sidebar_state="expanded"
)

# --- CSS 樣式微調 (注入自訂陰影與漸層美化) ---
st.markdown("""
    <style>
    .big-title { font-size:32px !important; font-weight: 700; color: #4E3629; margin-bottom: 5px; }
    .subtitle { font-size:16px !important; color: #707070; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #4E3629; color: white; height: 3em; font-size: 16px; font-weight: bold; }
    .stButton>button:hover { background-color: #74513E; color: white; border: 1px solid #4E3629; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 載入模型與特徵設定 ---
@st.cache_resource
def load_model():
    return joblib.load("coffee_model.joblib")

artifacts = load_model()
model = artifacts['model']
model_features = artifacts['model_features']
uv = artifacts['unique_values']

# --- 側邊欄：參數輸入區美化 ---
st.sidebar.markdown("## ⚙️ 營運參數設定")
st.sidebar.subheader("💰 定價與市場競合")
price = st.sidebar.slider("☕ 我方咖啡售價 ($)", 3.0, 8.0, 5.5, 0.1)
comp_price = st.sidebar.slider("🏪 競爭者咖啡售價 ($)", 3.0, 8.0, 5.2, 0.1)
price_diff = round(price - comp_price, 2)

st.sidebar.subheader("📈 行銷與營運")
ad_impressions = st.sidebar.number_input("📢 廣告曝光量 (Ad Impressions)", 100, 10000, 2500, step=100)
promo = st.sidebar.selectbox("🎁 促銷活動方案", uv['Promotion'])
coffee_type = st.sidebar.selectbox("🏷️ 預測咖啡品種", uv['Coffee_Type'])
staff_count = st.sidebar.slider("👥 當班店員人數配置", 1, 6, 3)

# --- 主畫面標頭 ---
st.markdown('<p class="big-title">☕ 咖啡銷量 AI 智慧決策與預測系統</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">整合歷史 8,000 筆大數據，利用隨機森林（Random Forest）演算法即時模擬銷量及市場動態</p>', unsafe_allow_html=True)

# --- 分欄排版：環境與時間參數 ---
st.markdown("### 🌦️ 環境與時間變數設定")
col_env1, col_env2, col_env3, col_env4 = st.columns(4)

with col_env1:
    weather = st.selectbox("天氣狀況", uv['Weather'])
with col_env2:
    temp = st.slider("預估氣溫 (°C)", 10, 40, 25)
with col_env3:
    day_of_week = st.selectbox("星期", uv['Day_of_Week'])
with col_env4:
    time_slot = st.selectbox("時段區間", uv['Time_Slot'])

col_env5, col_env6, col_env7 = st.columns([1, 1, 2])
with col_env5:
    month = st.slider("月份", 1, 12, 6)
with col_env6:
    hour = st.slider("時間 (24h)", 0, 23, 12)
with col_env7:
    st.write("") # 往下推一點對齊
    st.write("")
    is_holiday = st.checkbox("🔥 當天為「國定節假日 / 連假」", value=False)

st.markdown("---")

# --- 2. 建立標籤頁 (主要預測 vs 數據動態圖表) ---
tab1, tab2 = st.tabs(["🚀 AI 即時銷量預測", "📊 定價與市場動態圖分析"])

with tab1:
    # 建立動態大字卡（Metrics Row）
    st.markdown("#### 🎯 當前市場動態指標")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        # 定價與對手差異
        if price_diff > 0:
            st.metric(label="價格競爭力", value=f"${price}", delta=f"比對手貴 ${abs(price_diff)}", delta_color="inverse")
        elif price_diff < 0:
            st.metric(label="價格競爭力", value=f"${price}", delta=f"比對手便宜 ${abs(price_diff)}", delta_color="normal")
        else:
            st.metric(label="價格競爭力", value=f"${price}", delta="與對手同價", delta_color="off")
            
    with col_m2:
        st.metric(label="促銷刺激度", value=promo, delta="行銷活動進行中" if promo != "None" else "基本常態銷售", delta_color="normal" if promo != "None" else "off")
        
    with col_m3:
        st.metric(label="環境舒適度", value=f"{temp} °C", delta=weather)

    st.write("")
    st.write("")

    # --- 3. 執行預測計算 ---
    # 建立預測用 Dataframe
    input_data = pd.DataFrame([{
        'Temperature_C': temp, 'Is_Holiday': 1 if is_holiday else 0, 'Price': price,
        'Ad_Impressions': ad_impressions, 'Competitor_Price': comp_price, 
        'Price_Difference': price_diff, 'Month': month, 'Hour': hour, 'Staff_Count': staff_count,
        'Weather': weather, 'Day_of_Week': day_of_week, 'Time_Slot': time_slot, 
        'Promotion': promo, 'Coffee_Type': coffee_type
    }])

    input_encoded = pd.get_dummies(input_data)
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    # 按鈕觸發預測
    if st.button("🚀 點擊模擬 AI 預測銷量 Y"):
        with st.spinner('AI 正在計算最佳銷售決策中...'):
            prediction = int(np.round(model.predict(input_encoded)[0]))
            
        # 彈出慶祝氣球特效
        st.balloons()
        
        # 醒目的結果呈現大字型
        st.markdown(f"""
        <div style="background-color: #F4EBE1; padding: 25px; border-radius: 12px; border-left: 8px solid #4E3629; margin-top: 15px;">
            <span style="font-size: 18px; color: #555; font-weight: bold;">📊 AI 模型預測結果：</span><br>
            <span style="font-size: 20px; color: #333;">在當前設定的環境與商業交叉影響下，預估 <b>【{coffee_type}】</b> 單時段銷售量為：</span><br>
            <span style="font-size: 48px; font-weight: 800; color: #4E3629;">{prediction} <span style="font-size: 24px;">杯 / 件</span></span>
        </div>
        """, unsafe_allow_html=True)
        
        # 智慧商業洞察
        st.write("")
        if price_diff > 0 and prediction < 35:
            st.warning(f"💡 **AI 策略優化建議**：當前您的定價比競爭者貴了 {abs(price_diff)} 元，且預估銷量偏低。若非品牌溢價或咖啡品種具有獨特性，建議適度將定價調至 ${comp_price} 左右，或搭配更高強度的 **{promo if promo != 'None' else '折扣促銷'}** 活動，以活化客流量並防止客戶流失。")
        else:
            st.success(f"💡 **AI 策略優化建議**：當前的商業策略配置組合表現優異！在 **{weather}** 的環境與 **{promo}** 促銷的綜效帶動下，能為店內帶來相當可觀的爆發性銷量。請確保店內 **{staff_count}名店員** 的出餐備料速度，以因應尖峰人潮。")

with tab2:
    st.markdown("#### 📊 定價競合動態圖表 (互動式)")
    st.write("下方圖表會隨著你在左側拉桿微調定價而**即時動態波動**，幫助你直觀視覺化兩者價差關係。")
    
    # 製作一個簡單的對比動態 Dataframe
    chart_data = pd.DataFrame({
        '品牌': ['我方咖啡價格', '對手咖啡價格'],
        '價格 ($)': [price, comp_price],
        '色彩標記': ['#4E3629', '#C0A080']
    })
    
    # 使用 Altair 繪製精美的動態條形圖
    bar_chart = alt.Chart(chart_data).mark_bar(size=60, cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X('品牌:N', axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y('價格 ($):Q', scale=alt.Scale(domain=[0, 10])),
        color=alt.Color('品牌:N', scale=alt.Scale(domain=['我方咖啡價格', '對手咖啡價格'], range=['#4E3629', '#A68A78']), legend=None)
    ).properties(width=500, height=350)
    
    st.altair_chart(bar_chart, use_container_width=True)
    
    # 價差狀態雷達指示
    if price_diff > 0:
        st.info(f"🔍 **價差狀態**：當前處於「高價高毛利」策略模式，價格高於對手 {abs(price_diff)} 元。請特別注意行銷曝光以支撐高單價利潤。")
    elif price_diff < 0:
        st.success(f"🔍 **價差狀態**：當前處於「低價搶市」策略模式，價格低於對手 {abs(price_diff)} 元。此策略通常能有效擴大市場佔有率與銷售量。")
    else:
        st.warning(f"🔍 **價差狀態**：當前與對手定價完全持平。建議透過精準的「促銷活動」做出差異化，以爭取消費者青睞。")
    
