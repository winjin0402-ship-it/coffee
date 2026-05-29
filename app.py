import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="咖啡銷量 AI 預測系統", layout="centered")

st.title("☕ 咖啡每日銷售量 AI 預測系統")
st.write("請在下方輸入當前的環境與商業參數，系統將自動預測目標咖啡品種的銷售量。")

# 1. 載入模型與特徵設定
@st.cache_resource
def load_model():
    return joblib.load("coffee_model.joblib")

artifacts = load_model()
model = artifacts['model']
model_features = artifacts['model_features']
uv = artifacts['unique_values']

st.sidebar.header("📊 商業與市場參數")
st.sidebar.subheader("定價與廣告")
price = st.sidebar.slider("我方咖啡售價 ($)", 3.0, 8.0, 5.5, 0.1)
comp_price = st.sidebar.slider("競爭者咖啡售價 ($)", 3.0, 8.0, 5.2, 0.1)
ad_impressions = st.sidebar.number_input("廣告曝光量 (Ad Impressions)", 100, 10000, 2500)
promo = st.sidebar.selectbox("促銷活動", uv['Promotion'])

st.sidebar.subheader("店內配置")
coffee_type = st.sidebar.selectbox("咖啡品種", uv['Coffee_Type'])
staff_count = st.sidebar.slider("當班店員人數", 1, 6, 3)

st.header("🌦️ 環境與時間參數")
col1, col2 = st.columns(2)
with col1:
    weather = st.selectbox("當天天氣", uv['Weather'])
    temp = st.slider("預估氣溫 (°C)", 10, 40, 25)
    is_holiday = st.checkbox("是否為節假日？")
with col2:
    day_of_week = st.selectbox("星期幾", uv['Day_of_Week'])
    time_slot = st.selectbox("時間區段", uv['Time_Slot'])
    month = st.slider("月份", 1, 12, 6)
    hour = st.slider("小時 (24小時制)", 0, 23, 12)

# 計算衍生特徵
price_diff = price - comp_price

# 2. 建立預測用 Dataframe
input_data = pd.DataFrame([{
    'Temperature_C': temp, 'Is_Holiday': 1 if is_holiday else 0, 'Price': price,
    'Ad_Impressions': ad_impressions, 'Competitor_Price': comp_price, 
    'Price_Difference': price_diff, 'Month': month, 'Hour': hour, 'Staff_Count': staff_count,
    'Weather': weather, 'Day_of_Week': day_of_week, 'Time_Slot': time_slot, 
    'Promotion': promo, 'Coffee_Type': coffee_type
}])

# 3. 進行與訓練時一模一樣的 One-Hot Encoding
input_encoded = pd.get_dummies(input_data)

# 對齊欄位：補齊網頁輸入沒有產生的其他 0/1 欄位
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# 確保欄位順序跟訓練時完全一致
input_encoded = input_encoded[model_features]

# 4. 預測 Y 並顯示
if st.button("🚀 點擊生成預測銷量 Y", type="primary"):
    prediction = model.predict(input_encoded)[0]
    
    st.success("### 🎉 預測完成！")
    st.metric(label=f"預估【{coffee_type}】銷售量 (杯/件)", value=f"{int(np.round(prediction))} 杯")
    
    # 趣味洞察分析
    st.info(f"💡 **AI 商業洞察**：目前您設定的價格比競爭者{'貴' if price_diff > 0 else '便宜'} {abs(price_diff):.2f} 元。搭配 **{promo}** 促銷，在 **{weather}** 的環境下，預估能為您帶來穩定的客流量。")