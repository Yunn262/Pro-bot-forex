import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import io, base64
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from forex_python.converter import CurrencyRates
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

st.set_page_config(page_title="🚀 Bot Trading PRO — IA Preditiva", layout="wide")

# ===================== CSS =====================
st.markdown("""
<style>
@keyframes pulse {0% { box-shadow:0 0 0 0 rgba(0,255,0,0.6);} 70% {box-shadow:0 0 20px 10px rgba(0,255,0,0);} 100%{box-shadow:0 0 0 0 rgba(0,255,0,0);}}
.pulse-green { animation: pulse 1.5s infinite;}
.pulse-red { animation: pulse 1.5s infinite;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Bot Trading PRO — IA Preditiva (Versão PRO)")

# ===================== ALERTAS =====================
sound_up_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"
sound_down_b64 = "UklGRigAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQgAAAAA"

def play_sound(sound_b64):
    audio_bytes = base64.b64decode(sound_b64)
    st.audio(io.BytesIO(audio_bytes), format="audio/wav", start_time=0)

def show_signal_alert(signal: str, confidence: float, min_conf: float = 70):
    color_map = {"SUBIDA 🔼": "#1db954", "DESCIDA 🔽": "#e63946", "NEUTRAL ⚪": "#6c757d"}
    pulse_class = "pulse-green" if "SUBIDA" in signal else "pulse-red" if "DESCIDA" in signal else ""
    color = color_map.get(signal, "#6c757d")
    st.markdown(
        f"""
        <div class="{pulse_class}" style='background-color:{color};
        padding:1.3rem;border-radius:1rem;text-align:center;
        color:white;font-size:1.6rem;'>
        <b>{signal}</b><br>
        Confiança: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
    if confidence >= min_conf:
        if "SUBIDA" in signal:
            play_sound(sound_up_b64)
        elif "DESCIDA" in signal:
            play_sound(sound_down_b64)

# ===================== FUNÇÕES =====================
@st.cache_data(ttl=300)
def fetch_crypto(symbol="BTC/USDT", exchange_name='binance', timeframe='15m', limit=200):
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_forex_real(pair='EUR/USD', timeframe='15m', limit=200):
    c = CurrencyRates()
    end_time = datetime.now()
    delta_map = {'5m':5, '15m':15, '1h':60, '4h':240, '1d':1440}
    minutes = delta_map.get(timeframe, 15)
    timestamps = [end_time - timedelta(minutes=i*minutes) for i in reversed(range(limit))]
    rates = []
    base, target = pair.split('/')
    for ts in timestamps:
        try:
            rate = c.get_rate(base, target, ts)
            rates.append(rate)
        except:
            rates.append(np.nan)
    df = pd.DataFrame({'ts': timestamps, 'close': rates})
    df['open'] = df['close'].shift(1).fillna(method='bfill')
    df['high'] = df[['open','close']].max(axis=1) * 1.001
    df['low'] = df[['open','close']].min(axis=1) * 0.999
    df['volume'] = np.random.randint(100,500,size=len(df))
    df.set_index('ts', inplace=True)
    return df

def predict_next_move(df):
    df_feat = df.copy()
    df_feat['HL'] = df_feat['high'] - df_feat['low']
    df_feat['OC'] = df_feat['close'] - df_feat['open']
    df_feat['SMA'] = df_feat['close'].rolling(5).mean()
    df_feat['SMA_diff'] = df_feat['SMA'] - df_feat['close']
    df_feat.fillna(0, inplace=True)
    X = df_feat[['open','high','low','close','volume','HL','OC','SMA','SMA_diff']].iloc[:-1]
    y = df_feat['close'].shift(-1).iloc[:-1]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    next_feat = X.iloc[-1:].values
    pred_price = model.predict(next_feat)[0]
    last_price = df['close'].iloc[-1]
    diff = (pred_price - last_price)/last_price
    confidence = min(max(abs(diff)*1000, 60), 99)
    signal = "SUBIDA 🔼" if diff > 0.002 else "DESCIDA 🔽" if diff < -0.002 else "NEUTRAL ⚪"
    return last_price, pred_price, diff, signal, confidence

# ===================== INTERFACE =====================
market_type = st.sidebar.selectbox("Mercado", ["Criptomoeda","Forex"])

if market_type=="Criptomoeda":
    symbol = st.sidebar.text_input("Símbolo (ex: BTC/USDT)","BTC/USDT")
    exchange_name = st.sidebar.selectbox("Exchange", ["binance","coinbase","kraken","kucoin"])
else:
    symbol = st.sidebar.selectbox("Par Forex", ["EUR/USD","USD/JPY","GBP/USD","AUD/USD","USD/CHF"])

timeframe = st.sidebar.selectbox("Timeframe", ["5m","15m","1h","4h","1d"])
confidence_threshold = st.sidebar.slider("🔉 Nível mínimo de confiança p/ alerta", 50, 100, 75, 1)
auto_refresh = st.sidebar.checkbox("Atualizar automaticamente", value=False)
interval = st.sidebar.number_input("Intervalo (s)", min_value=10, max_value=300, value=60)

if auto_refresh:
    st_autorefresh(interval=interval*1000, key="data_refresh")

# ===================== EXECUÇÃO =====================
if st.button("▶️ Analisar mercado") or auto_refresh:
    with st.spinner("Analisando tendência com IA..."):
        if market_type=="Criptomoeda":
            df = fetch_crypto(symbol, exchange_name, timeframe)
        else:
            df = fetch_forex_real(symbol, timeframe)

        last_price, pred_price, diff, signal, confidence = predict_next_move(df)

        st.subheader(f"💰 Preço Atual: {last_price:.4f}")
        st.subheader(f"📈 Preço Previsto: {pred_price:.4f}")
        st.metric("Variação (%)", f"{diff*100:.2f}%")
        show_signal_alert(signal, confidence, confidence_threshold)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Preço Real'))
        fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1]+pd.Timedelta(minutes=15)],
                                 y=[last_price, pred_price], mode='lines+markers', name='Previsão'))
        st.plotly_chart(fig,use_container_width=True)
