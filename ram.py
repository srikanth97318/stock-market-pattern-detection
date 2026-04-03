import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="EigenStock AI Dashboard", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.main { background-color: #0e1117; }

div[data-testid="stMetricValue"] { font-size: calc(1.2vw + 12px); color: #00d4ff; }

.stTabs [data-baseweb="tab"] { 
    height: 50px; background-color: #161b22; border-radius: 5px; color: white;
}
.stTabs [aria-selected="true"] { background-color: #00d4ff; color: black; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Analysis Control Panel")

    default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA"
    tickers_input = st.text_input("Selected Stocks", default_tickers)

    period = st.selectbox("Time Range", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

    st.markdown("---")
    st.subheader("Analysis Modules")

    menu = st.radio("Choose View", [
        "Market Overview",
        "Data & Normalization",
        "Eigen Analysis",
        "Trend Insights"
    ])

    st.success("Live Data API Connected")

# ---------------- DATA ----------------
@st.cache_data
def fetch_data(t_list, p):
    ticker_clean = [t.strip().upper() for t in t_list.split(",")]

    raw_df = yf.download(ticker_clean, period=p, auto_adjust=True)

    if isinstance(raw_df.columns, pd.MultiIndex):
        df = raw_df["Close"]
    else:
        df = raw_df

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.ffill().bfill().dropna()

    return df

raw_data = fetch_data(tickers_input, period)

# ---------------- PROCESS ----------------
if not raw_data.empty and len(raw_data.columns) > 1:

    df_norm = (raw_data - raw_data.mean()) / raw_data.std()

    A = df_norm.values.T
    R = np.dot(A, A.T) / (len(raw_data) - 1)

    evals, evecs = np.linalg.eig(R)

    idx = evals.argsort()[::-1]
    evals = evals[idx].real
    evecs = evecs[:, idx].real

    explained_var = evals / np.sum(evals)

    # ---------------- UI ----------------

    if menu == "Market Overview":
        st.title(" Market Intelligence Dashboard")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Assets", len(raw_data.columns))
        m2.metric("Primary Trend Strength (PC1)", f"{explained_var[0]*100:.1f}%")
        m3.metric("Time Period (Days)", len(raw_data))

        st.subheader(" Normalized Stock Trend Comparison")
        fig = px.line(df_norm, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "Data & Normalization":
        st.title("🧮 Data Transformation Engine")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader(" Recent Price Data Matrix")
            st.dataframe(raw_data.tail(15), use_container_width=True)

        with c2:
            st.subheader(" Standardized Data Matrix (Z-Score)")
            st.dataframe(df_norm.tail(15), use_container_width=True)

        st.subheader(" Stock Correlation Matrix")
        fig = px.imshow(R, x=raw_data.columns, y=raw_data.columns,
                        text_auto=".2f", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "Eigen Analysis":
        st.title("🧬 Eigen Analysis Engine")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader(" Variance Explained by Components")
            fig = px.bar(x=[f"PC{i+1}" for i in range(len(evals))], y=evals)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader(" Dominant Market Influence Vector")
            weights = pd.DataFrame({"Influence": evecs[:, 0]}, index=raw_data.columns)
            st.dataframe(weights, use_container_width=True)

    elif menu == "Trend Insights":
        st.title(" Market Trend Interpretation")

        st.subheader(" Stock Influence on Market Trend")

        contribution = pd.DataFrame({
            'Stock': raw_data.columns,
            'Influence': evecs[:, 0]
        }).sort_values(by='Influence')

        fig = px.bar(contribution, x='Influence', y='Stock',
                     orientation='h', color='Influence',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Primary trend explains {explained_var[0]*100:.2f}% variance. "
            f"Top influencing stock: {contribution.iloc[-1]['Stock']}"
        )

else:
    st.error("⚠️ Data fetch failed. Check stock symbols or internet connection.")