
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime
import plotly.graph_objs as go

st.set_page_config(page_title="Stock Analysis App", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("final_df.csv")

@st.cache_data
def prepare_graph(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['ticker', 'Date'])
    price_matrix = df.pivot(index='Date', columns='ticker', values='Close').dropna()
    returns = np.log(price_matrix / price_matrix.shift(1)).dropna()

    def granger_matrix(data, maxlag=5):
        stock_ids = data.columns
        result = pd.DataFrame(np.zeros((len(stock_ids), len(stock_ids))), columns=stock_ids, index=stock_ids)
        for y in stock_ids:
            for x in stock_ids:
                if x == y: continue
                try:
                    test_result = grangercausalitytests(data[[y, x]], maxlag=maxlag, verbose=False)
                    p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(maxlag)]
                    result.loc[x, y] = min(p_values)
                except:
                    result.loc[x, y] = 1
        return result

    granger_pvals = granger_matrix(returns)
    threshold = 0.01
    G = nx.DiGraph()
    influence_data = []
    for from_node in granger_pvals.index:
        for to_node in granger_pvals.columns:
            p = granger_pvals.loc[from_node, to_node]
            if p < threshold:
                G.add_edge(from_node, to_node, weight=1 - p)
                influence_data.append({
                    "From": from_node,
                    "To": to_node,
                    "p-value": round(p, 6),
                    "Confidence": round((1 - p) * 100, 2)
                })
    influence_df = pd.DataFrame(influence_data).sort_values(by="Confidence", ascending=False)
    return G, influence_df

with open("AdaBoost_best_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

page = st.sidebar.radio("📚 Navigate", ["📈 Predict", "🕸️ Influence Graph", "📜 History", "🧪 Backtest"])

if page == "📈 Predict":
    st.title("📊 Predict Tomorrow's Stock Direction")
    uploaded_file = st.file_uploader("📤 Upload Feature CSV", type="csv")

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data:")
        st.dataframe(input_df)

        if st.button("📈 Predict"):
            try:
                input_scaled = scaler.transform(input_df)
                preds = model.predict(input_scaled)
                probas = model.predict_proba(input_scaled)

                results = input_df.copy()
                results['Prediction'] = ['📈 Up' if p == 1 else '📉 Down' for p in preds]
                results['Confidence'] = [f"{proba[p]*100:.2f}%" for p, proba in zip(preds, probas)]
                results['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.success("✅ Predictions Complete!")
                st.dataframe(results[['Prediction', 'Confidence', 'Timestamp']])
                st.download_button("📥 Download Predictions CSV", results.to_csv(index=False), "predictions.csv", "text/csv")

                try:
                    hist = pd.read_csv("prediction_history.csv")
                    pd.concat([hist, results]).to_csv("prediction_history.csv", index=False)
                except FileNotFoundError:
                    results.to_csv("prediction_history.csv", index=False)

                st.subheader("🧠 Natural Language Insights")
                input_df_sample = input_df.iloc[:3]
                for i, row in input_df_sample.iterrows():
                    st.markdown(f"### Row {i+1} - Prediction: **{results['Prediction'][i]}** | Confidence: **{results['Confidence'][i]}**")
                    insights = []

                    if 'RSI_14' in row:
                        if row['RSI_14'] < 30:
                            insights.append("RSI is below 30 — indicates oversold, possible reversal up.")
                        elif row['RSI_14'] > 70:
                            insights.append("RSI is above 70 — suggests overbought, possible pullback.")
                        else:
                            insights.append("RSI is in neutral zone.")

                    if 'MACD' in row and 'MACD_signal' in row:
                        if row['MACD'] > row['MACD_signal']:
                            insights.append("MACD above signal line — bullish signal.")
                        elif row['MACD'] < row['MACD_signal']:
                            insights.append("MACD below signal line — bearish signal.")

                    if 'Volume' in input_df.columns:
                        avg_vol = input_df['Volume'].mean()
                        if row['Volume'] > 1.5 * avg_vol:
                            insights.append("High trading volume — strong interest in the stock.")
                        elif row['Volume'] < 0.5 * avg_vol:
                            insights.append("Low trading volume — weaker conviction or sideways trend.")
                        else:
                            insights.append("Volume is average — no unusual activity.")

                    if 'Open' in row and 'Close' in row:
                        if row['Close'] > row['Open']:
                            insights.append("Bullish candlestick — closing higher than opening.")
                        elif row['Close'] < row['Open']:
                            insights.append("Bearish candlestick — closing lower than opening.")
                        else:
                            insights.append("Doji-like candle — indecision in market.")

                    st.markdown("📌 " + "<br>📌 ".join(insights), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

elif page == "🕸️ Influence Graph":
    st.title("🕸️ Granger Causality Stock Influence Graph")
    st.markdown("Shows which stocks **influence** others based on Granger Causality (p < 0.01).")
    with st.spinner("📊 Generating graph..."):
        df = load_data()
        G, influence_df = prepare_graph(df)

        st.subheader("🔗 Top Influencing Relationships")
        st.dataframe(influence_df.head(20), use_container_width=True)

        pos = nx.spring_layout(G, seed=42)
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            influences = len(list(G.successors(node)))
            followers = len(list(G.predecessors(node)))
            node_text.append(f"{node}<br>Influences: {influences}<br>Followers: {followers}")

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            textposition="bottom center",
            marker=dict(
                showscale=False,
                color='lightblue',
                size=15,
                line_width=2),
            text=list(G.nodes()),
            hovertext=node_text)

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="📌 Stock Influence Network (p < 0.01)",
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                       )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📜 History":
    st.title("📜 Prediction History")
    try:
        hist_df = pd.read_csv("prediction_history.csv")
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("📥 Download Full History", hist_df.to_csv(index=False), "prediction_history.csv")
    except FileNotFoundError:
        st.warning("No predictions made yet.")

elif page == "🧪 Backtest":
    st.title("🧪 Rule-Based Strategy Backtester")
    df = load_data()
    tickers = df['ticker'].unique()
    selected = st.selectbox("Choose Ticker", tickers)

    available_cols = df.columns.tolist()
    st.markdown(f"📌 Available Columns: `{', '.join(available_cols)}`")

    rule = st.text_input("Enter your rule (e.g. RSI_14 < 30 and MACD > 0)", "RSI_14 < 30 and MACD > 0")

    if st.button("Run Backtest"):
        try:
            sub = df[df['ticker'] == selected].copy()

            used_cols = [word for word in rule.replace(">", " ").replace("<", " ").replace("=", " ").split() if word in available_cols]
            undefined = [word for word in rule.replace(">", " ").replace("<", " ").replace("=", " ").split()
                         if word.isidentifier() and word not in available_cols and word not in ["and", "or", "not", "True", "False"]]

            if undefined:
                st.error(f"❌ These columns are not found in the data: {', '.join(undefined)}")
            else:
                sub['Signal'] = sub.eval(rule)
                sub['Return'] = sub['Close_next_day'] / sub['Close'] - 1
                sub['StrategyReturn'] = sub['Signal'] * sub['Return']

                st.metric("📊 Trades", int(sub['Signal'].sum()))
                st.metric("✅ Win %", f"{(sub['StrategyReturn'] > 0).mean() * 100:.2f}%")
                st.metric("📈 Avg Return", f"{sub['StrategyReturn'].mean() * 100:.2f}%")

                st.line_chart(sub[['Return', 'StrategyReturn']].cumsum())

        except Exception as e:
            st.error(f"❌ Rule evaluation failed: {e}")
