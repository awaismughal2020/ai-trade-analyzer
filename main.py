import streamlit as st
from trade_analyser import DirectTradeAnalyzer
import os
from dotenv import load_dotenv

load_dotenv()

analyzer = DirectTradeAnalyzer(
        coingecko_api_key=os.getenv('COINGECKO_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

st.title("AI Trading Coach - POC")
st.write("Upload transactions or use sample data")

if st.button("Analyze Sample Trades"):
    # Run your existing analysis
    analyses = analyzer.run_direct_analysis("data/sample_transactions.csv", max_trades=5)

    for trade in analyses:
        st.subheader(f"{trade.token} Trade")
        st.write(f"RSI at Buy: {trade.rsi_at_buy:.1f}")
        st.write(f"AI Advice: {trade.ai_recommendation}")
