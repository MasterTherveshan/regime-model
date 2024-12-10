import os
import io
import re
import json
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from openai import OpenAI

# Make sure OPENAI_API_KEY is set in your environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
client = OpenAI(api_key=OPENAI_API_KEY)

st.title("Market Regime Detection with LLM-Based Short Regime Names")

tabs = st.tabs(["Explanation", "Analysis"])

with tabs[0]:
    explanation_text = """
        **What Are We Doing Here?**  
        We’re trying to understand how markets behave under different "states" or "regimes."  
        Just like weather can change from sunny to rainy, financial markets can change between stable growth, boom times, sluggish periods, or crisis conditions. Identifying these regimes can help us make better investment decisions.

        **How Do We Identify Regimes?**  
        We use a statistical model called a "Markov Regime Switching" model. Think of it like a detector that looks at past returns of a group of equity ETFs (which represent different parts of the stock market) and tries to figure out which "regime" we're currently in. It doesn’t know these regimes ahead of time — it figures them out from the data itself.

        **What Variables Are We Using?**  
        - **Equity ETFs:** We take several Satrix ETFs (Top 40, Financials, Industrials, Resources) to form a combined market factor. This is just an average of their daily returns.
        - **Bonds (STXGOV):** We look at South African government bonds via the Satrix Bond ETF.
        - **FX (ZAR=X):** USD/ZAR exchange rate is included to see how the currency behaves under different regimes.
        - **FRED Data:** We also include two interest rate datasets from the US Federal Reserve’s FRED database, focusing on South African long-term yields and Treasury bill rates.

        **What Are Returns and Why Log Returns?**  
        We don’t use prices directly. We first compute "log returns," which is just a way of measuring percentage changes in a consistent manner. Log returns help in statistical modeling and interpreting growth rates of an investment over time.

        **How Many Regimes?**  
        We can choose how many regimes the model should detect. It could be 2, 3, 4, or 5.  
        - **2 Regimes:** Might just capture "good times" and "bad times."
        - **More Regimes:** We can capture more subtle differences. Maybe a stable growth regime, a speculative boom, a defensive slow-growth period, and a crisis period.

        **Limitations and Caveats:**  
        - **No Crystal Ball:** This model looks at historical data to guess regimes. It doesn’t predict the future with certainty.
        - **Statistical Assumptions:** Markov models assume that the transition from one regime to another depends only on the current state, not the full history. Real markets can be more complicated.
        - **Data Availability and Frequency:** We’re using daily data. Some interest rate data from FRED might be monthly or quarterly, so we "fill in" missing days which can introduce inaccuracies.
        - **Interpretation Required:** Even after we find regimes, naming or describing them is partly subjective. We rely on mean returns and volatility to guess what each regime represents.

        In short, this approach tries to segment the complex market environment into a few understandable "states," which can help when making decisions about investing, hedging, or risk management — but it should be one tool among many, not the final word on what will happen next.
        """
    st.markdown(explanation_text)

with tabs[1]:
    # Analysis tab
    st.sidebar.header("Analysis Settings")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
    k_regimes = st.sidebar.slider("Number of Regimes", min_value=2, max_value=5, value=3)

    st.markdown(f"Analyzing data from {start_date} to {end_date} with {k_regimes} regimes.")

    @st.cache_data
    def download_data_yf(ticker, start, end):
        return yf.download(ticker, start=start, end=end)

    @st.cache_data
    def download_data_fred(series, start, end):
        return web.DataReader(series, 'fred', start, end)

    equity_tickers = {
        'Top40': 'STX40.JO',
        'FINI': 'STXFIN.JO',
        'INDI': 'STXIND.JO',
        'RESI': 'STXRES.JO'
    }

    bond_ticker = 'STXGOV.JO'
    fx_ticker = 'ZAR=X'
    fred_series = {
        'LongTermYield': 'IRLTLT01ZAM156N',
        'TreasuryBill': 'INTGSTZAM193N'
    }

    st.markdown("### Data Retrieval")

    equity_data = {}
    failed_eq = []
    for name, tkr in equity_tickers.items():
        df = download_data_yf(tkr, start_date, end_date)
        if df.empty:
            failed_eq.append(name)
        else:
            equity_data[name] = df

    if failed_eq:
        st.warning(f"Failed to retrieve data for: {', '.join(failed_eq)}.")

    if not equity_data:
        st.error("No equity ETFs available for factor construction. Adjust tickers or date range.")
        st.stop()

    bond_data = download_data_yf(bond_ticker, start_date, end_date)
    if bond_data.empty:
        st.error("No bond data retrieved. Check ticker or date range.")
        st.stop()

    fx_data = download_data_yf(fx_ticker, start_date, end_date)
    if fx_data.empty:
        st.error("No FX data retrieved for USD/ZAR. Check ticker or date range.")
        st.stop()

    fred_data = {}
    failed_fred = []
    for name, series in fred_series.items():
        try:
            fdf = download_data_fred(series, start_date, end_date)
            if fdf.empty:
                failed_fred.append(name)
            else:
                fred_data[name] = fdf
        except Exception:
            failed_fred.append(name)

    if failed_fred:
        st.warning(f"Failed to retrieve FRED data for: {', '.join(failed_fred)}")

    # Compute returns
    for name, df in equity_data.items():
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
        equity_data[name] = df

    bond_data['Returns'] = np.log(bond_data['Close'] / bond_data['Close'].shift(1))
    bond_data.dropna(inplace=True)

    fx_data['Returns'] = np.log(fx_data['Close'] / fx_data['Close'].shift(1))
    fx_data.dropna(inplace=True)

    for name, fdf in fred_data.items():
        fdf = fdf.asfreq('B')
        fdf.ffill(inplace=True)
        fdf['Returns'] = np.log(fdf[fdf.columns[0]] / fdf[fdf.columns[0]].shift(1))
        fdf.dropna(inplace=True)
        fred_data[name] = fdf

    # Align dates
    common_index = None
    for df in equity_data.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    common_index = common_index.intersection(bond_data.index)
    common_index = common_index.intersection(fx_data.index)
    for fdf in fred_data.values():
        common_index = common_index.intersection(fdf.index)

    for name in equity_data:
        equity_data[name] = equity_data[name].loc[common_index]
    bond_data = bond_data.loc[common_index]
    fx_data = fx_data.loc[common_index]
    for name in fred_data:
        fred_data[name] = fred_data[name].loc[common_index]

    # Market factor
    all_eq_returns = pd.DataFrame({name: df['Returns'] for name, df in equity_data.items()}, index=common_index)
    market_factor = all_eq_returns.mean(axis=1)
    market_factor.name = 'MarketFactor'

    st.markdown("### Regime Detection on Combined Market Factor")
    st.write(f"Modeling with {k_regimes} regimes...")

    model = MarkovRegression(market_factor, k_regimes=k_regimes, switching_variance=True)
    results = model.fit()
    st.text(results.summary())

    regime_probs = results.smoothed_marginal_probabilities
    assigned_regime = regime_probs.idxmax(axis=1) + 1
    factor_df = pd.DataFrame({'Factor': market_factor, 'Regime': assigned_regime}, index=market_factor.index)

    bond_data['Regime'] = factor_df['Regime']
    fx_data['Regime'] = factor_df['Regime']
    for name in fred_data:
        fred_data[name]['Regime'] = factor_df['Regime']

    def calc_regime_stats(df, label):
        stats = []
        for r in range(1, k_regimes+1):
            subset = df[df['Regime'] == r]['Returns']
            stats.append({
                'Regime': r,
                f'{label} Mean Return': subset.mean(),
                f'{label} Volatility': subset.std(),
                'Count of Days': len(subset)
            })
        return pd.DataFrame(stats)

    st.subheader("Bond Return Statistics by Regime")
    bond_stats_df = calc_regime_stats(bond_data, 'Bond')
    st.dataframe(bond_stats_df.style.format({"Bond Mean Return": "{:.6f}", "Bond Volatility": "{:.6f}"}))

    st.subheader("USD/ZAR Return Statistics by Regime")
    fx_stats_df = calc_regime_stats(fx_data, 'USD/ZAR')
    st.dataframe(fx_stats_df.style.format({"USD/ZAR Mean Return": "{:.6f}", "USD/ZAR Volatility": "{:.6f}"}))

    for name, fdf in fred_data.items():
        st.subheader(f"{name} (FRED) Return Statistics by Regime")
        fred_stats_df = calc_regime_stats(fdf, name)
        mean_col = f"{name} Mean Return"
        vol_col = f"{name} Volatility"
        st.dataframe(fred_stats_df.style.format({mean_col: "{:.6f}", vol_col: "{:.6f}"}))

    st.markdown("**Note:** For interest rates, 'returns' are log changes in yields/rates.")

    st.subheader("Market Factor with Regime Probabilities")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(factor_df.index, factor_df['Factor'], color='black', label='Market Factor Returns')

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    labels = [f'Regime {i} Prob' for i in range(1, k_regimes+1)]
    bottom = np.zeros(len(factor_df))
    for i in range(k_regimes):
        ax.fill_between(factor_df.index, bottom, bottom + regime_probs[i], color=colors[i], alpha=0.3, label=labels[i])
        bottom += regime_probs[i]

    ax.set_title('Market Factor (Equity) Returns with Regime Probabilities')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Returns')
    ax.legend()
    st.pyplot(fig)

    # Prompt for simpler user-friendly format (no JSON)
    overall_vol = factor_df['Factor'].std()
    regime_info = []
    for r in range(1, k_regimes+1):
        subset = factor_df[factor_df['Regime'] == r]['Factor']
        mean_ret = subset.mean()
        vol = subset.std()
        regime_info.append((r, mean_ret, vol))

    prompt = (
        "You are a financial analyst with deep knowledge of South African markets. "
        "We have identified several regimes. For each regime, we have:\n"
        " - a regime number (an integer)\n"
        " - a mean return (positive = rising markets, negative = declining markets)\n"
        " - a volatility (standard deviation)\n\n"
        f"Overall Market Factor Volatility: {overall_vol:.6f}\n\n"
        "For each regime, provide a short name (1-3 words) and a brief, user-friendly description in South African context. "
        "Format your answer as a list of sections like:\n\n"
        "Regime 1: <Short Name>\nDescription: <Short Description>\n\n"
        "Regime 2: ...\nDescription: ...\n\n"
        "and so forth.\n\n"
        "Regimes:\n"
    )

    for (r, mean_ret, vol) in regime_info:
        prompt += f"- Regime {r}: Mean Return = {mean_ret:.6f}, Volatility = {vol:.6f}\n"

    with st.spinner("Consulting LLM for short regime names and descriptions..."):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

    llm_text = response.choices[0].message.content.strip()

    st.subheader("LLM-Generated Regime Names and Descriptions")
    st.markdown(llm_text)

    # Parse the LLM text to extract regime names
    # We'll look for lines like "Regime X: Name"
    regime_name_map = {}
    lines = llm_text.splitlines()
    for line in lines:
        # Check if line starts with "Regime"
        match = re.match(r"Regime\s+(\d+):\s+(.*)", line, re.IGNORECASE)
        if match:
            regime_num = int(match.group(1))
            regime_name = match.group(2).strip()
            regime_name_map[regime_num] = regime_name

    # Create final df with regime names
    final_df = factor_df.copy()
    final_df['RegimeName'] = final_df['Regime'].map(regime_name_map)

    st.subheader("Historical Dates with Regime Names")
    st.dataframe(final_df[['Regime', 'RegimeName']].head(10))

    # Provide a download button for Excel
    # Make sure openpyxl is installed: pip install openpyxl
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        final_df[['Regime', 'RegimeName']].to_excel(writer, sheet_name='Regimes', index=True)
    output.seek(0)

    st.download_button(
        label="Download Regime Data as Excel",
        data=output,
        file_name="regimes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("""
    **Interpretation:**
    The LLM provides concise regime names in a user-friendly manner, and we've mapped each date to its assigned regime.
    You can download this data as an Excel file.
    """)
