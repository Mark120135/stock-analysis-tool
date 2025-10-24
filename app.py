"""
Stock Analysis Web Application
A Streamlit web app that provides comprehensive stock analysis with AI-powered insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import traceback # For detailed error reporting

# Import your existing modules
from data_fetcher import YahooFinanceDataFetcher
from data_processor import StockDataProcessor
from valuation_model import StockValuationModel

# Try to import AI analyzers AND the new guidance data
try:
    # Use the ORIGINAL StockEvaluator from the user's file
    from stock_evaluator import (
        StockEvaluator, # Use the original class
        get_available_providers,
        is_provider_available,
        QUALITATIVE_QUESTIONS_GUIDANCE, # Import the qualitative data
        OPERATIONAL_QUESTIONS_GUIDANCE  # <-- ADD THIS IMPORT
    )
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    # Show error prominently if import fails, as it's critical
    st.error(f"Fatal Error: Could not import StockEvaluator or required data. Check stock_evaluator.py. Details: {e}")
    EVALUATOR_AVAILABLE = False
    # Define placeholder if import fails to prevent further errors during app load
    class StockEvaluator:
        def __init__(self, ticker, api_key=None, provider=None): self.ticker = ticker; self.use_ai = False; self.info = None
        def get_ai_qualitative_guidance(self, *args, **kwargs): return {'error': 'StockEvaluator class not loaded.'}
    def get_available_providers(): return []
    def is_provider_available(p): return False
    QUALITATIVE_QUESTIONS_GUIDANCE = {} # Define as empty dict


# Page configuration
st.set_page_config(
    page_title="Stock Analysis & Valuation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .stNumberInput > div > div > input { text-align: center; font-weight: bold; }
    .stMetric { text-align: center; border: 1px solid #eee; border-radius: 5px; padding: 10px; margin-bottom: 10px;}
    /* Style for guidance tab */
    .guidance-question { font-weight: bold; margin-top: 1rem; }
    .guidance-caption { color: #555; font-style: italic; }
    </style>
""", unsafe_allow_html=True)


# Initialize session state (Adding new ones needed for guidance workflow)
if 'fetched_data' not in st.session_state: st.session_state.fetched_data = {}
if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False
if 'stock_evaluator_instance' not in st.session_state: st.session_state.stock_evaluator_instance = None
if 'valuation_results_text' not in st.session_state: st.session_state.valuation_results_text = ""
# --- NEW/MODIFIED SESSION STATE ---
if 'ai_guidance_results' not in st.session_state: st.session_state.ai_guidance_results = {} # For detailed analysis from new tab
if 'pushed_qual_scores' not in st.session_state: st.session_state.pushed_qual_scores = {} # To hold scores pushed from the guidance tab
if 'evaluator_suggestion_data' not in st.session_state: st.session_state.evaluator_suggestion_data = {} # To store suggestions for final tab inputs
if 'industry_avg_pe' not in st.session_state: st.session_state.industry_avg_pe = 0.0 # <-- ADD THIS LINE

# Initialize components
@st.cache_resource # Keep caching for efficiency
def init_components():
    """Initialize data fetcher, processor, and valuation model"""
    fetcher = YahooFinanceDataFetcher()
    processor = StockDataProcessor()
    valuation_model = StockValuationModel(risk_free_rate=0.04, market_return=0.09)
    return fetcher, processor, valuation_model

# --- ORIGINAL DISPLAY/FETCH FUNCTIONS (Restored & Verified from uploaded file) ---
def create_price_chart(data_dict, title="Stock Price Comparison"):
    """Create interactive price comparison chart using Plotly"""
    fig = go.Figure()
    for ticker, data in data_dict.items():
        if data and 'processed_history' in data and isinstance(data['processed_history'], pd.DataFrame) and not data['processed_history'].empty:
            df = data['processed_history']
            if 'Close' in df.columns:
                 fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=ticker))
            else: print(f"Skipping {ticker} price: 'Close' missing.")
        else: print(f"Skipping {ticker} price: Invalid history data.")
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price ($)", hovermode='x unified', template='plotly_white', height=500)
    return fig

def create_technical_chart(df, ticker):
    """Create technical analysis chart with price and indicators"""
    if df is None or df.empty: return go.Figure().update_layout(title=f"No technical data for {ticker}")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f'{ticker} Price & MAs', 'RSI', 'MACD'), row_heights=[0.5, 0.25, 0.25])
    # Price and MAs
    if 'Close' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'), row=1, col=1)
    if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'), row=1, col=1)
    if 'SMA_200' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200'), row=1, col=1)
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
    # MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal'), row=3, col=1)
        if 'MACD_hist' in df.columns: fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Hist', marker_color=['green' if v>=0 else 'red' for v in df['MACD_hist']]), row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_layout(height=800, showlegend=False, template='plotly_white', hovermode='x unified')
    fig.update_xaxes(showspikes=True); fig.update_yaxes(showspikes=True)
    return fig

def fetch_and_process_stock(ticker, fetcher, processor):
    """Fetch and process data for a single stock"""
    try:
        info = fetcher.get_company_info(ticker)
        history = fetcher.get_stock_history(ticker, period="5y")
        financials_data = fetcher.get_financials(ticker)
        if not info: return {'info': None, 'error': f"Failed fetch info for {ticker}."}
        data = {'info': info, 'history': history, 'financials': financials_data}
        # Safely process history
        data['processed_history'] = processor.calculate_technical_indicators(history) if isinstance(history, pd.DataFrame) and not history.empty else pd.DataFrame()
        # Safely process financials
        if isinstance(financials_data, dict):
             data['annual_income'] = processor.get_yearly_financial_data(financials_data, 'income_stmt')
             data['annual_balance'] = processor.get_yearly_financial_data(financials_data, 'balance_sheet')
             data['annual_cash_flow'] = processor.get_yearly_financial_data(financials_data, 'cash_flow')
             # Calculate derived metrics safely
             data['fcf'] = processor.calculate_free_cash_flow(data.get('annual_cash_flow'))
             data['total_debt'] = processor.get_total_debt(data.get('annual_balance'))
             data['cash_equivalents'] = processor.get_cash_and_equivalents(data.get('annual_balance'))
             data['total_equity'] = processor.get_total_stockholder_equity(data.get('annual_balance'))
             data['eps_history'] = processor.get_eps_from_financials(data.get('annual_income'))
             data['ebitda'] = processor.get_ebitda(data.get('annual_income'))
        else: # Set defaults if financials missing
             for key in ['annual_income', 'annual_balance', 'annual_cash_flow', 'fcf', 'total_debt', 'cash_equivalents', 'total_equity', 'eps_history', 'ebitda']: data[key] = None
        return data
    except Exception as e:
        print(f"Error in fetch_and_process_stock for {ticker}: {e}") # Log error
        return {'info': None, 'error': str(e)}

def display_company_overview(info, ticker):
    """Display company overview metrics"""
    if not info: st.warning(f"Overview data unavailable for {ticker}."); return
    st.subheader(f"🏢 {info.get('longName', ticker)} Overview")
    col1, col2, col3, col4 = st.columns(4)
    # Formatting functions for safety
    def fmt_m(v): return f"${v/1e9:.2f}B" if isinstance(v,(int,float)) else "N/A"
    def fmt_p(v): return f"${v:.2f}" if isinstance(v,(int,float)) else "N/A"
    def fmt_r(v): return f"{v:.2f}" if isinstance(v,(int,float)) else "N/A"
    def fmt_pct(v): return f"{v*100:.2f}%" if isinstance(v,(int,float)) else "N/A"
    # Display metrics safely
    with col1: st.metric("Price", fmt_p(info.get('currentPrice'))); st.metric("Market Cap", fmt_m(info.get('marketCap')))
    with col2: st.metric("P/E", fmt_r(info.get('trailingPE'))); st.metric("EPS", fmt_p(info.get('trailingEps')))
    with col3: st.metric("52W High", fmt_p(info.get('fiftyTwoWeekHigh'))); st.metric("52W Low", fmt_p(info.get('fiftyTwoWeekLow')))
    with col4: st.metric("Beta", fmt_r(info.get('beta'))); st.metric("Div Yield", fmt_pct(info.get('dividendYield')))
    st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    if info.get('longBusinessSummary'):
        with st.expander("📖 Business Description"): st.write(info['longBusinessSummary'])

def calculate_and_display_valuation(main_ticker, competitor_tickers, fetched_data, valuation_model, fetcher):
    """Calculate and display valuation analysis, storing summary text."""
    output_lines = [] # To store lines for session state text
    st.header("💰 Valuation Analysis")
    main_data = fetched_data.get(main_ticker)
    if not main_data or not main_data.get('info'):
         st.warning(f"Valuation data missing for {main_ticker}.")
         st.session_state.valuation_results_text = "Valuation data missing."
         return

    dcf_value = None; avg_valuation = None
    history_df = main_data.get('history')
    current_price = history_df['Close'].iloc[-1] if isinstance(history_df, pd.DataFrame) and not history_df.empty else main_data['info'].get('currentPrice')
    if not current_price: st.warning("Could not determine current price."); current_price = 0

    st.subheader("1. Discounted Cash Flow (DCF) Valuation")
    try:
        # --- DCF Calculation (Ensure safe access) ---
        beta = main_data['info'].get('beta'); market_cap = main_data['info'].get('marketCap')
        shares = main_data['info'].get('sharesOutstanding'); fcf = main_data.get('fcf')
        current_fcf = fcf.iloc[-1] if isinstance(fcf, pd.Series) and not fcf.empty else None
        debt = main_data.get('total_debt'); total_debt = debt.iloc[-1] if isinstance(debt, pd.Series) and not debt.empty else None
        cash_eq = main_data.get('cash_equivalents'); cash = cash_eq.iloc[-1] if isinstance(cash_eq, pd.Series) and not cash_eq.empty else None

        if all(v is not None for v in [beta, market_cap, shares, current_fcf, total_debt, cash]):
            hist_growth = fcf.pct_change().mean() if len(fcf)>1 else 0.1
            growth = [max(min(hist_growth * (1-0.1*i), 0.3), 0.05) for i in range(5)]
            term_growth = 0.025; cost_debt = 0.055
            cost_equity = valuation_model.calculate_cost_of_equity(beta)
            wacc = valuation_model.calculate_wacc(market_cap, total_debt, cost_equity, cost_debt)

            col1, col2, col3 = st.columns(3) # Params
            with col1: st.metric("Cost of Equity", f"{cost_equity:.2%}")
            with col2: st.metric("WACC", f"{wacc:.2%}")
            with col3: st.metric("Terminal Growth", f"{term_growth:.2%}")
            st.write(f"**FCF Growth Est:** {', '.join([f'{g:.2%}' for g in growth])}")

            dcf_value = valuation_model.dcf_valuation(current_fcf, growth, term_growth, wacc, shares, total_debt, cash)
            upside = ((dcf_value - current_price) / current_price) * 100 if current_price and dcf_value is not None else 0

            col1, col2, col3 = st.columns(3) # Result
            with col1: st.metric("DCF Value", f"${dcf_value:.2f}")
            with col2: st.metric("Current Price", f"${current_price:.2f}")
            with col3: st.metric("Upside/Downside", f"{upside:+.2f}%", delta=f"{upside:+.2f}%")
            output_lines.append(f"DCF Value: {dcf_value:.2f}")
        else: st.warning("⚠️ Insufficient data for DCF."); output_lines.append("DCF Value: N/A")
    except Exception as e: st.error(f"❌ DCF Error: {e}"); st.code(traceback.format_exc()); output_lines.append(f"DCF Value: Error")

    st.divider()
    st.subheader("2. Relative Valuation (vs Competitors)")
    try:
        # --- Relative Calc (Safely collect, average, apply) ---
        pe_list, ps_list, pb_list, ev_ebitda_list = [], [], [], []
        for ticker in competitor_tickers: # Collect
             comp_data = fetched_data.get(ticker)
             if comp_data and comp_data.get('info'):
                info = comp_data['info']
                if info.get('trailingPE', 0) > 0: pe_list.append(info['trailingPE'])
                if info.get('priceToSalesTrailing12Months', 0) > 0: ps_list.append(info['priceToSalesTrailing12Months'])
                if info.get('priceToBook', 0) > 0: pb_list.append(info['priceToBook'])
                try: # EV/EBITDA
                    mkt_cap=info.get('marketCap'); debt=comp_data.get('total_debt'); cash=comp_data.get('cash_equivalents'); ebitda=comp_data.get('ebitda')
                    td=debt.iloc[-1] if isinstance(debt,pd.Series) and not debt.empty else None
                    ce=cash.iloc[-1] if isinstance(cash,pd.Series) and not cash.empty else None
                    eb=ebitda.iloc[-1] if isinstance(ebitda,pd.Series) and not ebitda.empty else None
                    if all(v is not None for v in [mkt_cap, td, ce, eb]) and eb != 0:
                         ev=mkt_cap + td - ce; ev_ebitda = ev / eb
                         if ev_ebitda > 0: ev_ebitda_list.append(ev_ebitda)
                except: continue # Ignore errors for one competitor
       # Averages
        avg_pe = np.mean(pe_list) if pe_list else None; avg_ps = np.mean(ps_list) if ps_list else None
        avg_pb = np.mean(pb_list) if pb_list else None; avg_ev_ebitda = np.mean(ev_ebitda_list) if ev_ebitda_list else None
        st.session_state.industry_avg_pe = float(avg_pe) if avg_pe is not None else 0.0 # <-- ADD THIS LINE
        # Display Averages
        col1, col2, col3, col4 = st.columns(4); col1.metric("Avg P/E", f"{avg_pe:.2f}" if avg_pe else "N/A"); col2.metric("Avg P/S", f"{avg_ps:.2f}" if avg_ps else "N/A"); col3.metric("Avg P/B", f"{avg_pb:.2f}" if avg_pb else "N/A"); col4.metric("Avg EV/EBITDA", f"{avg_ev_ebitda:.2f}" if avg_ev_ebitda else "N/A")
        # Target Metrics
        t_eps = main_data['info'].get('trailingEps'); t_shares = main_data['info'].get('sharesOutstanding')
        t_income = main_data.get('annual_income'); t_rev = t_income.loc['Total Revenue'].iloc[-1] if isinstance(t_income, pd.DataFrame) and 'Total Revenue' in t_income.index and not t_income.loc['Total Revenue'].empty else None
        t_equity = main_data.get('total_equity'); t_te = t_equity.iloc[-1] if isinstance(t_equity, pd.Series) and not t_equity.empty else None
        t_ebitda = main_data.get('ebitda'); t_eb = t_ebitda.iloc[-1] if isinstance(t_ebitda, pd.Series) and not t_ebitda.empty else None
        t_debt = main_data.get('total_debt'); t_td = t_debt.iloc[-1] if isinstance(t_debt, pd.Series) and not t_debt.empty else None
        t_cash = main_data.get('cash_equivalents'); t_ce = t_cash.iloc[-1] if isinstance(t_cash, pd.Series) and not t_cash.empty else None
        # Per Share Metrics
        t_sps = t_rev / t_shares if t_rev and t_shares else None; t_bps = t_te / t_shares if t_te and t_shares else None
        # Calculate Valuations
        rel_vals = valuation_model.relative_valuation(t_eps, t_sps, t_bps, avg_pe, avg_ps, avg_pb)
        if avg_ev_ebitda and t_eb and t_td is not None and t_ce is not None and t_shares:
            try: ev_val = valuation_model.ev_ebitda_valuation(t_eb, avg_ev_ebitda, t_td, t_ce, t_shares); rel_vals['EV/EBITDA'] = ev_val
            except: pass # Ignore EV/EBITDA calc error
        # Display Results
        if rel_vals:
            results_data = []; valid_vals = []
            for m, v in rel_vals.items():
                if v is not None: valid_vals.append(v); ups = ((v - current_price) / current_price * 100) if current_price else 0
                else: ups = None
                results_data.append({'Method':m, 'Value': f"${v:.2f}" if v else "N/A", 'Upside': f"{ups:+.2f}%" if ups else "N/A"})
            st.dataframe(pd.DataFrame(results_data), width='stretch', hide_index=True)
            if valid_vals: # Summary
                 avg_valuation = np.mean(valid_vals); avg_ups = ((avg_valuation - current_price) / current_price * 100) if current_price else 0
                 col1, col2, col3 = st.columns(3); col1.metric("Avg Value", f"${avg_valuation:.2f}"); col2.metric("Current Price", f"${current_price:.2f}"); col3.metric("Avg Upside", f"{avg_ups:+.2f}%", delta=f"{avg_ups:+.2f}%")
                 output_lines.append(f"Relative Avg Value: {avg_valuation:.2f}")
            else: output_lines.append("Relative Avg Value: N/A")
        else: st.warning("⚠️ Insufficient data for relative val."); output_lines.append("Relative Avg Value: N/A")
    except Exception as e: st.error(f"❌ Relative Error: {e}"); st.code(traceback.format_exc()); output_lines.append(f"Relative Avg Value: Error")

    output_lines.append(f"Current Price: {current_price:.2f}" if current_price is not None else "N/A")
    st.session_state.valuation_results_text = "\n".join(output_lines) # Store summary text
# --- END OF ORIGINAL FUNCTIONS ---


def main():
    """Main application function"""
    st.markdown('<p class="main-header">📊 Stock Analysis & Valuation Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive evaluation with AI suggestions</p>', unsafe_allow_html=True)
    fetcher, processor, valuation_model = init_components()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("📖 [App Guideline](https://www.notion.so/Welcome-to-the-Amazing-Stock-Analysis-Valuation-Platform-2966bdd054e980f5a0f8cbd422b1f0bc?source=copy_link)")
        st.subheader("📈 Stock Selection")
        main_ticker = st.text_input("Main Stock Symbol", value="NVDA").upper()
        competitors_input = st.text_input("Competitor Symbols", value="AMD,INTC")
        competitor_tickers = [t.strip().upper() for t in competitors_input.split(',') if t.strip()]
        st.divider()
        st.subheader("🤖 AI Configuration (Optional)")
        provider = "gemini"; api_key_input = "" # Define here
        if EVALUATOR_AVAILABLE:
            available_providers = get_available_providers()
            if available_providers:
                provider_display = {'gemini': '🔷 Google Gemini', 'openai': '🟢 OpenAI ChatGPT'}
                provider = st.radio("Select AI Provider:", options=available_providers, format_func=lambda x: provider_display.get(x, x.capitalize()))
                api_key_input = st.text_input("API Key:", type="password", help=f"Enter {provider.capitalize()} API key")
                if st.button("🔐 Set API Key & Initialize"):
                    if api_key_input:
                        try: # Instantiate original StockEvaluator
                            st.session_state.stock_evaluator_instance = StockEvaluator(ticker=main_ticker, api_key=api_key_input, provider=provider)
                            if st.session_state.stock_evaluator_instance.use_ai: st.success(f"✅ {provider.capitalize()} Initialized!")
                            else: st.warning("Init OK, but AI failed.")
                        except Exception as e: st.error(f"❌ Init Failed: {e}"); st.session_state.stock_evaluator_instance = None
                    else: st.warning("⚠️ Enter API key")
            if provider == "gemini": st.info("🔗 Get Gemini Key: [Google AI Studio](https://makersuite.google.com/app/apikey)")
            elif provider == "openai": st.info("🔗 Get OpenAI Key: [OpenAI Platform](https://platform.openai.com/api-keys)")
        else: st.warning("⚠️ stock_evaluator.py missing.")
        st.divider()
        # Analyze button
        if st.button("🚀 Run Basic Analysis", type="primary", width='stretch'):
            st.session_state.analysis_complete = False; st.session_state.ai_guidance_results = {}; st.session_state.pushed_qual_scores = {}; st.session_state.valuation_results_text = ""; st.session_state.evaluator_suggestion_data = {}; st.session_state.industry_avg_pe = 0.0 # <-- ADD THIS
            with st.spinner("Fetching and analyzing data..."):
                all_tickers = [main_ticker] + competitor_tickers; fetched_data = {}; has_errors = False
                progress_bar = st.progress(0)
                for idx, ticker in enumerate(all_tickers):
                    st.write(f"Fetching {ticker}...")
                    data = fetch_and_process_stock(ticker, fetcher, processor)
                    if data and data.get('info'): fetched_data[ticker] = data
                    else: st.warning(f"Failed fetch {ticker}. Error: {data.get('error', 'Unknown')}"); has_errors = has_errors or (ticker == main_ticker)
                    progress_bar.progress((idx + 1) / len(all_tickers))
                if has_errors or main_ticker not in fetched_data: st.error(f"❌ Failed fetch for {main_ticker}."); st.stop()
                st.session_state.fetched_data = fetched_data; st.session_state.analysis_complete = True
                st.success("✅ Basic data fetched!")
                # Init evaluator if not done
                if not st.session_state.stock_evaluator_instance and EVALUATOR_AVAILABLE:
                    try: st.session_state.stock_evaluator_instance = StockEvaluator(ticker=main_ticker, api_key=api_key_input or None, provider=provider); st.info("Evaluator initialized.")
                    except Exception as e: st.error(f"Failed evaluator init: {e}")
                # --- Get suggestion data ---
                evaluator_instance = st.session_state.get('stock_evaluator_instance')
                # Use evaluator.info if available, otherwise try fetching minimal info
                info_for_sugg = None
                if evaluator_instance and evaluator_instance.info:
                    info_for_sugg = evaluator_instance.info
                elif main_ticker in st.session_state.fetched_data:
                    info_for_sugg = st.session_state.fetched_data[main_ticker].get('info')

                if info_for_sugg:
                    st.write("Preparing suggestion data...")
                    sugg_data_raw = {
                        'pe': info_for_sugg.get('trailingPE'), # <-- ADD THIS LINE
                        'roe': info_for_sugg.get('returnOnEquity'), 'roa': info_for_sugg.get('returnOnAssets'), 'current_ratio': info_for_sugg.get('currentRatio'),
                        'total_debt_equity': info_for_sugg.get('debtToEquity'), 'revenue_growth_yoy': info_for_sugg.get('revenueGrowth'),
                        'eps_growth_yoy': info_for_sugg.get('earningsGrowth'), 'beta': info_for_sugg.get('beta'), 'volume': info_for_sugg.get('averageVolume'),
                        'ask': info_for_sugg.get('ask'), 'bid': info_for_sugg.get('bid'), 'forward_dividend_yield': info_for_sugg.get('forwardAnnualDividendYield')
                    }
                    sugg_data_clean = {}
                    for k, v in sugg_data_raw.items(): # Clean None/NaN to 0.0
                        sugg_data_clean[k] = float(v) if isinstance(v, (int, float)) and not np.isnan(v) else 0.0
                    try: # Calculate spread safely
                         sugg_data_clean['bid_ask_spread_pct'] = (sugg_data_clean.get('ask',0)-sugg_data_clean.get('bid',0))/sugg_data_clean['bid'] if sugg_data_clean.get('bid',0)>0 else 1.0
                    except: sugg_data_clean['bid_ask_spread_pct'] = 1.0 # Default high spread
                    st.session_state.evaluator_suggestion_data = sugg_data_clean
                    st.success("✅ Suggestion data prepared.")
                else: st.warning("Could not get data for suggestions.")
    

    # --- Main Content Area ---
    if st.session_state.analysis_complete and st.session_state.fetched_data:
        main_data = st.session_state.fetched_data[main_ticker]
        evaluator = st.session_state.stock_evaluator_instance

        # --- Define Tabs ---
        tab_list = ["📊 Overview", "📈 Charts", "📋 Financials", "💰 Valuation",
                    "🤔 Qualitative Guidance", "🎯 Comprehensive Result"] # Added Guidance
        tabs = st.tabs(tab_list)

        # --- Populate Original Tabs (0-3) ---
        with tabs[0]: display_company_overview(main_data.get('info'), main_ticker)
        with tabs[1]:
            st.header("📈 Technical Analysis"); st.plotly_chart(create_price_chart(st.session_state.fetched_data), use_container_width=True)
            tech_data = main_data.get('processed_history');
            if tech_data is not None and not tech_data.empty: st.plotly_chart(create_technical_chart(tech_data, main_ticker), use_container_width=True)
            else: st.info("Technical data unavailable.")
        with tabs[2]:
            st.header("🏢 Financial Statements")
            financial_type = st.selectbox("Select Statement:", ["Income Statement", "Balance Sheet", "Cash Flow"], key=f"fin_select_{main_ticker}")
            fin_map = {"Income Statement": "income_stmt", "Balance Sheet": "balance_sheet", "Cash Flow": "cash_flow"}
            fin_dict = main_data.get('financials', {})
            df_key = fin_map.get(financial_type)
            df = fin_dict.get(df_key) if isinstance(fin_dict, dict) and df_key else None
            if isinstance(df, pd.DataFrame) and not df.empty: st.dataframe(df, use_container_width=True)
            else: st.warning("⚠️ Financial data unavailable.")
        with tabs[3]: calculate_and_display_valuation(main_ticker, competitor_tickers, st.session_state.fetched_data, valuation_model, fetcher)

        # --- New Tab: Qualitative Guidance (4) ---
        with tabs[4]:
            st.header("🤔 Qualitative & Operational Guidance")
            st.markdown("Analyze factors, load AI guidance, then push scores.")
            ai_configured = bool(evaluator and evaluator.use_ai)
            if not ai_configured: st.warning("Configure AI to load guidance.", icon="🤖")

            # Unique key for button using ticker
            load_guidance_key = f"load_guidance_{main_ticker}"
            if st.button("🤖 Load AI Guidance", disabled=not ai_configured, key=load_guidance_key):
                with st.spinner(f"Running AI analysis on {main_ticker}..."):
                    # Use the get_ai_qualitative_guidance method added to the original StockEvaluator
                    guidance_results = evaluator.get_ai_qualitative_guidance(ticker_override=main_ticker) # Pass ticker override
                    if isinstance(guidance_results, dict) and 'error' in guidance_results: st.error(f"AI Error: {guidance_results['error']}")
                    elif isinstance(guidance_results, dict): st.session_state.ai_guidance_results = guidance_results; st.success("✅ AI loaded!"); st.rerun()
                    else: st.error(f"Unexpected AI result type: {type(guidance_results)}")
            st.divider()

            current_guidance_scores = {} # Temp dict to hold scores entered in this tab
            # --- START OF QUALITATIVE BLOCK ---
            if QUALITATIVE_QUESTIONS_GUIDANCE:
                for category, questions in QUALITATIVE_QUESTIONS_GUIDANCE.items():
                    st.subheader(f"{category.replace('_', ' ').title()}")
                    for q in questions:
                        q_key = q['key']
                        # Display question and guidance
                        st.markdown(f"<p class='guidance-question'>Q: {q['question']}</p>", unsafe_allow_html=True)
                        st.caption(f"Guidance: {q['guidance']}")
                        
                        # Get AI data for this question
                        ai_data = st.session_state.ai_guidance_results.get(q_key)
                        ai_score = 5 # Default
                        ai_analysis = "No AI analysis loaded."
                        if ai_data and isinstance(ai_data, dict):
                            ai_score = int(ai_data['suggested_score']) if str(ai_data.get('suggested_score', '')).isdigit() else 5
                            ai_analysis = ai_data.get('analysis', "AI analysis format error.")

                        # Get default value from pushed scores, or use AI score
                        default_value = st.session_state.pushed_qual_scores.get(q_key, ai_score)

                        # Create UI elements
                        col1, col2 = st.columns([1, 4])
                        with col1: # Score Input
                             current_guidance_scores[q_key] = st.number_input("Score", 1, 10, value=default_value, key=f"guidance_{main_ticker}_{q_key}", label_visibility="collapsed")
                        with col2: # AI Details Checkbox
                             show_ai = st.checkbox("Show AI Details", key=f"show_ai_{main_ticker}_{q_key}", value=False)
                             if show_ai:
                                 st.metric("AI Suggested Score", f"{ai_score}/10")
                                 st.markdown(ai_analysis)
                        st.markdown("---") # Separator
            # --- END OF QUALITATIVE BLOCK ---
            
            st.divider() # Divider between sections

            # --- START OF OPERATIONAL BLOCK (Correct Indentation) ---
            if OPERATIONAL_QUESTIONS_GUIDANCE:
                st.subheader("Operational Factors")
                for category, questions in OPERATIONAL_QUESTIONS_GUIDANCE.items():
                    for q in questions:
                        q_key = q['key']
                        # Display question and guidance
                        st.markdown(f"<p class='guidance-question'>Q: {q['question']}</p>", unsafe_allow_html=True)
                        st.caption(f"Guidance: {q['guidance']}")
                        
                        # Get AI data for this question
                        ai_data = st.session_state.ai_guidance_results.get(q_key)
                        ai_score = 5 # Default
                        ai_analysis = "No AI analysis loaded."
                        if ai_data and isinstance(ai_data, dict):
                            ai_score = int(ai_data['suggested_score']) if str(ai_data.get('suggested_score', '')).isdigit() else 5
                            ai_analysis = ai_data.get('analysis', "AI analysis format error.")

                        # Get default value from pushed scores, or use AI score
                        default_value = st.session_state.pushed_qual_scores.get(q_key, ai_score)

                        # Create UI elements
                        col1, col2 = st.columns([1, 4])
                        with col1: # Score Input
                             current_guidance_scores[q_key] = st.number_input("Score", 1, 10, value=default_value, key=f"guidance_{main_ticker}_{q_key}", label_visibility="collapsed")
                        with col2: # AI Details Checkbox
                             show_ai = st.checkbox("Show AI Details", key=f"show_ai_{main_ticker}_{q_key}", value=False)
                             if show_ai:
                                 st.metric("AI Suggested Score", f"{ai_score}/10")
                                 st.markdown(ai_analysis)
                        st.markdown("---") # Separator
            # --- END OF OPERATIONAL BLOCK ---

            st.divider()
            # Push Scores Button
            push_scores_key = f"push_scores_{main_ticker}"
            if st.button("➡️ Push All Scores to Final Tab", key=push_scores_key, type="primary"):
                st.session_state.pushed_qual_scores = current_guidance_scores.copy()
                st.success(f"Pushed {len(st.session_state.pushed_qual_scores)} scores!")
                # Optional: Show what was pushed
                with st.expander("Show Pushed Scores"):
                    st.json(st.session_state.pushed_qual_scores)
            
        with tabs[5]:
            st.header("🎯 Comprehensive Stock Evaluation Result")
            st.markdown("Enter **Quant/Val/Ops** metrics. **Qual** scores used from 'Push Scores'.")
            if not evaluator: st.warning("Run analysis first."); st.stop()

            user_inputs_final = {'quant_metrics': {}, 'val_inputs': {}, 'ops_scores': {}, 'industry_avgs': {}}
            sugg_data = st.session_state.evaluator_suggestion_data
            # Parse valuation defaults from stored text
            val_defaults = {'current_price': 0.0, 'dcf_value': 0.0, 'relative_value': 0.0}
            try:
                lines = st.session_state.valuation_results_text.split('\n')

                # Helper function to safely parse the value
                def safe_parse_float(line_text):
                    try:
                        # Split at the colon, take the last part, strip whitespace
                        value_str = line_text.split(':')[-1].strip()
                        # Return the float value
                        return float(value_str)
                    except (ValueError, TypeError):
                        # If it's "N/A", "Error", or empty, return 0.0
                        return 0.0

                for line in lines:
                    line_l = line.lower().strip()
                    if line_l.startswith("current price:"):
                        val_defaults['current_price'] = safe_parse_float(line)
                    elif line_l.startswith("dcf value:"):
                        val_defaults['dcf_value'] = safe_parse_float(line)
                    elif line_l.startswith("relative avg value:"):
                        val_defaults['relative_value'] = safe_parse_float(line)
            except Exception as parse_e: print(f"Could not parse val defaults: {parse_e}") # Log error
            st.subheader("1. Quantitative Factors")
            with st.expander("Enter Quantitative Metrics"):
                 c1, c2 = st.columns(2)
                 with c1:
                      st.markdown("**Profitability**")
                      # Unique keys for final inputs
                      user_inputs_final['quant_metrics']['avg_margins'] = st.number_input("Avg. Margins", 0.0, format="%.4f", key=f"final_avg_margins_{main_ticker}")
                      user_inputs_final['quant_metrics']['roe'] = st.number_input("ROE", value=sugg_data.get('roe',0.0), format="%.4f", help=f"Sugg: {sugg_data.get('roe'):.4f}", key=f"final_roe_{main_ticker}")
                      user_inputs_final['quant_metrics']['roa'] = st.number_input("ROA", value=sugg_data.get('roa',0.0), format="%.4f", help=f"Sugg: {sugg_data.get('roa'):.4f}", key=f"final_roa_{main_ticker}")
                      st.markdown("**Solvency**")
                      user_inputs_final['quant_metrics']['current_ratio'] = st.number_input("Current Ratio", value=sugg_data.get('current_ratio',0.0), format="%.4f", help=f"Sugg: {sugg_data.get('current_ratio'):.2f}", key=f"final_current_ratio_{main_ticker}")
                      user_inputs_final['quant_metrics']['quick_ratio'] = st.number_input("Quick Ratio", 0.0, format="%.4f", key=f"final_quick_ratio_{main_ticker}")
                      user_inputs_final['quant_metrics']['interest_coverage'] = st.number_input("Interest Coverage", 0.0, format="%.4f", key=f"final_int_cov_{main_ticker}")
                      user_inputs_final['quant_metrics']['total_debt_equity'] = st.number_input("Debt/Equity", value=sugg_data.get('total_debt_equity',0.0), format="%.4f", help=f"Sugg: {sugg_data.get('total_debt_equity'):.2f}", key=f"final_de_{main_ticker}")
                 with c2:
                      st.markdown("**Growth Potential**")
                      # Expect % input, convert later
                      user_inputs_final['quant_metrics']['revenue_growth_yoy'] = st.number_input("Revenue Growth Rate %", value=sugg_data.get('revenue_growth_yoy',0.0)*100, format="%.2f", help=f"Sugg: {sugg_data.get('revenue_growth_yoy'):.2%}", key=f"final_rev_g_{main_ticker}")
                      user_inputs_final['quant_metrics']['eps_growth_yoy'] = st.number_input("EPS Growth %", value=sugg_data.get('eps_growth_yoy',0.0)*100, format="%.2f", help=f"Sugg: {sugg_data.get('eps_growth_yoy'):.2%}", key=f"final_eps_g_{main_ticker}")
                      st.markdown("**Risk Metrics**")
                      user_inputs_final['quant_metrics']['beta'] = st.number_input("Beta", value=sugg_data.get('beta',0.0), format="%.4f", help=f"Sugg: {sugg_data.get('beta'):.2f}", key=f"final_beta_{main_ticker}")
                      user_inputs_final['quant_metrics']['volatility'] = st.number_input("Volatility", 0.0, format="%.4f", key=f"final_vol_{main_ticker}")
                      st.markdown("**Metrics for Credits**")
                      user_inputs_final['quant_metrics']['pe'] = st.number_input("P/E", value=sugg_data.get('pe',0.0), format="%.2f", help=f"Sugg: {sugg_data.get('pe'):.2f}", key=f"final_pe_{main_ticker}")
                      user_inputs_final['industry_avgs']['pe'] = st.number_input("Industry Avg P/E", value=st.session_state.industry_avg_pe, format="%.2f", key=f"final_ind_pe_{main_ticker}")
            with st.expander("Enter Op Efficiency Metrics"):
                 c1, c2 = st.columns(2)
                 user_inputs_final['quant_metrics']['days_inventory'] = c1.number_input("Days Inventory", 0.0, format="%.2f", key=f"final_days_inv_{main_ticker}")
                 user_inputs_final['industry_avgs']['days_inventory'] = c2.number_input("Industry Avg Days Inventory", 0.0, format="%.2f", key=f"final_ind_inv_{main_ticker}")
                 user_inputs_final['quant_metrics']['days_sales'] = c1.number_input("Days Sales Outstanding", 0.0, format="%.2f", key=f"final_days_sales_{main_ticker}")
                 user_inputs_final['industry_avgs']['days_sales'] = c2.number_input("Industry Avg Days Sales Outstanding", 0.0, format="%.2f", key=f"final_ind_dso_{main_ticker}")

            st.subheader("3. Valuation Factors")
            with st.expander("Enter Valuation Inputs"):
                 user_inputs_final['val_inputs']['current_price'] = st.number_input("Current Price", value=val_defaults['current_price'], format="%.2f", key=f"final_current_price_{main_ticker}")
                 user_inputs_final['val_inputs']['dcf_value'] = st.number_input("DCF Value", value=val_defaults['dcf_value'], format="%.2f", key=f"final_dcf_{main_ticker}")
                 user_inputs_final['val_inputs']['relative_value'] = st.number_input("Relative Value (Avg)", value=val_defaults['relative_value'], format="%.2f", key=f"final_relative_{main_ticker}")


            st.divider()
            # --- Final Calculation Button ---
            calc_final_key = f"calc_final_{main_ticker}"
            if st.button("🧮 Calculate Final Score", key=calc_final_key):
                # 1. Get Pushed Qual Scores
                final_qual_scores = st.session_state.pushed_qual_scores.copy()
                if not final_qual_scores: st.warning("⚠️ No Qual scores pushed. Using default 5."); final_qual_scores = {q['key']: 5 for cat in QUALITATIVE_QUESTIONS_GUIDANCE.values() for q in cat}

                # 2. Convert Growth % inputs
                user_inputs_final['quant_metrics']['revenue_growth_yoy'] /= 100.0
                user_inputs_final['quant_metrics']['eps_growth_yoy'] /= 100.0

                with st.spinner("Running final evaluation..."):
                    try:
                        # 3. Prepare data dict for evaluator methods
                        calc_input_data = st.session_state.fetched_data.get(main_ticker, {}).copy()
                        calc_input_data.update(user_inputs_final['quant_metrics'])
                        calc_input_data.update(user_inputs_final['val_inputs'])

                        # --- MODIFIED: Add PUSHED operational scores ---
                        # The evaluator expects keys like 'user_tax_score', 'user_liquidity_score' etc.
                        calc_input_data['user_liquidity_score'] = final_qual_scores.get('liquidity', 5)
                        calc_input_data['user_tax_score'] = final_qual_scores.get('tax', 5)
                        calc_input_data['user_dividend_score'] = final_qual_scores.get('dividend', 5)
                        calc_input_data['user_portfolio_fit_score'] = final_qual_scores.get('portfolio_fit', 5)
                    # --- END OF MODIFICATION ---
                        # Ensure evaluator uses current ticker data if needed
                        # evaluator.ticker = main_ticker; evaluator.info = calc_input_data.get('info')

                        # --- 4. Call ORIGINAL StockEvaluator scoring methods ---
                        # These methods need to exist in your StockEvaluator class and accept the prepared calc_input_data dict and potentially industry_avgs
                        quant_result = evaluator._calculate_quantitative_score(calc_input_data, user_inputs_final['industry_avgs'])
                        qual_scores_cat = {cat: {q['key']: final_qual_scores.get(q['key'], 5) for q in qs} for cat, qs in QUALITATIVE_QUESTIONS_GUIDANCE.items()}
                        qual_result = evaluator._calculate_qualitative_score(qual_scores_cat) # This uses the pushed scores
                        val_result = evaluator._calculate_valuation_score(calc_input_data)
                        ops_result = evaluator._calculate_operational_score(calc_input_data)
                        credits_result = evaluator._calculate_justification_credits(quant_result.get('scores',{}), qual_result.get('categories',{}), val_result.get('scores',{}), ops_result.get('scores',{}), calc_input_data)
                        final_result_data = evaluator._calculate_final_score_and_recommendation(quant_result.get('total_score',0), qual_result.get('normalized_score',0), val_result.get('total_score',0), ops_result.get('total_score',0), credits_result.get('total_credits',0))
                        report = evaluator._generate_report(quant_result, qual_result, val_result, ops_result, credits_result, final_result_data, calc_input_data)

                        # --- 5. Display Results ---
                        st.success("✅ Evaluation complete!")
                        final_score = final_result_data.get('final_score', 'N/A'); recommendation = final_result_data.get('recommendation', 'N/A')
                        col1, col2 = st.columns(2); col1.metric("Final Score", f"{final_score:.2f}" if isinstance(final_score, (int, float)) else 'N/A'); col2.metric("Recommendation", recommendation)
                        st.text_area("Full Report", report, height=600)
                        st.download_button(label="💾 Download Report", data=report, file_name=f"{main_ticker}_eval_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain")
                    except AttributeError as ae: st.error(f"❌ Calc Error: Method missing in stock_evaluator? {ae}"); st.code(traceback.format_exc())
                    except Exception as e: st.error(f"❌ Error during final calculation: {e}"); st.code(traceback.format_exc())

    else:
        st.info("👋 Configure analysis in sidebar & click 'Run Basic Analysis'.")

    # Footer
    st.divider()
    st.markdown("<div style='text-align: center; color: #666;'>Disclaimer: Educational. Not financial advice.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()