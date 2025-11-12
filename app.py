"""
Stock & ETF Analysis Web Application
A Streamlit web app that provides comprehensive stock/ETF analysis with AI-powered insights
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

# --- MODIFIED: Import BOTH evaluators and guidance data ---
try:
    from stock_evaluator import (
        StockEvaluator,
        QUALITATIVE_QUESTIONS_GUIDANCE,
        OPERATIONAL_QUESTIONS_GUIDANCE
    )
    STOCK_EVAL_AVAILABLE = True
except ImportError as e:
    st.error(f"Fatal Error: Could not import StockEvaluator. Details: {e}")
    STOCK_EVAL_AVAILABLE = False
    class StockEvaluator: pass
    QUALITATIVE_QUESTIONS_GUIDANCE = {}
    OPERATIONAL_QUESTIONS_GUIDANCE = {}

try:
    from etf_evaluator import (
        ETFEvaluator,
        ETF_QUALITATIVE_QUESTIONS_GUIDANCE,
        ETF_OPERATIONAL_QUESTIONS_GUIDANCE, # <-- IMPORT NEW ETF OPS DATA
        get_available_providers,
        is_provider_available
    )
    ETF_EVAL_AVAILABLE = True
except ImportError as e:
    st.error(f"Fatal Error: Could not import ETFEvaluator. Details: {e}")
    ETF_EVAL_AVAILABLE = False
    class ETFEvaluator: pass
    ETF_QUALITATIVE_QUESTIONS_GUIDANCE = {}
    ETF_OPERATIONAL_QUESTIONS_GUIDANCE = {} # Placeholder
    def get_available_providers(): return []
# --- END MODIFICATION ---


# Page configuration
st.set_page_config(
    page_title="Stock & ETF Analysis Tool",
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
    .guidance-question { font-weight: bold; margin-top: 1rem; }
    .guidance-caption { color: #555; font-style: italic; }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'analysis_type' not in st.session_state: st.session_state.analysis_type = "Stock"
if 'fetched_data' not in st.session_state: st.session_state.fetched_data = {}
if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False
if 'evaluator_instance' not in st.session_state: st.session_state.evaluator_instance = None # Will hold Stock or ETF evaluator
if 'valuation_results_text' not in st.session_state: st.session_state.valuation_results_text = ""
if 'ai_guidance_results' not in st.session_state: st.session_state.ai_guidance_results = {}
if 'pushed_scores' not in st.session_state: st.session_state.pushed_scores = {}
if 'evaluator_suggestion_data' not in st.session_state: st.session_state.evaluator_suggestion_data = {}
if 'industry_avg_pe' not in st.session_state: st.session_state.industry_avg_pe = 0.0

# Initialize components
@st.cache_resource
def init_components():
    fetcher = YahooFinanceDataFetcher()
    processor = StockDataProcessor()
    valuation_model = StockValuationModel(risk_free_rate=0.04, market_return=0.09)
    return fetcher, processor, valuation_model

# --- CHARTING/FETCHING FUNCTIONS (Unchanged from previous corrected version) ---
def create_price_chart(data_dict, title="Price Comparison"):
    fig = go.Figure()
    for ticker, data in data_dict.items():
        if data and 'processed_history' in data and isinstance(data['processed_history'], pd.DataFrame) and not data['processed_history'].empty:
            df = data['processed_history']
            if 'Close' in df.columns:
                 fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=ticker))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price ($)", hovermode='x unified', template='plotly_white', height=500)
    return fig

def create_technical_chart(df, ticker):
    if df is None or df.empty: return go.Figure().update_layout(title=f"No technical data for {ticker}")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f'{ticker} Price & MAs', 'RSI'), row_heights=[0.7, 0.3])
    if 'Close' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'), row=1, col=1)
    if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'), row=1, col=1)
    if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'), row=1, col=1)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_layout(height=700, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template='plotly_white', hovermode='x unified')
    fig.update_xaxes(showgrid=True); fig.update_yaxes(showgrid=True)
    return fig

def fetch_and_process_stock(ticker, fetcher, processor):
    """Fetch and process data for a single stock OR ETF"""
    try:
        info = fetcher.get_company_info(ticker)
        history = fetcher.get_stock_history(ticker, period="10y") # Get 10y for ETF
        financials_data = fetcher.get_financials(ticker)
        if not info:
            print(f"fetch_and_process_stock: No info found for {ticker}")
            return {'info': None, 'error': f"Failed fetch info for {ticker}."}

        data = {'info': info, 'history': history, 'financials': financials_data}
        data['processed_history'] = processor.calculate_technical_indicators(history) if isinstance(history, pd.DataFrame) and not history.empty else pd.DataFrame()

        # --- Process based on type ---
        is_etf = ('etfIndustry' in info or 'legalType' in info and 'Exchange Traded Fund' in info['legalType'] or 'fundFamily' in info) and 'sector' not in info
        if not is_etf and ('quoteType' in info and info['quoteType'] == 'ETF'):
             is_etf = True
             print(f"Info: {ticker} identified as ETF via quoteType.")
        if not is_etf and 'sector' in info:
             print(f"Info: {ticker} identified as Stock via 'sector'.")
             is_etf = False # Ensure it's false
        elif not is_etf:
             print(f"Warning: Could not definitively classify {ticker}. Assuming Stock based on lack of ETF keys.")
             is_etf = False # Default assumption

        data['is_etf'] = is_etf
        print(f"fetch_and_process_stock: {ticker} classified as {'ETF' if is_etf else 'Stock'}")

        if is_etf:
            data['etf_metrics'] = processor.get_etf_metrics(info)
        else: # Stock
            if isinstance(financials_data, dict):
                 data['annual_income'] = processor.get_yearly_financial_data(financials_data, 'income_stmt')
                 data['annual_balance'] = processor.get_yearly_financial_data(financials_data, 'balance_sheet')
                 data['annual_cash_flow'] = processor.get_yearly_financial_data(financials_data, 'cash_flow')
                 data['fcf'] = processor.calculate_free_cash_flow(data.get('annual_cash_flow'))
                 data['total_debt'] = processor.get_total_debt(data.get('annual_balance'))
                 data['cash_equivalents'] = processor.get_cash_and_equivalents(data.get('annual_balance'))
                 data['total_equity'] = processor.get_total_stockholder_equity(data.get('annual_balance'))
                 data['eps_history'] = processor.get_eps_from_financials(data.get('annual_income'))
                 data['ebitda'] = processor.get_ebitda(data.get('annual_income'))
            else:
                 print(f"Warning: Financials data for stock {ticker} is not a dictionary.")

        return data
    except Exception as e:
        print(f"Error in fetch_and_process_stock for {ticker}: {e}\n{traceback.format_exc()}")
        return {'info': None, 'error': str(e)}

def display_company_overview(info, ticker, is_etf=False):
    """Display company/ETF overview metrics"""
    if not info: st.warning(f"Overview data unavailable for {ticker}."); return

    st.subheader(f"🏢 {info.get('longName', ticker)} Overview")
    def fmt_m(v): return f"${v/1e9:.2f}B" if isinstance(v,(int,float)) and v is not None else "N/A"
    def fmt_p(v): return f"${v:.2f}" if isinstance(v,(int,float)) and v is not None else "N/A"
    def fmt_r(v): return f"{v:.2f}" if isinstance(v,(int,float)) and v is not None else "N/A"
    def fmt_pct(v): return f"{v*100:.2f}%" if isinstance(v,(int,float)) and v is not None else "N/A"

    if is_etf:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Price", fmt_p(info.get('currentPrice', info.get('previousClose')))); st.metric("NAV", fmt_p(info.get('navPrice')))
        with col2: st.metric("Total Assets", fmt_m(info.get('totalAssets'))); st.metric("Yield", fmt_pct(info.get('yield')))
        with col3: st.metric("Expense Ratio", fmt_pct(info.get('expenseRatio'))); st.metric("Beta (3Y)", fmt_r(info.get('beta3Year')))
        with col4: st.metric("P/E (Trailing)", fmt_r(info.get('trailingPE'))); st.metric("Turnover", fmt_pct(info.get('turnover')))
        st.write(f"**Family:** {info.get('fundFamily', info.get('family', 'N/A'))} | **Category:** {info.get('category', 'N/A')}")
    else: # Stock
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Price", fmt_p(info.get('currentPrice'))); st.metric("Market Cap", fmt_m(info.get('marketCap')))
        with col2: st.metric("P/E", fmt_r(info.get('trailingPE'))); st.metric("EPS", fmt_p(info.get('trailingEps')))
        with col3: st.metric("52W High", fmt_p(info.get('fiftyTwoWeekHigh'))); st.metric("52W Low", fmt_p(info.get('fiftyTwoWeekLow')))
        with col4: st.metric("Beta", fmt_r(info.get('beta'))); st.metric("Div Yield", fmt_pct(info.get('dividendYield')))
        st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")

    if info.get('longBusinessSummary'):
        with st.expander("📖 Description"): st.write(info['longBusinessSummary'])

# --- (Stock-specific UI functions - UNCHANGED) ---
def display_stock_guidance_tab(main_ticker, evaluator):
    st.header("🤔 Qualitative & Operational Guidance (Stock)")
    st.markdown("Analyze factors, load AI guidance, then push scores.")
    ai_configured = bool(evaluator and evaluator.use_ai)
    if not ai_configured: st.warning("Configure AI to load guidance.", icon="🤖")

    load_guidance_key = f"load_guidance_{main_ticker}"
    if st.button("🤖 Load AI Guidance", disabled=not ai_configured, key=load_guidance_key):
        with st.spinner(f"Running AI analysis on {main_ticker}..."):
            guidance_results = evaluator.get_ai_qualitative_guidance(ticker_override=main_ticker)
            if isinstance(guidance_results, dict) and 'error' in guidance_results: st.error(f"AI Error: {guidance_results['error']}")
            elif isinstance(guidance_results, dict): st.session_state.ai_guidance_results = guidance_results; st.success("✅ AI loaded!"); st.rerun()
            else: st.error(f"Unexpected AI result type: {type(guidance_results)}")
    st.divider()

    current_guidance_scores = {}

    # --- QUALITATIVE BLOCK ---
    if QUALITATIVE_QUESTIONS_GUIDANCE:
        for category, questions in QUALITATIVE_QUESTIONS_GUIDANCE.items():
            st.subheader(f"{category.replace('_', ' ').title()}")
            for q in questions:
                _display_guidance_widget(q, main_ticker, current_guidance_scores)

    st.divider()

    # --- OPERATIONAL BLOCK ---
    if OPERATIONAL_QUESTIONS_GUIDANCE:
        st.subheader("Operational Factors")
        for category, questions in OPERATIONAL_QUESTIONS_GUIDANCE.items():
            for q in questions:
                _display_guidance_widget(q, main_ticker, current_guidance_scores)

    st.divider()
    push_scores_key = f"push_scores_{main_ticker}"
    if st.button("➡️ Push All Scores to Final Tab", key=push_scores_key, type="primary"):
        st.session_state.pushed_scores = current_guidance_scores.copy()
        st.success(f"Pushed {len(st.session_state.pushed_scores)} scores!")

def display_stock_final_tab(main_ticker, evaluator):
    st.header("🎯 Comprehensive Stock Evaluation Result")
    st.markdown("Enter **Quant/Val** metrics. **Qual/Ops** scores used from 'Push Scores'.")
    if not evaluator: st.warning("Run analysis first."); st.stop()

    user_inputs_final = {'quant_metrics': {}, 'val_inputs': {}, 'industry_avgs': {}}
    sugg_data = st.session_state.evaluator_suggestion_data

    # Helper for suggestions
    def sugg(key, default=0.0): return sugg_data.get(key, default) if sugg_data.get(key) is not None else default

    # Parse valuation defaults from stored text
    val_defaults = {'current_price': 0.0, 'dcf_value': 0.0, 'relative_value': 0.0}
    try:
        lines = st.session_state.valuation_results_text.split('\n')
        def safe_parse_float(line_text):
            try: return float(line_text.split(':')[-1].strip())
            except: return 0.0
        for line in lines:
            line_l = line.lower().strip()
            if line_l.startswith("current price:"): val_defaults['current_price'] = safe_parse_float(line)
            elif line_l.startswith("dcf value:"): val_defaults['dcf_value'] = safe_parse_float(line)
            elif line_l.startswith("relative avg value:"): val_defaults['relative_value'] = safe_parse_float(line)
    except Exception as e: print(f"Could not parse val defaults: {e}")

    st.subheader("1. Quantitative Factors")
    with st.expander("Enter Quantitative Metrics"):
         c1, c2 = st.columns(2)
         with c1:
              st.markdown("**Profitability**")
              user_inputs_final['quant_metrics']['avg_margins'] = st.number_input("Net Margins", 0.0, format="%.4f", key=f"final_avg_margins_{main_ticker}")
              user_inputs_final['quant_metrics']['roe'] = st.number_input("ROE", value=float(sugg('roe')), format="%.4f", help=f"Sugg: {sugg('roe'):.4f}", key=f"final_roe_{main_ticker}")
              user_inputs_final['quant_metrics']['roa'] = st.number_input("ROA", value=float(sugg('roa')), format="%.4f", help=f"Sugg: {sugg('roa'):.4f}", key=f"final_roa_{main_ticker}")
              st.markdown("**Solvency**")
              user_inputs_final['quant_metrics']['current_ratio'] = st.number_input("Current Ratio", value=float(sugg('current_ratio')), format="%.2f", help=f"Sugg: {sugg('current_ratio'):.2f}", key=f"final_current_ratio_{main_ticker}")
              user_inputs_final['quant_metrics']['quick_ratio'] = st.number_input("Quick Ratio", 0.0, format="%.2f", key=f"final_quick_ratio_{main_ticker}")
              user_inputs_final['quant_metrics']['interest_coverage'] = st.number_input("Interest Coverage", 0.0, format="%.2f", key=f"final_int_cov_{main_ticker}")
              user_inputs_final['quant_metrics']['total_debt_equity'] = st.number_input("Debt/Equity", value=float(sugg('total_debt_equity')), format="%.2f", help=f"Sugg: {sugg('total_debt_equity'):.2f}", key=f"final_de_{main_ticker}")
         with c2:
              st.markdown("**Growth Potential**")
              user_inputs_final['quant_metrics']['revenue_growth_yoy'] = st.number_input("Revenue Growth %", value=float(sugg('revenue_growth_yoy')*100), format="%.2f", help=f"Sugg: {sugg('revenue_growth_yoy'):.2%}", key=f"final_rev_g_{main_ticker}")
              user_inputs_final['quant_metrics']['eps_growth_yoy'] = st.number_input("EPS Growth %", value=float(sugg('eps_growth_yoy')*100), format="%.2f", help=f"Sugg: {sugg('eps_growth_yoy'):.2%}", key=f"final_eps_g_{main_ticker}")
              st.markdown("**Risk Metrics**")
              user_inputs_final['quant_metrics']['beta'] = st.number_input("Beta", value=float(sugg('beta')), format="%.2f", help=f"Sugg: {sugg('beta'):.2f}", key=f"final_beta_{main_ticker}")
              user_inputs_final['quant_metrics']['volatility'] = st.number_input("Volatility %", 0.0, format="%.2f", key=f"final_vol_{main_ticker}")
              st.markdown("**Metrics for Credits**")
              user_inputs_final['quant_metrics']['pe'] = st.number_input("P/E", value=float(sugg('pe')), format="%.2f", help=f"Sugg: {sugg('pe'):.2f}", key=f"final_pe_{main_ticker}")
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
    calc_final_key = f"calc_final_stock_{main_ticker}"
    if st.button("🧮 Calculate Final Stock Score", key=calc_final_key):
        final_scores = st.session_state.pushed_scores.copy()
        if not final_scores: st.warning("⚠️ No Qual/Ops scores pushed. Using default 5.");

        # Convert % inputs back to decimals for calculation
        user_inputs_final['quant_metrics']['revenue_growth_yoy'] /= 100.0
        user_inputs_final['quant_metrics']['eps_growth_yoy'] /= 100.0
        user_inputs_final['quant_metrics']['volatility'] /= 100.0 # Also convert volatility if entered as %

        with st.spinner("Running final stock evaluation..."):
            try:
                # Prepare data dict, ensure all needed keys exist even if None/0
                calc_input_data = st.session_state.fetched_data.get(main_ticker, {}).get('info', {}).copy() # Start with info
                calc_input_data.update(user_inputs_final['quant_metrics'])
                calc_input_data.update(user_inputs_final['val_inputs'])

                # Add PUSHED operational scores, ensuring they exist
                calc_input_data['user_liquidity_score'] = final_scores.get('liquidity', 5)
                calc_input_data['user_tax_score'] = final_scores.get('tax', 5)
                calc_input_data['user_dividend_score'] = final_scores.get('dividend', 5)
                calc_input_data['user_portfolio_fit_score'] = final_scores.get('portfolio_fit', 5)

                # Ensure industry averages dict is passed, even if empty
                industry_avgs = user_inputs_final.get('industry_avgs', {})

                # Call evaluator methods
                quant_result = evaluator._calculate_quantitative_score(calc_input_data, industry_avgs)

                # Prepare qualitative scores correctly (pass flat dict)
                qual_scores_flat = {q['key']: final_scores.get(q['key'], 5)
                                    for cat_qs in QUALITATIVE_QUESTIONS_GUIDANCE.values()
                                    for q in cat_qs}
                # Add operational scores to the flat dict passed to qual score calc
                for cat_qs in OPERATIONAL_QUESTIONS_GUIDANCE.values():
                     for q in cat_qs:
                          qual_scores_flat[q['key']] = final_scores.get(q['key'], 5)

                qual_result = evaluator._calculate_qualitative_score(qual_scores_flat)

                val_result = evaluator._calculate_valuation_score(calc_input_data)
                ops_result = evaluator._calculate_operational_score(calc_input_data) # Uses user scores from calc_input_data

                # Ensure all inputs for credits are prepared
                credits_input_data = calc_input_data.copy()
                credits_input_data['industry_avgs'] = industry_avgs # Add industry avgs needed for credits

                credits_result = evaluator._calculate_justification_credits(
                     quant_result.get('scores',{}), qual_result.get('categories',{}),
                     val_result.get('scores',{}), ops_result.get('scores',{}),
                     credits_input_data # Pass combined data
                )
                final_result_data = evaluator._calculate_final_score_and_recommendation(
                     quant_result.get('total_score',0), qual_result.get('normalized_score',0),
                     val_result.get('total_score',0), ops_result.get('total_score',0),
                     credits_result.get('total_credits',0)
                )
                report = evaluator._generate_report(quant_result, qual_result, val_result, ops_result, credits_result, final_result_data, calc_input_data)

                _display_final_report(final_result_data, report, main_ticker)
            except AttributeError as ae: st.error(f"❌ Calc Error: Method missing in stock_evaluator? {ae}"); st.code(traceback.format_exc())
            except Exception as e: st.error(f"❌ Error during final calculation: {e}"); st.code(traceback.format_exc())

# --- MODIFIED: ETF-specific UI functions ---
def display_etf_guidance_tab(main_ticker, evaluator):
    st.header("🤔 Qualitative & Operational Guidance (ETF)")
    st.markdown("Analyze factors, load AI guidance, then push scores.")
    ai_configured = bool(evaluator and evaluator.use_ai)
    if not ai_configured: st.warning("Configure AI to load guidance.", icon="🤖")

    load_guidance_key = f"load_guidance_etf_{main_ticker}"
    if st.button("🤖 Load AI Guidance", disabled=not ai_configured, key=load_guidance_key):
        with st.spinner(f"Running AI analysis on {main_ticker}..."):
            # This call now gets BOTH qual and ops guidance
            guidance_results = evaluator.get_ai_qualitative_guidance(ticker_override=main_ticker)
            if isinstance(guidance_results, dict) and 'error' in guidance_results: st.error(f"AI Error: {guidance_results['error']}")
            elif isinstance(guidance_results, dict): st.session_state.ai_guidance_results = guidance_results; st.success("✅ AI loaded!"); st.rerun()
            else: st.error(f"Unexpected AI result type: {type(guidance_results)}")
    st.divider()

    current_guidance_scores = {}

    # --- ETF QUALITATIVE BLOCK ---
    if ETF_QUALITATIVE_QUESTIONS_GUIDANCE:
        for category, questions in ETF_QUALITATIVE_QUESTIONS_GUIDANCE.items():
            st.subheader(f"{category.replace('_', ' ').title()}")
            for q in questions:
                _display_guidance_widget(q, main_ticker, current_guidance_scores)

    st.divider()

    # --- ETF OPERATIONAL QUESTIONS (NOW uses imported data) ---
    if ETF_OPERATIONAL_QUESTIONS_GUIDANCE:
        st.subheader("Operational Factors")
        # Loop through the imported operational questions
        for category, questions in ETF_OPERATIONAL_QUESTIONS_GUIDANCE.items():
            for q in questions:
                _display_guidance_widget(q, main_ticker, current_guidance_scores)

    st.divider()
    push_scores_key = f"push_scores_etf_{main_ticker}"
    if st.button("➡️ Push All Scores to Final Tab", key=push_scores_key, type="primary"):
        st.session_state.pushed_scores = current_guidance_scores.copy()
        st.success(f"Pushed {len(st.session_state.pushed_scores)} scores!")

def display_etf_final_tab(main_ticker, evaluator):
    st.header("🎯 Comprehensive ETF Evaluation Result")
    st.markdown("Enter **Quant/Val** metrics. **Qual/Ops** scores are used from the 'Push Scores' button.")
    if not evaluator: st.warning("Run analysis first."); st.stop()

    user_inputs_final = {'quant_metrics': {}, 'val_inputs': {}}
    sugg_data = st.session_state.evaluator_suggestion_data

    # Helper for suggestions
    def sugg(key, default=0.0): return sugg_data.get(key, default) if sugg_data.get(key) is not None else default

    st.subheader("1. Quantitative Factors ")
    with st.expander("Enter Quantitative Metrics"):
         c1, c2 = st.columns(2)
         with c1:
            user_inputs_final['quant_metrics']['annualized_return'] = st.number_input("10Y Ann. Return %", value=float(sugg('annualized_return_10y')*100), format="%.2f", help=f"Sugg: {sugg('annualized_return_10y'):.2%}")
            user_inputs_final['quant_metrics']['benchmark_return'] = st.number_input("Benchmark Ann. Return %", value=13.0, format="%.2f", help="S&P 500 ~13%")
            user_inputs_final['quant_metrics']['beta'] = st.number_input("Beta (3Y)", value=float(sugg('beta')), format="%.2f", help=f"Sugg: {sugg('beta'):.2f}")
         with c2:
            user_inputs_final['quant_metrics']['sharpe_ratio'] = st.number_input("Sharpe Ratio", value=float(sugg('sharpe_ratio')), format="%.2f", help=f"Sugg: {sugg('sharpe_ratio'):.2f}")
            user_inputs_final['quant_metrics']['expense_ratio'] = st.number_input("Expense Ratio %", value=float(sugg('expense_ratio')*100), format="%.2f", help=f"Sugg: {sugg('expense_ratio'):.2%}")
            user_inputs_final['quant_metrics']['tracking_error'] = st.number_input("Tracking Error %", value=0.1, format="%.2f")

    st.subheader("3. Valuation Factors ")
    with st.expander("Enter Valuation Inputs"):
         c1, c2 = st.columns(2)
         with c1:
            user_inputs_final['val_inputs']['forward_pe'] = st.number_input("Forward P/E", value=float(sugg('forward_pe')), format="%.2f", help=f"Sugg: {sugg('forward_pe'):.2f}")
            user_inputs_final['val_inputs']['peer_avg_pe'] = st.number_input("Peer Avg P/E", value=float(sugg('forward_pe', 20.0)), format="%.2f", help="Enter peer average")
            user_inputs_final['val_inputs']['current_price'] = st.number_input("Current Price", value=float(sugg('current_price')), format="%.2f", help=f"Sugg: {sugg('current_price'):.2f}")
         with c2:
            user_inputs_final['val_inputs']['nav_price'] = st.number_input("NAV Price", value=float(sugg('nav_price')), format="%.2f", help=f"Sugg: {sugg('nav_price'):.2f}")
            user_inputs_final['val_inputs']['dividend_yield'] = st.number_input("Dividend Yield %", value=float(sugg('dividend_yield')*100), format="%.2f", help=f"Sugg: {sugg('dividend_yield'):.2%}")
            user_inputs_final['val_inputs']['eps_growth'] = st.number_input("Holdings EPS Growth %", value=10.0, format="%.2f", help="Est. growth of underlying holdings")

    st.divider()
    calc_final_key = f"calc_final_etf_{main_ticker}"
    if st.button("🧮 Calculate Final ETF Score", key=calc_final_key):
        final_scores = st.session_state.pushed_scores.copy() # Contains pushed Qual AND Ops scores
        if not final_scores: st.warning("⚠️ No Qual/Ops scores pushed. Using default 5.");

        # Convert % inputs to decimals
        user_inputs_final['quant_metrics']['annualized_return'] /= 100.0
        user_inputs_final['quant_metrics']['benchmark_return'] /= 100.0
        user_inputs_final['quant_metrics']['expense_ratio'] /= 100.0
        user_inputs_final['quant_metrics']['tracking_error'] /= 100.0
        user_inputs_final['val_inputs']['dividend_yield'] /= 100.0
        user_inputs_final['val_inputs']['eps_growth'] /= 100.0

        with st.spinner("Running final ETF evaluation..."):
            try:
                # Prepare data dict for calculations
                calc_input_data = st.session_state.fetched_data.get(main_ticker, {}).get('etf_metrics', {}).copy() # Start with base metrics
                calc_input_data.update(user_inputs_final['quant_metrics'])
                calc_input_data.update(user_inputs_final['val_inputs'])
                # Note: Operational scores are passed directly below, not added here

                # Call evaluator methods
                quant_result = evaluator._calculate_quantitative_score(calc_input_data)

                # Qualitative score calculation needs the flat dict of ALL pushed scores
                qual_result = evaluator._calculate_qualitative_score(final_scores)

                val_result = evaluator._calculate_valuation_score(calc_input_data)

                # Operational score calculation just needs the pushed scores
                ops_result = evaluator._calculate_operational_score(final_scores) # Pass flat dict

                final_result_data = evaluator._calculate_final_score_and_recommendation(
                     quant_result.get('total_score',0), qual_result.get('normalized_score',0),
                     val_result.get('total_score',0), ops_result.get('total_score',0)
                )
                report = evaluator._generate_report(quant_result, qual_result, val_result, ops_result, final_result_data)

                _display_final_report(final_result_data, report, main_ticker)
            except AttributeError as ae: st.error(f"❌ Calc Error: Method missing in etf_evaluator? {ae}"); st.code(traceback.format_exc())
            except Exception as e: st.error(f"❌ Error during final calculation: {e}"); st.code(traceback.format_exc())

# --- Shared Helper Functions ---
def _display_guidance_widget(q_dict, main_ticker, scores_dict):
    """Helper to display a single guidance question widget."""
    q_key = q_dict['key']
    st.markdown(f"<p class='guidance-question'>Q: {q_dict['question']}</p>", unsafe_allow_html=True)
    st.caption(f"Guidance: {q_dict['guidance']}")

    ai_data = st.session_state.ai_guidance_results.get(q_key)
    ai_score = 5
    ai_analysis = "No AI analysis loaded."
    if ai_data and isinstance(ai_data, dict):
        # Robust score parsing
        try: ai_score = int(ai_data.get('suggested_score', 5))
        except (ValueError, TypeError): ai_score = 5
        ai_analysis = ai_data.get('analysis', "AI analysis format error.")

    default_value = st.session_state.pushed_scores.get(q_key, ai_score)

    col1, col2 = st.columns([1, 4])
    with col1:
         # Ensure key is unique using q_key
         scores_dict[q_key] = st.number_input("Score", 1, 10, value=default_value, key=f"guidance_{main_ticker}_{q_key}", label_visibility="collapsed")
    with col2:
         # Ensure key is unique using q_key
         show_ai = st.checkbox("Show AI Details", key=f"show_ai_{main_ticker}_{q_key}", value=False)
         if show_ai:
             st.metric("AI Suggested Score", f"{ai_score}/10")
             st.markdown(ai_analysis)
    st.markdown("---")

def _display_final_report(final_result_data, report, main_ticker):
    """Helper to display the final report area."""
    st.success("✅ Evaluation complete!")
    final_score = final_result_data.get('final_score', 'N/A'); recommendation = final_result_data.get('recommendation', 'N/A')
    col1, col2 = st.columns(2); col1.metric("Final Score", f"{final_score:.2f}" if isinstance(final_score, (int, float)) else 'N/A'); col2.metric("Recommendation", recommendation)
    st.text_area("Full Report", report, height=600, key=f"report_{main_ticker}")
    st.download_button(label="💾 Download Report", data=report, file_name=f"{main_ticker}_eval_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", key=f"download_{main_ticker}")

# --- calculate_and_display_valuation (STOCK SPECIFIC - UNCHANGED) ---
def calculate_and_display_valuation(main_ticker, competitor_tickers, fetched_data, valuation_model, fetcher):
    """Calculates and displays DCF and Relative Valuation for a STOCK."""
    output_lines = []
    st.header("💰 Valuation Analysis (Stock)")
    main_data = fetched_data.get(main_ticker)
    if not main_data or not main_data.get('info'):
         st.warning(f"Valuation data missing for {main_ticker}.")
         st.session_state.valuation_results_text = "Valuation data missing."
         return

    dcf_value = None; avg_valuation = None
    history_df = main_data.get('history')
    current_price = history_df['Close'].iloc[-1] if isinstance(history_df, pd.DataFrame) and not history_df.empty else main_data['info'].get('currentPrice')
    if not current_price: st.warning("Could not determine current price."); current_price = 0
    output_lines.append(f"Current Price: {current_price:.2f}" if current_price is not None else "Current Price: N/A")


    st.subheader("1. Discounted Cash Flow (DCF) Valuation")
    try:
        beta = main_data['info'].get('beta')
        market_cap = main_data['info'].get('marketCap')
        shares = main_data['info'].get('sharesOutstanding')
        fcf = main_data.get('fcf') # This should be a Pandas Series
        current_fcf = fcf.iloc[-1] if isinstance(fcf, pd.Series) and not fcf.empty else None
        debt = main_data.get('total_debt') # Pandas Series
        total_debt = debt.iloc[-1] if isinstance(debt, pd.Series) and not debt.empty else None
        cash_eq = main_data.get('cash_equivalents') # Pandas Series
        cash = cash_eq.iloc[-1] if isinstance(cash_eq, pd.Series) and not cash_eq.empty else None

        # Check if all required DCF inputs are valid numbers
        dcf_inputs_valid = all(isinstance(v, (int, float)) and v is not None for v in [beta, market_cap, shares, current_fcf, total_debt, cash])

        if dcf_inputs_valid and shares > 0:
            hist_growth = 0.10 # Default growth if history is short/missing
            if isinstance(fcf, pd.Series) and len(fcf) > 1:
                growth_series = fcf.pct_change().dropna()
                growth_series = growth_series[np.isfinite(growth_series)]
                if not growth_series.empty: hist_growth = np.median(growth_series)
            hist_growth = max(min(hist_growth, 0.30), -0.10)
            growth_rates = [max(min(hist_growth * (1 - 0.1 * i), 0.3), 0.03) for i in range(5)]
            term_growth = 0.025; cost_debt = 0.055
            cost_equity = valuation_model.calculate_cost_of_equity(beta)
            wacc = valuation_model.calculate_wacc(market_cap, total_debt, cost_equity, cost_debt)

            if wacc is None or np.isnan(wacc) or wacc <= term_growth:
                 st.warning(f"⚠️ Invalid WACC ({wacc:.2%}) or WACC <= Terminal Growth ({term_growth:.2%}). Cannot calculate DCF.")
                 output_lines.append("DCF Value: Invalid WACC")
            else:
                col1, col2, col3 = st.columns(3); col1.metric("Cost of Equity", f"{cost_equity:.2%}"); col2.metric("WACC", f"{wacc:.2%}"); col3.metric("Terminal Growth", f"{term_growth:.2%}")
                dcf_value = valuation_model.dcf_valuation(current_fcf, growth_rates, term_growth, wacc, shares, total_debt, cash)
                upside = ((dcf_value - current_price) / current_price) * 100 if current_price and dcf_value is not None and not np.isnan(dcf_value) else 0

                col1, col2, col3 = st.columns(3); col1.metric("DCF Value", f"${dcf_value:.2f}"); col2.metric("Current Price", f"${current_price:.2f}"); col3.metric("Upside/Downside", f"{upside:+.2f}%", delta=f"{upside:+.2f}%" if abs(upside)>0.1 else None)
                output_lines.append(f"DCF Value: {dcf_value:.2f}")
        else:
            st.warning("⚠️ Insufficient or invalid data for DCF (Beta, Mkt Cap, Shares, FCF, Debt, Cash). Check fetched data.")
            output_lines.append("DCF Value: N/A")
    except Exception as e:
        st.error(f"❌ DCF Calculation Error: {e}"); st.code(traceback.format_exc()); output_lines.append(f"DCF Value: Error")

    st.divider()
    st.subheader("2. Relative Valuation (vs Competitors)")
    try:
        pe_list, ps_list, pb_list = [], [], []
        for ticker in competitor_tickers:
             comp_data = fetched_data.get(ticker)
             if comp_data and comp_data.get('info'):
                info = comp_data['info']
                if info.get('trailingPE') and info['trailingPE'] > 0: pe_list.append(info['trailingPE'])
                if info.get('priceToSalesTrailing12Months') and info['priceToSalesTrailing12Months'] > 0: ps_list.append(info['priceToSalesTrailing12Months'])
                if info.get('priceToBook') and info['priceToBook'] > 0: pb_list.append(info['priceToBook'])

        avg_pe = np.median(pe_list) if pe_list else None
        avg_ps = np.median(ps_list) if ps_list else None
        avg_pb = np.median(pb_list) if pb_list else None
        st.session_state.industry_avg_pe = float(avg_pe) if avg_pe is not None else 0.0

        col1, col2, col3 = st.columns(3); col1.metric("Median Peer P/E", f"{avg_pe:.2f}" if avg_pe else "N/A"); col2.metric("Median Peer P/S", f"{avg_ps:.2f}" if avg_ps else "N/A"); col3.metric("Median Peer P/B", f"{avg_pb:.2f}" if avg_pb else "N/A")

        t_info = main_data.get('info', {})
        t_eps = t_info.get('trailingEps'); t_shares = t_info.get('sharesOutstanding')
        t_rev = None; t_te = None
        if 'annual_income' in main_data and isinstance(main_data['annual_income'], pd.DataFrame) and 'Total Revenue' in main_data['annual_income'].index:
            t_rev = main_data['annual_income'].loc['Total Revenue'].iloc[-1] if not main_data['annual_income'].loc['Total Revenue'].empty else None
        if 'total_equity' in main_data and isinstance(main_data['total_equity'], pd.Series):
             t_te = main_data['total_equity'].iloc[-1] if not main_data['total_equity'].empty else None

        t_sps = (t_rev / t_shares) if t_rev and t_shares else None
        t_bps = (t_te / t_shares) if t_te and t_shares else None

        rel_vals_dict = valuation_model.relative_valuation(t_eps, t_sps, t_bps, avg_pe, avg_ps, avg_pb)

        if rel_vals_dict:
            results_data = []; valid_vals = []
            for method, value in rel_vals_dict.items():
                if value is not None and not np.isnan(value) and value > 0:
                     valid_vals.append(value)
                     upside = ((value - current_price) / current_price * 100) if current_price else 0
                else: upside = None; value=None
                results_data.append({'Method': method, 'Value': f"${value:.2f}" if value else "N/A", 'Upside': f"{upside:+.2f}%" if upside else "N/A"})

            if results_data: st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
            if valid_vals:
                 avg_valuation = np.mean(valid_vals)
                 avg_ups = ((avg_valuation - current_price) / current_price * 100) if current_price else 0
                 col1, col2, col3 = st.columns(3); col1.metric("Relative Avg Value", f"${avg_valuation:.2f}"); col2.metric("Current Price", f"${current_price:.2f}"); col3.metric("Avg Upside", f"{avg_ups:+.2f}%", delta=f"{avg_ups:+.2f}%" if abs(avg_ups)>0.1 else None)
                 output_lines.append(f"Relative Avg Value: {avg_valuation:.2f}")
            else:
                 st.warning("⚠️ No valid relative valuations could be calculated.")
                 output_lines.append("Relative Avg Value: N/A")
        else:
            st.warning("⚠️ Insufficient data for relative valuation (Peers or Target).")
            output_lines.append("Relative Avg Value: N/A")
    except Exception as e:
        st.error(f"❌ Relative Valuation Error: {e}"); st.code(traceback.format_exc()); output_lines.append(f"Relative Avg Value: Error")

    st.session_state.valuation_results_text = "\n".join(output_lines)


# --- MAIN APP ---
def main():
    st.markdown(f'<p class="main-header">📊 EMPATHY: Stock & ETF Analysis Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive evaluation with AI suggestions</p>', unsafe_allow_html=True)
    fetcher, processor, valuation_model = init_components()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("📖 [App Guideline](https://www.notion.so/Welcome-to-the-Amazing-Stock-Analysis-Valuation-Platform-2966bdd054e980f5a0f8cbd422b1f0bc?source=copy_link)")

        # --- Analysis Type Selector ---
        analysis_type = st.radio("Select Analysis Type:", ["Stock", "ETF"], key="analysis_type")

        st.subheader("📈 Ticker Selection")
        main_ticker = st.text_input("Main Ticker Symbol", value="NVDA" if analysis_type == "Stock" else "QQQ").upper()

        competitor_tickers = []
        if analysis_type == "Stock":
            competitors_input = st.text_input("Competitor Symbols", value="AMD,INTC")
            competitor_tickers = [t.strip().upper() for t in competitors_input.split(',') if t.strip()]

        st.divider()
        st.subheader("🤖 AI Configuration (Optional)")
        provider = "gemini"; api_key_input = "" # Keep placeholders

        # Check if the required evaluator for the selected type is available
        EVAL_AVAILABLE = (analysis_type == "Stock" and STOCK_EVAL_AVAILABLE) or (analysis_type == "ETF" and ETF_EVAL_AVAILABLE)

        if EVAL_AVAILABLE:
            available_providers = get_available_providers()
            if available_providers:
                provider_display = {'gemini': '🔷 Google Gemini', 'openai': '🟢 OpenAI ChatGPT'}
                provider = st.radio("Select AI Provider:", options=available_providers, format_func=lambda x: provider_display.get(x, x.capitalize()), key="ai_provider_select") # Added key
                # Use session state to store API key temporarily if needed, or get from secrets
                api_key_input = st.text_input("API Key:", type="password", help=f"Enter {provider.capitalize()} API key", key="api_key_input") # Added key

                # Initialize button separate from API key input
                if st.button("🔐 Set API Key & Initialize", key="init_api_button"): # Added key
                    if api_key_input:
                        try:
                            # Init correct evaluator INSTANCE
                            print(f"Initializing {analysis_type} evaluator with API key...")
                            if analysis_type == "Stock":
                                st.session_state.evaluator_instance = StockEvaluator(ticker=main_ticker, api_key=api_key_input, provider=provider)
                            else: # ETF
                                st.session_state.evaluator_instance = ETFEvaluator(ticker=main_ticker, api_key=api_key_input, provider=provider)

                            # Check if AI was successfully configured within the instance
                            if st.session_state.evaluator_instance and st.session_state.evaluator_instance.use_ai:
                                 st.success(f"✅ {provider.capitalize()} Initialized for {analysis_type}!")
                                 print(f"API Init Success: {provider} for {analysis_type}")
                            elif st.session_state.evaluator_instance:
                                 st.warning(f"Evaluator Initialized for {analysis_type}, but AI configuration failed or API key invalid.")
                                 print(f"API Init Warning: Evaluator OK, but AI failed for {provider}/{analysis_type}")
                            else:
                                 st.error(f"❌ Failed to initialize {analysis_type} Evaluator instance.")
                                 print(f"API Init Error: Failed to create {analysis_type} evaluator instance.")

                        except Exception as e:
                             st.error(f"❌ Evaluator Init Failed: {e}"); st.code(traceback.format_exc())
                             print(f"API Init Exception: {e}")
                             st.session_state.evaluator_instance = None
                    else:
                        st.warning("⚠️ Please enter an API key before initializing.")
                        print("API Init Warning: No key entered.")
            else:
                 st.info("No AI provider libraries (google-generativeai or openai) found. AI features disabled.")


            # Display links based on selected provider
            if provider == "gemini": st.info("🔗 Get Gemini Key: [Google AI Studio](https://makersuite.google.com/app/apikey)")
            elif provider == "openai": st.info("🔗 Get OpenAI Key: [OpenAI Platform](https://platform.openai.com/api-keys)")
        else:
            st.warning(f"⚠️ {analysis_type} evaluator files (`{'stock' if analysis_type=='Stock' else 'etf'}_evaluator.py`) missing or failed to import. Cannot use AI.")

        st.divider()
        # --- Run Analysis Button ---
        if st.button("🚀 Run Basic Analysis", type="primary", use_container_width=True, key="run_analysis_button"): # Added key
            # Reset state variables
            st.session_state.analysis_complete = False
            st.session_state.ai_guidance_results = {}
            st.session_state.pushed_scores = {}
            st.session_state.valuation_results_text = ""
            st.session_state.evaluator_suggestion_data = {}
            st.session_state.industry_avg_pe = 0.0
            st.session_state.evaluator_instance = None # Crucially reset evaluator
            print("\n--- Running Basic Analysis ---")
            print(f"Selected Type: {analysis_type}, Ticker: {main_ticker}")

            with st.spinner("Fetching and analyzing data... This may take a moment."):
                progress_bar = st.progress(0, text="Starting...")
                all_tickers = [main_ticker] + competitor_tickers # Competitors list is empty for ETF
                total_tickers = len(all_tickers)
                fetched_data = {}; has_errors = False; main_ticker_type_is_etf = None

                # Fetch data loop
                for idx, ticker in enumerate(all_tickers):
                    progress_text = f"Fetching {ticker} ({idx + 1}/{total_tickers})..."
                    progress_bar.progress((idx + 1) / total_tickers, text=progress_text)
                    print(progress_text)
                    data = fetch_and_process_stock(ticker, fetcher, processor)
                    if data and data.get('info'):
                         fetched_data[ticker] = data
                         print(f"Success fetching {ticker}")
                         if ticker == main_ticker: # Store the detected type of the main ticker
                              main_ticker_type_is_etf = data.get('is_etf', False)
                    else:
                         error_msg = data.get('error', 'Unknown error')
                         st.warning(f"Failed fetch {ticker}. Error: {error_msg}");
                         print(f"Failed fetch {ticker}. Error: {error_msg}")
                         if ticker == main_ticker: has_errors = True # Critical if main ticker fails

                progress_bar.progress(1.0, text="Processing complete.")
                time.sleep(1)
                progress_bar.empty()

                if has_errors or main_ticker not in fetched_data:
                     st.error(f"❌ Critical error fetching data for {main_ticker}. Cannot proceed.");
                     print(f"Critical error fetching data for {main_ticker}.")
                     st.stop()

                # Store fetched data
                st.session_state.fetched_data = fetched_data
                st.session_state.analysis_complete = True
                st.success("✅ Basic data fetched successfully!")
                print("Basic data fetched successfully.")

                # --- Initialize Evaluator Instance and Prepare Suggestion Data ---
                # Use the DETECTED type of the main ticker, overriding sidebar selection if needed
                current_analysis_type = "ETF" if main_ticker_type_is_etf else "Stock"
                # Determine type based on fetched data
                detected_type = "ETF" if main_ticker_type_is_etf else "Stock"

                # Ensure analysis_type exists in state and update ONLY if detected type differs from selection
                if 'analysis_type' not in st.session_state:
                    st.session_state.analysis_type = detected_type # Initialize if somehow missing
                    print(f"Initialized missing analysis_type to detected: {detected_type}")
                elif st.session_state.analysis_type != detected_type:
                    st.session_state.analysis_type = detected_type # Update state if type detected differs
                    print(f"Corrected analysis_type based on fetched data to: {detected_type}")
                    st.warning(f"Note: Ticker {main_ticker} was detected as {detected_type}. Switched analysis type.") # Inform user
                else:
                    # The selected type matches the detected type, no change needed.
                    print(f"Analysis type selection ({st.session_state.analysis_type}) matches detected type.")
                    pass 

                # Use the potentially updated state value going forward in this run
                current_analysis_type = st.session_state.analysis_type 
                print(f"Proceeding with analysis type: {current_analysis_type}")
                # --- Initialize Evaluator Instance and Prepare Suggestion Data --- (This comment follows)
                print(f"Detected actual type for {main_ticker}: {current_analysis_type}")

                evaluator_available_for_type = (current_analysis_type == "Stock" and STOCK_EVAL_AVAILABLE) or \
                                               (current_analysis_type == "ETF" and ETF_EVAL_AVAILABLE)

                if evaluator_available_for_type:
                    try:
                        print(f"Initializing {current_analysis_type} evaluator...")
                        # Use API key if it was entered AND initialized successfully before
                        current_api_key = api_key_input if 'evaluator_instance' in st.session_state and st.session_state.evaluator_instance and st.session_state.evaluator_instance.use_ai else None

                        if current_analysis_type == "Stock":
                            st.session_state.evaluator_instance = StockEvaluator(ticker=main_ticker, api_key=current_api_key, provider=provider)
                        else: # ETF
                            st.session_state.evaluator_instance = ETFEvaluator(ticker=main_ticker, api_key=current_api_key, provider=provider)

                        # Check instance creation
                        if st.session_state.evaluator_instance:
                            st.info(f"{current_analysis_type} Evaluator initialized.")
                            print(f"{current_analysis_type} Evaluator initialized successfully.")
                        else:
                             raise RuntimeError("Evaluator instance is None after init attempt.")

                    except Exception as e:
                        st.error(f"Failed evaluator initialization during analysis run: {e}")
                        print(f"Failed evaluator initialization during analysis run: {e}\n{traceback.format_exc()}")
                        st.session_state.evaluator_instance = None # Ensure it's None on failure
                else:
                    st.warning(f"Evaluator module for {current_analysis_type} not available. Final analysis disabled.")
                    print(f"Evaluator module for {current_analysis_type} not available.")
                    st.session_state.evaluator_instance = None # Ensure it's None


                # --- Prepare suggestion data based on ACTUAL type ---
                main_data = st.session_state.fetched_data[main_ticker]
                info_for_sugg = main_data.get('info')

                if info_for_sugg:
                    st.write("Preparing suggestion data...")
                    sugg_data_clean = {}
                    if current_analysis_type == "Stock":
                        sugg_data_raw = {
                            'pe': info_for_sugg.get('trailingPE'), 'roe': info_for_sugg.get('returnOnEquity'), 'roa': info_for_sugg.get('returnOnAssets'),
                            'current_ratio': info_for_sugg.get('currentRatio'), 'total_debt_equity': info_for_sugg.get('debtToEquity'),
                            'revenue_growth_yoy': info_for_sugg.get('revenueGrowth'), 'eps_growth_yoy': info_for_sugg.get('earningsGrowth'),
                            'beta': info_for_sugg.get('beta')
                        }
                        for k, v in sugg_data_raw.items():
                            try: sugg_data_clean[k] = float(v) if isinstance(v, (int, float)) and not np.isnan(v) else 0.0
                            except (ValueError, TypeError): sugg_data_clean[k] = 0.0
                        print(f"Prepared Stock suggestion data: {sugg_data_clean}")
                    else: # ETF
                        sugg_data_clean = main_data.get('etf_metrics', {})
                        expected_etf_keys = ['beta', 'sharpe_ratio', 'expense_ratio', 'nav_price', 'current_price', 'forward_pe', 'dividend_yield', 'turnover', 'total_assets', 'annualized_return_10y']
                        for k in expected_etf_keys:
                             val = sugg_data_clean.get(k)
                             try: sugg_data_clean[k] = float(val) if isinstance(val, (int, float)) and not np.isnan(val) else 0.0
                             except (ValueError, TypeError): sugg_data_clean[k] = 0.0
                        print(f"Prepared ETF suggestion data: {sugg_data_clean}")

                    st.session_state.evaluator_suggestion_data = sugg_data_clean
                    st.success("✅ Suggestion data prepared.")
                else:
                    st.warning("Could not get 'info' data for suggestions.")
                    print("Could not get 'info' data for suggestions.")

                # Force rerun to update tabs correctly after analysis
                st.rerun()


    # --- Main Content Area ---
    # Display tabs only if analysis is complete and data exists for the main ticker
    if st.session_state.analysis_complete and main_ticker in st.session_state.fetched_data:
        main_data = st.session_state.fetched_data[main_ticker]
        evaluator = st.session_state.evaluator_instance # This instance corresponds to the *actual* type
        # Determine type based on fetched data for rendering logic
        is_etf = main_data.get('is_etf', False)
        print(f"Rendering Main Content Area. Detected type: {'ETF' if is_etf else 'Stock'}")

        # --- Define Tabs based on detected type ---
        if is_etf:
            if not ETF_EVAL_AVAILABLE:
                st.error("ETF Evaluator module (`etf_evaluator.py`) is missing or has errors. Cannot display ETF tabs.")
            elif not isinstance(evaluator, ETFEvaluator):
                 st.error("Mismatch: Data indicates ETF, but ETF Evaluator is not loaded. Try re-running analysis.")
            else: # OK to display ETF tabs
                tab_list = ["📊 Overview", "📈 Charts", "🤔 Qualitative Guidance", "🎯 Comprehensive Result"]
                tabs = st.tabs(tab_list)
                with tabs[0]: display_company_overview(main_data.get('info'), main_ticker, is_etf=True)
                with tabs[1]:
                    st.header("📈 Price Analysis"); st.plotly_chart(create_price_chart(st.session_state.fetched_data), use_container_width=True)
                    tech_data = main_data.get('processed_history');
                    if tech_data is not None and not tech_data.empty: st.plotly_chart(create_technical_chart(tech_data, main_ticker), use_container_width=True)
                with tabs[2]:
                    display_etf_guidance_tab(main_ticker, evaluator)
                with tabs[3]:
                    display_etf_final_tab(main_ticker, evaluator)

        else: # Stock
             if not STOCK_EVAL_AVAILABLE:
                  st.error("Stock Evaluator module (`stock_evaluator.py`) is missing or has errors. Cannot display Stock tabs.")
             elif not isinstance(evaluator, StockEvaluator):
                  st.error("Mismatch: Data indicates Stock, but Stock Evaluator is not loaded. Try re-running analysis.")
             else: # OK to display Stock tabs
                 tab_list = ["📊 Overview", "📈 Charts", "📋 Financials", "💰 Valuation", "🤔 Qualitative Guidance", "🎯 Comprehensive Result"]
                 tabs = st.tabs(tab_list)
                 with tabs[0]: display_company_overview(main_data.get('info'), main_ticker, is_etf=False)
                 with tabs[1]:
                    st.header("📈 Technical Analysis"); st.plotly_chart(create_price_chart(st.session_state.fetched_data), use_container_width=True)
                    tech_data = main_data.get('processed_history');
                    if tech_data is not None and not tech_data.empty: st.plotly_chart(create_technical_chart(tech_data, main_ticker), use_container_width=True)
                 with tabs[2]:
                    st.header("🏢 Financial Statements")
                    financial_type = st.selectbox("Select Statement:", ["Income Statement", "Balance Sheet", "Cash Flow"], key=f"fin_select_{main_ticker}")
                    fin_map = {"Income Statement": "income_stmt", "Balance Sheet": "balance_sheet", "Cash Flow": "cash_flow"}
                    fin_dict = main_data.get('financials', {})
                    df = fin_dict.get(fin_map.get(financial_type)) if isinstance(fin_dict, dict) else None
                    if isinstance(df, pd.DataFrame) and not df.empty: st.dataframe(df, use_container_width=True)
                    else: st.warning("⚠️ Financial data unavailable.")
                 with tabs[3]:
                    calculate_and_display_valuation(main_ticker, competitor_tickers, st.session_state.fetched_data, valuation_model, fetcher)
                 with tabs[4]:
                    display_stock_guidance_tab(main_ticker, evaluator)
                 with tabs[5]:
                    display_stock_final_tab(main_ticker, evaluator)

    else:
        # Initial state before analysis is run
        st.info("👋 Select analysis type, configure in sidebar & click 'Run Basic Analysis'.")

    st.divider()
    st.markdown("<div style='text-align: center; color: #666;'>Disclaimer: Educational. Not financial advice.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
