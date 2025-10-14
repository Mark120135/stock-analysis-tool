"""
Stock Analysis Web Application
A Streamlit web app that provides comprehensive stock analysis with AI-powered insights

To run this app:
    streamlit run app.py

To deploy online:
    1. Push your code to GitHub
    2. Go to streamlit.io/cloud
    3. Connect your GitHub repository
    4. Deploy!
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Import your existing modules
from data_fetcher import YahooFinanceDataFetcher
from data_processor import StockDataProcessor
from valuation_model import StockValuationModel

# Try to import AI analyzers
try:
    from gemini_analyzer import AIQualitativeAnalyzer, get_available_providers, is_provider_available
    INDUSTRY_ANALYZER_AVAILABLE = True
except ImportError:
    INDUSTRY_ANALYZER_AVAILABLE = False

try:
    from governance_analyzer import GovernanceQualitativeAnalyzer
    GOVERNANCE_ANALYZER_AVAILABLE = True
except ImportError:
    GOVERNANCE_ANALYZER_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Stock Analysis & Valuation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'industry_analyzer' not in st.session_state:
    st.session_state.industry_analyzer = None
if 'governance_analyzer' not in st.session_state:
    st.session_state.governance_analyzer = None


# Initialize components
@st.cache_resource
def init_components():
    """Initialize data fetcher, processor, and valuation model"""
    fetcher = YahooFinanceDataFetcher()
    processor = StockDataProcessor()
    valuation_model = StockValuationModel(risk_free_rate=0.04, market_return=0.09)
    return fetcher, processor, valuation_model


def create_price_chart(data_dict, title="Stock Price Comparison"):
    """Create interactive price comparison chart using Plotly"""
    fig = go.Figure()
    
    for ticker, data in data_dict.items():
        if 'processed_history' in data and data['processed_history'] is not None:
            df = data['processed_history']
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_technical_chart(df, ticker):
    """Create technical analysis chart with price and indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} Price & Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red')), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, template='plotly_white', hovermode='x unified')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


def fetch_and_process_stock(ticker, fetcher, processor):
    """Fetch and process data for a single stock"""
    try:
        data = {
            'info': fetcher.get_company_info(ticker),
            'history': fetcher.get_stock_history(ticker, period="5y"),
            'financials': fetcher.get_financials(ticker)
        }
        
        if data['info']:
            data['processed_history'] = processor.calculate_technical_indicators(data['history'])
            data['annual_cash_flow'] = processor.get_yearly_financial_data(data['financials'], 'cash_flow')
            data['annual_balance'] = processor.get_yearly_financial_data(data['financials'], 'balance_sheet')
            data['annual_income'] = processor.get_yearly_financial_data(data['financials'], 'income_stmt')
            data['fcf'] = processor.calculate_free_cash_flow(data['annual_cash_flow'])
            data['total_debt'] = processor.get_total_debt(data['annual_balance'])
            data['cash_equivalents'] = processor.get_cash_and_equivalents(data['annual_balance'])
            data['total_equity'] = processor.get_total_stockholder_equity(data['annual_balance'])
            data['eps_history'] = processor.get_eps_from_financials(data['annual_income'])
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def display_company_overview(info, ticker):
    """Display company overview metrics"""
    st.subheader(f"📋 {info.get('longName', ticker)} Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
        st.metric("Market Cap", f"${info.get('marketCap', 0) / 1e9:.2f}B" if info.get('marketCap') else "N/A")
    
    with col2:
        st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A")
        st.metric("EPS", f"${info.get('trailingEps', 'N/A'):.2f}" if info.get('trailingEps') else "N/A")
    
    with col3:
        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
    
    with col4:
        st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A")
        st.metric("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")
    
    st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    
    if info.get('longBusinessSummary'):
        with st.expander("📄 Business Description"):
            st.write(info['longBusinessSummary'])


def calculate_and_display_valuation(main_ticker, competitor_tickers, fetched_data, valuation_model, fetcher):
    """Calculate and display valuation analysis"""
    st.header("💰 Valuation Analysis")
    
    main_data = fetched_data[main_ticker]
    
    # DCF Valuation
    st.subheader("1. Discounted Cash Flow (DCF) Valuation")
    
    try:
        beta = main_data['info'].get('beta')
        market_cap = main_data['info'].get('marketCap')
        shares_outstanding = main_data['info'].get('sharesOutstanding')
        current_fcf = main_data['fcf'].iloc[-1] if not main_data['fcf'].empty else None
        total_debt = main_data['total_debt'].iloc[-1] if not main_data['total_debt'].empty else None
        cash_equivalents = main_data['cash_equivalents'].iloc[-1] if not main_data['cash_equivalents'].empty else None
        
        required_data_map = {
            'Beta Value': beta,
            'Market Cap': market_cap,
            'Shares Outstanding': shares_outstanding,
            'Latest Free Cash Flow': current_fcf,
            'Total Debt': total_debt,
            'Cash & Equivalents': cash_equivalents
        }
        
        missing_items = [name for name, value in required_data_map.items() if value is None]
        
        if not missing_items:
            # Calculate growth rates
            if len(main_data['fcf']) > 1:
                hist_growth = main_data['fcf'].pct_change().mean()
                growth_rates_high = [max(min(hist_growth * (1 - 0.1 * i), 0.3), 0.05) for i in range(5)]
            else:
                growth_rates_high = [0.15, 0.12, 0.10, 0.08, 0.05]
            
            terminal_growth_rate = 0.025
            cost_of_debt = 0.055
            cost_of_equity = valuation_model.calculate_cost_of_equity(beta)
            wacc = valuation_model.calculate_wacc(market_cap, total_debt, cost_of_equity, cost_of_debt)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cost of Equity", f"{cost_of_equity:.2%}")
            with col2:
                st.metric("WACC", f"{wacc:.2%}")
            with col3:
                st.metric("Terminal Growth", f"{terminal_growth_rate:.2%}")
            
            st.write(f"**FCF Growth Assumptions (5 years):** {', '.join([f'{g:.2%}' for g in growth_rates_high])}")
            
            dcf_value = valuation_model.dcf_valuation(
                current_fcf, growth_rates_high, terminal_growth_rate,
                wacc, shares_outstanding, total_debt, cash_equivalents
            )
            
            current_price = main_data['history']['Close'].iloc[-1]
            upside = ((dcf_value - current_price) / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DCF Intrinsic Value", f"${dcf_value:.2f}")
            with col2:
                st.metric("Current Price", f"${current_price:.2f}")
            with col3:
                st.metric("Upside/Downside", f"{upside:+.2f}%", delta=f"{upside:+.2f}%")
            
        else:
            st.warning(f"⚠️ Insufficient data for DCF valuation. Missing: {', '.join(missing_items)}")
    
    except Exception as e:
        st.error(f"❌ DCF calculation error: {str(e)}")
    
    # Relative Valuation
    st.subheader("2. Relative Valuation (vs Competitors)")
    
    try:
        pe_list, ps_list, pb_list = [], [], []
        
        for ticker in competitor_tickers:
            if ticker in fetched_data:
                info = fetched_data[ticker]['info']
                if info.get('trailingPE'):
                    pe_list.append(info['trailingPE'])
                if info.get('priceToSalesTrailing12Months'):
                    ps_list.append(info['priceToSalesTrailing12Months'])
                if info.get('priceToBook'):
                    pb_list.append(info['priceToBook'])
        
        avg_pe = np.mean(pe_list) if pe_list else None
        avg_ps = np.mean(ps_list) if ps_list else None
        avg_pb = np.mean(pb_list) if pb_list else None
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Competitor P/E", f"{avg_pe:.2f}" if avg_pe else "N/A")
        with col2:
            st.metric("Avg Competitor P/S", f"{avg_ps:.2f}" if avg_ps else "N/A")
        with col3:
            st.metric("Avg Competitor P/B", f"{avg_pb:.2f}" if avg_pb else "N/A")
        
        target_eps = main_data['info'].get('trailingEps')
        total_revenue = main_data['annual_income'].loc['Total Revenue'].iloc[-1] if 'Total Revenue' in main_data['annual_income'].index else None
        total_equity = main_data['total_equity'].iloc[-1] if not main_data['total_equity'].empty else None
        shares_outstanding = main_data['info'].get('sharesOutstanding')
        
        target_sps = total_revenue / shares_outstanding if total_revenue and shares_outstanding else None
        target_bps = total_equity / shares_outstanding if total_equity and shares_outstanding else None
        
        relative_values = valuation_model.relative_valuation(target_eps, target_sps, target_bps, avg_pe, avg_ps, avg_pb)
        
        if relative_values:
            st.write("**Relative Valuation Estimates:**")
            for method, value in relative_values.items():
                st.write(f"• {method}: **${value:.2f}**")
        else:
            st.warning("⚠️ Insufficient data for relative valuation")
    
    except Exception as e:
        st.error(f"❌ Relative valuation error: {str(e)}")


def main():
    """Main application function"""
    
    # Header
    st.markdown('<p class="main-header">📊 Stock Analysis & Valuation Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive Financial Analysis with AI-Powered Insights</p>', unsafe_allow_html=True)
    
    # Initialize components
    fetcher, processor, valuation_model = init_components()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock inputs
        st.subheader("📈 Stock Selection")
        main_ticker = st.text_input("Main Stock Symbol", value="NVDA", help="Enter the primary stock ticker to analyze").upper()
        competitors_input = st.text_input("Competitor Symbols", value="AMD,INTC", help="Comma-separated list of competitor tickers")
        competitor_tickers = [t.strip().upper() for t in competitors_input.split(',') if t.strip()]
        
        st.divider()
        
        # AI Configuration
        st.subheader("🤖 AI Analysis (Optional)")
        
        # Show available providers
        available_providers = []
        if INDUSTRY_ANALYZER_AVAILABLE:
            from gemini_analyzer import get_available_providers
            available_providers = get_available_providers()
        
        if available_providers:
            provider = st.radio(
                "Select AI Provider:",
                options=available_providers,
                format_func=lambda x: "🔷 Google Gemini 2.0 (Free tier)" if x == "gemini" else "🟢 OpenAI ChatGPT (Paid)",
                help="Choose your preferred AI provider for qualitative analysis"
            )
            
            # API Key input
            api_key = st.text_input(
                "API Key:",
                type="password",
                help=f"Enter your {provider.upper()} API key"
            )
            
            if st.button("🔑 Set API Key"):
                if api_key:
                    try:
                        if INDUSTRY_ANALYZER_AVAILABLE:
                            st.session_state.industry_analyzer = AIQualitativeAnalyzer(api_key, provider=provider)
                        if GOVERNANCE_ANALYZER_AVAILABLE:
                            st.session_state.governance_analyzer = GovernanceQualitativeAnalyzer(api_key, provider=provider)
                        st.success(f"✅ {provider.upper()} API configured successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to configure API: {str(e)}")
                else:
                    st.warning("⚠️ Please enter an API key")
            
            # API key links
            if provider == "gemini":
                st.info("🔗 Get free API key: [Google AI Studio](https://makersuite.google.com/app/apikey)")
            else:
                st.info("🔗 Get API key: [OpenAI Platform](https://platform.openai.com/api-keys)")
        else:
            st.warning("⚠️ No AI packages installed. Install google-generativeai or openai to enable AI analysis.")
        
        st.divider()
        
        # Analyze button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_complete = False
            with st.spinner("Fetching and analyzing data..."):
                # Fetch data for all tickers
                all_tickers = [main_ticker] + competitor_tickers
                fetched_data = {}
                
                progress_bar = st.progress(0)
                for idx, ticker in enumerate(all_tickers):
                    st.write(f"Fetching {ticker}...")
                    data = fetch_and_process_stock(ticker, fetcher, processor)
                    if data and data['info']:
                        fetched_data[ticker] = data
                    progress_bar.progress((idx + 1) / len(all_tickers))
                
                if main_ticker in fetched_data:
                    st.session_state.fetched_data = fetched_data
                    st.session_state.analysis_complete = True
                    st.success("✅ Data fetched successfully!")
                else:
                    st.error(f"❌ Failed to fetch data for {main_ticker}")
    
    # Main content area
    if st.session_state.analysis_complete and st.session_state.fetched_data:
        main_data = st.session_state.fetched_data[main_ticker]
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "📈 Charts",
            "📋 Financials",
            "💰 Valuation",
            "📊 Quantitative Analysis",
            "🎯 Qualitative Analysis"
        ])
        
        # Tab 1: Overview
        with tab1:
            display_company_overview(main_data['info'], main_ticker)
        
        # Tab 2: Charts
        with tab2:
            st.header("📈 Technical Analysis")
            
            # Price comparison chart
            st.subheader("Price Comparison (5 Years)")
            price_chart = create_price_chart(st.session_state.fetched_data)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Technical indicators chart
            st.subheader(f"Technical Indicators - {main_ticker}")
            if 'processed_history' in main_data:
                tech_chart = create_technical_chart(main_data['processed_history'], main_ticker)
                st.plotly_chart(tech_chart, use_container_width=True)
        
        # Tab 3: Financials
        with tab3:
            st.header("📋 Financial Statements")
            
            financial_type = st.selectbox(
                "Select Financial Statement:",
                ["Income Statement", "Balance Sheet", "Cash Flow"]
            )
            
            key_map = {
                "Income Statement": "income_stmt",
                "Balance Sheet": "balance_sheet",
                "Cash Flow": "cash_flow"
            }
            
            df = main_data['financials'].get(key_map[financial_type])
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("⚠️ Data not available")
        
        # Tab 4: Valuation
        with tab4:
            calculate_and_display_valuation(
                main_ticker,
                competitor_tickers,
                st.session_state.fetched_data,
                valuation_model,
                fetcher
            )
        
        # Tab 5: Quantitative Analysis
        with tab5:
            st.header("📊 Quantitative Analysis (Industry Comparison)")
            
            if not INDUSTRY_ANALYZER_AVAILABLE:
                st.warning("⚠️ Industry analyzer not installed. Install google-generativeai or openai to enable this feature.")
            elif not st.session_state.industry_analyzer:
                st.info("ℹ️ Configure AI API in the sidebar to enable quantitative analysis")
            else:
                if st.button("🔄 Fetch Quantitative Analysis"):
                    with st.spinner("Analyzing... This may take 30-90 seconds"):
                        try:
                            quantitative_data = st.session_state.industry_analyzer.fetch_gurufocus_data(main_ticker)
                            report = st.session_state.industry_analyzer.format_qualitative_report(quantitative_data)
                            st.text(report)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        # Tab 6: Qualitative Analysis
        with tab6:
            st.header("🎯 Qualitative Analysis (Governance & Strategy)")
            
            if not GOVERNANCE_ANALYZER_AVAILABLE:
                st.warning("⚠️ Governance analyzer not installed. Install google-generativeai or openai to enable this feature.")
            elif not st.session_state.governance_analyzer:
                st.info("ℹ️ Configure AI API in the sidebar to enable qualitative analysis")
            else:
                if st.button("🔄 Fetch Qualitative Analysis"):
                    with st.spinner("Analyzing... This may take 20-40 seconds"):
                        try:
                            governance_data = st.session_state.governance_analyzer.fetch_governance_analysis(main_ticker)
                            report = st.session_state.governance_analyzer.format_governance_report(governance_data)
                            st.text(report)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    else:
        # Welcome screen
        st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis' to get started!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📊 Technical Analysis
            - Price charts & comparisons
            - Moving averages
            - RSI & MACD indicators
            - 5-year historical data
            """)
        
        with col2:
            st.markdown("""
            ### 💰 Valuation Models
            - DCF valuation
            - Relative valuation
            - Peer comparison
            - Financial statements
            """)
        
        with col3:
            st.markdown("""
            ### 🤖 AI-Powered Insights
            - Industry comparison
            - Governance analysis
            - Strategic assessment
            - Risk evaluation
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 Stock Analysis & Valuation Platform | Built with Streamlit</p>
        <p style='font-size: 0.8rem;'>Disclaimer: This tool is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()