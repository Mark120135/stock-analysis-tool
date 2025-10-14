# Try to import both APIs
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Optional, List, Tuple
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class AIQualitativeAnalyzer:
    """Supports both Gemini and OpenAI ChatGPT APIs with multi-source data fetching and industry comparison"""
    
    def __init__(self, api_key: str, provider: str = "gemini"):
        """
        Initialize AI API for qualitative analysis
        :param api_key: API key (Gemini or OpenAI)
        :param provider: "gemini" or "openai"
        """
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider == "gemini":
            if not GENAI_AVAILABLE:
                raise ImportError("google-generativeai package is not installed. Install with: pip install google-generativeai")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is not installed. Install with: pip install openai")
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
            
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'")
    
    def get_industry_peers(self, ticker: str, max_peers: int = 20) -> List[str]:
        """
        Get industry peer tickers by fetching companies in the same industry
        Uses Yahoo Finance screener approach
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            market_cap = info.get('marketCap', 0)
            
            print(f"  → Sector: {sector}, Industry: {industry}")
            print(f"  → Finding industry peers (this may take a moment)...")
            
            # Get a broader list of tickers to search through
            # Focus on major exchanges
            peers = []
            
            # Try to get similar companies by searching industry keywords
            # This is a simplified approach - you could enhance with screener APIs
            try:
                # Use Wikipedia's S&P 500 list as a starting point for US stocks
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                tables = pd.read_html(url)
                sp500_df = tables[0]
                
                # Filter by sector if available
                if sector:
                    sector_companies = sp500_df[sp500_df['GICS Sector'] == sector]['Symbol'].tolist()
                    peers.extend(sector_companies[:max_peers])
            except:
                print("  → Note: Could not fetch S&P 500 list, using limited peer set")
            
            # Remove the original ticker from peers
            peers = [p for p in peers if p != ticker]
            
            print(f"  → Found {len(peers)} potential peers from same sector")
            return peers[:max_peers]
            
        except Exception as e:
            print(f"  → Warning: Could not fetch peers: {e}")
            return []
    
    def get_industry_benchmarks(self, ticker: str, peers: List[str]) -> Dict:
        """
        Fetch key metrics from peer companies and calculate industry medians
        Returns dictionary with median values for comparison
        """
        print(f"\n  📊 Calculating industry benchmarks from {len(peers)} peers...")
        
        metrics_data = {
            'grossMargins': [],
            'operatingMargins': [],
            'profitMargins': [],
            'returnOnEquity': [],
            'returnOnAssets': [],
            'revenueGrowth': [],
            'earningsGrowth': [],
            'currentRatio': [],
            'quickRatio': [],
            'debtToEquity': [],
            'trailingPE': [],
            'priceToBook': [],
            'priceToSalesTrailing12Months': [],
            'enterpriseToEbitda': [],
            'freeCashflow': [],
            'operatingCashflow': []
        }
        
        successful_peers = 0
        for i, peer in enumerate(peers):
            try:
                if i % 5 == 0:
                    print(f"  → Processing peer {i+1}/{len(peers)}...")
                
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                
                # Collect metrics that exist
                for metric in metrics_data.keys():
                    value = peer_info.get(metric)
                    if value is not None and isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        metrics_data[metric].append(value)
                
                successful_peers += 1
                
            except Exception as e:
                # Silently skip failed peers
                continue
        
        print(f"  ✓ Successfully processed {successful_peers} peers")
        
        # Calculate medians
        benchmarks = {}
        for metric, values in metrics_data.items():
            if len(values) >= 3:  # Need at least 3 data points
                benchmarks[metric] = np.median(values)
            else:
                benchmarks[metric] = None
        
        return benchmarks
    
    def compare_to_industry(self, value, benchmark, metric_name: str, higher_is_better: bool = True) -> Tuple[str, str]:
        """
        Compare a metric to industry benchmark
        Returns (comparison_symbol, comparison_text)
        """
        if value is None or value == 'N/A' or benchmark is None:
            return '', ''
        
        try:
            # Convert percentage strings to float if needed
            if isinstance(value, str):
                value = float(value.rstrip('%'))
            
            # Handle different metric types
            if metric_name in ['grossMargins', 'operatingMargins', 'profitMargins', 'returnOnEquity', 'returnOnAssets', 'revenueGrowth', 'earningsGrowth']:
                # These are typically decimals that need to be converted to percentages
                if value < 1 and benchmark < 1:
                    value = value * 100
                    benchmark = benchmark * 100
            
            # Calculate difference
            diff_pct = ((value - benchmark) / abs(benchmark)) * 100 if benchmark != 0 else 0
            
            # Determine if better or worse
            if higher_is_better:
                if value > benchmark:
                    symbol = "✅"
                    text = f"Above industry median ({benchmark:.2f})"
                else:
                    symbol = "⚠️"
                    text = f"Below industry median ({benchmark:.2f})"
            else:  # Lower is better (e.g., debt ratios, P/E)
                if value < benchmark:
                    symbol = "✅"
                    text = f"Below industry median ({benchmark:.2f}) - Better"
                else:
                    symbol = "⚠️"
                    text = f"Above industry median ({benchmark:.2f}) - Higher"
            
            return symbol, text
            
        except Exception as e:
            return '', ''
    
    def fetch_yfinance_data(self, ticker: str, days_ago: int = 0, include_industry_comparison: bool = True) -> str:
        """
        Fetch comprehensive data from Yahoo Finance with industry comparison
        :param ticker: Stock ticker symbol
        :param days_ago: Number of days ago to fetch historical price (0 = current)
        :param include_industry_comparison: Whether to fetch and compare to industry peers
        :return: Formatted financial data with industry benchmarks
        """
        try:
            print(f"Fetching comprehensive data from Yahoo Finance for {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            
            # Get industry info
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            # Get industry benchmarks if requested
            industry_benchmarks = {}
            if include_industry_comparison:
                peers = self.get_industry_peers(ticker)
                if peers:
                    industry_benchmarks = self.get_industry_benchmarks(ticker, peers)
                    print(f"  ✓ Industry benchmarks calculated from {len(peers)} peers\n")
            
            # Get historical price data
            if days_ago > 0:
                target_date = datetime.now() - timedelta(days=days_ago)
                start_date = target_date - timedelta(days=5)
                end_date = target_date + timedelta(days=1)
                hist_prices = stock.history(start=start_date, end=end_date)
                
                if not hist_prices.empty:
                    current_price = hist_prices['Close'].iloc[-1]
                    price_date = hist_prices.index[-1].strftime('%Y-%m-%d')
                    price_note = f"Historical price from {days_ago} days ago ({price_date})"
                else:
                    current_price = info.get('currentPrice', 'N/A')
                    price_note = "Current price (historical data not available)"
            else:
                current_price = info.get('currentPrice', 'N/A')
                price_note = "Current market price"
            
            # Calculate additional metrics
            market_cap = info.get('marketCap', 0)
            total_debt = info.get('totalDebt', 0)
            total_cash = info.get('totalCash', 0)
            net_debt = total_debt - total_cash if total_debt and total_cash else None
            
            # Helper function for formatting with industry comparison
            def format_metric(value, metric_key, higher_is_better=True):
                """Format value with industry comparison"""
                if value is None or value == 'N/A':
                    return 'N/A'
                
                # Format the base value
                if isinstance(value, str):
                    formatted = value
                elif metric_key in ['grossMargins', 'operatingMargins', 'profitMargins', 'returnOnEquity', 'returnOnAssets', 'revenueGrowth', 'earningsGrowth']:
                    # These are decimals that need percentage conversion
                    if value < 1:
                        formatted = f"{value * 100:.2f}%"
                    else:
                        formatted = f"{value:.2f}%"
                elif metric_key in ['currentRatio', 'quickRatio', 'debtToEquity', 'trailingPE', 'priceToBook', 'priceToSalesTrailing12Months', 'enterpriseToEbitda']:
                    formatted = f"{value:.2f}"
                elif isinstance(value, (int, float)) and value > 1000:
                    formatted = f"${value:,.0f}"
                else:
                    formatted = f"{value:.2f}" if isinstance(value, float) else str(value)
                
                # Add industry comparison if available
                if metric_key in industry_benchmarks and industry_benchmarks[metric_key] is not None:
                    symbol, text = self.compare_to_industry(value, industry_benchmarks[metric_key], metric_key, higher_is_better)
                    if symbol:
                        return f"{formatted} | {symbol} {text}"
                
                return formatted
            
            # Helper for safe get
            def safe_get(key, default='N/A'):
                return info.get(key, default)
            
            # Format the data summary
            data_summary = f"""
=== YAHOO FINANCE DATA FOR {ticker} ===
Data Retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}
Price Data: {price_note}
Industry Benchmarks: {'✓ Calculated from peer companies' if industry_benchmarks else 'Not available'}

**COMPANY OVERVIEW:**
- Company Name: {info.get('longName', 'N/A')}
- Sector: {sector}
- Industry: {industry}
- Current Price: ${current_price}
- Market Cap: ${market_cap:,.0f}
- Shares Outstanding: {info.get('sharesOutstanding', 0):,.0f}

**PROFITABILITY METRICS (vs Industry Median):**
- Gross Margin: {format_metric(safe_get('grossMargins'), 'grossMargins', True)}
- Operating Margin: {format_metric(safe_get('operatingMargins'), 'operatingMargins', True)}
- Net Profit Margin: {format_metric(safe_get('profitMargins'), 'profitMargins', True)}
- ROE (Return on Equity): {format_metric(safe_get('returnOnEquity'), 'returnOnEquity', True)}
- ROA (Return on Assets): {format_metric(safe_get('returnOnAssets'), 'returnOnAssets', True)}

**REVENUE & EARNINGS:**
- Total Revenue (TTM): ${safe_get('totalRevenue'):,.0f if isinstance(safe_get('totalRevenue'), (int, float)) else safe_get('totalRevenue')}
- Revenue per Share: ${safe_get('revenuePerShare')}
- EPS (TTM): ${safe_get('trailingEps')}
- Forward EPS: ${safe_get('forwardEps')}
- EBITDA: ${safe_get('ebitda'):,.0f if isinstance(safe_get('ebitda'), (int, float)) else safe_get('ebitda')}

**GROWTH METRICS (vs Industry Median):**
- Revenue Growth (YoY): {format_metric(safe_get('revenueGrowth'), 'revenueGrowth', True)}
- Earnings Growth: {format_metric(safe_get('earningsGrowth'), 'earningsGrowth', True)}
- Earnings Quarterly Growth: {safe_get('earningsQuarterlyGrowth')}

**CASH FLOW:**
- Operating Cash Flow: {format_metric(safe_get('operatingCashflow'), 'operatingCashflow', True)}
- Free Cash Flow: {format_metric(safe_get('freeCashflow'), 'freeCashflow', True)}

**SOLVENCY & LIQUIDITY (vs Industry Median):**
- Current Ratio: {format_metric(safe_get('currentRatio'), 'currentRatio', True)}
- Quick Ratio: {format_metric(safe_get('quickRatio'), 'quickRatio', True)}
- Total Debt: ${total_debt:,.0f}
- Total Cash: ${total_cash:,.0f}
- Net Debt: ${net_debt:,.0f if net_debt else 'N/A'}
- Debt to Equity: {format_metric(safe_get('debtToEquity'), 'debtToEquity', False)}

**VALUATION METRICS (vs Industry Median):**
- P/E Ratio (Trailing): {format_metric(safe_get('trailingPE'), 'trailingPE', False)}
- P/B Ratio: {format_metric(safe_get('priceToBook'), 'priceToBook', False)}
- P/S Ratio: {format_metric(safe_get('priceToSalesTrailing12Months'), 'priceToSalesTrailing12Months', False)}
- EV/EBITDA: {format_metric(safe_get('enterpriseToEbitda'), 'enterpriseToEbitda', False)}

**DIVIDENDS:**
- Dividend Rate: ${safe_get('dividendRate')}
- Dividend Yield: {safe_get('dividendYield')}
- Payout Ratio: {safe_get('payoutRatio')}

**ANALYST RECOMMENDATIONS:**
- Target Mean Price: ${safe_get('targetMeanPrice')}
- Recommendation: {safe_get('recommendationKey', 'N/A').upper()}
- Number of Analysts: {safe_get('numberOfAnalystOpinions')}

**INDUSTRY COMPARISON LEGEND:**
✅ = Above/Better than industry median (outperforming peers)
⚠️ = Below/Worse than industry median (underperforming peers)

Note: Industry benchmarks calculated from median values of peer companies in the same sector.
For valuation metrics (P/E, P/B, etc.), lower values are generally better (less expensive).
"""
            
            return data_summary
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data: {e}")
            return f"Error: Unable to fetch data from Yahoo Finance - {str(e)}"
    
    def fetch_gurufocus_data(self, ticker: str, days_ago: int = 0) -> Dict:
        """
        Fetch and analyze financial data using selected AI API
        :param ticker: Stock ticker symbol
        :param days_ago: Number of days ago for historical price (0 = current)
        :return: Dictionary containing analyzed metrics
        """
        # Fetch data from Yahoo Finance with industry comparison
        yfinance_content = self.fetch_yfinance_data(ticker, days_ago, include_industry_comparison=True)
        
        prompt = f"""You are an expert financial analyst. Analyze the data below for {ticker}.

{yfinance_content}

IMPORTANT: The data above already includes industry comparisons with ✅ (above median) and ⚠️ (below median) indicators.
Your job is to interpret these comparisons and provide investment insights.

Create a structured analysis following this format:

📊 **PROFITABILITY ANALYSIS**
[For each metric shown above, copy the exact value and industry comparison indicator]
- Gross Margin: [COPY EXACT VALUE with ✅ or ⚠️ if shown]
- Operating Margin: [COPY EXACT VALUE with ✅ or ⚠️ if shown]
- Net Profit Margin: [COPY EXACT VALUE with ✅ or ⚠️ if shown]
- ROE: [COPY EXACT VALUE with ✅ or ⚠️ if shown]
- ROA: [COPY EXACT VALUE with ✅ or ⚠️ if shown]

Overall Rating: [Count the ✅ vs ⚠️ indicators above and rate as Strong/Average/Weak]
Summary: [2-3 sentences explaining what the industry comparisons reveal about profitability competitive position]

💰 **FINANCIAL HEALTH & SOLVENCY**
- Current Ratio: [COPY EXACT VALUE with ✅ or ⚠️]
- Quick Ratio: [COPY EXACT VALUE with ✅ or ⚠️]
- Debt to Equity: [COPY EXACT VALUE with ✅ or ⚠️]

Rating: [Based on ✅ vs ⚠️ count]
Summary: [2-3 sentences on financial strength vs peers]

📈 **GROWTH ANALYSIS**
- Revenue Growth: [COPY EXACT VALUE with ✅ or ⚠️]
- Earnings Growth: [COPY EXACT VALUE with ✅ or ⚠️]

Rating: [Based on indicators]
Summary: [2-3 sentences on growth vs industry]

💵 **VALUATION ASSESSMENT**
- P/E Ratio: [COPY EXACT VALUE with ✅ or ⚠️]
- P/B Ratio: [COPY EXACT VALUE with ✅ or ⚠️]
- P/S Ratio: [COPY EXACT VALUE with ✅ or ⚠️]
- EV/EBITDA: [COPY EXACT VALUE with ✅ or ⚠️]

Note: For valuation, ✅ means cheaper than industry (better value), ⚠️ means more expensive
Rating: [Undervalued/Fair/Overvalued based on indicators]
Summary: [2-3 sentences on valuation vs peers]

🎯 **INVESTMENT THESIS**

**Competitive Strengths (✅ indicators):**
• [List each metric that has ✅, showing it outperforms industry]
• [Continue for all ✅ metrics]

**Competitive Weaknesses (⚠️ indicators):**
• [List each metric that has ⚠️, showing it underperforms industry]
• [Continue for all ⚠️ metrics]

**Industry Position Summary:**
[3-4 sentences summarizing: 
- Total ✅ count vs ⚠️ count
- Which categories (profitability, growth, valuation, solvency) are strongest/weakest
- Overall competitive position in industry]

**Overall Investment Rating:**
✅ STRONG BUY - Majority ✅ indicators, outperforms on key metrics
✔ BUY/HOLD - Mixed indicators, competitive with industry
⚠️ AVOID/SELL - Majority ⚠️ indicators, underperforms industry

**Final Recommendation:**
[4-5 sentences providing:
1. Clear statement on whether company outperforms or underperforms industry based on indicator count
2. Key competitive advantages or disadvantages from the data
3. Valuation attractiveness
4. Investment recommendation with reasoning based on the reliable industry comparisons]

CRITICAL: Base your entire analysis on the ✅ and ⚠️ indicators provided in the data. These are calculated from actual peer company data, not estimates. Count them accurately and let them guide your recommendations."""
        
        try:
            if self.provider == "gemini":
                response_text = self._fetch_with_gemini(prompt)
            else:  # openai
                response_text = self._fetch_with_openai(prompt)
            
            parsed_data = self._parse_response(response_text, ticker)
            return parsed_data
            
        except Exception as e:
            print(f"Error fetching data via {self.provider.upper()}: {e}")
            return self._get_empty_data_structure(ticker, str(e))
    
    def _fetch_with_gemini(self, prompt: str) -> str:
        """Fetch data using Gemini API"""
        response = self.model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 4096,
            }
        )
        return response.text
    
    def _fetch_with_openai(self, prompt: str) -> str:
        """Fetch data using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst. Interpret the industry comparison indicators (✅ and ⚠️) provided in the data to give reliable investment insights. Do not make up comparisons - only use what's explicitly shown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        return response.choices[0].message.content
    
    def _parse_response(self, response_text: str, ticker: str) -> Dict:
        """Parse AI API response and structure the data"""
        data = {
            'ticker': ticker,
            'provider': self.provider,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'Yahoo Finance with Industry Peer Comparison',
            'raw_response': response_text
        }
        return data
    
    def _get_empty_data_structure(self, ticker: str, error_msg: str = '') -> Dict:
        """Return empty data structure when fetch fails"""
        return {
            'ticker': ticker,
            'provider': self.provider,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'Error',
            'error': error_msg or 'Failed to fetch data',
            'raw_response': ''
        }
    
    def format_qualitative_report(self, data: Dict) -> str:
        """Format the qualitative data into a readable report"""
        if 'error' in data and data['error']:
            return f"❌ Error fetching data for {data['ticker']}: {data.get('error', 'Unknown error')}\n"
        
        provider_name = data.get('provider', 'AI').upper()
        if provider_name == 'GEMINI':
            provider_display = "Google Gemini 2.0 Flash"
        elif provider_name == 'OPENAI':
            provider_display = "OpenAI GPT-4o-mini"
        else:
            provider_display = provider_name
        
        report = f"{'='*80}\n"
        report += f"  📊 COMPREHENSIVE INVESTMENT ANALYSIS FOR {data['ticker']}\n"
        report += f"{'='*80}\n"
        report += f"  🤖 AI Provider: {provider_display}\n"
        report += f"  📡 Data Source: {data.get('data_source', 'Yahoo Finance')}\n"
        report += f"  🕒 Analysis Timestamp: {data.get('timestamp', 'N/A')}\n"
        report += f"{'='*80}\n\n"
        
        if 'raw_response' in data and data['raw_response']:
            report += data['raw_response']
        else:
            report += "No data available.\n"
        
        report += f"\n\n{'='*80}\n"
        report += "INDUSTRY COMPARISON METHODOLOGY:\n"
        report += "✅ = Above industry median (calculated from actual peer companies)\n"
        report += "⚠️ = Below industry median (calculated from actual peer companies)\n"
        report += f"\nPeer companies identified from same sector via Yahoo Finance\n"
        report += f"Analysis powered by {provider_display}\n"
        report += f"{'='*80}\n"
        
        return report


# Helper functions
def get_available_providers():
    """Returns list of available AI providers"""
    providers = []
    if GENAI_AVAILABLE:
        providers.append("gemini")
    if OPENAI_AVAILABLE:
        providers.append("openai")
    return providers

def is_provider_available(provider: str) -> bool:
    """Check if a specific provider is available"""
    if provider.lower() == "gemini":
        return GENAI_AVAILABLE
    elif provider.lower() == "openai":
        return OPENAI_AVAILABLE
    return False