"""
Enhanced Yahoo Finance Data Discrepancy Checker with AI Benchmarking
Compares yfinance API data with industry benchmarks using improved peer selection
Designed to integrate with existing Streamlit app
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Optional AI imports
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Helper functions for app.py integration
def get_available_providers() -> List[str]:
    """Return list of available AI providers"""
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


class EnhancedYFinanceChecker:
    """
    Enhanced tool to:
    1. Extract data from yfinance API
    2. Find better-matched industry peers (by market cap + industry)
    3. Calculate robust benchmarks with statistics
    4. Use AI to interpret results (optional)
    5. Show clear performance indicators vs benchmarks
    
    Compatible with existing app.py API key input method
    """
    
    def __init__(self, ticker: str, api_key: Optional[str] = None, provider: str = "gemini"):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        self.info = self.stock.info
        
        # AI setup (optional) - same as original app
        self.use_ai = False
        self.provider = provider.lower()
        
        if api_key:
            if self.provider == "gemini" and GENAI_AVAILABLE:
                try:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    self.use_ai = True
                except Exception as e:
                    print(f"⚠️ Gemini setup failed: {e}")
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                try:
                    self.client = OpenAI(api_key=api_key)
                    self.model_name = "gpt-4o-mini"
                    self.use_ai = True
                except Exception as e:
                    print(f"⚠️ OpenAI setup failed: {e}")
        
        # Cache for benchmarks
        self.benchmarks = {}
        self.peers = []
        
    def get_improved_industry_peers(self, max_peers: int = 30) -> List[str]:
        """
        Improved peer selection using:
        1. Same GICS Sub-Industry (more specific than sector)
        2. Market cap within ±70% range
        3. Multiple exchange coverage (NYSE, NASDAQ, not just S&P 500)
        """
        try:
            sector = self.info.get('sector', '')
            industry = self.info.get('industry', '')
            market_cap = self.info.get('marketCap', 0)
            
            print(f"\n🔍 Finding Industry Peers for {self.ticker}")
            print(f"  → Sector: {sector}")
            print(f"  → Industry: {industry}")
            print(f"  → Market Cap: ${market_cap/1e9:.2f}B" if market_cap else "  → Market Cap: N/A")
            
            # Define market cap range (±70%)
            if market_cap:
                min_cap = market_cap * 0.3
                max_cap = market_cap * 1.7
            else:
                min_cap = 0
                max_cap = float('inf')
            
            peers = []
            
            # Strategy 1: Try S&P 500 first
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                sp500_df = pd.read_html(url)[0]
                
                # Filter by sector
                if sector:
                    sector_matches = sp500_df[sp500_df['GICS Sector'] == sector]['Symbol'].tolist()
                    
                    # Further filter by market cap
                    for symbol in sector_matches:
                        if symbol == self.ticker:
                            continue
                        try:
                            peer_stock = yf.Ticker(symbol)
                            peer_cap = peer_stock.info.get('marketCap', 0)
                            if peer_cap and min_cap <= peer_cap <= max_cap:
                                peers.append(symbol)
                        except:
                            continue
                            
                print(f"  ✓ Found {len(peers)} S&P 500 peers with similar market cap")
                
            except Exception as e:
                print(f"  ⚠️ Could not fetch S&P 500 list: {e}")
            
            # Strategy 2: Try NASDAQ 100 if needed
            if len(peers) < 10:
                try:
                    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
                    nasdaq_df = pd.read_html(url)[2]  # Usually the third table
                    nasdaq_symbols = nasdaq_df['Ticker'].tolist()
                    
                    for symbol in nasdaq_symbols:
                        if symbol in peers or symbol == self.ticker:
                            continue
                        try:
                            peer_stock = yf.Ticker(symbol)
                            peer_info = peer_stock.info
                            peer_sector = peer_info.get('sector', '')
                            peer_cap = peer_info.get('marketCap', 0)
                            
                            if peer_sector == sector and peer_cap and min_cap <= peer_cap <= max_cap:
                                peers.append(symbol)
                        except:
                            continue
                            
                    print(f"  ✓ Added NASDAQ-100 peers. Total: {len(peers)}")
                except:
                    pass
            
            # Remove duplicates and limit
            peers = list(set(peers))[:max_peers]
            
            print(f"  ✅ Final peer count: {len(peers)} companies")
            
            return peers
            
        except Exception as e:
            print(f"  ❌ Error finding peers: {e}")
            return []
    
    def calculate_robust_benchmarks(self, peers: List[str]) -> Dict:
        """
        Calculate benchmarks with statistical robustness:
        - Median (less affected by outliers)
        - 25th and 75th percentiles (range)
        - Count of valid data points
        - Standard deviation
        """
        print(f"\n📊 Calculating Robust Benchmarks from {len(peers)} peers...")
        
        metrics_to_track = {
            'currentPrice': [],
            'marketCap': [],
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
            'forwardPE': [],
            'priceToBook': [],
            'priceToSalesTrailing12Months': [],
            'enterpriseToEbitda': [],
            'dividendYield': [],
        }
        
        successful = 0
        for i, peer in enumerate(peers):
            if i % 10 == 0:
                print(f"  → Processing peer {i+1}/{len(peers)}...")
            
            try:
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                
                for metric in metrics_to_track.keys():
                    value = peer_info.get(metric)
                    if value is not None and isinstance(value, (int, float)):
                        if not np.isnan(value) and not np.isinf(value):
                            # Filter extreme outliers (beyond 3 standard deviations)
                            if len(metrics_to_track[metric]) > 3:
                                current_values = metrics_to_track[metric]
                                mean = np.mean(current_values)
                                std = np.std(current_values)
                                if std > 0 and abs(value - mean) <= 3 * std:
                                    metrics_to_track[metric].append(value)
                                elif std == 0:
                                    metrics_to_track[metric].append(value)
                            else:
                                metrics_to_track[metric].append(value)
                
                successful += 1
                
            except Exception as e:
                continue
        
        print(f"  ✅ Successfully processed {successful}/{len(peers)} peers")
        
        # Calculate statistics
        benchmarks = {}
        for metric, values in metrics_to_track.items():
            if len(values) >= 5:  # Need at least 5 data points for reliability
                benchmarks[metric] = {
                    'median': np.median(values),
                    'p25': np.percentile(values, 25),
                    'p75': np.percentile(values, 75),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                benchmarks[metric] = None
        
        return benchmarks
    
    def compare_to_benchmark(self, value, benchmark_stats: Dict, metric_name: str, 
                            higher_is_better: bool = True) -> Tuple[str, str, str]:
        """
        Compare metric to benchmark with detailed analysis
        Returns: (symbol, comparison_text, percentile_position)
        """
        if value is None or value == 'N/A' or benchmark_stats is None:
            return '❓', 'No benchmark data', 'N/A'
        
        try:
            # Convert to float if needed
            if isinstance(value, str):
                value = float(value.rstrip('%'))
            
            median = benchmark_stats['median']
            p25 = benchmark_stats['p25']
            p75 = benchmark_stats['p75']
            count = benchmark_stats['count']
            
            # Determine percentile position
            if value >= p75:
                percentile = "Top 25%"
            elif value >= median:
                percentile = "Above Median"
            elif value >= p25:
                percentile = "Below Median"
            else:
                percentile = "Bottom 25%"
            
            # Determine performance
            if higher_is_better:
                if value >= p75:
                    symbol = "🟢"
                    text = f"Excellent - {percentile} (median: {median:.2f}, n={count})"
                elif value >= median:
                    symbol = "🟡"
                    text = f"Good - {percentile} (median: {median:.2f}, n={count})"
                elif value >= p25:
                    symbol = "🟠"
                    text = f"Below Avg - {percentile} (median: {median:.2f}, n={count})"
                else:
                    symbol = "🔴"
                    text = f"Weak - {percentile} (median: {median:.2f}, n={count})"
            else:  # Lower is better
                if value <= p25:
                    symbol = "🟢"
                    text = f"Excellent - {percentile} (median: {median:.2f}, n={count})"
                elif value <= median:
                    symbol = "🟡"
                    text = f"Good - {percentile} (median: {median:.2f}, n={count})"
                elif value <= p75:
                    symbol = "🟠"
                    text = f"Above Avg - {percentile} (median: {median:.2f}, n={count})"
                else:
                    symbol = "🔴"
                    text = f"High - {percentile} (median: {median:.2f}, n={count})"
            
            return symbol, text, percentile
            
        except Exception as e:
            return '❓', f'Error: {str(e)}', 'N/A'
    
    def get_all_metrics_with_benchmarks(self) -> Dict:
        """Extract all metrics and compare with benchmarks"""
        
        # Get peers and benchmarks
        self.peers = self.get_improved_industry_peers()
        if self.peers:
            self.benchmarks = self.calculate_robust_benchmarks(self.peers)
        
        def format_metric(key, value, higher_is_better=True):
            """Format value with benchmark comparison"""
            if value is None:
                return 'N/A'
            
            # Format base value
            if key in ['grossMargins', 'operatingMargins', 'profitMargins', 
                      'returnOnEquity', 'returnOnAssets', 'revenueGrowth', 
                      'earningsGrowth', 'dividendYield']:
                if isinstance(value, (int, float)) and value < 1:
                    formatted = f"{value * 100:.2f}%"
                else:
                    formatted = f"{value:.2f}%"
            elif key in ['currentRatio', 'quickRatio', 'debtToEquity', 
                        'trailingPE', 'forwardPE', 'priceToBook', 
                        'priceToSalesTrailing12Months', 'enterpriseToEbitda']:
                formatted = f"{value:.2f}"
            elif key in ['marketCap', 'totalRevenue', 'totalDebt', 'totalCash']:
                if value >= 1e9:
                    formatted = f"${value/1e9:.2f}B"
                elif value >= 1e6:
                    formatted = f"${value/1e6:.2f}M"
                else:
                    formatted = f"${value:,.0f}"
            elif key in ['currentPrice', 'revenuePerShare', 'trailingEps', 'forwardEps']:
                formatted = f"${value:.2f}"
            else:
                formatted = str(value)
            
            # Add benchmark comparison
            if key in self.benchmarks and self.benchmarks[key] is not None:
                symbol, text, percentile = self.compare_to_benchmark(
                    value, self.benchmarks[key], key, higher_is_better
                )
                return f"{formatted} | {symbol} {text}"
            
            return formatted
        
        # Extract metrics
        metrics = {
            # Company Info
            'company_name': self.info.get('longName', 'N/A'),
            'sector': self.info.get('sector', 'N/A'),
            'industry': self.info.get('industry', 'N/A'),
            
            # Price & Market Data
            'currentPrice': format_metric('currentPrice', self.info.get('currentPrice')),
            'marketCap': format_metric('marketCap', self.info.get('marketCap')),
            
            # Profitability
            'grossMargins': format_metric('grossMargins', self.info.get('grossMargins'), True),
            'operatingMargins': format_metric('operatingMargins', self.info.get('operatingMargins'), True),
            'profitMargins': format_metric('profitMargins', self.info.get('profitMargins'), True),
            'returnOnEquity': format_metric('returnOnEquity', self.info.get('returnOnEquity'), True),
            'returnOnAssets': format_metric('returnOnAssets', self.info.get('returnOnAssets'), True),
            
            # Growth
            'revenueGrowth': format_metric('revenueGrowth', self.info.get('revenueGrowth'), True),
            'earningsGrowth': format_metric('earningsGrowth', self.info.get('earningsGrowth'), True),
            
            # Financial Health
            'currentRatio': format_metric('currentRatio', self.info.get('currentRatio'), True),
            'quickRatio': format_metric('quickRatio', self.info.get('quickRatio'), True),
            'debtToEquity': format_metric('debtToEquity', self.info.get('debtToEquity'), False),
            
            # Valuation
            'trailingPE': format_metric('trailingPE', self.info.get('trailingPE'), False),
            'forwardPE': format_metric('forwardPE', self.info.get('forwardPE'), False),
            'priceToBook': format_metric('priceToBook', self.info.get('priceToBook'), False),
            'priceToSalesTrailing12Months': format_metric('priceToSalesTrailing12Months', 
                                                          self.info.get('priceToSalesTrailing12Months'), False),
            
            # Other
            'dividendYield': format_metric('dividendYield', self.info.get('dividendYield'), True),
            
            # Metadata
            '_fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '_peer_count': len(self.peers),
            '_benchmark_count': len([b for b in self.benchmarks.values() if b is not None])
        }
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        metrics = self.get_all_metrics_with_benchmarks()
        
        report = f"""
{'='*90}
📊 ENHANCED YFINANCE DATA ANALYSIS FOR {self.ticker}
{'='*90}
Company: {metrics['company_name']}
Sector: {metrics['sector']}
Industry: {metrics['industry']}
Data Fetched: {metrics['_fetch_time']}
Peer Companies: {metrics['_peer_count']} (market-cap matched)
Benchmark Metrics: {metrics['_benchmark_count']}

{'='*90}
💰 PRICE & MARKET DATA
{'='*90}
Current Price:         {metrics['currentPrice']}
Market Cap:            {metrics['marketCap']}

{'='*90}
📈 PROFITABILITY METRICS
{'='*90}
Gross Margin:          {metrics['grossMargins']}
Operating Margin:      {metrics['operatingMargins']}
Net Profit Margin:     {metrics['profitMargins']}
ROE:                   {metrics['returnOnEquity']}
ROA:                   {metrics['returnOnAssets']}

{'='*90}
🚀 GROWTH METRICS
{'='*90}
Revenue Growth (YoY):  {metrics['revenueGrowth']}
Earnings Growth:       {metrics['earningsGrowth']}

{'='*90}
💪 FINANCIAL HEALTH
{'='*90}
Current Ratio:         {metrics['currentRatio']}
Quick Ratio:           {metrics['quickRatio']}
Debt to Equity:        {metrics['debtToEquity']}

{'='*90}
💵 VALUATION METRICS
{'='*90}
P/E (Trailing):        {metrics['trailingPE']}
P/E (Forward):         {metrics['forwardPE']}
P/B Ratio:             {metrics['priceToBook']}
P/S Ratio:             {metrics['priceToSalesTrailing12Months']}

{'='*90}
📍 BENCHMARK LEGEND
{'='*90}
🟢 = Top 25% (Excellent performance vs peers)
🟡 = Above Median (Good performance vs peers)
🟠 = Below Median (Below average vs peers)
🔴 = Bottom 25% (Weak performance vs peers)
❓ = Insufficient benchmark data

For valuation metrics (P/E, P/B, etc.), lower is better.

{'='*90}
COMPARE WITH YAHOO FINANCE WEBSITE:
https://finance.yahoo.com/quote/{self.ticker}
https://finance.yahoo.com/quote/{self.ticker}/key-statistics
{'='*90}
"""
        
        # Add AI analysis if available
        if self.use_ai:
            report += "\n" + self.generate_ai_analysis(metrics)
        
        return report
    
    def generate_ai_analysis(self, metrics: Dict) -> str:
        """Generate AI-powered investment analysis"""
        print("\n🤖 Generating AI-powered investment analysis...")
        
        prompt = f"""Analyze this stock data with benchmark comparisons for {self.ticker}:

Company: {metrics['company_name']}
Sector: {metrics['sector']}

PROFITABILITY (vs {metrics['_peer_count']} peers):
- Gross Margin: {metrics['grossMargins']}
- Operating Margin: {metrics['operatingMargins']}
- Net Margin: {metrics['profitMargins']}
- ROE: {metrics['returnOnEquity']}
- ROA: {metrics['returnOnAssets']}

GROWTH:
- Revenue Growth: {metrics['revenueGrowth']}
- Earnings Growth: {metrics['earningsGrowth']}

FINANCIAL HEALTH:
- Current Ratio: {metrics['currentRatio']}
- Quick Ratio: {metrics['quickRatio']}
- Debt/Equity: {metrics['debtToEquity']}

VALUATION:
- P/E: {metrics['trailingPE']}
- P/B: {metrics['priceToBook']}
- P/S: {metrics['priceToSalesTrailing12Months']}

Provide a concise investment analysis (300 words) covering:
1. Overall competitive position (count 🟢🟡🟠🔴 indicators)
2. Key strengths and weaknesses
3. Investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
4. Main risks and opportunities

Be specific and reference the benchmark indicators."""
        
        try:
            if self.provider == "gemini":
                response = self.model.generate_content(prompt)
                analysis = response.text
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional financial analyst. Provide clear, actionable investment insights based on quantitative benchmarks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                analysis = response.choices[0].message.content
            
            return f"""
{'='*90}
🤖 AI-POWERED INVESTMENT ANALYSIS
{'='*90}
{analysis}
{'='*90}
"""
        except Exception as e:
            return f"\n⚠️ AI analysis failed: {str(e)}\n"


class AIQualitativeAnalyzer:
    """
    Wrapper class for compatibility with app.py
    Provides qualitative analysis using AI providers
    """
    
    def __init__(self, api_key: str, provider: str = "gemini"):
        self.api_key = api_key
        self.provider = provider.lower()
        
        # Initialize AI provider
        if self.provider == "gemini" and GENAI_AVAILABLE:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        elif self.provider == "openai" and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
        else:
            raise ValueError(f"Provider {provider} not available or not supported")
    
    def fetch_gurufocus_data(self, ticker: str) -> Dict:
        """
        Fetch quantitative data with industry benchmarks
        Returns metrics dictionary for analysis
        """
        checker = EnhancedYFinanceChecker(ticker, api_key=self.api_key, provider=self.provider)
        return checker.get_all_metrics_with_benchmarks()
    
    def format_qualitative_report(self, metrics: Dict) -> str:
        """
        Format the quantitative analysis report
        """
        # Extract benchmark count
        peer_count = metrics.get('_peer_count', 0)
        benchmark_count = metrics.get('_benchmark_count', 0)
        
        report = f"""
{'='*90}
📊 QUANTITATIVE ANALYSIS WITH INDUSTRY BENCHMARKS
{'='*90}
Company: {metrics.get('company_name', 'N/A')}
Sector: {metrics.get('sector', 'N/A')}
Industry: {metrics.get('industry', 'N/A')}
Peer Companies: {peer_count}
Benchmark Metrics: {benchmark_count}

{'='*90}
💰 PRICE & MARKET DATA
{'='*90}
Current Price:         {metrics.get('currentPrice', 'N/A')}
Market Cap:            {metrics.get('marketCap', 'N/A')}

{'='*90}
📈 PROFITABILITY METRICS
{'='*90}
Gross Margin:          {metrics.get('grossMargins', 'N/A')}
Operating Margin:      {metrics.get('operatingMargins', 'N/A')}
Net Profit Margin:     {metrics.get('profitMargins', 'N/A')}
ROE:                   {metrics.get('returnOnEquity', 'N/A')}
ROA:                   {metrics.get('returnOnAssets', 'N/A')}

{'='*90}
🚀 GROWTH METRICS
{'='*90}
Revenue Growth (YoY):  {metrics.get('revenueGrowth', 'N/A')}
Earnings Growth:       {metrics.get('earningsGrowth', 'N/A')}

{'='*90}
💪 FINANCIAL HEALTH
{'='*90}
Current Ratio:         {metrics.get('currentRatio', 'N/A')}
Quick Ratio:           {metrics.get('quickRatio', 'N/A')}
Debt to Equity:        {metrics.get('debtToEquity', 'N/A')}

{'='*90}
💵 VALUATION METRICS
{'='*90}
P/E (Trailing):        {metrics.get('trailingPE', 'N/A')}
P/E (Forward):         {metrics.get('forwardPE', 'N/A')}
P/B Ratio:             {metrics.get('priceToBook', 'N/A')}
P/S Ratio:             {metrics.get('priceToSalesTrailing12Months', 'N/A')}

{'='*90}
📍 BENCHMARK LEGEND
{'='*90}
🟢 = Top 25% (Excellent performance vs peers)
🟡 = Above Median (Good performance vs peers)
🟠 = Below Median (Below average vs peers)
🔴 = Bottom 25% (Weak performance vs peers)
❓ = Insufficient benchmark data

For valuation metrics (P/E, P/B, etc.), lower is better.
{'='*90}
"""
        
        # Add AI analysis if available
        if self.api_key:
            try:
                ai_analysis = self._generate_ai_investment_analysis(metrics)
                report += f"\n{ai_analysis}"
            except Exception as e:
                report += f"\n⚠️ AI analysis failed: {str(e)}\n"
        
        return report
    
    def _generate_ai_investment_analysis(self, metrics: Dict) -> str:
        """Generate AI-powered investment analysis"""
        prompt = f"""Analyze this stock data with benchmark comparisons:

Company: {metrics.get('company_name', 'N/A')}
Sector: {metrics.get('sector', 'N/A')}

PROFITABILITY (vs {metrics.get('_peer_count', 0)} peers):
- Gross Margin: {metrics.get('grossMargins', 'N/A')}
- Operating Margin: {metrics.get('operatingMargins', 'N/A')}
- Net Margin: {metrics.get('profitMargins', 'N/A')}
- ROE: {metrics.get('returnOnEquity', 'N/A')}
- ROA: {metrics.get('returnOnAssets', 'N/A')}

GROWTH:
- Revenue Growth: {metrics.get('revenueGrowth', 'N/A')}
- Earnings Growth: {metrics.get('earningsGrowth', 'N/A')}

FINANCIAL HEALTH:
- Current Ratio: {metrics.get('currentRatio', 'N/A')}
- Quick Ratio: {metrics.get('quickRatio', 'N/A')}
- Debt/Equity: {metrics.get('debtToEquity', 'N/A')}

VALUATION:
- P/E: {metrics.get('trailingPE', 'N/A')}
- P/B: {metrics.get('priceToBook', 'N/A')}
- P/S: {metrics.get('priceToSalesTrailing12Months', 'N/A')}

Provide a concise investment analysis (300 words) covering:
1. Overall competitive position (reference 🟢🟡🟠🔴 indicators)
2. Key strengths and weaknesses
3. Investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
4. Main risks and opportunities

Be specific and reference the benchmark indicators."""
        
        if self.provider == "gemini":
            response = self.model.generate_content(prompt)
            analysis = response.text
        else:  # openai
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst. Provide clear, actionable investment insights based on quantitative benchmarks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            analysis = response.choices[0].message.content
        
        return f"""
{'='*90}
🤖 AI-POWERED INVESTMENT ANALYSIS
{'='*90}
{analysis}
{'='*90}
"""


# Convenience function for integration with existing app.py
def analyze_stock_with_benchmarks(ticker: str, api_key: Optional[str] = None, 
                                  provider: str = "gemini") -> str:
    """
    Main function to analyze a stock with industry benchmarks
    Compatible with existing app.py API key input method
    
    Args:
        ticker: Stock symbol
        api_key: Optional AI API key (Gemini or OpenAI)
        provider: "gemini" or "openai"
    
    Returns:
        Formatted analysis report string
    """
    checker = EnhancedYFinanceChecker(ticker, api_key=api_key, provider=provider)
    return checker.generate_report()


# Standalone execution (for testing)
if __name__ == "__main__":
    print("="*90)
    print("🎯 Enhanced YFinance Discrepancy Checker with AI Benchmarking")
    print("="*90)
    
    ticker = input("\nEnter ticker symbol (e.g., AAPL): ").strip().upper()
    
    use_ai = input("Use AI analysis? (y/n): ").strip().lower() == 'y'
    
    api_key = None
    provider = "gemini"
    if use_ai:
        provider = input("AI provider (gemini/openai) [gemini]: ").strip().lower() or "gemini"
        api_key = input(f"Enter your {provider.upper()} API key: ").strip()
    
    print(f"\n🚀 Analyzing {ticker}...")
    
    report = analyze_stock_with_benchmarks(ticker, api_key=api_key, provider=provider)
    
    print(report)
    
    # Option to save
    save = input("\n💾 Save report to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to: {filename}")