import yfinance as yf
import pandas as pd
import requests

class YahooFinanceDataFetcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_stock_history(self, ticker, period="1y", interval="1d"):
        """Get stock historical data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            return hist
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def get_company_info(self, ticker):
        """Get company basic information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            print(f"Error fetching company information for {ticker}: {e}")
            return {}

    def get_financials(self, ticker):
        """Get company financial statements (annual and quarterly)"""
        try:
            stock = yf.Ticker(ticker)
            financials = {
                'income_stmt': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_income_stmt': stock.quarterly_financials,
                'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                'quarterly_cash_flow': stock.quarterly_cashflow
            }
            return financials
        except Exception as e:
            print(f"Error fetching financial statements for {ticker}: {e}")
            return {}

    def get_key_stats(self, ticker):
        """
        Get key statistical data.
        The `info` object from `yfinance` already contains most key statistical data.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Extract key statistical data we care about from the info dictionary
            key_stats = {
                'Market Cap': info.get('marketCap'),
                'Forward P/E': info.get('forwardPE'),
                'Trailing P/E': info.get('trailingPE'),
                'PEG Ratio': info.get('pegRatio'),
                'Price/Sales': info.get('priceToSalesTrailing12Months'),
                'Price/Book': info.get('priceToBook'),
                'EPS TTM': info.get('trailingEps'),
                'Beta': info.get('beta'),
                'Dividend Yield': info.get('dividendYield'),
                'Revenue Growth (YoY)': info.get('revenueGrowth'),
                'Profit Margins': info.get('profitMargins')
            }
            return key_stats
        except Exception as e:
            print(f"Error fetching key statistics for {ticker}: {e}")
            return {}

    def search_ticker(self, company_name):
        """Search for stock ticker based on company name"""
        search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}&quotesCount=1&newsCount=0"
        try:
            response = requests.get(search_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if data and 'quotes' in data and len(data['quotes']) > 0:
                return data['quotes'][0]['symbol']
            return None
        except Exception as e:
            print(f"Error searching ticker for {company_name}: {e}")
            return None

    def get_current_yield(self, ticker="^TNX"):
        """Get current yield/price for specified ticker, defaults to US 10-Year Treasury yield"""
        try:
            # Get last 5 days of data, sufficient to find latest closing price
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except Exception as e:
            print(f"Error fetching current yield for {ticker}: {e}")
            return None