import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import yfinance as yf

from data_fetcher import YahooFinanceDataFetcher
from data_processor import StockDataProcessor
from visualizer import StockVisualizer
from valuation_model import StockValuationModel

# Try to import AI analyzers
try:
    from gemini_analyzer import AIQualitativeAnalyzer, get_available_providers, is_provider_available
    INDUSTRY_ANALYZER_AVAILABLE = True
except ImportError:
    INDUSTRY_ANALYZER_AVAILABLE = False
    print("Warning: Industry comparison analyzer not available.")

try:
    from governance_analyzer import GovernanceQualitativeAnalyzer
    GOVERNANCE_ANALYZER_AVAILABLE = True
except ImportError:
    GOVERNANCE_ANALYZER_AVAILABLE = False
    print("Warning: Governance analyzer not available.")


class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Valuation & Visualization Tool")
        self.root.geometry("1200x900")

        self.fetcher = YahooFinanceDataFetcher()
        self.processor = StockDataProcessor()
        self.valuation_model = StockValuationModel(risk_free_rate=0.04, market_return=0.09)
        
        # Initialize AI Analyzers
        self.industry_analyzer = None
        self.governance_analyzer = None
        self.selected_provider = tk.StringVar(value="gemini")

        self.fetched_data = {}
        self._create_widgets()

    def _create_widgets(self):
        # === AI API Configuration Frame ===
        api_frame = ttk.LabelFrame(self.root, text="🤖 AI API Configuration (Optional - for Qualitative Analysis)", padding=10)
        api_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Provider selection
        provider_frame = ttk.Frame(api_frame)
        provider_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=(0, 10), sticky="w")
        
        ttk.Label(provider_frame, text="Select AI Provider:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        # Gemini radio button
        self.gemini_radio = ttk.Radiobutton(
            provider_frame, 
            text="🔷 Google Gemini 2.0 (Free tier available)",
            variable=self.selected_provider, 
            value="gemini",
            command=self._on_provider_change
        )
        self.gemini_radio.pack(side=tk.LEFT, padx=5)
        
        # OpenAI radio button
        self.openai_radio = ttk.Radiobutton(
            provider_frame, 
            text="🟢 OpenAI ChatGPT (Paid, more stable)",
            variable=self.selected_provider, 
            value="openai",
            command=self._on_provider_change
        )
        self.openai_radio.pack(side=tk.LEFT, padx=5)
        
        # Instructions label (will change based on provider)
        self.instruction_label = ttk.Label(
            api_frame, 
            text="Get Gemini API key at: https://makersuite.google.com/app/apikey",
            foreground="blue",
            cursor="hand2"
        )
        self.instruction_label.grid(row=1, column=0, columnspan=4, padx=5, pady=(0, 5), sticky="w")
        
        # API Key input
        ttk.Label(api_frame, text="API Key:", font=("Arial", 10, "bold")).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.api_key_entry = ttk.Entry(api_frame, width=60, show="*", font=("Arial", 10))
        self.api_key_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        self.set_api_button = ttk.Button(api_frame, text="✓ Set API Key", command=self.set_api_key)
        self.set_api_button.grid(row=2, column=2, padx=5, pady=5)
        
        # Status indicator
        self.api_status_label = ttk.Label(api_frame, text="● Not configured", foreground="red")
        self.api_status_label.grid(row=2, column=3, padx=5, pady=5)
        
        # Make column 1 expandable
        api_frame.columnconfigure(1, weight=1)
        
        # Check availability and update UI
        self._update_provider_availability()
        
        # Stock Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Stock Information Input", padding=5)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(input_frame, text="Main Stock Symbol/Name:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.main_ticker_entry = ttk.Entry(input_frame, width=20)
        self.main_ticker_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.main_ticker_entry.insert(0, "NVDA")

        ttk.Label(input_frame, text="Competitor Symbols (comma-separated):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.competitors_entry = ttk.Entry(input_frame, width=40)
        self.competitors_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.competitors_entry.insert(0, "AMD,INTC")

        self.run_button = ttk.Button(input_frame, text="🚀 Fetch Data & Analyze", command=self.analyze_stock)
        self.run_button.grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky="ns")

        # Notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Chart Analysis Tab
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="📊 Chart Analysis")
        self.visualizer = StockVisualizer(master=self.plot_frame)

        # Financial Statements Tab
        self.financials_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.financials_frame, text="📋 Financial Statements")
        self.financials_text = scrolledtext.ScrolledText(self.financials_frame, wrap=tk.WORD, width=100, height=20, font=("Courier", 9))
        self.financials_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Valuation Results Tab
        self.valuation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.valuation_frame, text="💰 Valuation Results")
        self.valuation_output = scrolledtext.ScrolledText(self.valuation_frame, wrap=tk.WORD, width=100, height=20, font=("Courier", 9))
        self.valuation_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Quantitative Analysis Tab (Industry Comparison)
        self.quantitative_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.quantitative_frame, text="📈 Quantitative Analysis (Vs Industry Comparisons)")
        self.quantitative_text = scrolledtext.ScrolledText(self.quantitative_frame, wrap=tk.WORD, width=100, height=20, font=("Courier", 9))
        self.quantitative_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Qualitative Analysis Tab (Governance & Strategy)
        self.qualitative_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.qualitative_frame, text="🎯 Qualitative Analysis")
        self.qualitative_text = scrolledtext.ScrolledText(self.qualitative_frame, wrap=tk.WORD, width=100, height=20, font=("Courier", 9))
        self.qualitative_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _update_provider_availability(self):
        """Update UI based on which providers are available"""
        if not INDUSTRY_ANALYZER_AVAILABLE and not GOVERNANCE_ANALYZER_AVAILABLE:
            return
        
        if INDUSTRY_ANALYZER_AVAILABLE:
            from gemini_analyzer import get_available_providers
            available_providers = get_available_providers()
            
            if "gemini" not in available_providers:
                self.gemini_radio.config(state="disabled")
                self.gemini_radio.config(text="🔷 Google Gemini 2.0 (Not installed)")
            
            if "openai" not in available_providers:
                self.openai_radio.config(state="disabled")
                self.openai_radio.config(text="🟢 OpenAI ChatGPT (Not installed)")
            
            # If neither is available, select the first available one
            if available_providers:
                if self.selected_provider.get() not in available_providers:
                    self.selected_provider.set(available_providers[0])
                self._on_provider_change()

    def _on_provider_change(self):
        """Update instructions when provider changes"""
        provider = self.selected_provider.get()
        
        if provider == "gemini":
            self.instruction_label.config(
                text="Get free Gemini API key at: https://makersuite.google.com/app/apikey"
            )
        else:  # openai
            self.instruction_label.config(
                text="Get OpenAI API key at: https://platform.openai.com/api-keys (requires credits)"
            )
        
        # Reset status if provider changed
        if self.industry_analyzer or self.governance_analyzer:
            self.api_status_label.config(text="● Provider changed - please reset API key", foreground="orange")
            self.industry_analyzer = None
            self.governance_analyzer = None

    def set_api_key(self):
        """Set the AI API key for both analyzers"""
        if not INDUSTRY_ANALYZER_AVAILABLE and not GOVERNANCE_ANALYZER_AVAILABLE:
            messagebox.showerror(
                "Package Not Installed", 
                "AI analyzer packages are not installed.\n\n"
                "Please install required packages:\n"
                "- For Gemini: pip install google-generativeai\n"
                "- For OpenAI: pip install openai"
            )
            return
        
        provider = self.selected_provider.get()
        
        # Check if selected provider is available
        if INDUSTRY_ANALYZER_AVAILABLE:
            from gemini_analyzer import is_provider_available
            if not is_provider_available(provider):
                package_name = "google-generativeai" if provider == "gemini" else "openai"
                messagebox.showerror(
                    "Package Not Installed",
                    f"The {package_name} package is not installed.\n\n"
                    f"Please install it using:\n"
                    f"pip install {package_name}"
                )
                return
            
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter a valid API key.")
            return
        
        try:
            # Initialize both analyzers
            if INDUSTRY_ANALYZER_AVAILABLE:
                self.industry_analyzer = AIQualitativeAnalyzer(api_key, provider=provider)
            
            if GOVERNANCE_ANALYZER_AVAILABLE:
                self.governance_analyzer = GovernanceQualitativeAnalyzer(api_key, provider=provider)
            
            self.api_status_label.config(text=f"● Connected ({provider.upper()})", foreground="green")
            
            provider_name = "Google Gemini 2.0" if provider == "gemini" else "OpenAI ChatGPT"
            messagebox.showinfo(
                "Success", 
                f"{provider_name} API key set successfully!\n\n"
                "You can now use both AI-powered analysis features:\n"
                "• Quantitative (Industry Comparison)\n"
                "• Qualitative (Governance & Strategy)"
            )
        except Exception as e:
            self.api_status_label.config(text="● Error", foreground="red")
            messagebox.showerror("Error", f"Failed to initialize {provider.upper()} API:\n\n{str(e)}\n\nPlease check your API key.")

    def analyze_stock(self):
        # Disable button and show analyzing state
        self.run_button.config(state="disabled", text="⏳ Analyzing...")
        self.root.update_idletasks()

        try:
            main_ticker = self.main_ticker_entry.get().strip().upper()
            if not main_ticker:
                messagebox.showerror("Error", "Please enter a main stock symbol.")
                return

            competitor_tickers = [t.strip() for t in self.competitors_entry.get().strip().upper().split(',') if t.strip()]
            all_tickers = [main_ticker] + competitor_tickers
            self.fetched_data = {}

            successful_fetches = 0

            for ticker in all_tickers:
                print(f"Fetching data for {ticker}...")
                try:
                    data = self._fetch_and_process_single_stock(ticker)
                    if not data['info']:
                        messagebox.showwarning("Warning", f"Unable to fetch data for {ticker}, please check the stock symbol.")
                        continue
                    self.fetched_data[ticker] = data
                    successful_fetches += 1
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    messagebox.showwarning("Warning", f"Error processing {ticker}: {str(e)}")
                    continue

            if main_ticker not in self.fetched_data:
                messagebox.showerror("Error", f"Unable to fetch data for main stock {main_ticker}. Analysis aborted.")
                return

            print("All data fetched, starting display...")
            self._display_financials(main_ticker)
            self._calculate_and_display_valuation(main_ticker, competitor_tickers)
            self._plot_data(main_ticker, competitor_tickers)
            
            # Fetch and display both types of analysis
            self._fetch_and_display_quantitative_analysis(main_ticker)
            self._fetch_and_display_qualitative_analysis(main_ticker)
            
            print(f"Analysis completed successfully! Processed {successful_fetches} stocks.")

        except Exception as e:
            print(f"Unexpected error during analysis: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            
        finally:
            self.run_button.config(state="normal", text="🚀 Fetch Data & Analyze")

    def _fetch_and_process_single_stock(self, ticker):
        data = {
            'info': self.fetcher.get_company_info(ticker),
            'history': self.fetcher.get_stock_history(ticker, period="5y"),
            'financials': self.fetcher.get_financials(ticker)
        }
        if data['info']:
            data['processed_history'] = self.processor.calculate_technical_indicators(data['history'])
            data['annual_cash_flow'] = self.processor.get_yearly_financial_data(data['financials'], 'cash_flow')
            data['annual_balance'] = self.processor.get_yearly_financial_data(data['financials'], 'balance_sheet')
            data['annual_income'] = self.processor.get_yearly_financial_data(data['financials'], 'income_stmt')
            data['fcf'] = self.processor.calculate_free_cash_flow(data['annual_cash_flow'])
            data['total_debt'] = self.processor.get_total_debt(data['annual_balance'])
            data['cash_equivalents'] = self.processor.get_cash_and_equivalents(data['annual_balance'])
            data['total_equity'] = self.processor.get_total_stockholder_equity(data['annual_balance'])
            data['eps_history'] = self.processor.get_eps_from_financials(data['annual_income'])
        return data

    def _plot_data(self, main_ticker, competitor_tickers):
        plot_dfs = {}
        if main_ticker in self.fetched_data:
            plot_dfs[main_ticker] = self.fetched_data[main_ticker]['processed_history']
        for ticker in competitor_tickers:
            if ticker in self.fetched_data:
                plot_dfs[ticker] = self.fetched_data[ticker]['processed_history']
        self.visualizer.plot_multi_stock_comparison(plot_dfs, title="Closing Price Comparison (5 Years)")

    def _display_financials(self, main_ticker):
        self.financials_text.delete(1.0, tk.END)
        data = self.fetched_data[main_ticker]
        self.financials_text.insert(tk.END, f"--- {main_ticker} Key Financial Data ---\n\n")
        for key in ['income_stmt', 'balance_sheet', 'cash_flow']:
            self.financials_text.insert(tk.END, f"--- {key.replace('_', ' ').title()} ---\n")
            df = data['financials'].get(key)
            if df is not None and not df.empty:
                self.financials_text.insert(tk.END, df.to_string() + "\n\n")
            else:
                self.financials_text.insert(tk.END, "Data not available\n\n")

    def _calculate_and_display_valuation(self, main_ticker, competitor_tickers):
        self.valuation_output.delete(1.0, tk.END)
        main_data = self.fetched_data[main_ticker]
        output = f"--- {main_ticker} Valuation Analysis ---\n\n"

        current_yield_Y = self.fetcher.get_current_yield("^TNX")

        # --- DCF Valuation ---
        output += "### 1. Discounted Cash Flow (DCF) Valuation ###\n"
        try:
            beta = main_data['info'].get('beta')
            market_cap = main_data['info'].get('marketCap')
            shares_outstanding = main_data['info'].get('sharesOutstanding')
            current_fcf = main_data['fcf'].iloc[-1] if not main_data['fcf'].empty else None
            total_debt = main_data['total_debt'].iloc[-1] if not main_data['total_debt'].empty else None
            cash_equivalents = main_data['cash_equivalents'].iloc[-1] if not main_data['cash_equivalents'].empty else None
            required_data_map = {'Beta Value': beta, 'Market Cap': market_cap, 'Shares Outstanding': shares_outstanding,
                                 'Latest Free Cash Flow': current_fcf, 'Total Debt': total_debt, 'Cash & Equivalents': cash_equivalents}
            missing_items = [name for name, value in required_data_map.items() if value is None]
            if not missing_items:
                if len(main_data['fcf']) > 1:
                    hist_growth = main_data['fcf'].pct_change().mean()
                    growth_rates_high = [max(min(hist_growth * (1 - 0.1 * i), 0.3), 0.05) for i in range(5)]
                else:
                    growth_rates_high = [0.15, 0.12, 0.10, 0.08, 0.05]
                terminal_growth_rate = 0.025
                cost_of_debt = 0.055
                cost_of_equity = self.valuation_model.calculate_cost_of_equity(beta)
                wacc = self.valuation_model.calculate_wacc(market_cap, total_debt, cost_of_equity, cost_of_debt)
                output += f"  - WACC Calculation Parameters: Beta={beta:.2f}, Cost of Equity={cost_of_equity:.2%}, WACC={wacc:.2%}\n"
                output += f"  - FCF Growth Assumptions (5 years): {[f'{g:.2%}' for g in growth_rates_high]}\n"
                dcf_value = self.valuation_model.dcf_valuation(current_fcf, growth_rates_high, terminal_growth_rate,
                                                               wacc, shares_outstanding, total_debt, cash_equivalents)
                output += f"  >>> DCF Intrinsic Value per Share: ${dcf_value:.2f}\n"
            else:
                output += f"  - Insufficient key data for DCF valuation.\n"
                output += f"  - Missing Items: {', '.join(missing_items)}\n"
        except Exception as e:
            output += f"  - DCF valuation calculation error: {e}\n"

        # --- Relative Valuation ---
        output += "\n### 2. Relative Valuation ###\n"
        try:
            pe_list, ps_list, pb_list = [], [], []
            for ticker in competitor_tickers:
                if ticker in self.fetched_data:
                    info = self.fetched_data[ticker]['info']
                    if info.get('trailingPE'): pe_list.append(info['trailingPE'])
                    if info.get('priceToSalesTrailing12Months'): ps_list.append(info['priceToSalesTrailing12Months'])
                    if info.get('priceToBook'): pb_list.append(info['priceToBook'])
            avg_pe = np.mean(pe_list) if pe_list else None
            avg_ps = np.mean(ps_list) if ps_list else None
            avg_pb = np.mean(pb_list) if pb_list else None
            pe_str = f"{avg_pe:.2f}" if avg_pe is not None else "N/A"
            ps_str = f"{avg_ps:.2f}" if avg_ps is not None else "N/A"
            pb_str = f"{avg_pb:.2f}" if avg_pb is not None else "N/A"
            output += f"  - Competitor Average Multiples: P/E={pe_str}, P/S={ps_str}, P/B={pb_str}\n"
            target_eps = main_data['info'].get('trailingEps')
            total_revenue = main_data['annual_income'].loc['Total Revenue'].iloc[-1] if 'Total Revenue' in main_data['annual_income'].index else None
            total_equity = main_data['total_equity'].iloc[-1] if not main_data['total_equity'].empty else None
            shares_outstanding = main_data['info'].get('sharesOutstanding')
            target_sps = total_revenue / shares_outstanding if total_revenue and shares_outstanding else None
            target_bps = total_equity / shares_outstanding if total_equity and shares_outstanding else None
            relative_values = self.valuation_model.relative_valuation(target_eps, target_sps, target_bps, avg_pe, avg_ps, avg_pb)
            if not relative_values:
                output += "  - Insufficient key data (such as EPS, stockholder equity, etc.) for relative valuation.\n"
            else:
                for method, value in relative_values.items(): 
                    output += f"  >>> Valuation based on {method}: ${value:.2f}\n"
        except Exception as e:
            output += f"  - Relative valuation calculation error: {e}\n"

        output += f"\n### Current Market Price ###\n  - {main_ticker} Current Stock Price: ${main_data['history']['Close'].iloc[-1]:.2f}\n"
        self.valuation_output.insert(tk.END, output)

    def _fetch_and_display_quantitative_analysis(self, main_ticker):
        """Fetch and display quantitative analysis with industry comparison"""
        self.quantitative_text.delete(1.0, tk.END)
        
        if not INDUSTRY_ANALYZER_AVAILABLE:
            self.quantitative_text.insert(tk.END, "⚠️  INDUSTRY ANALYZER NOT INSTALLED\n\n")
            self.quantitative_text.insert(tk.END, "Quantitative analysis with industry comparison requires:\n\n")
            self.quantitative_text.insert(tk.END, "Option 1 - Google Gemini 2.0 (Free tier available):\n")
            self.quantitative_text.insert(tk.END, "    pip install google-generativeai\n\n")
            self.quantitative_text.insert(tk.END, "Option 2 - OpenAI ChatGPT (Paid, more stable):\n")
            self.quantitative_text.insert(tk.END, "    pip install openai\n\n")
            self.quantitative_text.insert(tk.END, "After installation, restart the application.\n")
            return
        
        if not self.industry_analyzer:
            self.quantitative_text.insert(tk.END, "⚠️  AI API NOT CONFIGURED\n\n")
            self.quantitative_text.insert(tk.END, "Please configure your AI API key at the top of the window.\n\n")
            self.quantitative_text.insert(tk.END, "This analysis compares the company's financial metrics against\n")
            self.quantitative_text.insert(tk.END, "industry peers to show competitive strengths and weaknesses.\n")
            return
        
        provider = self.selected_provider.get()
        provider_display = "Google Gemini 2.0" if provider == "gemini" else "OpenAI ChatGPT"
        
        self.quantitative_text.insert(tk.END, f"📊 Fetching quantitative analysis with industry comparison for {main_ticker}...\n")
        self.quantitative_text.insert(tk.END, f"🤖 Using {provider_display} for AI analysis\n")
        self.quantitative_text.insert(tk.END, "⏳ This may take 30-90 seconds (fetching peer data). Please wait...\n\n")
        self.root.update_idletasks()
        
        try:
            quantitative_data = self.industry_analyzer.fetch_gurufocus_data(main_ticker)
            report = self.industry_analyzer.format_qualitative_report(quantitative_data)
            
            self.quantitative_text.delete(1.0, tk.END)
            self.quantitative_text.insert(tk.END, report)
            
        except Exception as e:
            self.quantitative_text.delete(1.0, tk.END)
            self.quantitative_text.insert(tk.END, f"❌ ERROR FETCHING QUANTITATIVE ANALYSIS\n\n")
            self.quantitative_text.insert(tk.END, f"Error: {str(e)}\n\n")
            self.quantitative_text.insert(tk.END, "Please check:\n")
            self.quantitative_text.insert(tk.END, "1. Your API key is valid and has available quota/credits\n")
            self.quantitative_text.insert(tk.END, "2. You have internet connectivity\n")
            self.quantitative_text.insert(tk.END, "3. The ticker symbol is correct\n")

    def _fetch_and_display_qualitative_analysis(self, main_ticker):
        """Fetch and display qualitative governance analysis"""
        self.qualitative_text.delete(1.0, tk.END)
        
        if not GOVERNANCE_ANALYZER_AVAILABLE:
            self.qualitative_text.insert(tk.END, "⚠️  GOVERNANCE ANALYZER NOT INSTALLED\n\n")
            self.qualitative_text.insert(tk.END, "Qualitative governance analysis requires:\n\n")
            self.qualitative_text.insert(tk.END, "Option 1 - Google Gemini 2.0 (Free tier available):\n")
            self.qualitative_text.insert(tk.END, "    pip install google-generativeai\n\n")
            self.qualitative_text.insert(tk.END, "Option 2 - OpenAI ChatGPT (Paid, more stable):\n")
            self.qualitative_text.insert(tk.END, "    pip install openai\n\n")
            self.qualitative_text.insert(tk.END, "After installation, restart the application.\n")
            return
        
        if not self.governance_analyzer:
            self.qualitative_text.insert(tk.END, "⚠️  AI API NOT CONFIGURED\n\n")
            self.qualitative_text.insert(tk.END, "Please configure your AI API key at the top of the window.\n\n")
            self.qualitative_text.insert(tk.END, "This analysis evaluates:\n")
            self.qualitative_text.insert(tk.END, "  • Corporate Governance\n")
            self.qualitative_text.insert(tk.END, "  • Business Model & Competitive Moat\n")
            self.qualitative_text.insert(tk.END, "  • Industry Outlook & Position\n")
            self.qualitative_text.insert(tk.END, "  • R&D and Innovation Capability\n")
            return
        
        provider = self.selected_provider.get()
        provider_display = "Google Gemini 2.0" if provider == "gemini" else "OpenAI ChatGPT"
        
        self.qualitative_text.insert(tk.END, f"🎯 Fetching qualitative governance analysis for {main_ticker}...\n")
        self.qualitative_text.insert(tk.END, f"🤖 Using {provider_display} for AI analysis\n")
        self.qualitative_text.insert(tk.END, "⏳ This may take 20-40 seconds. Please wait...\n\n")
        self.root.update_idletasks()
        
        try:
            governance_data = self.governance_analyzer.fetch_governance_analysis(main_ticker)
            report = self.governance_analyzer.format_governance_report(governance_data)
            
            self.qualitative_text.delete(1.0, tk.END)
            self.qualitative_text.insert(tk.END, report)
            
        except Exception as e:
            self.qualitative_text.delete(1.0, tk.END)
            self.qualitative_text.insert(tk.END, f"❌ ERROR FETCHING GOVERNANCE ANALYSIS\n\n")
            self.qualitative_text.insert(tk.END, f"Error: {str(e)}\n\n")
            self.qualitative_text.insert(tk.END, "Please check:\n")
            self.qualitative_text.insert(tk.END, "1. Your API key is valid and has available quota/credits\n")
            self.qualitative_text.insert(tk.END, "2. You have internet connectivity\n")
            self.qualitative_text.insert(tk.END, "3. The ticker symbol is correct\n")
            
            if provider == "openai":
                self.qualitative_text.insert(tk.END, "4. Your OpenAI account has sufficient credits\n")
            else:
                self.qualitative_text.insert(tk.END, "4. You haven't exceeded Gemini's free tier quota\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()