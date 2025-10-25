import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, Canvas, Frame, Scrollbar
import numpy as np
import yfinance as yf

from data_fetcher import YahooFinanceDataFetcher
from data_processor import StockDataProcessor
from visualizer import StockVisualizer
from valuation_model import StockValuationModel

# Try to import AI analyzers
try:
    from stock_evaluator import ComprehensiveStockAnalyzer, get_available_providers, is_provider_available
    COMPREHENSIVE_ANALYZER_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_ANALYZER_AVAILABLE = False
    print("Warning: Comprehensive analyzer not available.")


class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analysis and Valuation Tool")
        self.root.geometry("1200x800")

        self.fetched_data = {}
        self.fetcher = YahooFinanceDataFetcher()
        self.processor = StockDataProcessor()
        self.valuation_model = StockValuationModel(risk_free_rate=0.04, market_return=0.09)

        # Analyzers
        self.comprehensive_analyzer = None
        
        # NEW: Dictionaries for comprehensive tab widgets
        self.quant_entries = {}
        self.quant_suggestions = {}
        self.quant_industry_entries = {}
        self.qual_entries = {}
        self.qual_suggestions = {}
        self.val_entries = {}
        self.ops_entries = {}
        self.ops_suggestions = {}

        self._create_widgets()

    def _create_widgets(self):
        # === AI API Configuration Frame ===
        api_frame = ttk.LabelFrame(self.root, text="🤖 AI API Configuration (Optional)", padding=10)
        api_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Provider selection
        provider_frame = ttk.Frame(api_frame)
        provider_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=(0, 10), sticky="w")
        
        ttk.Label(provider_frame, text="Select AI Provider:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        self.selected_provider = tk.StringVar(value="gemini")
        
        # Gemini radio button
        self.gemini_radio = ttk.Radiobutton(
            provider_frame, 
            text="🔷 Google Gemini",
            variable=self.selected_provider, 
            value="gemini",
            command=self._on_provider_change
        )
        self.gemini_radio.pack(side=tk.LEFT, padx=5)
        
        # OpenAI radio button
        self.openai_radio = ttk.Radiobutton(
            provider_frame, 
            text="🟢 OpenAI ChatGPT",
            variable=self.selected_provider, 
            value="openai",
            command=self._on_provider_change
        )
        self.openai_radio.pack(side=tk.LEFT, padx=5)
        
        # Instructions label
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
        
        api_frame.columnconfigure(1, weight=1)
        
        # Check availability
        self._update_provider_availability()
        
        # Stock Input Frame (Kept from original)
        input_frame = ttk.LabelFrame(self.root, text="Stock Information Input", padding=5)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(input_frame, text="Main Ticker:").pack(side=tk.LEFT, padx=5)
        self.main_ticker_entry = ttk.Entry(input_frame, width=10)
        self.main_ticker_entry.pack(side=tk.LEFT, padx=5)
        self.main_ticker_entry.insert(0, "AAPL") # Default value

        ttk.Label(input_frame, text="Competitors (comma-sep):").pack(side=tk.LEFT, padx=5)
        self.competitors_entry = ttk.Entry(input_frame, width=20)
        self.competitors_entry.pack(side=tk.LEFT, padx=5)
        self.competitors_entry.insert(0, "MSFT,GOOG") # Default value

        self.run_analysis_button = ttk.Button(input_frame, text="Run Basic Analysis", command=self.run_analysis)
        self.run_analysis_button.pack(side=tk.LEFT, padx=10, pady=5)

        # Notebook with tabs - UPDATED
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Chart Analysis Tab (Kept from original)
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="📊 Chart Analysis")
        self.visualizer = StockVisualizer(master=self.plot_frame)

        # Financial Statements Tab (Kept from original)
        self.financials_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.financials_frame, text="📋 Financial Statements")
        self.financials_text = scrolledtext.ScrolledText(self.financials_frame, wrap=tk.WORD, width=100, height=20, font=("Courier", 9))
        self.financials_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Valuation Results Tab (Kept from original)
        self.valuation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.valuation_frame, text="💰 Valuation Results")
        self.valuation_output = scrolledtext.ScrolledText(self.valuation_frame, wrap=tk.WORD, width=100, height=20, font=("Courier", 9))
        self.valuation_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # NEW: Comprehensive Evaluation Tab (Replaces Quantitative + Qualitative tabs)
        self.comprehensive_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comprehensive_frame, text="🎯 Comprehensive Evaluation (PDF)")
        
        # Create comprehensive evaluation interface
        self._create_comprehensive_evaluation_tab()

    def _create_comprehensive_evaluation_tab(self):
        """Create the comprehensive evaluation tab as a scrollable form"""
        
        # Top button frame
        button_frame = ttk.Frame(self.comprehensive_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.load_suggestions_button = ttk.Button(
            button_frame, 
            text="🚀 Load Suggestions (Yahoo + AI)",
            command=self._load_suggestions
        )
        self.load_suggestions_button.pack(side=tk.LEFT, padx=5)

        self.calculate_score_button = ttk.Button(
            button_frame, 
            text="🧮 Calculate Final Score",
            command=self._calculate_comprehensive_score
        )
        self.calculate_score_button.pack(side=tk.LEFT, padx=5)

        # Main frame with scrollbar
        main_frame = Frame(self.comprehensive_frame)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = Canvas(main_frame)
        scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- Frame for results ---
        results_frame = ttk.LabelFrame(scrollable_frame, text="Final Report", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.comprehensive_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, width=100, height=20, font=("Courier", 9)
        )
        self.comprehensive_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Frame 1: Quantitative Factors ---
        quant_frame = ttk.LabelFrame(scrollable_frame, text="1. Quantitative Factors (User Entry)", padding=10)
        quant_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_quant_frame(quant_frame)

        # --- Frame 2: Qualitative Factors ---
        qual_frame = ttk.LabelFrame(scrollable_frame, text="2. Qualitative Factors (User Score 1-10)", padding=10)
        qual_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_qual_frame(qual_frame)
        
        # --- Frame 3: Valuation Factors ---
        val_frame = ttk.LabelFrame(scrollable_frame, text="3. Valuation Factors (User Entry)", padding=10)
        val_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_val_frame(val_frame)

        # --- Frame 4: Operational Factors ---
        ops_frame = ttk.LabelFrame(scrollable_frame, text="4. Operational Factors (User Score 1-10)", padding=10)
        ops_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_ops_frame(ops_frame)

    def _create_labeled_entry(self, parent, text, row, var_dict, var_key, 
                              sugg_dict=None, sugg_key=None, col_offset=0):
        """Helper function to create a labeled entry widget"""
        ttk.Label(parent, text=text).grid(row=row, column=col_offset, sticky='w', padx=5, pady=2)
        entry = ttk.Entry(parent, width=12)
        entry.grid(row=row, column=col_offset + 1, padx=5, pady=2)
        var_dict[var_key] = entry
        
        if sugg_dict is not None and sugg_key is not None:
            sugg_label = ttk.Label(parent, text="Yahoo: N/A", foreground="blue")
            sugg_label.grid(row=row, column=col_offset + 2, sticky='w', padx=5)
            sugg_dict[sugg_key] = sugg_label

    def _build_quant_frame(self, parent):
        """Builds the quantitative factors input frame"""
        # Profitability
        ttk.Label(parent, text="Profitability", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky='w', pady=5)
        self._create_labeled_entry(parent, "Avg. Margins (e.g., 0.3):", 1, self.quant_entries, 'avg_margins')
        self._create_labeled_entry(parent, "ROE (e.g., 0.15):", 2, self.quant_entries, 'roe', self.quant_suggestions, 'roe')
        self._create_labeled_entry(parent, "ROA (e.g., 0.08):", 3, self.quant_entries, 'roa', self.quant_suggestions, 'roa')
        
        # Operational Efficiency
        ttk.Label(parent, text="Operational Efficiency", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=3, sticky='w', pady=5)
        self._create_labeled_entry(parent, "Days Inventory:", 5, self.quant_entries, 'days_inventory')
        self._create_labeled_entry(parent, "Days Sales Outstanding:", 6, self.quant_entries, 'days_sales')
        ttk.Label(parent, text="Industry Avg Days Inv:").grid(row=5, column=3, sticky='w', padx=5)
        self.quant_industry_entries['days_inventory'] = ttk.Entry(parent, width=10)
        self.quant_industry_entries['days_inventory'].grid(row=5, column=4, padx=5)
        ttk.Label(parent, text="Industry Avg DSO:").grid(row=6, column=3, sticky='w', padx=5)
        self.quant_industry_entries['days_sales'] = ttk.Entry(parent, width=10)
        self.quant_industry_entries['days_sales'].grid(row=6, column=4, padx=5)
        
        # Solvency
        ttk.Label(parent, text="Solvency", font=("Arial", 10, "bold")).grid(row=7, column=0, columnspan=3, sticky='w', pady=5)
        self._create_labeled_entry(parent, "Current Ratio:", 8, self.quant_entries, 'current_ratio', self.quant_suggestions, 'current_ratio')
        self._create_labeled_entry(parent, "Quick Ratio:", 9, self.quant_entries, 'quick_ratio')
        self._create_labeled_entry(parent, "Interest Coverage:", 10, self.quant_entries, 'interest_coverage')
        self._create_labeled_entry(parent, "Total Debt/Equity:", 11, self.quant_entries, 'total_debt_equity', self.quant_suggestions, 'total_debt_equity')

        # Growth Potential
        ttk.Label(parent, text="Growth Potential", font=("Arial", 10, "bold")).grid(row=0, column=3, columnspan=3, sticky='w', pady=5, padx=20)
        self._create_labeled_entry(parent, "Revenue Growth (YoY):", 1, self.quant_entries, 'revenue_growth_yoy', self.quant_suggestions, 'revenue_growth_yoy', col_offset=3)
        self._create_labeled_entry(parent, "EPS Growth (YoY):", 2, self.quant_entries, 'eps_growth_yoy', self.quant_suggestions, 'eps_growth_yoy', col_offset=3)

        # Risk Metrics
        ttk.Label(parent, text="Risk Metrics", font=("Arial", 10, "bold")).grid(row=3, column=3, columnspan=3, sticky='w', pady=5, padx=20)
        self._create_labeled_entry(parent, "Beta:", 4, self.quant_entries, 'beta', self.quant_suggestions, 'beta', col_offset=3)
        self._create_labeled_entry(parent, "Volatility (e.g., 0.3):", 5, self.quant_entries, 'volatility', col_offset=3)
        
        # Other metrics for credits
        ttk.Label(parent, text="Metrics for Credits", font=("Arial", 10, "bold")).grid(row=7, column=3, columnspan=3, sticky='w', pady=5, padx=20)
        self._create_labeled_entry(parent, "P/E (for credit 1):", 8, self.quant_entries, 'pe', col_offset=3)
        ttk.Label(parent, text="Industry Avg P/E:").grid(row=8, column=5, sticky='w', padx=5)
        self.quant_industry_entries['pe'] = ttk.Entry(parent, width=10)
        self.quant_industry_entries['pe'].grid(row=8, column=6, padx=5)
        
        self.quant_entries['margin_expanding'] = tk.BooleanVar()
        ttk.Checkbutton(parent, text="Margins Expanding? (Credit 6)", variable=self.quant_entries['margin_expanding']).grid(row=9, column=3, columnspan=2, sticky='w', padx=20)

    def _build_qual_frame(self, parent):
        """Builds the qualitative factors input frame"""
        qual_questions = {
            'governance': ['board_structure', 'management_integrity', 'scandal_history'],
            'business_model': ['product_uniqueness', 'competitive_moat'],
            'industry_outlook': ['industry_growth', 'market_position', 'global_presence'],
            'innovation': ['rd_investment', 'innovation_revenue', 'product_launches'],
            'esg': ['environmental', 'social', 'governance_esg'],
            'macro': ['macro_resilience']
        }
        
        row = 0
        for category, questions in qual_questions.items():
            self.qual_entries[category] = {}
            self.qual_suggestions[category] = {}
            
            ttk.Label(parent, text=category.replace('_', ' ').title(), font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=3, sticky='w', pady=(10,2))
            row += 1
            
            for q_key in questions:
                q_text = q_key.replace('_', ' ').title()
                ttk.Label(parent, text=f"{q_text}:").grid(row=row, column=0, sticky='w', padx=10, pady=2)
                entry = ttk.Entry(parent, width=5)
                entry.insert(0, "5") # Default score
                entry.grid(row=row, column=1, padx=5, pady=2)
                self.qual_entries[category][q_key] = entry
                
                sugg_label = ttk.Label(parent, text="AI Suggestion: N/A", foreground="green", wraplength=700)
                sugg_label.grid(row=row+1, column=0, columnspan=3, sticky='w', padx=20, pady=(0,5))
                self.qual_suggestions[category][q_key] = sugg_label
                row += 2

    def _build_val_frame(self, parent):
        """Builds the valuation factors input frame"""
        self._create_labeled_entry(parent, "Current Price:", 0, self.val_entries, 'current_price')
        self._create_labeled_entry(parent, "DCF Value:", 1, self.val_entries, 'dcf_value')
        self._create_labeled_entry(parent, "Relative Value (Avg):", 2, self.val_entries, 'relative_value')
        
        def populate_from_val_tab():
            try:
                val_text = self.valuation_output.get(1.0, tk.END)
                price = dcf = rel = None
                
                for line in val_text.split('\n'):
                    if "Current Stock Price:" in line:
                        price = line.split('$')[-1].strip()
                    if "DCF Intrinsic Value per Share:" in line:
                        dcf = line.split('$')[-1].strip()
                    if "Average Valuation:" in line: # Assumes _calculate_and_display_valuation outputs this
                        rel = line.split('$')[-1].strip()
                
                if price: 
                    self.val_entries['current_price'].delete(0, tk.END)
                    self.val_entries['current_price'].insert(0, price)
                if dcf: 
                    self.val_entries['dcf_value'].delete(0, tk.END)
                    self.val_entries['dcf_value'].insert(0, dcf)
                if rel: 
                    self.val_entries['relative_value'].delete(0, tk.END)
                    self.val_entries['relative_value'].insert(0, rel)
            except Exception as e:
                messagebox.showwarning("Populate Error", f"Could not auto-populate values. Run basic analysis first.\n{e}")

        ttk.Button(parent, text="Populate from 💰Valuation Tab", command=populate_from_val_tab).grid(row=1, column=3, padx=10)

    def _build_ops_frame(self, parent):
        """Builds the operational factors input frame"""
        self._create_labeled_entry(parent, "Liquidity Score (1-10):", 0, self.ops_entries, 'liquidity', self.ops_suggestions, 'liquidity')
        self._create_labeled_entry(parent, "Tax & Regulatory (1-10):", 1, self.ops_entries, 'tax')
        self._create_labeled_entry(parent, "Dividend Score (1-10):", 2, self.ops_entries, 'dividend', self.ops_suggestions, 'dividend')
        self._create_labeled_entry(parent, "Portfolio Fit (1-10):", 3, self.ops_entries, 'portfolio_fit')
        
        # Set defaults
        self.ops_entries['liquidity'].insert(0, "5")
        self.ops_entries['tax'].insert(0, "5")
        self.ops_entries['dividend'].insert(0, "5")
        self.ops_entries['portfolio_fit'].insert(0, "5")

    def _update_provider_availability(self):
        """Update UI based on which providers are available"""
        if not COMPREHENSIVE_ANALYZER_AVAILABLE:
            return
        
        available_providers = get_available_providers()
        
        if "gemini" not in available_providers:
            self.gemini_radio.config(state="disabled")
            self.gemini_radio.config(text="🔷 Google Gemini (Not installed)")
        
        if "openai" not in available_providers:
            self.openai_radio.config(state="disabled")
            self.openai_radio.config(text="🟢 OpenAI ChatGPT (Not installed)")
        
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
                text="Get OpenAI API key at: https://platform.openai.com/api-keys"
            )
        
        # Reset status if provider changed
        if self.comprehensive_analyzer:
            self.api_status_label.config(text="● Provider changed - please reset API key", foreground="orange")
            self.comprehensive_analyzer = None

    def set_api_key(self):
        """Set the AI API key for comprehensive analyzer"""
        if not COMPREHENSIVE_ANALYZER_AVAILABLE:
            messagebox.showerror("Package Not Installed", "Comprehensive analyzer package is not installed.")
            return
        
        provider = self.selected_provider.get()
        
        if not is_provider_available(provider):
            package_name = "google-generativeai" if provider == "gemini" else "openai"
            messagebox.showerror("Package Not Installed", f"The {package_name} package is not installed.")
            return
            
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter a valid API key.")
            return
        
        try:
            # This is the only analyzer we instantiate now
            self.comprehensive_analyzer = ComprehensiveStockAnalyzer(api_key, provider=provider)
            self.api_status_label.config(text=f"● Connected ({provider.upper()})", foreground="green")
            
            provider_name = "Google Gemini" if provider == "gemini" else "OpenAI ChatGPT"
            messagebox.showinfo("Success", f"{provider_name} API key set successfully!")
        except Exception as e:
            self.api_status_label.config(text="● Error", foreground="red")
            messagebox.showerror("Error", f"Failed to initialize {provider.upper()} API:\n\n{str(e)}")

    def run_analysis(self):
        """Main function to fetch data and populate all tabs"""
        main_ticker = self.main_ticker_entry.get().strip().upper()
        competitors_input = self.competitors_entry.get().strip().upper()
        competitor_tickers = [t.strip() for t in competitors_input.split(',') if t.strip()]
        
        if not main_ticker:
            messagebox.showerror("Error", "Please enter a main stock ticker.")
            return

        # Clear old suggestion data
        for sugg_dict in [self.quant_suggestions, self.ops_suggestions]:
            for label in sugg_dict.values():
                label.config(text="Yahoo: N/A")
        for cat_dict in self.qual_suggestions.values():
            for label in cat_dict.values():
                label.config(text="AI Suggestion: N/A")
        self.comprehensive_text.delete(1.0, tk.END)

        try:
            all_tickers = [main_ticker] + competitor_tickers
            self.fetched_data.clear()
            
            for ticker in all_tickers:
                self.root.update_idletasks()
                data = self._fetch_and_process_single_stock(ticker)
                if data and data['info']:
                    self.fetched_data[ticker] = data
                else:
                    messagebox.showwarning("Data Error", f"Could not fetch complete data for {ticker}")

            if main_ticker not in self.fetched_data:
                messagebox.showerror("Error", f"Failed to fetch data for main ticker {main_ticker}")
                return

            # Populate tabs
            self._plot_data(main_ticker, competitor_tickers)
            self._display_financials(main_ticker)
            self._calculate_and_display_valuation(main_ticker, competitor_tickers)
            
            messagebox.showinfo("Success", "Basic analysis complete. Data loaded into tabs.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")

    def _load_suggestions(self):
        """Loads suggestions for the comprehensive tab"""
        main_ticker = self.main_ticker_entry.get().strip().upper()
        if not main_ticker:
            messagebox.showerror("Error", "Please enter a main ticker.")
            return

        self.comprehensive_text.delete(1.0, tk.END)
        self.comprehensive_text.insert(tk.END, f"🚀 Loading suggestions for {main_ticker}...\n")
        self.root.update_idletasks()

        if not self.comprehensive_analyzer:
            self.comprehensive_analyzer = ComprehensiveStockAnalyzer() # Init without AI

        # 1. Fetch Yahoo Finance Suggestions
        try:
            sugg_data = self.comprehensive_analyzer.get_suggestion_data(main_ticker)
            if 'error' in sugg_data:
                raise Exception(sugg_data['error'])
            
            # Populate Quant suggestions
            for key, label in self.quant_suggestions.items():
                val = sugg_data.get(key)
                if val is not None:
                    label.config(text=f"Yahoo: {val:.4f}")
                else:
                    label.config(text="Yahoo: N/A")
            
            # Populate Ops suggestions
            vol = sugg_data.get('volume', 0)
            spread = sugg_data.get('bid_ask_spread_pct', 1.0)
            self.ops_suggestions['liquidity'].config(text=f"Yahoo: Vol={vol or 0:,.0f}, Spread={spread or 0:.3%}")
            
            yld = sugg_data.get('forward_dividend_yield', 0)
            self.ops_suggestions['dividend'].config(text=f"Yahoo: Fwd Yield={yld or 0:.2%}")
            
            self.comprehensive_text.insert(tk.END, f"✅ Yahoo Finance suggestions loaded.\n")
        except Exception as e:
            self.comprehensive_text.insert(tk.END, f"❌ Error loading Yahoo data: {e}\n")
            
        self.root.update_idletasks()

        # 2. Fetch AI Suggestions
        if self.comprehensive_analyzer and self.comprehensive_analyzer.use_ai:
            try:
                self.comprehensive_text.insert(tk.END, f"🤖 Requesting AI suggestions... (this may take a moment)\n")
                self.root.update_idletasks()
                
                ai_suggs = self.comprehensive_analyzer.get_ai_suggestions(main_ticker)
                if 'error' in ai_suggs:
                    raise Exception(ai_suggs['error'])
                
                for category, questions in self.qual_suggestions.items():
                    for q_key, label in questions.items():
                        sugg = ai_suggs.get(q_key, "AI suggestion not found.")
                        label.config(text=f"AI: {sugg}")

                self.comprehensive_text.insert(tk.END, f"✅ AI suggestions loaded.\n")
            except Exception as e:
                self.comprehensive_text.insert(tk.END, f"❌ Error loading AI data: {e}\n")
        else:
            self.comprehensive_text.insert(tk.END, f"ℹ️ AI not configured. Skipping AI suggestions.\n")

    def _calculate_comprehensive_score(self):
        """Gathers all user inputs and calculates the final score"""
        if not self.comprehensive_analyzer:
            messagebox.showerror("Error", "Analyzer not initialized. Try setting API key or loading suggestions.")
            return

        main_ticker = self.main_ticker_entry.get().strip().upper()
        if not main_ticker:
            messagebox.showerror("Error", "Please enter a main ticker.")
            return

        user_inputs = {
            'quant_metrics': {}, 'qual_scores': {}, 'val_inputs': {},
            'ops_scores': {}, 'industry_avgs': {}
        }

        try:
            # Helper to safely get float from entry
            def get_float(entry_widget):
                val_str = entry_widget.get().strip()
                if not val_str: return 0.0
                return float(val_str)
            
            def get_int(entry_widget):
                val_str = entry_widget.get().strip()
                if not val_str: return 0
                return int(val_str)

            # 1. Gather Quant Metrics
            for key, entry in self.quant_entries.items():
                if key == 'margin_expanding':
                    user_inputs['quant_metrics'][key] = entry.get()
                else:
                    user_inputs['quant_metrics'][key] = get_float(entry)
            
            for key, entry in self.quant_industry_entries.items():
                user_inputs['industry_avgs'][key] = get_float(entry)

            # 2. Gather Qual Scores
            for category, questions in self.qual_entries.items():
                user_inputs['qual_scores'][category] = {}
                for q_key, entry in questions.items():
                    user_inputs['qual_scores'][category][q_key] = get_int(entry) # Scores are 1-10

            # 3. Gather Val Inputs
            for key, entry in self.val_entries.items():
                user_inputs['val_inputs'][key] = get_float(entry)
                
            # 4. Gather Ops Scores
            for key, entry in self.ops_entries.items():
                user_inputs['ops_scores'][key] = get_int(entry) # Scores are 1-10

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid number entered. Please check all fields.\n{e}")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to gather inputs: {e}")
            return

        # Run final calculation
        self.comprehensive_text.delete(1.0, tk.END)
        self.comprehensive_text.insert(tk.END, f"🧮 Calculating final score for {main_ticker}...\n\n")
        self.root.update_idletasks()

        try:
            # Need suggestion data for calculation
            sugg_data = self.comprehensive_analyzer.get_suggestion_data(main_ticker)
            if 'error' in sugg_data:
                raise Exception(f"Failed to get suggestion data for calculation: {sugg_data['error']}")
            
            result = self.comprehensive_analyzer.run_final_evaluation(user_inputs, sugg_data)
            
            if result['success']:
                self.comprehensive_text.delete(1.0, tk.END)
                self.comprehensive_text.insert(tk.END, result['report'])
                messagebox.showinfo("Success", "Comprehensive evaluation completed successfully!")
            else:
                self.comprehensive_text.delete(1.0, tk.END)
                self.comprehensive_text.insert(tk.END, f"❌ ERROR\n\n{result.get('error')}")
                messagebox.showerror("Error", f"Evaluation failed: {result.get('error')}")

        except Exception as e:
            import traceback
            self.comprehensive_text.delete(1.0, tk.END)
            self.comprehensive_text.insert(tk.END, f"❌ UNEXPECTED ERROR\n\n{str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # --- KEPT ORIGINAL FUNCTIONS ---

    def _fetch_and_process_single_stock(self, ticker):
        """Fetches and processes data for a single stock (from original file)"""
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
            data['ebitda'] = self.processor.get_ebitda(data['annual_income'])
        return data

    def _plot_data(self, main_ticker, competitor_tickers):
        """Plots stock comparison chart (from original file)"""
        plot_dfs = {}
        if main_ticker in self.fetched_data:
            plot_dfs[main_ticker] = self.fetched_data[main_ticker]['processed_history']
        for ticker in competitor_tickers:
            if ticker in self.fetched_data:
                plot_dfs[ticker] = self.fetched_data[ticker]['processed_history']
        self.visualizer.plot_multi_stock_comparison(plot_dfs, title="Closing Price Comparison (5 Years)")

    def _display_financials(self, main_ticker):
        """Displays financial statements (from original file)"""
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
        """Calculates and displays DCF and Relative Valuation (from original file)"""
        self.valuation_output.delete(1.0, tk.END)
        main_data = self.fetched_data[main_ticker]
        output = f"--- {main_ticker} Valuation Analysis ---\n\n"

        # --- DCF Valuation ---
        output += "### 1. Discounted Cash Flow (DCF) Valuation ###\n"
        dcf_value = None # To pass to auto-populator
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
        avg_valuation = None # To pass to auto-populator
        try:
            pe_list, ps_list, pb_list, ev_ebitda_list = [], [], [], []
            
            for ticker in competitor_tickers:
                if ticker in self.fetched_data:
                    info = self.fetched_data[ticker]['info']
                    comp_data = self.fetched_data[ticker]
                    
                    if info.get('trailingPE'): pe_list.append(info['trailingPE'])
                    if info.get('priceToSalesTrailing12Months'): ps_list.append(info['priceToSalesTrailing12Months'])
                    if info.get('priceToBook'): pb_list.append(info['priceToBook'])
                    
                    comp_market_cap = info.get('marketCap')
                    comp_total_debt = comp_data['total_debt'].iloc[-1] if not comp_data['total_debt'].empty else None
                    comp_cash = comp_data['cash_equivalents'].iloc[-1] if not comp_data['cash_equivalents'].empty else None
                    comp_ebitda = comp_data['ebitda'].iloc[-1] if not comp_data['ebitda'].empty else None
                    
                    if all([comp_market_cap, comp_total_debt is not None, comp_cash is not None, 
                           comp_ebitda is not None, comp_ebitda > 0]):
                        comp_ev = comp_market_cap + comp_total_debt - comp_cash
                        comp_ev_ebitda = comp_ev / comp_ebitda
                        ev_ebitda_list.append(comp_ev_ebitda)
            
            avg_pe = np.mean(pe_list) if pe_list else None
            avg_ps = np.mean(ps_list) if ps_list else None
            avg_pb = np.mean(pb_list) if pb_list else None
            avg_ev_ebitda = np.mean(ev_ebitda_list) if ev_ebitda_list else None
            
            output += f"  - Competitor Average Multiples:\n"
            output += f"      P/E={avg_pe or 'N/A':.2f}, P/S={avg_ps or 'N/A':.2f}, P/B={avg_pb or 'N/A':.2f}, EV/EBITDA={avg_ev_ebitda or 'N/A':.2f}\n"
            
            target_eps = main_data['info'].get('trailingEps')
            total_revenue = main_data['annual_income'].loc['Total Revenue'].iloc[-1] if 'Total Revenue' in main_data['annual_income'].index else None
            total_equity = main_data['total_equity'].iloc[-1] if not main_data['total_equity'].empty else None
            shares_outstanding = main_data['info'].get('sharesOutstanding')
            target_ebitda = main_data['ebitda'].iloc[-1] if not main_data['ebitda'].empty else None
            total_debt = main_data['total_debt'].iloc[-1] if not main_data['total_debt'].empty else None
            cash_equivalents = main_data['cash_equivalents'].iloc[-1] if not main_data['cash_equivalents'].empty else None
            
            target_sps = total_revenue / shares_outstanding if total_revenue and shares_outstanding else None
            target_bps = total_equity / shares_outstanding if total_equity and shares_outstanding else None
            
            relative_values = self.valuation_model.relative_valuation(
                target_eps, target_sps, target_bps, avg_pe, avg_ps, avg_pb
            )
            
            if avg_ev_ebitda and target_ebitda and total_debt is not None and cash_equivalents is not None:
                ev_ebitda_value = self.valuation_model.ev_ebitda_valuation(
                    target_ebitda, avg_ev_ebitda, total_debt, cash_equivalents, shares_outstanding
                )
                if ev_ebitda_value:
                    relative_values['EV/EBITDA Multiple'] = ev_ebitda_value
            
            if not relative_values:
                output += "  - Insufficient key data for relative valuation.\n"
            else:
                output += "\n  Relative Valuation Results:\n"
                for method, value in relative_values.items(): 
                    output += f"  >>> Valuation based on {method}: ${value:.2f}\n"
                
                avg_valuation = np.mean(list(relative_values.values()))
                output += f"\n  >>> Average Valuation: ${avg_valuation:.2f}\n"
                
        except Exception as e:
            output += f"  - Relative valuation calculation error: {e}\n"

        current_price = main_data['history']['Close'].iloc[-1]
        output += f"\n### Current Market Price ###\n  - {main_ticker} Current Stock Price: ${current_price:.2f}\n"
        self.valuation_output.insert(tk.END, output)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()