import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, Canvas, Frame, Scrollbar
import numpy as np
import yfinance as yf
from datetime import datetime
import threading # For running AI calls without blocking UI
import traceback # For error details

from data_fetcher import YahooFinanceDataFetcher
from data_processor import StockDataProcessor
from visualizer import StockVisualizer
from valuation_model import StockValuationModel

# Try to import AI analyzers AND the new guidance data
try:
    from stock_evaluator import ( StockEvaluator, get_available_providers, is_provider_available, QUALITATIVE_QUESTIONS_GUIDANCE )
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    print(f"Import Error: {e}"); EVALUATOR_AVAILABLE = False
    class StockEvaluator: pass; QUALITATIVE_QUESTIONS_GUIDANCE = {} # Placeholders


class StockAnalysisApp:
    def __init__(self, root):
        self.root = root; self.root.title("Stock Analysis Tool"); self.root.geometry("1200x850") # Slightly larger height
        self.fetched_data = {}; self.fetcher = YahooFinanceDataFetcher(); self.processor = StockDataProcessor()
        self.valuation_model = StockValuationModel(risk_free_rate=0.04, market_return=0.09)
        self.stock_evaluator_instance = None; self.selected_provider = tk.StringVar(value="gemini")

        # --- Widget Dictionaries ---
        # Guidance Tab
        self.qual_guidance_entries = {} # {q_key: widget}
        self.qual_guidance_ai_widgets = {} # {q_key: (score_label, text_widget)}
        # Final Result Tab (Now includes inputs again)
        self.final_quant_entries = {}
        self.final_quant_industry_entries = {}
        self.final_val_entries = {}
        self.final_ops_entries = {}
        # Internal storage
        self._pushed_scores_temp = {} # {q_key: score}

        self._create_widgets()

    def _create_widgets(self):
        # AI Config Frame
        api_frame = ttk.LabelFrame(self.root, text="🤖 AI Config", padding=5); api_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        # ... (API frame widgets layout - kept concise for brevity) ...
        provider_frame = ttk.Frame(api_frame); provider_frame.grid(row=0, column=0, columnspan=4, sticky="w") # Grid layout
        ttk.Label(provider_frame, text="Provider:").pack(side=tk.LEFT)
        self.gemini_radio=ttk.Radiobutton(provider_frame, text="G", variable=self.selected_provider, value="gemini", command=self._on_provider_change); self.gemini_radio.pack(side=tk.LEFT)
        self.openai_radio=ttk.Radiobutton(provider_frame, text="O", variable=self.selected_provider, value="openai", command=self._on_provider_change); self.openai_radio.pack(side=tk.LEFT)
        ttk.Label(api_frame, text="Key:", anchor='w').grid(row=1, column=0, sticky="w", padx=2)
        self.api_key_entry = ttk.Entry(api_frame, width=30, show="*"); self.api_key_entry.grid(row=1, column=1, sticky="ew", padx=2)
        self.set_api_button = ttk.Button(api_frame, text="Set & Init", command=self.set_api_key, width=8); self.set_api_button.grid(row=1, column=2, padx=2)
        self.api_status_label = ttk.Label(api_frame, text="Stat: None", foreground="red", width=15, anchor='w'); self.api_status_label.grid(row=1, column=3, sticky='w', padx=2)
        self.instruction_label = ttk.Label(api_frame, text="Key Instructions", foreground="blue", cursor="hand2", anchor='w'); self.instruction_label.grid(row=0, column=4, rowspan=2, sticky='w', padx=5) # Place instruction link
        api_frame.columnconfigure(1, weight=1) # Allow key entry to expand
        self._on_provider_change(); self._update_provider_availability()

        # Stock Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Stock Input", padding=5); input_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        # ... (Ticker, Competitors, Run button layout) ...
        ttk.Label(input_frame, text="Ticker:").pack(side=tk.LEFT, padx=2)
        self.main_ticker_entry = ttk.Entry(input_frame, width=8); self.main_ticker_entry.pack(side=tk.LEFT, padx=2); self.main_ticker_entry.insert(0, "AAPL")
        ttk.Label(input_frame, text="Competitors:").pack(side=tk.LEFT, padx=2)
        self.competitors_entry = ttk.Entry(input_frame, width=15); self.competitors_entry.pack(side=tk.LEFT, padx=2); self.competitors_entry.insert(0, "MSFT,GOOG")
        self.run_analysis_button = ttk.Button(input_frame, text="Run Basic Analysis", command=self.run_analysis); self.run_analysis_button.pack(side=tk.LEFT, padx=5)


        # Notebook
        self.notebook = ttk.Notebook(self.root); self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tabs (Chart, Financials, Valuation)
        self.plot_frame = ttk.Frame(self.notebook); self.notebook.add(self.plot_frame, text="📊 Charts")
        self.visualizer = StockVisualizer(master=self.plot_frame)
        self.financials_frame = ttk.Frame(self.notebook); self.notebook.add(self.financials_frame, text="📋 Financials")
        self.financials_text = scrolledtext.ScrolledText(self.financials_frame, height=15); self.financials_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.valuation_frame = ttk.Frame(self.notebook); self.notebook.add(self.valuation_frame, text="💰 Valuation")
        self.valuation_output = scrolledtext.ScrolledText(self.valuation_frame, height=15); self.valuation_output.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Qualitative Guidance Tab
        self.qual_guidance_frame = ttk.Frame(self.notebook); self.notebook.add(self.qual_guidance_frame, text="🤔 Qual Guidance")
        self._create_qualitative_guidance_tab()

        # Comprehensive Result Tab (with Inputs Restored)
        self.comprehensive_frame = ttk.Frame(self.notebook); self.notebook.add(self.comprehensive_frame, text="🎯 Comprehensive Result")
        self._create_comprehensive_result_tab_with_inputs() # Use new builder

    def _create_qualitative_guidance_tab(self):
        # ... (Guidance Tab UI - Kept as is from previous version) ...
        # Includes: Load AI Button, Push Scores Button, Status Label, Scrollable Frame
        # Inside scrollable frame: Loops through QUALITATIVE_QUESTIONS_GUIDANCE
        # Creates: Question Label, Guidance Label, Score Entry (self.qual_guidance_entries), AI Frame (self.qual_guidance_ai_widgets)
        button_frame = ttk.Frame(self.qual_guidance_frame); button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.load_ai_guidance_button = ttk.Button(button_frame, text="🤖 Load AI Guidance", command=self._load_ai_guidance, state="disabled"); self.load_ai_guidance_button.pack(side=tk.LEFT, padx=5)
        self.push_scores_button = ttk.Button(button_frame, text="➡️ Push Scores", command=self._push_qualitative_scores); self.push_scores_button.pack(side=tk.LEFT, padx=5)
        self.ai_guidance_status = ttk.Label(button_frame, text="Status: Idle", width=50); self.ai_guidance_status.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        main_frame = Frame(self.qual_guidance_frame); main_frame.pack(fill=tk.BOTH, expand=True)
        canvas = Canvas(main_frame); scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=10)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        row = 0
        if QUALITATIVE_QUESTIONS_GUIDANCE:
            # ... (Loop creating labels, entries, AI frames - code identical to previous version) ...
            pass # Placeholder
        else: ttk.Label(scrollable_frame, text="Error loading structure.").pack()


    # --- NEW: Builder for Final Tab WITH Inputs ---
    def _create_comprehensive_result_tab_with_inputs(self):
        """Builds UI for final tab, including Quant, Val, Ops inputs."""

        # --- Top Frame: Calculate Button ---
        top_frame = ttk.Frame(self.comprehensive_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.calculate_final_button = ttk.Button(top_frame, text="🧮 Calculate Final Score (using Inputs & Pushed Qual Scores)", command=self._calculate_final_comprehensive_score); self.calculate_final_button.pack(side=tk.LEFT, padx=5)
        # Optional: Add a status label here if needed

        # --- Main Scrollable Area ---
        main_frame = Frame(self.comprehensive_frame); main_frame.pack(fill=tk.BOTH, expand=True)
        canvas = Canvas(main_frame); scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=10)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")

        # --- Input Frames ---
        # Quantitative
        quant_frame = ttk.LabelFrame(scrollable_frame, text="1. Quantitative Factors (User Entry)", padding=10)
        quant_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')
        self._build_final_quant_frame(quant_frame) # Call dedicated builder

        # Valuation
        val_frame = ttk.LabelFrame(scrollable_frame, text="3. Valuation Factors (User Entry)", padding=10) # PDF Numbering
        val_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')
        self._build_final_val_frame(val_frame)

        # Operational
        ops_frame = ttk.LabelFrame(scrollable_frame, text="4. Operational Factors (User Score 1-10)", padding=10) # PDF Numbering
        ops_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')
        self._build_final_ops_frame(ops_frame)

        # --- Report Output Frame ---
        results_frame = ttk.LabelFrame(scrollable_frame, text="Final Comprehensive Report", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=10, anchor='n')
        self.comprehensive_report_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=25, font=("Courier", 9)); self.comprehensive_report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.comprehensive_report_text.insert(tk.END, "Enter Quant/Val/Ops data above, push Qual scores, then Calculate."); self.comprehensive_report_text.config(state="disabled")

    # --- NEW: Helper builders for final tab inputs ---
    def _build_final_quant_frame(self, parent):
        """Builds quantitative input widgets for the final tab."""
        # This structure mirrors the one from the fully refactored version
        # Uses self.final_quant_entries and self.final_quant_industry_entries
        self.final_quant_entries = {} # Reset specific dict for this frame
        self.final_quant_industry_entries = {}

        # Profitability
        ttk.Label(parent, text="Profitability", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky='w', pady=2)
        self._create_labeled_entry_final(parent, "Avg. Margins:", 1, self.final_quant_entries, 'avg_margins')
        self._create_labeled_entry_final(parent, "ROE:", 2, self.final_quant_entries, 'roe')
        self._create_labeled_entry_final(parent, "ROA:", 3, self.final_quant_entries, 'roa')
        # Operational Efficiency
        ttk.Label(parent, text="Op. Efficiency", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=3, sticky='w', pady=2)
        self._create_labeled_entry_final(parent, "Days Inv:", 5, self.final_quant_entries, 'days_inventory')
        self._create_labeled_entry_final(parent, "Days Sales:", 6, self.final_quant_entries, 'days_sales')
        ttk.Label(parent, text="Ind Avg Inv:").grid(row=5, column=3, sticky='w'); e_ind_inv = ttk.Entry(parent, width=8); e_ind_inv.grid(row=5, column=4); self.final_quant_industry_entries['days_inventory'] = e_ind_inv
        ttk.Label(parent, text="Ind Avg DSO:").grid(row=6, column=3, sticky='w'); e_ind_dso = ttk.Entry(parent, width=8); e_ind_dso.grid(row=6, column=4); self.final_quant_industry_entries['days_sales'] = e_ind_dso
        # Solvency
        ttk.Label(parent, text="Solvency", font=("Arial", 10, "bold")).grid(row=7, column=0, columnspan=3, sticky='w', pady=2)
        self._create_labeled_entry_final(parent, "Current R:", 8, self.final_quant_entries, 'current_ratio')
        self._create_labeled_entry_final(parent, "Quick R:", 9, self.final_quant_entries, 'quick_ratio')
        self._create_labeled_entry_final(parent, "Interest Cov:", 10, self.final_quant_entries, 'interest_coverage')
        self._create_labeled_entry_final(parent, "Debt/Equity:", 11, self.final_quant_entries, 'total_debt_equity')
        # Growth
        ttk.Label(parent, text="Growth", font=("Arial", 10, "bold")).grid(row=0, column=3, columnspan=3, sticky='w', pady=2, padx=10)
        self._create_labeled_entry_final(parent, "Rev Growth:", 1, self.final_quant_entries, 'revenue_growth_yoy', col_offset=3)
        self._create_labeled_entry_final(parent, "EPS Growth:", 2, self.final_quant_entries, 'eps_growth_yoy', col_offset=3)
        # Risk
        ttk.Label(parent, text="Risk", font=("Arial", 10, "bold")).grid(row=3, column=3, columnspan=3, sticky='w', pady=2, padx=10)
        self._create_labeled_entry_final(parent, "Beta:", 4, self.final_quant_entries, 'beta', col_offset=3)
        self._create_labeled_entry_final(parent, "Volatility:", 5, self.final_quant_entries, 'volatility', col_offset=3)
        # Credits Metrics
        ttk.Label(parent, text="For Credits", font=("Arial", 10, "bold")).grid(row=7, column=3, columnspan=3, sticky='w', pady=2, padx=10)
        self._create_labeled_entry_final(parent, "P/E:", 8, self.final_quant_entries, 'pe', col_offset=3)
        ttk.Label(parent, text="Ind Avg PE:").grid(row=8, column=5, sticky='w'); e_ind_pe = ttk.Entry(parent, width=8); e_ind_pe.grid(row=8, column=6); self.final_quant_industry_entries['pe'] = e_ind_pe
        self.final_quant_entries['margin_expanding'] = tk.BooleanVar() # Store BooleanVar
        ttk.Checkbutton(parent, text="Margins Expanding?", variable=self.final_quant_entries['margin_expanding']).grid(row=9, column=3, columnspan=2, sticky='w', padx=10)

    def _build_final_val_frame(self, parent):
        """Builds valuation input widgets for the final tab."""
        self.final_val_entries = {} # Reset specific dict
        self._create_labeled_entry_final(parent, "Current Price:", 0, self.final_val_entries, 'current_price')
        self._create_labeled_entry_final(parent, "DCF Value:", 1, self.final_val_entries, 'dcf_value')
        self._create_labeled_entry_final(parent, "Relative Value (Avg):", 2, self.final_val_entries, 'relative_value')
        # Add populate button
        ttk.Button(parent, text="Populate from Valuation Tab", command=self._populate_final_val_inputs).grid(row=1, column=3, padx=10)

    def _build_final_ops_frame(self, parent):
        """Builds operational input widgets for the final tab."""
        self.final_ops_entries = {} # Reset specific dict
        self._create_labeled_entry_final(parent, "Liquidity (1-10):", 0, self.final_ops_entries, 'liquidity')
        self._create_labeled_entry_final(parent, "Tax/Reg (1-10):", 1, self.final_ops_entries, 'tax')
        self._create_labeled_entry_final(parent, "Dividend (1-10):", 2, self.final_ops_entries, 'dividend')
        self._create_labeled_entry_final(parent, "Portfolio Fit (1-10):", 3, self.final_ops_entries, 'portfolio_fit')
        # Set defaults
        for key in ['liquidity', 'tax', 'dividend', 'portfolio_fit']:
             if key in self.final_ops_entries: self.final_ops_entries[key].insert(0, "5")

    # --- NEW: Helper for creating final tab entries ---
    def _create_labeled_entry_final(self, parent, text, row, var_dict, var_key, col_offset=0):
         """Helper to create labeled entry FOR FINAL TAB ONLY."""
         ttk.Label(parent, text=text).grid(row=row, column=col_offset, sticky='w', padx=5, pady=2)
         entry = ttk.Entry(parent, width=10) # Consistent width
         entry.grid(row=row, column=col_offset + 1, padx=5, pady=2)
         var_dict[var_key] = entry # Store the widget

    # --- NEW: Helper to populate final valuation inputs ---
    def _populate_final_val_inputs(self):
        """Populates final valuation inputs from valuation tab output."""
        try:
            val_text = self.valuation_output.get(1.0, tk.END) # Get text from valuation output widget
            price = dcf = rel = None
            for line in val_text.split('\n'):
                if "Current Stock Price:" in line: price = line.split('$')[-1].strip()
                elif "DCF Intrinsic Value per Share:" in line: dcf = line.split('$')[-1].strip()
                elif "Average Valuation:" in line: rel = line.split('$')[-1].strip() # Check if Average Val is outputted

            # Update final tab widgets
            if price and 'current_price' in self.final_val_entries: self.final_val_entries['current_price'].delete(0,tk.END); self.final_val_entries['current_price'].insert(0, price)
            if dcf and 'dcf_value' in self.final_val_entries: self.final_val_entries['dcf_value'].delete(0,tk.END); self.final_val_entries['dcf_value'].insert(0, dcf)
            if rel and 'relative_value' in self.final_val_entries: self.final_val_entries['relative_value'].delete(0,tk.END); self.final_val_entries['relative_value'].insert(0, rel)
            messagebox.showinfo("Populate", "Valuation fields populated.")
        except Exception as e: messagebox.showwarning("Populate Error", f"Could not parse values: {e}")

    # --- END NEW HELPERS ---

    def set_api_key(self):
        # ... (API key logic - kept as is) ...
        self._update_provider_availability() # Ensure AI button state is correct

    def run_analysis(self):
        # ... (Basic analysis logic - kept as is) ...
        # --- Clear relevant UI elements ---
        self.ai_guidance_status.config(text="Status: Idle", foreground='black')
        # ... (Clear guidance AI widgets) ...
        self.comprehensive_report_text.config(state="normal"); self.comprehensive_report_text.delete(1.0, tk.END); self.comprehensive_report_text.insert(tk.END, "Run analysis, use Guidance tab, then calculate."); self.comprehensive_report_text.config(state="disabled")
        self._pushed_scores_temp = {}
        # ... (Clear final input fields if desired, or leave populated) ...
        try:
             # ... (Fetch data) ...
             # ... (Populate basic tabs) ...
             # ... (Initialize evaluator if needed) ...
            messagebox.showinfo("Success", "Basic analysis complete.")
        except Exception as e: messagebox.showerror("Error", f"Analysis failed: {e}")


    def _load_ai_guidance(self):
        # ... (Load AI Guidance logic - kept as is, uses threading) ...
        pass

    def _fetch_guidance_thread(self, ticker):
        # ... (Worker thread logic - kept as is) ...
        pass

    def _update_guidance_ui_error(self, error_msg):
        # ... (Update UI on error - kept as is) ...
        pass

    def _update_guidance_ui_success(self, guidance_results):
        # ... (Update UI on success - kept as is) ...
        pass

    def _push_qualitative_scores(self):
        # ... (Push Scores logic - stores to self._pushed_scores_temp - kept as is) ...
        pass

    # --- UPDATED FINAL CALCULATION METHOD ---
    def _calculate_final_comprehensive_score(self):
        """Calculates the final score using evaluator methods, USER INPUTS, and pushed scores."""
        if not self.stock_evaluator_instance: messagebox.showerror("Error", "Evaluator not initialized."); return
        if not hasattr(self, '_pushed_scores_temp') or not self._pushed_scores_temp: messagebox.showerror("Error", "Push scores from Guidance tab first."); return
        main_ticker = self.main_ticker_entry.get().strip().upper();
        if not main_ticker: return

        # Re-initialize evaluator if ticker changed
        if self.stock_evaluator_instance.ticker != main_ticker or not self.stock_evaluator_instance.info:
            messagebox.showinfo("Info", f"Re-initializing evaluator for {main_ticker}...")
            self.set_api_key() # Re-init with current ticker and API key
            if not self.stock_evaluator_instance: return # Stop if re-init failed

        self.comprehensive_report_text.config(state="normal"); self.comprehensive_report_text.delete(1.0, tk.END)
        self.comprehensive_report_text.insert(tk.END, f"🧮 Calculating for {main_ticker} using inputs & stored scores...\n\n"); self.root.update_idletasks()

        try:
            evaluator = self.stock_evaluator_instance
            current_data = self.fetched_data.get(main_ticker, {}) # Use already fetched data

            # --- Gather USER INPUTS from final tab widgets ---
            user_quant_metrics = {}
            user_industry_avgs = {}
            user_val_inputs = {}
            user_ops_scores = {}

            # Helper to safely get float/int/bool from Entry/BooleanVar
            def get_widget_val(widget, dtype=float):
                try:
                    if isinstance(widget, tk.BooleanVar): return widget.get()
                    val_str = widget.get().strip()
                    if not val_str: return 0.0 if dtype == float else 0 # Default for empty
                    return dtype(val_str)
                except (ValueError, tk.TclError): # Handle invalid input or missing widget
                    return 0.0 if dtype == float else 0

            # Gather Quant
            for key, widget in self.final_quant_entries.items():
                user_quant_metrics[key] = get_widget_val(widget, dtype=bool if key=='margin_expanding' else float)
            for key, widget in self.final_quant_industry_entries.items():
                user_industry_avgs[key] = get_widget_val(widget, dtype=float)

            # Gather Valuation
            for key, widget in self.final_val_entries.items():
                user_val_inputs[key] = get_widget_val(widget, dtype=float)

            # Gather Operational
            for key, widget in self.final_ops_entries.items():
                user_ops_scores[key] = get_widget_val(widget, dtype=int) # Scores are int 1-10

            # --- Call internal scoring methods WITH USER INPUTS ---
            # Quantitative (Pass user inputs and industry averages)
            # Need to merge user metrics into a dict evaluator might expect
            eval_quant_input = current_data.copy()
            eval_quant_input.update(user_quant_metrics)
            quant_result = evaluator._calculate_quantitative_score(eval_quant_input, user_industry_avgs) # Pass industry avg dict

            # Qualitative (Use PUSHED SCORES)
            qual_scores_cat = {cat: {q['key']: self._pushed_scores_temp.get(q['key'], 5) for q in qs} for cat, qs in QUALITATIVE_QUESTIONS_GUIDANCE.items()}
            qual_result = evaluator._calculate_qualitative_score(qual_scores_cat)

            # Valuation (Pass user inputs)
            eval_val_input = current_data.copy() # Start with base data
            eval_val_input.update(user_val_inputs) # Add user inputs
            val_result = evaluator._calculate_valuation_score(eval_val_input)

            # Operational (Pass user inputs)
            eval_ops_input = current_data.copy()
            eval_ops_input['user_tax_score'] = user_ops_scores.get('tax', 5) # Update from user input
            eval_ops_input['user_portfolio_fit_score'] = user_ops_scores.get('portfolio_fit', 5)
            # Include liquidity/dividend user scores if the method uses them
            eval_ops_input['user_liquidity_score'] = user_ops_scores.get('liquidity', 5)
            eval_ops_input['user_dividend_score'] = user_ops_scores.get('dividend', 5)
            ops_result = evaluator._calculate_operational_score(eval_ops_input)

            # Credits (Pass necessary data and results)
            # Combine data for credits check
            credits_input_data = current_data.copy()
            credits_input_data.update(user_quant_metrics)
            credits_input_data.update(user_val_inputs)
            credits_result = evaluator._calculate_justification_credits(
                 quant_result.get('scores',{}), qual_result.get('categories',{}),
                 val_result.get('scores',{}), ops_result.get('scores',{}),
                 credits_input_data # Pass combined data
            )
            # Final Score
            final_result_data = evaluator._calculate_final_score_and_recommendation(
                 quant_result.get('total_score', 0), qual_result.get('normalized_score', 0),
                 val_result.get('total_score', 0), ops_result.get('total_score', 0),
                 credits_result.get('total_credits', 0)
            )
            # Report
            report = evaluator._generate_report(quant_result, qual_result, val_result, ops_result, credits_result, final_result_data, credits_input_data)

            # --- Display Results ---
            self.comprehensive_report_text.delete(1.0, tk.END); self.comprehensive_report_text.insert(tk.END, report); self.comprehensive_report_text.config(state="disabled")
            messagebox.showinfo("Success", "Final evaluation complete!")

        except AttributeError as ae: error_msg = f"Calc Error: Method missing? {ae}"; print(traceback.format_exc())
        except Exception as e: error_msg = f"Calculation Error: {e}"; print(traceback.format_exc())
        if 'error_msg' in locals():
             self.comprehensive_report_text.delete(1.0, tk.END); self.comprehensive_report_text.insert(tk.END, f"❌ ERROR:\n\n{error_msg}"); self.comprehensive_report_text.config(state="disabled")
             messagebox.showerror("Error", error_msg)


    # --- KEPT ORIGINAL HELPER/DISPLAY FUNCTIONS ---
    def _fetch_and_process_single_stock(self, ticker):
        # ... (Keep original implementation) ...
        pass
    def _plot_data(self, main_ticker, competitor_tickers):
        # ... (Keep original implementation) ...
        pass
    def _display_financials(self, main_ticker):
         # ... (Keep original implementation) ...
         pass
    def _calculate_and_display_valuation(self, main_ticker, competitor_tickers):
         # ... (Keep original implementation) ...
         pass
    # --- END KEPT FUNCTIONS ---

if __name__ == "__main__":
    root = tk.Tk(); style = ttk.Style(root)
    try: # Apply theme
        if "aqua" in style.theme_names(): style.theme_use("aqua")
        elif "vista" in style.theme_names(): style.theme_use("vista")
        else: style.theme_use('clam')
    except tk.TclError: style.theme_use('clam')
    app = StockAnalysisApp(root); root.mainloop()