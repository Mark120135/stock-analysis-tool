"""
Comprehensive Stock Evaluation System
Based on the refined Stock Evaluation Criteria with:
- Quantitative Factors (23% weight)
- Qualitative Factors (27% weight)
- Valuation Factors (40% weight)
- Operational & Practical Factors (10% weight)
- Justification Credits (cross-check bonuses)
"""

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import json # Added for the new AI guidance function


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

# --- ADDED: Static data for the Qualitative Guidance tab ---
QUALITATIVE_QUESTIONS_GUIDANCE = {
    'governance': [
        { 'key': 'board_structure', 'question': 'Is the board structure sound (e.g., independent directors > 50%)?', 'guidance': 'Strong = Yes, diverse/independent. Adequate = Partial. Weak = No, insider-dominated.'},
        { 'key': 'management_integrity', 'question': 'Does management demonstrate integrity and long-term vision (e.g., aligned incentives)?', 'guidance': 'Strong = High insider ownership, clear strategy. Adequate = Average. Weak = Misalignment.'},
        { 'key': 'scandal_history', 'question': 'Has the company been involved in fraud or scandals?', 'guidance': 'Strong = None in 5+ years. Adequate = Minor/resolved. Weak = Recent/major.'}
    ],
    'business_model': [
        { 'key': 'product_uniqueness', 'question': 'Do products/services have uniqueness (e.g., patents, brand)?', 'guidance': 'Strong = Clear differentiators. Adequate = Some. Weak = Commoditized.'},
        { 'key': 'competitive_moat', 'question': 'Does it benefit from moats (network effects, switching costs)?', 'guidance': 'Strong = Multiple durable moats. Adequate = One/temporary. Weak = None.'}
    ],
    'industry_outlook': [
        { 'key': 'industry_growth', 'question': 'Is the industry in a growth phase (e.g., >10% CAGR)?', 'guidance': 'Strong = High growth. Adequate = Stable. Weak = Declining.'},
        { 'key': 'market_position', 'question': 'Does the company hold a strong competitive position in its industry (e.g., market share >15-20% depending on sector)?', 'guidance': 'Strong = Leader/dominant. Adequate = Mid-tier. Weak = Laggard.'},
        { 'key': 'global_presence', 'question': 'Does it have global presence or regional dominance?', 'guidance': 'Strong = Global scale. Adequate = Regional. Weak = Local/niche.'}
    ],
    'innovation': [
        { 'key': 'rd_investment', 'question': 'Is the company investing in R&D (e.g., >5% of revenue)?', 'guidance': 'Strong = Consistent/high. Adequate = Moderate. Weak = Low/declining.'},
        { 'key': 'innovation_revenue', 'question': 'Does the company generate significant revenue from recent innovations (e.g., new products or services launched in the last 3-5 years)?', 'guidance': 'Strong = Proven track record. Adequate = Emerging. Weak = None.'},
        { 'key': 'product_launches', 'question': 'Can it launch new products in changing markets?', 'guidance': 'Strong = Agile/innovative. Adequate = Reactive. Weak = Stagnant.'}
    ],
    'esg': [
        { 'key': 'environmental', 'question': 'Environmental: Does it manage sustainability (e.g., low carbon footprint)?', 'guidance': 'Strong = Leading practices (e.g., net-zero goals). Adequate = Compliant. Weak = High risks/polluter.'},
        { 'key': 'social', 'question': 'Social: Does the company demonstrate strong social responsibility in labor practices, diversity, and community engagement?', 'guidance': 'Strong = Positive impact (e.g., high employee satisfaction). Adequate = Neutral. Weak = Controversies.'},
        { 'key': 'governance_esg', 'question': 'Governance: Does the company demonstrate advanced governance practices through transparency and robust ESG policies (e.g., ethical policies)?', 'guidance': 'Strong = Transparent/ESG-integrated. Adequate = Standard. Weak = Issues.'}
    ],
    'macro': [
        { 'key': 'macro_resilience', 'question': 'Is the stock resilient to macro factors (e.g., rates, inflation)?', 'guidance': 'Strong = Low sensitivity. Adequate = Moderate. Weak = High vulnerability.'}
    ]
}
# --- END OF ADDED DICTIONARY ---

# --- ADDED: Static data for the Operational Guidance tab ---
OPERATIONAL_QUESTIONS_GUIDANCE = {
    'liquidity': [
        { 'key': 'liquidity', 'question': 'Liquidity & Trading: Is the stock highly liquid?', 'guidance': 'Strong (8-10) = High liquidity (e.g., Vol > 500k, Spread < 0.1%). Adequate (4-7) = Moderate. Weak (1-3) = Illiquid.'}
    ],
    'tax_regulatory': [
        { 'key': 'tax', 'question': 'Tax & Regulatory: Are tax implications and regulatory risks favorable?', 'guidance': 'Strong (8-10) = Favorable. Adequate (4-7) = Neutral. Weak (1-3) = High risks/unfavorable.'}
    ],
    'dividend': [
        { 'key': 'dividend', 'question': 'Dividend Yield: Is the dividend yield attractive and sustainable?', 'guidance': 'Strong (8-10) = >2.5%. Adequate (4-7) = 1.0-2.5%. Weak (1-3) = <1.0% or unsustainable.'}
    ],
    'portfolio_fit': [
        { 'key': 'portfolio_fit', 'question': 'Portfolio Fit: Does this stock align with your personal goals and diversification?', 'guidance': 'Strong (8-10) = Core fit. Adequate (4-7) = Satellite. Weak (1-3) = Mismatch.'}
    ]
}
# --- END OF ADDED DICTIONARY ---

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


class StockEvaluator:
    """
    Comprehensive stock evaluation system implementing the refined criteria
    """

    def __init__(self, ticker: str, industry_avg_data: Optional[Dict] = None,
                 api_key: Optional[str] = None, provider: str = "gemini"):
        """
        Initializes the evaluator with ticker symbol and optional industry/API data.

        Args:
            ticker (str): The stock ticker symbol.
            industry_avg_data (Optional[Dict]): Dictionary containing industry average metrics.
            api_key (Optional[str]): API key for the selected AI provider.
            provider (str): AI provider name ('gemini' or 'openai'). Defaults to 'gemini'.
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = self._safe_get_info()
        self.financials = self._safe_get_financials()
        self.balance_sheet = self._safe_get_balance_sheet()
        self.cash_flow = self._safe_get_cash_flow()
        self.history = self._safe_get_history()
        self.industry_avg_data = industry_avg_data if industry_avg_data else {}

        self.api_key = api_key
        self.provider = provider
        self.use_ai = False
        self.model = None
        self.client = None
        self.model_name = "gpt-4-turbo-preview" # Default model for OpenAI

        if not self.info:
             print(f"⚠️ Warning: Could not fetch basic info for {ticker}. Some evaluations may fail.")

        if api_key:
            if self.provider == "gemini" and GENAI_AVAILABLE:
                try:
                    genai.configure(api_key=api_key)
                    # Use a powerful model for analysis
                    self.model = genai.GenerativeModel('gemini-2.5-flash') # Updated model
                    self.use_ai = True
                    print("✅ Gemini API Configured")
                except Exception as e:
                    print(f"⚠️ Failed to configure Gemini API: {e}")
                    self.use_ai = False
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                try:
                    self.client = OpenAI(api_key=api_key)
                    # Could add a check here to verify the model exists if needed
                    self.use_ai = True
                    print("✅ OpenAI API Configured")
                except Exception as e:
                    print(f"⚠️ Failed to configure OpenAI API: {e}")
                    self.use_ai = False
            else:
                print(f"⚠️ AI Provider '{provider}' selected but package not installed or supported.")
                self.use_ai = False # Explicitly set to false
        else:
             print("ℹ️ No API key provided. AI features disabled.")
             self.use_ai = False
    @staticmethod
    def _scale_score(value, min_val, max_val, min_score=1, max_score=10):
        """
        Scales a value from one range to another, clamping it within the bounds.
        
        Example: Scale a DCF upside of 15% (0.15)
        - min_val = -0.50 (for -50% upside)
        - max_val = 0.50 (for +50% upside)
        - This will map -50% to 1, +50% to 10, and 15% to 7.0
        """
        if value is None or not np.isfinite(value):
            return min_score  # Return worst score if data is bad
            
        # 1. Clamp the value to be within the min/max bounds
        clamped_val = max(min_val, min(value, max_val))
        
        # 2. Normalize the clamped value to a 0.0 - 1.0 range
        # Avoid division by zero if min_val == max_val
        if (max_val - min_val) == 0:
            return min_score
            
        normalized_val = (clamped_val - min_val) / (max_val - min_val)
        
        # 3. Scale the normalized value to the desired score range (1-10)
        score = min_score + (normalized_val * (max_score - min_score))
        
        return score
    # --- ADDED METHOD: get_ai_qualitative_guidance ---
    # --- START OF REPLACEMENT ---
    def get_ai_qualitative_guidance(self, ticker_override: Optional[str] = None) -> Dict:
        """
        Generates DETAILED text-based AI analysis by making SEPARATE
        calls for qualitative and operational factors and merging them.
        """
        ticker_to_use = ticker_override if ticker_override else self.ticker

        if not self.use_ai:
            return {'error': 'AI provider not configured or available.'}

        # Use existing info if available, otherwise fetch minimal needed
        temp_info = self.info
        if ticker_override and ticker_override != self.ticker:
             try:
                 temp_stock = yf.Ticker(ticker_override)
                 temp_info = temp_stock.info
                 if not temp_info: raise ValueError(f"Could not fetch info for {ticker_override}")
             except Exception as e:
                  return {'error': f'Failed to fetch info for {ticker_override}: {e}'}
        elif not temp_info: # If original info fetch failed
             return {'error': f'Stock info for {ticker_to_use} not available.'}

        # --- 1. Build Qualitative Prompt Data ---
        qual_question_list = ""
        for category, questions in QUALITATIVE_QUESTIONS_GUIDANCE.items():
            qual_question_list += f"\n## {category.replace('_', ' ').title()}\n"
            for q in questions:
                qual_question_list += f"- {q['key']}: {q['question']} (Guidance: {q['guidance']})\n"
        qual_expected_keys = {q['key'] for cat in QUALITATIVE_QUESTIONS_GUIDANCE.values() for q in cat}

        # --- 2. Build Operational Prompt Data ---
        ops_question_list = "\n## Operational Factors\n"
        for category, questions in OPERATIONAL_QUESTIONS_GUIDANCE.items():
            for q in questions:
                ops_question_list += f"- {q['key']}: {q['question']} (Guidance: {q['guidance']})\n"
        ops_expected_keys = {q['key'] for cat in OPERATIONAL_QUESTIONS_GUIDANCE.values() for q in cat}

        # --- 3. Run AI Calls ---
        try:
            print(f"Fetching Qualitative AI guidance for {ticker_to_use} (1/2)...")
            qual_results = self._fetch_ai_analysis_for_questions(
                ticker_to_use, temp_info, qual_question_list, qual_expected_keys
            )
            if 'error' in qual_results:
                print(f"Error in Qualitative call: {qual_results['error']}")
                return qual_results # Propagate error

            print(f"Fetching Operational AI guidance for {ticker_to_use} (2/2)...")
            ops_results = self._fetch_ai_analysis_for_questions(
                ticker_to_use, temp_info, ops_question_list, ops_expected_keys
            )
            if 'error' in ops_results:
                print(f"Error in Operational call: {ops_results['error']}")
                return ops_results # Propagate error

            # --- 4. Merge and Return ---
            print("AI guidance fetched successfully. Merging results.")
            qual_results.update(ops_results) # Merge ops results into qual results
            return qual_results

        except Exception as e:
            import traceback
            error_msg = f"AI guidance orchestration failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {'error': f"AI guidance orchestration failed: {e}"}
    # --- END OF REPLACEMENT ---
    # --- ADD THIS NEW HELPER FUNCTION ---
    def _fetch_ai_analysis_for_questions(self, ticker_to_use: str, temp_info: Dict, question_list_for_prompt: str, expected_keys: set) -> Dict:
        """
        Internal helper to run a single AI analysis call for a given set of questions.
        """
        num_questions = len(expected_keys)
        try:
            context = f"""
            Company: {temp_info.get('longName', ticker_to_use)} ({ticker_to_use})
            Sector: {temp_info.get('sector', 'N/A')}
            Industry: {temp_info.get('industry', 'N/A')}
            Description: {temp_info.get('longBusinessSummary', 'N/A')[:1500]}...

            Key Data (might be incomplete):
            - Beta: {temp_info.get('beta', 'N/A')}
            - auditRisk: {temp_info.get('auditRisk', 'N/A')}
            - boardRisk: {temp_info.get('boardRisk', 'N/A')}
            - compensationRisk: {temp_info.get('compensationRisk', 'N/A')}
            - overallRisk: {temp_info.get('overallRisk', 'N/A')}
            - governanceEpochDate: {datetime.fromtimestamp(temp_info['governanceEpochDate']).strftime('%Y-%m-%d') if temp_info.get('governanceEpochDate') else 'N/A'}
            - compensationAsOfEpochDate: {datetime.fromtimestamp(temp_info['compensationAsOfEpochDate']).strftime('%Y-%m-%d') if temp_info.get('compensationAsOfEpochDate') else 'N/A'}
            """

            prompt = f"""
            Act as a senior equity analyst providing guidance for evaluating {ticker_to_use}.
            Use the provided context and your internal knowledge base.

            Context:
            {context}

            For EACH of the following {num_questions} criteria:
            1.  Provide a detailed 6-8 sentence analysis explaining your reasoning and assessment relevant to the question and guidance.
            2.  Suggest a score ('suggested_score') from 1 (Weak) to 10 (Strong) based SOLELY on your analysis, reflecting the guidance definitions (Strong=8-10, Adequate=4-7, Weak=1-3).

            {question_list_for_prompt}

            VERY IMPORTANT: Return your response as a single, valid JSON object (an array of {num_questions} objects, NO introductory text or markdown formatting before or after).
            Use this EXACT format for each object in the array:
            {{
                "key": "criteria_key_name",
                "analysis": "Your detailed 6-8 sentence analysis here...",
                "suggested_score": <your_score_1_to_10>
            }}
            """

            analysis_text = ""
            max_retries = 3
            retry_delay = 2 # seconds

            for attempt in range(max_retries):
                try:
                    if self.provider == "gemini":
                        model_config = {"temperature": 0.3, "max_output_tokens": 8192, "response_mime_type": "application/json"} # Request JSON
                        response = self.model.generate_content(prompt, generation_config=model_config)
                        analysis_text = response.text
                    else: # openai
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=4096,
                            response_format={"type": "json_object"}
                        )
                        analysis_text = response.choices[0].message.content

                    # Attempt to parse
                    if analysis_text.strip().startswith("```json"):
                        analysis_text = analysis_text.strip()[7:-3].strip()
                    elif analysis_text.strip().startswith("```"):
                         analysis_text = analysis_text.strip()[3:-3].strip()

                    try:
                        parsed_json_list = json.loads(analysis_text)
                        if not isinstance(parsed_json_list, list): raise ValueError("Expected list")
                    except (json.JSONDecodeError, ValueError):
                         wrapper_obj = json.loads(analysis_text)
                         found_list = None
                         if isinstance(wrapper_obj, list): found_list = wrapper_obj
                         elif isinstance(wrapper_obj, dict):
                             for key, value in wrapper_obj.items():
                                 if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and 'key' in value[0]:
                                     found_list = value; break
                         if found_list is None: raise ValueError("Could not find list in AI response")
                         parsed_json_list = found_list

                    # Validation and conversion to dict
                    suggestions_dict = {}
                    received_keys = set()
                    if not isinstance(parsed_json_list, list): raise ValueError("AI response not list")

                    for item in parsed_json_list:
                         if not isinstance(item, dict): continue
                         key = item.get('key'); analysis = item.get('analysis'); score = item.get('suggested_score')
                         if key and analysis and score is not None:
                             if key in expected_keys:
                                suggestions_dict[key] = {'analysis': analysis, 'suggested_score': int(score)}
                                received_keys.add(key)
                             else: print(f"Warning: Unexpected key '{key}' from AI.")
                         else: print(f"Warning: Malformed item from AI: {item}")

                    missing_keys = expected_keys - received_keys
                    if missing_keys:
                        print(f"Warning: AI response missing keys: {missing_keys}")
                        for k in missing_keys: suggestions_dict[k] = {'analysis': 'AI analysis missing.', 'suggested_score': 5}

                    return suggestions_dict # Success!

                except Exception as inner_e:
                    print(f"Attempt {attempt + 1} failed for {self.provider}: {inner_e}")
                    if attempt + 1 == max_retries:
                         print("Max retries reached.")
                         raise
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

            return {'error': 'AI guidance failed after multiple retries.'}

        except Exception as e:
            import traceback
            error_msg = f"AI helper function failed: {str(e)}\n{traceback.format_exc()}\n\nAI Response was:\n{analysis_text[:500]}..."
            print(error_msg)
            return {'error': f"AI generation failed. Check logs. (Error: {str(e)})"}
    # --- END OF NEW HELPER FUNCTION ---


    # ==================== Data Fetching Helpers ====================
    def _safe_get_info(self) -> Optional[Dict]:
        """Safely fetches stock info."""
        try: return self.stock.info
        except Exception as e: print(f"Error fetching info for {self.ticker}: {e}"); return None

    def _safe_get_financials(self) -> Optional[pd.DataFrame]:
        """Safely fetches financials."""
        try: return self.stock.financials
        except Exception as e: print(f"Error fetching financials for {self.ticker}: {e}"); return None

    def _safe_get_balance_sheet(self) -> Optional[pd.DataFrame]:
        """Safely fetches balance sheet."""
        try: return self.stock.balance_sheet
        except Exception as e: print(f"Error fetching balance sheet for {self.ticker}: {e}"); return None

    def _safe_get_cash_flow(self) -> Optional[pd.DataFrame]:
        """Safely fetches cash flow."""
        try: return self.stock.cashflow
        except Exception as e: print(f"Error fetching cash flow for {self.ticker}: {e}"); return None

    def _safe_get_history(self, period="5y") -> Optional[pd.DataFrame]:
        """Safely fetches historical data."""
        try: return self.stock.history(period=period)
        except Exception as e: print(f"Error fetching history for {self.ticker}: {e}"); return None

    def _get_metric_avg(self, metric_name: str, years: int = 3) -> Optional[float]:
        """Calculates the average of a metric over recent years."""
        # NOTE: This function relies on self.info, self.financials etc.
        # It should be refactored to accept 'data' dict if used in complex flows
        data = None
        if metric_name in ['operatingMargins', 'profitMargins', 'returnOnEquity', 'returnOnAssets']:
            # These might be directly in info as recent values, or need calculation from financials
            if self.info and metric_name in self.info:
                return self.info[metric_name] # Use most recent if available
            # Fallback: Attempt calculation (example for profit margin)
            if metric_name == 'profitMargins' and self.financials is not None and 'Net Income' in self.financials.index and 'Total Revenue' in self.financials.index:
                net_income = self.financials.loc['Net Income'].iloc[:years]
                revenue = self.financials.loc['Total Revenue'].iloc[:years]
                if not revenue.empty and not net_income.empty:
                    margins = net_income / revenue
                    return margins.mean()
            return self.info.get(metric_name) # Return None if not found/calculable

        # Handle metrics potentially found in different statements
        elif metric_name == 'Current Ratio':
            if self.balance_sheet is not None and 'Total Current Assets' in self.balance_sheet.index and 'Total Current Liabilities' in self.balance_sheet.index:
                assets = self.balance_sheet.loc['Total Current Assets'].iloc[:years]
                liabilities = self.balance_sheet.loc['Total Current Liabilities'].iloc[:years]
                if not liabilities.empty and not liabilities.eq(0).any():
                     ratios = assets / liabilities
                     return ratios.mean()
            return self.info.get('currentRatio') # Fallback to info

        elif metric_name == 'Total Debt/Equity':
            if self.balance_sheet is not None and 'Total Liab' in self.balance_sheet.index and 'Total Stockholder Equity' in self.balance_sheet.index:
                 debt = self.balance_sheet.loc['Total Liab'].iloc[:years] # Using Total Liab as proxy for Debt
                 equity = self.balance_sheet.loc['Total Stockholder Equity'].iloc[:years]
                 if not equity.empty and not equity.eq(0).any():
                     ratios = debt / equity
                     return ratios.mean()
            return self.info.get('debtToEquity') # Fallback to info

        elif metric_name == 'Revenue Growth Rate (YoY)':
            if self.financials is not None and 'Total Revenue' in self.financials.index:
                revenue = self.financials.loc['Total Revenue'].iloc[:years+1] # Need one extra year for pct_change
                if len(revenue) > 1:
                     growth = revenue.pct_change().dropna()
                     return growth.mean()
            return self.info.get('revenueGrowth') # Fallback

        elif metric_name == 'EPS Growth Rate (YoY)':
             # Basic EPS is often available
             if self.financials is not None and 'Basic EPS' in self.financials.index:
                 eps = self.financials.loc['Basic EPS'].iloc[:years+1]
                 if len(eps) > 1:
                     # Handle potential zero/negative EPS before calculating growth
                     eps_clean = eps[eps > 0] # Only calculate growth on positive EPS? Or handle differently?
                     if len(eps_clean) > 1:
                          growth = eps_clean.pct_change().dropna()
                          # Be cautious of extreme outliers in growth rates
                          growth_capped = growth.clip(lower=-1, upper=5) # Cap growth rates
                          return growth_capped.mean()
             return self.info.get('earningsGrowth') # Fallback

        elif metric_name == 'Beta':
             return self.info.get('beta') # Usually in info

        elif metric_name == 'Volatility':
            # Calculate rolling standard deviation of daily returns
            if self.history is not None and not self.history.empty:
                 daily_returns = self.history['Close'].pct_change().dropna()
                 # Annualized volatility (std dev * sqrt(trading days))
                 volatility = daily_returns.std() * np.sqrt(252)
                 return volatility
            return None # Cannot calculate

        return None # Metric not found


    # ==================== Scoring Functions ====================
    # These functions implement the logic from the PDF tables
    # NOTE: These functions should ideally be refactored to take *all* data
    # from the 'data' dict passed in, not rely on self.info, self.financials etc.
    # For now, we only fix the argument mismatch.

    def _calculate_profitability_score(self, data: Dict, years: int = 3) -> Tuple[float, str]:
        """Calculates profitability score (1-10) and provides explanation."""
        # Uses _get_metric_avg, which relies on self.financials etc.
        margins_avg = data.get('avg_margins', self._get_metric_avg('operatingMargins', years)) # Use data dict first
        if margins_avg is None: margins_avg = self._get_metric_avg('profitMargins', years) # Fallback
        roe_avg = data.get('roe', self._get_metric_avg('returnOnEquity', years))
        roa_avg = data.get('roa', self._get_metric_avg('returnOnAssets', years))

        if margins_avg is None or roe_avg is None or roa_avg is None:
            return 0, "Missing profitability data"

        margin_score = 0
        if margins_avg > 0.30: margin_score = 10
        elif margins_avg >= 0.15: margin_score = 7
        else: margin_score = 3

        roe_roa_score = 0
        if roe_avg > 0.15 and roa_avg > 0.08: roe_roa_score = 10
        elif roe_avg >= 0.08 and roa_avg >= 0.04: roe_roa_score = 7
        elif roe_avg < 0.08 and roa_avg < 0.04: roe_roa_score = 3
        else: roe_roa_score = 5 # Mixed case

        score = (margin_score + roe_roa_score) / 2
        explanation = f"Margins ({margins_avg:.1%}) score={margin_score}, ROE/ROA ({roe_avg:.1%}/{roa_avg:.1%}) score={roe_roa_score} -> Avg={score:.2f}"
        return score, explanation

    # --- FIXED: Added industry_avg_data argument ---
    def _calculate_efficiency_score(self, data: Dict, industry_avg_data: Optional[Dict] = None, years: int = 3) -> Tuple[float, str]:
        """Calculates operational efficiency score (1-10) and explanation."""
        days_inv = data.get('days_inventory', -1) # Get from user input
        days_sales = data.get('days_sales', -1) # Get from user input

        # --- FIXED: Use passed industry_avg_data first, fallback to self.industry_avg_data ---
        industry_data = industry_avg_data if industry_avg_data is not None else self.industry_avg_data
        ind_avg_inv = industry_data.get('days_inventory', -1)
        ind_avg_dso = industry_data.get('days_sales', -1) # Key was 'daysSalesOutstanding' before, changed to 'days_sales' for consistency

        inv_score = 5 # Default if data missing
        inv_exp = "N/A"
        if days_inv != -1 and ind_avg_inv != -1:
             if days_inv < ind_avg_inv * 0.8: inv_score = 10
             elif days_inv <= ind_avg_inv * 1.2: inv_score = 7
             else: inv_score = 3
             inv_exp = f"Inv Days ({days_inv:.0f} vs Ind {ind_avg_inv:.0f})"

        dso_score = 5 # Default if data missing
        dso_exp = "N/A"
        if days_sales != -1 and ind_avg_dso != -1:
             if days_sales < ind_avg_dso * 0.8: dso_score = 10
             elif days_sales <= ind_avg_dso * 1.2: dso_score = 7
             else: dso_score = 3
             dso_exp = f"DSO ({days_sales:.0f} vs Ind {ind_avg_dso:.0f})"

        score = (inv_score + dso_score) / 2
        explanation = f"{inv_exp} score={inv_score}, {dso_exp} score={dso_score} -> Avg={score:.2f}"
        return score, explanation

    def _calculate_solvency_score(self, data: Dict, years: int = 3) -> Tuple[float, str]:
        """Calculates solvency score (1-10) and explanation."""
        current_ratio = data.get('current_ratio', self._get_metric_avg('Current Ratio', years))
        debt_equity = data.get('total_debt_equity', self._get_metric_avg('Total Debt/Equity', years))
        quick_ratio = data.get('quick_ratio', self.info.get('quickRatio')) # Often in info if available
        interest_coverage = data.get('interest_coverage', -1) # Get from user input

        scores = []
        exps = []

        if current_ratio is not None:
             if current_ratio > 2.0: scores.append(10)
             elif current_ratio >= 1.5: scores.append(7)
             else: scores.append(3)
             exps.append(f"Current ({current_ratio:.1f})={scores[-1]}")
        else: exps.append("Current=N/A")

        if quick_ratio is not None:
             if quick_ratio > 1.5: scores.append(10)
             elif quick_ratio >= 1.0: scores.append(7)
             else: scores.append(3)
             exps.append(f"Quick ({quick_ratio:.1f})={scores[-1]}")
        else: exps.append("Quick=N/A")

        int_cov_score = 5 # Default
        if interest_coverage != -1:
             if interest_coverage > 5: int_cov_score = 10
             elif interest_coverage >= 3: int_cov_score = 7
             else: int_cov_score = 3
             exps.append(f"IntCov ({interest_coverage:.1f}x)={int_cov_score}")
        else: exps.append(f"IntCov=N/A (User Input)")
        scores.append(int_cov_score)


        if debt_equity is not None:
             if debt_equity < 0.5: scores.append(10)
             elif debt_equity <= 1.0: scores.append(7)
             else: scores.append(3)
             exps.append(f"D/E ({debt_equity:.1f})={scores[-1]}")
        else: exps.append("D/E=N/A")

        score = np.mean(scores) if scores else 0
        explanation = f"{', '.join(exps)} -> Avg={score:.2f}"
        return score, explanation

    def _calculate_growth_score(self, data: Dict, years: int = 3) -> Tuple[float, str]:
        """Calculates growth potential score (1-10) and explanation."""
        rev_growth = data.get('revenue_growth_yoy', self._get_metric_avg('Revenue Growth Rate (YoY)', years))
        eps_growth = data.get('eps_growth_yoy', self._get_metric_avg('EPS Growth Rate (YoY)', years))

        scores = []
        exps = []

        if rev_growth is not None:
             if rev_growth > 0.15: scores.append(10)
             elif rev_growth >= 0.05: scores.append(7)
             else: scores.append(3)
             exps.append(f"Rev ({rev_growth:.1%})={scores[-1]}")
        else: exps.append("Rev=N/A")

        if eps_growth is not None:
             if eps_growth > 0.15: scores.append(10)
             elif eps_growth >= 0.05: scores.append(7)
             else: scores.append(3)
             exps.append(f"EPS ({eps_growth:.1%})={scores[-1]}")
        else: exps.append("EPS=N/A")

        score = np.mean(scores) if scores else 0
        explanation = f"{', '.join(exps)} -> Avg={score:.2f}"
        return score, explanation

    def _calculate_risk_score(self, data: Dict, years: int = 3) -> Tuple[float, str]:
        """Calculates risk score (1-10) and explanation."""
        beta = data.get('beta', self._get_metric_avg('Beta', years))
        volatility = data.get('volatility', self._get_metric_avg('Volatility', years))

        scores = []
        exps = []

        if beta is not None:
             if 0.8 <= beta <= 1.2: scores.append(10)
             elif beta < 0.8 or beta <= 1.5: scores.append(7)
             else: scores.append(3)
             exps.append(f"Beta ({beta:.2f})={scores[-1]}")
        else: exps.append("Beta=N/A")

        if volatility is not None:
             if volatility < 0.20: scores.append(10)
             elif volatility <= 0.30: scores.append(7)
             else: scores.append(3)
             exps.append(f"Vol ({volatility:.1%})={scores[-1]}")
        else: exps.append("Vol=N/A (User Input)")

        score = np.mean(scores) if scores else 0
        explanation = f"{', '.join(exps)} -> Avg={score:.2f}"
        return score, explanation

    # --- FIXED: Added industry_avg_data argument ---
    def _calculate_quantitative_score(self, data: Dict, industry_avg_data: Optional[Dict] = None, years: int = 3) -> Dict:
        """Calculates the overall quantitative score and sub-scores."""
        scores = {}
        explanations = {}

        scores['profitability'], explanations['profitability'] = self._calculate_profitability_score(data, years)
        # --- FIXED: Pass industry_avg_data to efficiency calc ---
        scores['efficiency'], explanations['efficiency'] = self._calculate_efficiency_score(data, industry_avg_data, years)
        scores['solvency'], explanations['solvency'] = self._calculate_solvency_score(data, years)
        scores['growth'], explanations['growth'] = self._calculate_growth_score(data, years)
        scores['risk'], explanations['risk'] = self._calculate_risk_score(data, years)

        total_score = np.mean(list(scores.values()))

        return {
            'total_score': total_score,
            'scores': scores,
            'explanations': explanations
        }


    def _calculate_qualitative_score(self, scores_dict: Dict) -> Dict:
        """
        Calculates qualitative score based on AI analysis or default scores.
        Input `scores_dict` should be structured like:
        {'governance': {'board_structure': 8, 'management_integrity': 7, ...}, 'business_model': {...}, ...}
        """
        all_scores = []
        category_scores = {}
        category_explanations = {} # Store AI analysis snippets
        num_questions = 0

        if not scores_dict: # If no manual scores provided
             # Return default mid-scores if no input possible
             print("Warning: No qualitative scores provided. Using default score 5.")
             default_score = 5.0
             categories = QUALITATIVE_QUESTIONS_GUIDANCE.keys()
             for cat, qs in QUALITATIVE_QUESTIONS_GUIDANCE.items():
                 q_count = len(qs)
                 category_scores[cat] = default_score
                 category_explanations[cat] = "Default score 5 (No Input)"
                 all_scores.extend([default_score] * q_count)
                 num_questions += q_count

        else: # If manual scores are provided (e.g., from Guidance tab push)
             print("Using provided qualitative scores.")
             for category, questions in QUALITATIVE_QUESTIONS_GUIDANCE.items():
                 # Get scores for questions in this category from the flat scores_dict
                 cat_q_scores = []
                 if category in scores_dict: # Check if scores_dict is nested
                     cat_q_scores = list(scores_dict[category].values())
                 else: # Assume scores_dict is flat {q_key: score}
                     cat_q_scores = [scores_dict.get(q['key'], 5) for q in questions if q['key'] in scores_dict]
                 
                 # Handle case where scores_dict is flat but only some keys present
                 if not cat_q_scores:
                     cat_q_scores = [scores_dict.get(q['key'], 5) for q in questions]


                 if cat_q_scores:
                      category_scores[category] = np.mean(cat_q_scores)
                      category_explanations[category] = f"Avg User Score: {category_scores[category]:.2f}"
                      all_scores.extend(cat_q_scores)
                      num_questions += len(cat_q_scores)
                 else:
                      category_scores[category] = 0
                      category_explanations[category] = "No User Scores"
        
        # This AI block is likely deprecated by the new UI flow, but kept for standalone
        if self.use_ai and not scores_dict: 
            print("Requesting AI for qualitative scoring...")
            # This call is problematic as it's not the new detailed guidance
            ai_qual_result = self._get_ai_qualitative_analysis() 
            if ai_qual_result and 'error' not in ai_qual_result:
                 for category, result in ai_qual_result.items():
                      score = result.get('score', 5) 
                      category_scores[category] = score
                      category_explanations[category] = result.get('analysis', 'AI analysis failed')[:200] + "..."
                      q_count = len(QUALITATIVE_QUESTIONS_GUIDANCE.get(category, []))
                      all_scores.extend([score] * q_count)
                 num_questions = len(all_scores)
            else:
                 print(f"AI qualitative analysis failed: {ai_qual_result.get('error', 'Unknown')}.")
                 # Fallback handled by the 'if not scores_dict' block


        # Normalize total score to 1-10
        raw_total_score = sum(all_scores)
        # Expecting 16 questions based on PDF
        expected_questions = 16 
        if num_questions == 0:
            normalized_score = 0 # Avoid division by zero
        elif num_questions != expected_questions:
            print(f"Warning: Expected {expected_questions} qualitative scores, found {num_questions}. Normalizing based on found count.")
            normalized_score = (raw_total_score / num_questions) 
        else: # num_questions == expected_questions
             normalized_score = raw_total_score / expected_questions
        

        return {
            'normalized_score': normalized_score,
            'raw_total': raw_total_score,
            'categories': category_scores,
            'explanations': category_explanations
        }

    # Placeholder for AI call - needs detailed implementation
    def _get_ai_qualitative_analysis(self) -> Dict:
        """Internal helper to call AI for qualitative factors. Requires detailed prompt."""
        # This is the OLD AI call, get_ai_qualitative_guidance is the new one
        if not self.use_ai: return {'error': 'AI not configured'}
        print("Warning: Using deprecated _get_ai_qualitative_analysis method.")
        # --- Build a detailed prompt asking for scores & analysis per category ---
        prompt = f"Analyze the qualitative factors for {self.ticker} ({self.info.get('longName', '')})..."
        prompt += "\nBased on Corporate Governance, Business Model & Moat, Industry Outlook, R&D/Innovation, ESG, and Macro Resilience..."
        prompt += "\nProvide a score (1-10) and brief analysis snippet for each category."
        prompt += "\nFormat as JSON: {'governance': {'score': 8, 'analysis': '...'}, 'business_model': {...}, ...}"

        try:
            if self.provider == "gemini":
                response = self.model.generate_content(prompt)
                try:
                    json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
                    ai_result = json.loads(json_str)
                    return ai_result
                except Exception as parse_e:
                    return {'error': f'Failed to parse Gemini response: {parse_e}. Response: {response.text[:200]}'}
            else: # openai
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} # Use JSON mode
                )
                ai_result = json.loads(response.choices[0].message.content)
                return ai_result
        except Exception as e:
            return {'error': f'AI call failed: {e}'}


    def _calculate_valuation_score(self, data: Dict) -> Dict:
        """Calculates valuation score (1-10) using continuous scaling."""
        dcf_value = data.get('dcf_value', 0) 
        relative_value = data.get('relative_value', 0) 
        current_price = data.get('current_price', self.history['Close'].iloc[-1] if self.history is not None and not self.history.empty else 0)

        scores = {}
        exps = {}

        # --- 1. DCF Scoring ---
        if dcf_value > 0 and current_price > 0:
            dcf_upside = (dcf_value - current_price) / current_price
            
            # === YOUR NEW DCF RANGE ===
            DCF_MIN_UPSIDE = -0.50  # -30% (Score 1)
            DCF_MAX_UPSIDE = 0.50   # +30% (Score 10)
            
            scores['dcf'] = self._scale_score(
                dcf_upside, 
                DCF_MIN_UPSIDE, 
                DCF_MAX_UPSIDE
            )
            exps['dcf'] = f"DCF Upside: {dcf_upside:.2%} -> Score: {scores['dcf']:.2f}"
        else:
            scores['dcf'] = 1.0  # Give bottom score if data is missing
            exps['dcf'] = "DCF=N/A (User Input)"

        # --- 2. Relative (Comparative) Scoring ---
        if relative_value > 0 and current_price > 0:
            rel_upside = (relative_value - current_price) / current_price
            
            # === YOUR NEW RELATIVE RANGE ===
            REL_MIN_UPSIDE = -0.50  # -30% (Score 1)
            REL_MAX_UPSIDE = 0.50   # +30% (Score 10)

            scores['relative'] = self._scale_score(
                rel_upside, 
                REL_MIN_UPSIDE, 
                REL_MAX_UPSIDE
            )
            exps['relative'] = f"Relative Upside: {rel_upside:.2%} -> Score: {scores['relative']:.2f}"
        else:
            scores['relative'] = 1.0 # Give bottom score if data is missing
            exps['relative'] = "Relative=N/A (User Input)"

        # --- 3. Final Score ---
        total_score = np.mean(list(scores.values())) if scores else 1.0

        return {
            'total_score': total_score,
            'scores': scores,
            'explanations': exps
        }
    def _calculate_operational_score(self, data: Dict) -> Dict:
        """
        Calculates operational factors score (1-10) using PUSHED user scores.
        """
        # Get user scores from the data dict passed from the UI. Use 5 as default.
        liq_score = data.get('user_liquidity_score', data.get('liquidity', 5))
        tax_score = data.get('user_tax_score', data.get('tax', 5))
        div_score = data.get('user_dividend_score', data.get('dividend', 5))
        fit_score = data.get('user_portfolio_fit_score', data.get('portfolio_fit', 5))

        scores = {
            'liquidity': liq_score,
            'tax': tax_score, # Key matches PDF and guidance
            'dividend': div_score,
            'portfolio_fit': fit_score
        }
        
        # Calculate the average score
        total_score = np.mean(list(scores.values()))
        
        explanations = {
            'liquidity': f"User Score -> {liq_score}",
            'tax': f"User Score -> {tax_score}",
            'dividend': f"User Score -> {div_score}",
            'portfolio_fit': f"User Score -> {fit_score}"
        }
        
        return {
            'total_score': total_score, 
            'scores': scores, 
            'explanations': explanations
        }

    def _calculate_justification_credits(self, quant_scores, qual_scores, val_scores, ops_scores, data) -> Dict:
         """Calculates justification credits (max 7.0)."""
         credits = {}
         total_credits = 0

         # Simplify access to sub-scores
         qs = quant_scores # Quantitative sub-scores dict
         ql_cat = qual_scores # Qualitative category scores dict
         vs = val_scores # Valuation sub-scores dict
         os = ops_scores # Operational sub-scores dict

         # Metrics needed
         pe = data.get('pe', self.info.get('trailingPE', 0) or 0) # Get from user input first
         peer_pe = data.get('industry_avgs', {}).get('pe', pe if pe > 0 else 1) # Get from user input
         rev_growth = data.get('revenue_growth_yoy', self._get_metric_avg('Revenue Growth Rate (YoY)', 3) or 0)
         eps_growth = data.get('eps_growth_yoy', self._get_metric_avg('EPS Growth Rate (YoY)', 3) or 0)
         roe = data.get('roe', self._get_metric_avg('returnOnEquity', 3) or 0)
         op_margin = data.get('avg_margins', self._get_metric_avg('operatingMargins', 3) or 0)
         debt_equity = data.get('total_debt_equity', self._get_metric_avg('Total Debt/Equity', 3) or 10) # Default high
         interest_coverage = data.get('interest_coverage', -1)
         interest_coverage_ok = interest_coverage > 3 # Check from user input
         beta = data.get('beta', self.info.get('beta', 1.0) or 1.0)
         volatility = data.get('volatility', self._get_metric_avg('Volatility', 3) or 0.5) # Default high
         margin_expanding = data.get('margin_expanding', False) # Get from user input
         div_yield = data.get('user_dividend_score', 5) # Use user score as proxy
         dcf_value = data.get('dcf_value', 0)
         current_price = data.get('current_price', 0)

         # 1. Moat-Premium
         if peer_pe > 0 and pe > (peer_pe * 1.2) and ql_cat.get('business_model', 0) >= 8:
             credits['Moat-Premium'] = 1.0

         # 2. Growth-Multiple
         if (rev_growth > 0.15 or eps_growth > 0.15) and peer_pe > 0 and pe > (peer_pe * 1.15):
             if ql_cat.get('industry_outlook', 0) >= 8 and ql_cat.get('innovation', 0) >= 8:
                 credits['Growth-Multiple'] = 0.8

         # 3. Profit-Gov
         if (roe > 0.15 or op_margin > 0.30) and ql_cat.get('governance', 0) >= 8:
             credits['Profit-Gov'] = 0.7

         # 4. Debt-Macro
         if (debt_equity > 0.5 or not interest_coverage_ok) and ql_cat.get('macro', 0) >= 8:
              credits['Debt-Macro'] = 0.6

         # 5. Vol-Innovate
         if (beta > 1.5 or (volatility > 0.30 and volatility < 10.0)) and ql_cat.get('innovation', 0) >= 8: # Add upper bound sanity check
             credits['Vol-Innovate'] = 0.7

         # 6. ESG-Margin
         if margin_expanding and ql_cat.get('esg', 0) >= 8:
             credits['ESG-Margin'] = 0.6

         # 7. Scale-Liquidity
         if os.get('liquidity', 10) < 5: # If operational liquidity score is weak
             if ql_cat.get('industry_outlook', 0) >= 9: # Approximation
                 credits['Scale-Liquidity'] = 0.5

         # 8. Reinvest-Yield
         if div_yield < 5: # Check dividend score
              if ql_cat.get('innovation', 0) >= 8 and eps_growth > 0.15 and roe > 0.15 and ql_cat.get('business_model', 0) >= 8:
                  credits['Reinvest-Yield'] = 0.7

         # 9. DCF-Conviction
         if dcf_value > 0 and current_price > 0 and (current_price < dcf_value * 0.8):
              high_qual_cats = sum(1 for cat_score in [
                   ql_cat.get('business_model', 0), ql_cat.get('governance', 0),
                   ql_cat.get('industry_outlook', 0), ql_cat.get('innovation', 0),
                   ql_cat.get('esg', 0)
              ] if cat_score >= 9)
              if high_qual_cats >= 3:
                  credits['DCF-Conviction'] = 0.8

         # 10. Fit-Override
         # Check if any quant sub-score is weak
         if any(score <= 3 for score in qs.values()):
              if os.get('portfolio_fit', 0) == 10:
                  credits['Fit-Override'] = 0.5


         total_credits = min(sum(credits.values()), 7.0) # Cap at 7.0

         return {
             'total_credits': total_credits,
             'details': credits
         }


    def _calculate_final_score_and_recommendation(self, quant_total, qual_norm, val_total, ops_total, credits_total) -> Dict:
         """Calculates the final weighted score and recommendation."""
         analysis_score_10 = (
             (quant_total * 0.23) +
             (qual_norm * 0.27) +
             (val_total * 0.40) +
             (ops_total * 0.10)
         )
         analysis_score_100 = analysis_score_10 * 10
         final_score = analysis_score_100 + credits_total

         if final_score >= 85: recommendation = "STRONG BUY"
         elif final_score >= 70: recommendation = "BUY / Core"
         elif final_score >= 55: recommendation = "HOLD / Satellite"
         else: recommendation = "AVOID"

         return {
             'final_score': final_score,
             'analysis_score_100': analysis_score_100,
             'recommendation': recommendation,
             'weights': {'quant': 0.23, 'qual': 0.27, 'val': 0.40, 'ops': 0.10}
         }

    # ==================== Main Evaluation Function ====================
    def evaluate_stock(self) -> Dict:
        """
        Runs the complete stock evaluation process using internal data.
        NOTE: This method is for standalone use and does NOT use external user inputs
        from the Streamlit UI. The Streamlit app calls the _calculate methods directly.
        """
        if not self.info: # Basic check if data loaded
             return {'success': False, 'error': f"Failed to load initial data for {self.ticker}"}

        # --- Prepare Data ---
        calc_data = {'currentPrice': self.info.get('currentPrice')}
        # Add other data from self.info that scoring functions might expect
        calc_data.update(self.info) 
        
        # User scores would ideally be passed into this function if using externally
        calc_data['user_tax_score'] = 5 # Default/Placeholder
        calc_data['user_portfolio_fit_score'] = 5 # Default/Placeholder
        calc_data['user_liquidity_score'] = 5 # Default/Placeholder
        calc_data['user_dividend_score'] = 5 # Default/Placeholder


        # --- Run Scoring ---
        try:
             # Pass self.industry_avg_data to quantitative
             quant_result = self._calculate_quantitative_score(calc_data, self.industry_avg_data) 
             
             # Pass empty dict to use AI/defaults, or pass actual scores if available
             # In standalone mode, we pass {} and let it use AI if configured
             qual_result = self._calculate_qualitative_score({}) 
             
             val_result = self._calculate_valuation_score(calc_data)
             ops_result = self._calculate_operational_score(calc_data)
             credits_result = self._calculate_justification_credits(
                  quant_result['scores'], qual_result['categories'],
                  val_result['scores'], ops_result['scores'], calc_data
             )
             final_result = self._calculate_final_score_and_recommendation(
                  quant_result['total_score'], qual_result['normalized_score'],
                  val_result['total_score'], ops_result['total_score'],
                  credits_result['total_credits']
             )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR DURING SCORING for {self.ticker}:\n{error_details}")
            return {'success': False, 'error': f"Scoring calculation failed: {e}"}


        # --- Generate Report ---
        report = self._generate_report(
             quant_result, qual_result, val_result, ops_result,
             credits_result, final_result, calc_data
        )

        return {
            'success': True,
            'report': report,
            **final_result # Include final score, recommendation etc. at top level
        }


    def _generate_report(self, quant, qual, val, ops, credits, final, data) -> str:
         """Generates a formatted text report of the evaluation."""
         report = f"""
{'='*60}
COMPREHENSIVE STOCK EVALUATION: {self.ticker} ({self.info.get('longName', '')})
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINAL SCORE: {final['final_score']:.2f} / 100
RECOMMENDATION: {final['recommendation']}
(Analysis Score: {final['analysis_score_100']:.2f} + Credits: {credits['total_credits']:.2f})

{'='*60}
SCORE BREAKDOWN (Weights: Quant={final['weights']['quant']:.0%}, Qual={final['weights']['qual']:.0%}, Val={final['weights']['val']:.0%}, Ops={final['weights']['ops']:.0%})
{'='*60}
1. Quantitative........: {quant['total_score']:.2f}/10 (Weight: {final['weights']['quant']:.0%})
   - Profitability.....: {quant['scores']['profitability']:.2f}/10 ({quant['explanations']['profitability']})
   - Efficiency........: {quant['scores']['efficiency']:.2f}/10 ({quant['explanations']['efficiency']})
   - Solvency..........: {quant['scores']['solvency']:.2f}/10 ({quant['explanations']['solvency']})
   - Growth............: {quant['scores']['growth']:.2f}/10 ({quant['explanations']['growth']})
   - Risk..............: {quant['scores']['risk']:.2f}/10 ({quant['explanations']['risk']})

2. Qualitative........: {qual['normalized_score']:.2f}/10 (Weight: {final['weights']['qual']:.0%}) (Raw Total: {qual['raw_total']:.1f})
"""
         if qual['categories']:
             for cat, score in qual['categories'].items():
                  exp = qual['explanations'].get(cat, '')
                  report += f"   - {cat.replace('_',' ').title():<18}: {score:.2f}/10 ({exp})\n"
         else:
             report += "   - No category scores calculated.\n"
         report += f"""
3. Valuation..........: {val['total_score']:.2f}/10 (Weight: {final['weights']['val']:.0%})
"""
         if val['scores']:
             report += f"   - DCF...............: {val['scores'].get('dcf', 0):.2f}/10 ({val['explanations'].get('dcf', 'N/A')})\n"
             report += f"   - Relative..........: {val['scores'].get('relative', 0):.2f}/10 ({val['explanations'].get('relative', 'N/A')})\n"
         else:
             report += "   - No valuation scores calculated.\n"
         
         report += f"""
4. Operational........: {ops['total_score']:.2f}/10 (Weight: {final['weights']['ops']:.0%})
"""
         if ops['scores']:
             report += f"   - Liquidity.........: {ops['scores']['liquidity']:.2f}/10 ({ops['explanations']['liquidity']})\n"
             report += f"   - Tax/Regulatory....: {ops['scores']['tax']:.2f}/10 ({ops['explanations']['tax']})\n"
             report += f"   - Dividend..........: {ops['scores']['dividend']:.2f}/10 ({ops['explanations']['dividend']})\n"
             report += f"   - Portfolio Fit.....: {ops['scores']['portfolio_fit']:.2f}/10 ({ops['explanations']['portfolio_fit']})\n"
         else:
             report += "   - No operational scores calculated.\n"

         report += f"""
{'='*60}
5. JUSTIFICATION CREDITS (Total: +{credits['total_credits']:.2f})
{'='*60}
"""
         if not credits['details']:
              report += "   No credits awarded.\n"
         else:
              for name, value in credits['details'].items():
                   report += f"   - {name}: +{value:.1f}\n"

         report += f"\n{'='*60}\nEND OF REPORT\n{'='*60}"
         return report


# ==================== Helper Wrapper for Streamlit/Tkinter ====================
# This class is not directly used by app.py or main_app.py, but was in the original file.
# Kept for completeness.
class ComprehensiveStockAnalyzer:
     """
     Simplified wrapper primarily focused on using StockEvaluator.
     This maintains compatibility with the UI code structure previously generated.
     """
     def __init__(self, api_key: Optional[str] = None, provider: str = "gemini"):
          self.api_key = api_key
          self.provider = provider
          self._evaluator_instance = None 

     def _get_evaluator(self, ticker: str) -> Optional[StockEvaluator]:
         """Gets or creates the StockEvaluator instance for a specific ticker."""
         if EVALUATOR_AVAILABLE:
              try:
                  instance = StockEvaluator(ticker=ticker, api_key=self.api_key, provider=self.provider)
                  if instance.info:
                      return instance
                  else:
                      print(f"Warning: Failed to initialize StockEvaluator fully for {ticker} (missing info).")
                      return instance

              except Exception as e:
                  print(f"Error creating StockEvaluator for {ticker}: {e}")
                  return None
         else:
              print("StockEvaluator class not available.")
              return None

     @property 
     def use_ai(self) -> bool:
         if self.api_key and EVALUATOR_AVAILABLE:
             temp_eval = StockEvaluator(ticker="AAPL", api_key=self.api_key, provider=self.provider) # Use dummy ticker
             return temp_eval.use_ai
         return False

     def get_suggestion_data(self, ticker: str) -> Dict:
         """Fetches Yahoo Finance suggestion data (delegates)."""
         evaluator = self._get_evaluator(ticker)
         if evaluator and evaluator.info:
             info = evaluator.info
             data = {}
             data['operating_margin'] = info.get('operatingMargins')
             data['profit_margin'] = info.get('profitMargins')
             data['roe'] = info.get('returnOnEquity')
             data['roa'] = info.get('returnOnAssets')
             data['current_ratio'] = info.get('currentRatio')
             data['total_debt_equity'] = info.get('debtToEquity')
             data['revenue_growth_yoy'] = info.get('revenueGrowth')
             data['eps_growth_yoy'] = info.get('earningsGrowth')
             data['beta'] = info.get('beta')
             data['volume'] = info.get('averageDailyVolume10Day') or info.get('averageVolume')
             data['bid'] = info.get('bid')
             data['ask'] = info.get('ask')
             try: data['bid_ask_spread_pct'] = (data['ask'] - data['bid']) / data['bid'] if data.get('bid') and data['bid'] > 0 else None
             except: data['bid_ask_spread_pct'] = None
             data['forward_dividend_yield'] = info.get('forwardAnnualDividendYield')
             # Clean None/NaN
             return {k: v if v is not None and (not isinstance(v, float) or not np.isnan(v)) else None for k, v in data.items()}
         elif evaluator:
             return {'error': f'Could not fetch info for {ticker} via StockEvaluator.'}
         else:
              return {'error': 'StockEvaluator unavailable.'}

     def get_ai_suggestions(self, ticker: str) -> Dict:
          """Generates SHORT AI hints (delegates)."""
          evaluator = self._get_evaluator(ticker)
          if evaluator and evaluator.use_ai:
              print("Warning: Using detailed guidance call for short hints - this is inefficient.")
              detailed_guidance = evaluator.get_ai_qualitative_guidance(ticker)
              if 'error' in detailed_guidance: return detailed_guidance
              short_hints = {}
              for key, data in detailed_guidance.items():
                   first_sentence = data['analysis'].split('.')[0] + '.' if '.' in data['analysis'] else data['analysis']
                   short_hints[key] = first_sentence
              return short_hints
          elif evaluator: return {'error': 'AI not configured in evaluator.'}
          else: return {'error': 'StockEvaluator unavailable.'}


     def get_ai_qualitative_guidance(self, ticker: str) -> Dict:
          """Generates DETAILED AI guidance (delegates)."""
          evaluator = self._get_evaluator(ticker)
          if evaluator:
              return evaluator.get_ai_qualitative_guidance(ticker) # Call the new method
          else:
              return {'error': 'StockEvaluator unavailable.'}

     def run_final_evaluation(self, user_inputs: Dict, suggestion_data: Dict) -> Dict:
          """Runs the final evaluation using StockEvaluator's logic."""
          ticker = user_inputs.get('ticker', None) # Assume ticker might be passed in user_inputs
          if not ticker: return {'success': False, 'error': 'Ticker missing in user_inputs'}

          evaluator = self._get_evaluator(ticker)
          if not evaluator: return {'success': False, 'error': 'StockEvaluator unavailable.'}
          
          print("Warning: Calling original evaluate_stock. Pushed user scores might be ignored unless evaluate_stock is modified.")

          try:
               pushed_scores = user_inputs.get('qual_scores', {}) 
               evaluation_result = evaluator.evaluate_stock() # Call original method
               
               if evaluation_result.get('success'):
                    pass

               return evaluation_result 

          except Exception as e:
               import traceback
               return {'success': False, 'error': f"Wrapper evaluation failed: {e}\n{traceback.format_exc()}"}


# ==================== Simplified Execution for Testing ====================
def run_evaluation_standalone(ticker, api_key=None, provider="gemini"):
    """Helper for standalone testing of the original StockEvaluator."""
    print(f"\n--- Running Standalone Evaluation for {ticker} ---")
    evaluator = StockEvaluator(ticker, api_key=api_key, provider=provider)
    result = evaluator.evaluate_stock()
    if result.get('success'):
        print(result.get('report', "No report generated."))
        print(f"Final Score: {result.get('final_score')}, Recommendation: {result.get('recommendation')}")
    else:
        print(f"Evaluation Failed: {result.get('error', 'Unknown error')}")

# ==================== Main Execution Block (Original) ====================

if __name__ == "__main__":
    print("="*90)
    print("🎯 Comprehensive Stock Evaluation System (Command Line)")
    print("="*90)

    ticker = input("\nEnter ticker symbol (e.g., AAPL): ").strip().upper()

    use_ai = input("Use AI for qualitative evaluation? (y/n) [y]: ").strip().lower() or 'y'
    use_ai = (use_ai == 'y')

    api_key = None
    provider = "gemini"
    if use_ai:
        available_providers = get_available_providers()
        if not available_providers:
             print("⚠️ No AI providers installed. Cannot use AI features.")
             use_ai = False
        else:
             provider_options = "/".join(available_providers)
             provider = input(f"AI provider ({provider_options}) [{available_providers[0]}]: ").strip().lower() or available_providers[0]
             if provider not in available_providers:
                  print(f"Invalid provider. Defaulting to {available_providers[0]}.")
                  provider = available_providers[0]
             api_key = input(f"Enter your {provider.upper()} API key: ").strip()
             if not api_key:
                  print("⚠️ No API key entered. Disabling AI features.")
                  use_ai = False

    print(f"\n🚀 Evaluating {ticker}...")

    evaluator_instance = StockEvaluator(
        ticker=ticker,
        api_key=api_key if use_ai else None,
        provider=provider
    )

    evaluation_result = evaluator_instance.evaluate_stock()

    if evaluation_result['success']:
        print("\n--- Evaluation Report ---")
        print(evaluation_result['report'])

        save = input("\n💾 Save report to file? (y/n) [n]: ").strip().lower() or 'n'
        if save == 'y':
            filename = f"{ticker}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(evaluation_result['report'])
                print(f"✅ Report saved as {filename}")
            except Exception as e:
                print(f"❌ Failed to save report: {e}")
    else:
        print("\n--- Evaluation Failed ---")
        print(f"Error: {evaluation_result.get('error', 'Unknown error')}")

    print("\n🏁 Evaluation finished.")