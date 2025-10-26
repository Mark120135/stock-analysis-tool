"""
ETF Evaluation System
Based on the refined ETF Evaluation Criteria with:
- Quantitative Factors (30% weight)
- Qualitative Factors (30% weight)
- Valuation Factors (20% weight)
- Operational & Practical Factors (20% weight)
"""

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import json

# Optional AI imports (mirrors stock_evaluator)
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

# --- ETF Qualitative Guidance Data ---
ETF_QUALITATIVE_QUESTIONS_GUIDANCE = {
    'sector_outlook': [
        { 'key': 'sector_growth', 'question': 'Does the ETF target a sector or theme with strong long-term growth potential?', 'guidance': 'Strong (8-10) = Megatrends >10% CAGR. Adequate (4-7) = Moderate growth/cyclical. Weak (1-3) = Stagnant/mature.'},
        { 'key': 'sector_tailwinds', 'question': 'Are there tailwinds (e.g., policy support, innovation) supporting this sector?', 'guidance': 'Strong (8-10) = Clear catalysts (e.g., CHIPS Act). Adequate (4-7) = Neutral/mixed. Weak (1-3) = Headwinds.'}
    ],
    'index_holdings': [
        { 'key': 'index_methodology', 'question': 'Does the ETF\'s index methodology (e.g., market-cap vs. equal-weight) align with your goals?', 'guidance': 'Strong (8-10) = Aligned (e.g., market-cap for growth). Adequate (4-7) = Slight mismatch. Weak (1-3) = Misaligned.'},
        { 'key': 'diversification', 'question': 'Is the ETF appropriately diversified (e.g., number of holdings, top holding concentration)?', 'guidance': 'Strong (8-10) = 50-300 holdings, <30% in top 10. Adequate (4-7) = 30-50 holdings or 30-50% conc. Weak (1-3) = <30 holdings or >50% conc.'},
        { 'key': 'exposure', 'question': 'Does the ETF have global or sector-specific exposure suitable for your strategy?', 'guidance': 'Strong (8-10) = Balanced U.S./global. Adequate (4-7) = U.S.-heavy. Weak (1-3) = Overly narrow/single-country.'}
    ],
    'issuer_quality': [
        { 'key': 'issuer_reputation', 'question': 'Is the ETF managed by a reputable issuer with a strong track record?', 'guidance': 'Strong (8-10) = Reputable (Vanguard, BlackRock). Adequate (4-7) = Newer/growing. Weak (1-3) = Small/untested.'},
        { 'key': 'innovation_strategy', 'question': 'Does the ETF incorporate innovative strategies (e.g., smart-beta, ESG) without high costs?', 'guidance': 'Strong (8-10) = Innovative with <0.3% fee. Adequate (4-7) = Standard/reliable. Weak (1-3) = Gimmicky/high-fee.'}
    ],
    'sustainability': [
        { 'key': 'resilience', 'question': 'Is the ETF resilient to economic downturns or market cycles?', 'guidance': 'Strong (8-10) = <50% drawdown in past crashes. Adequate (4-7) = 50-70% drawdown. Weak (1-3) = High volatility/slow recovery.'},
        { 'key': 'geopolitical_risk', 'question': 'Are there minimal regulatory or geopolitical risks to the ETF\'s holdings?', 'guidance': 'Strong (8-10) = Low risk (diversified tech). Adequate (4-7) = Moderate (e.g., semis/China). Weak (1-3) = High risk.'}
    ]
}

# --- NEW: Static data for ETF Operational Guidance ---
ETF_OPERATIONAL_QUESTIONS_GUIDANCE = {
    'operational': [ # Group under a single category for simplicity
        { 'key': 'liquidity', 'question': 'Liquidity & Trading: Is the ETF highly liquid?', 'guidance': 'Strong (8-10) = High vol (>1M), tight spread (<0.1%). Adequate (4-7) = 500k-1M vol. Weak (1-3) = Low liquidity.'},
        { 'key': 'tax', 'question': 'Tax Efficiency: Is the stock tax-efficient with low regulatory risks?', 'guidance': 'Strong (8-10) = <10% turnover, minimal distributions. Adequate (4-7) = 10-20% turnover. Weak (1-3) = High turnover/distributions.'},
        { 'key': 'portfolio_fit', 'question': 'Portfolio Fit: Does this ETF align with your goals and risk tolerance?', 'guidance': 'Strong (8-10) = Clear fit (e.g., core holding). Adequate (4-7) = Partial fit. Weak (1-3) = Mismatch.'}
    ]
}
# --- END OF NEW DICTIONARY ---

def get_available_providers() -> List[str]:
    """Return list of available AI providers"""
    providers = []
    if GENAI_AVAILABLE: providers.append("gemini")
    if OPENAI_AVAILABLE: providers.append("openai")
    return providers

def is_provider_available(provider: str) -> bool:
    """Check if a specific provider is available"""
    if provider.lower() == "gemini": return GENAI_AVAILABLE
    elif provider.lower() == "openai": return OPENAI_AVAILABLE
    return False


class ETFEvaluator:
    """
    Comprehensive ETF evaluation system implementing the refined criteria
    """

    def __init__(self, ticker: str, api_key: Optional[str] = None, provider: str = "gemini"):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.info = self._safe_get_info()
        self.history = self._safe_get_history()

        self.api_key = api_key
        self.provider = provider
        self.use_ai = False
        self.model = None
        self.client = None
        self.model_name = "gpt-4-turbo-preview" # Ensure this is updated if needed

        if not self.info:
             print(f"⚠️ Warning: Could not fetch basic info for {ticker}. Some evaluations may fail.")

        if api_key:
            if self.provider == "gemini" and GENAI_AVAILABLE:
                try:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel('gemini-2.5-flash') # Use appropriate model
                    self.use_ai = True
                    print("✅ ETF Evaluator: Gemini API Configured")
                except Exception as e:
                    print(f"⚠️ Failed to configure Gemini API: {e}")
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                try:
                    self.client = OpenAI(api_key=api_key)
                    self.use_ai = True
                    print("✅ ETF Evaluator: OpenAI API Configured")
                except Exception as e:
                    print(f"⚠️ Failed to configure OpenAI API: {e}")
            else:
                print(f"⚠️ AI Provider '{provider}' selected but package not installed or supported.")
        else:
             print("ℹ️ No API key provided. AI features disabled.")

    # --- Data Fetching Helpers ---
    def _safe_get_info(self) -> Optional[Dict]:
        try: return self.stock.info
        except Exception as e: print(f"Error fetching info for {self.ticker}: {e}"); return None

    def _safe_get_history(self, period="10y") -> Optional[pd.DataFrame]:
        try: return self.stock.history(period=period)
        except Exception as e: print(f"Error fetching history for {self.ticker}: {e}"); return None

    # --- AI Guidance Function (MODIFIED) ---
    def get_ai_qualitative_guidance(self, ticker_override: Optional[str] = None) -> Dict:
        """
        Generates DETAILED text-based AI analysis for BOTH ETF qualitative AND operational factors.
        """
        ticker_to_use = ticker_override if ticker_override else self.ticker
        if not self.use_ai:
            return {'error': 'AI provider not configured or available.'}

        # Fetch info if needed
        temp_info = self.info
        if ticker_override and ticker_override != self.ticker:
             try:
                 temp_stock = yf.Ticker(ticker_override)
                 temp_info = temp_stock.info
                 if not temp_info: raise ValueError(f"Could not fetch info for {ticker_override}")
             except Exception as e:
                  return {'error': f'Failed to fetch info for {ticker_override}: {e}'}
        elif not temp_info:
             return {'error': f'ETF info for {ticker_to_use} not available.'}

        # --- Build COMBINED Prompt Data ---
        question_list = ""
        qual_expected_keys = set()
        for category, questions in ETF_QUALITATIVE_QUESTIONS_GUIDANCE.items():
            question_list += f"\n## {category.replace('_', ' ').title()}\n"
            for q in questions:
                question_list += f"- {q['key']}: {q['question']} (Guidance: {q['guidance']})\n"
                qual_expected_keys.add(q['key'])

        ops_expected_keys = set()
        question_list += f"\n## Operational Factors\n" # Add section header
        for category, questions in ETF_OPERATIONAL_QUESTIONS_GUIDANCE.items():
            for q in questions:
                question_list += f"- {q['key']}: {q['question']} (Guidance: {q['guidance']})\n"
                ops_expected_keys.add(q['key'])

        # Combine expected keys
        all_expected_keys = qual_expected_keys.union(ops_expected_keys)
        total_questions = len(all_expected_keys) # Should be 12

        # --- Run SINGLE AI Call ---
        try:
            print(f"Fetching ETF Qual & Ops AI guidance for {ticker_to_use} ({total_questions} items)...")
            results = self._fetch_ai_analysis_for_questions(
                ticker_to_use, temp_info, question_list, all_expected_keys
            )
            return results # This dict contains results for all keys
        except Exception as e:
            import traceback
            error_msg = f"AI guidance orchestration failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {'error': f"AI guidance orchestration failed: {e}"}

    # --- Helper Function (_fetch_ai_analysis_for_questions - UNCHANGED from previous version) ---
    # This function now takes the combined question list and expected keys
    def _fetch_ai_analysis_for_questions(self, ticker_to_use: str, temp_info: Dict, question_list_for_prompt: str, expected_keys: set) -> Dict:
        """
        Internal helper to run a single AI analysis call for a given set of questions.
        (Ensure this function exists and is correctly implemented as provided before)
        """
        num_questions = len(expected_keys)
        print(f"Calling AI helper for {num_questions} questions.")
        try:
            # Prepare context for ETF
            context = f"""
            ETF: {temp_info.get('longName', ticker_to_use)} ({ticker_to_use})
            Family: {temp_info.get('fundFamily', temp_info.get('family', 'N/A'))}
            Category: {temp_info.get('category', 'N/A')}
            Description: {temp_info.get('longBusinessSummary', 'N/A')[:1500]}...

            Key Data:
            - NAV: {temp_info.get('navPrice', 'N/A')}
            - Total Assets: {temp_info.get('totalAssets', 'N/A')}
            - Yield: {temp_info.get('yield', 'N/A')}
            - Beta (3Y): {temp_info.get('beta3Year', 'N/A')}
            - P/E Ratio: {temp_info.get('trailingPE', 'N/A')}
            - Expense Ratio: {temp_info.get('expenseRatio', 'N/A')}
            """

            # Update prompt to expect the correct number of questions
            prompt = f"""
            Act as a senior ETF analyst providing guidance for evaluating {ticker_to_use}.
            Use the provided context and your internal knowledge base.

            Context:
            {context}

            For EACH of the following {num_questions} criteria (covering qualitative and operational aspects):
            1.  Provide a detailed 6-8 sentence analysis explaining your reasoning.
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
                    print(f"AI Call Attempt {attempt + 1}/{max_retries}...")
                    if self.provider == "gemini":
                        model_config = {"temperature": 0.3, "max_output_tokens": 8192, "response_mime_type": "application/json"}
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

                    # --- Robust JSON Parsing ---
                    print("Attempting to parse AI response...")
                    # 1. Remove markdown backticks if present
                    if analysis_text.strip().startswith("```json"):
                        analysis_text = analysis_text.strip()[7:-3].strip()
                    elif analysis_text.strip().startswith("```"):
                         analysis_text = analysis_text.strip()[3:-3].strip()

                    # 2. Try direct parsing as list
                    try:
                        parsed_json_list = json.loads(analysis_text)
                        if not isinstance(parsed_json_list, list):
                             raise ValueError("Expected a list, got dict/other")
                        print("Successfully parsed as list.")
                    except (json.JSONDecodeError, ValueError) as e1:
                        print(f"Direct list parsing failed ({e1}). Trying to parse as object and find list...")
                        # 3. Try parsing as object and finding the list within it
                        try:
                            wrapper_obj = json.loads(analysis_text)
                            found_list = None
                            if isinstance(wrapper_obj, dict):
                                for key, value in wrapper_obj.items():
                                     # Check if value is a list of dicts with 'key'
                                     if (isinstance(value, list) and len(value) > 0 and
                                             isinstance(value[0], dict) and 'key' in value[0]):
                                         found_list = value
                                         print(f"Found list under key '{key}' in wrapper object.")
                                         break
                            if found_list is None:
                                 raise ValueError("Could not find a valid list within the JSON object")
                            parsed_json_list = found_list
                        except (json.JSONDecodeError, ValueError) as e2:
                            print(f"Parsing as object also failed ({e2}). Final attempt failed.")
                            raise ValueError(f"AI response is not valid JSON list or expected object structure. Response start: {analysis_text[:200]}...") # Raise final error

                    # --- Validation ---
                    print("Validating parsed JSON list...")
                    suggestions_dict = {}
                    received_keys = set()
                    if not isinstance(parsed_json_list, list):
                        raise ValueError("Parsed result is not a list")

                    for item_idx, item in enumerate(parsed_json_list):
                         if not isinstance(item, dict):
                              print(f"Warning: Item {item_idx} is not a dict: {item}")
                              continue
                         key = item.get('key'); analysis = item.get('analysis'); score_val = item.get('suggested_score')
                         score = None
                         # Try converting score robustly
                         try: score = int(score_val) if score_val is not None else None
                         except (ValueError, TypeError): score = None

                         if key and analysis and score is not None:
                             if key in expected_keys:
                                suggestions_dict[key] = {'analysis': str(analysis), 'suggested_score': score}
                                received_keys.add(key)
                             else: print(f"Warning: Unexpected key '{key}' from AI.")
                         else: print(f"Warning: Malformed item {item_idx} from AI (missing key/analysis/score): {item}")

                    missing_keys = expected_keys - received_keys
                    if missing_keys:
                        print(f"Warning: AI response missing keys: {missing_keys}")
                        for k in missing_keys: suggestions_dict[k] = {'analysis': 'AI analysis missing.', 'suggested_score': 5}

                    print(f"Successfully processed AI response. Found {len(received_keys)}/{len(expected_keys)} keys.")
                    return suggestions_dict # Success!

                except Exception as inner_e:
                    print(f"Attempt {attempt + 1} failed for {self.provider}: {inner_e}")
                    if attempt + 1 == max_retries:
                         print("Max retries reached.")
                         # Log the final failed response for debugging
                         print(f"Final failed AI response text:\n{analysis_text}")
                         raise # Reraise the last exception
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff

            # Should not be reached if max_retries exception is raised
            return {'error': 'AI guidance failed after multiple retries.'}

        except Exception as e:
            import traceback
            error_msg = f"AI helper function failed: {str(e)}\n{traceback.format_exc()}\n\nAI Response was:\n{analysis_text[:500]}..."
            print(error_msg)
            return {'error': f"AI generation failed. Check logs. (Error: {str(e)})"}
    # --- END OF HELPER ---


    # ==================== Scoring Functions ====================

    def _calculate_quantitative_score(self, data: Dict) -> Dict:
        """Calculates the overall quantitative score based on user inputs."""
        scores = {}
        explanations = {}

        # 1. Annualized Returns
        returns = data.get('annualized_return', 0)
        benchmark = data.get('benchmark_return', 0)
        if returns > (benchmark + 0.03): scores['returns'] = 10
        elif returns >= benchmark: scores['returns'] = 7
        else: scores['returns'] = 3
        explanations['returns'] = f"Return ({returns:.1%}) vs Benchmark ({benchmark:.1%})"

        # 2. Beta
        beta = data.get('beta', 1.0)
        if 1.0 <= beta <= 1.5: scores['beta'] = 10
        elif (0.8 <= beta < 1.0) or (1.5 < beta <= 2.0): scores['beta'] = 7
        else: scores['beta'] = 3
        explanations['beta'] = f"Beta ({beta:.2f})"

        # 3. Sharpe Ratio
        sharpe = data.get('sharpe_ratio', 0)
        if sharpe > 1.2: scores['sharpe'] = 10
        elif sharpe >= 0.8: scores['sharpe'] = 7
        else: scores['sharpe'] = 3
        explanations['sharpe'] = f"Sharpe Ratio ({sharpe:.2f})"

        # 4. Expense Ratio
        expense = data.get('expense_ratio', 1.0) # Assume high if missing
        if expense < 0.0015: scores['expense'] = 10 # < 0.15%
        elif expense <= 0.004: scores['expense'] = 7  # 0.15% - 0.4%
        else: scores['expense'] = 3                  # > 0.4%
        explanations['expense'] = f"Expense ({expense:.2%})"

        # 5. Tracking Error
        tracking_error = data.get('tracking_error', 1.0) # Assume high if missing
        if tracking_error < 0.003: scores['tracking_error'] = 10 # < 0.3%
        elif tracking_error <= 0.007: scores['tracking_error'] = 7 # 0.3% - 0.7%
        else: scores['tracking_error'] = 3                 # > 0.7%
        explanations['tracking_error'] = f"Tracking Error ({tracking_error:.2%})"

        total_score = np.mean(list(scores.values()))
        return {'total_score': total_score, 'scores': scores, 'explanations': explanations}

    def _calculate_qualitative_score(self, scores_dict: Dict) -> Dict:
        """Calculates qualitative score based on pushed scores (now includes Ops)."""
        all_scores = []
        category_scores = {}
        category_explanations = {}
        num_questions = 0

        # Process Qualitative Questions
        for category, questions in ETF_QUALITATIVE_QUESTIONS_GUIDANCE.items():
            cat_q_scores = [scores_dict.get(q['key'], 5) for q in questions] # Use pushed scores or default 5
            if cat_q_scores:
                category_scores[category] = np.mean(cat_q_scores)
                category_explanations[category] = f"Avg User Score: {category_scores[category]:.2f}"
                all_scores.extend(cat_q_scores)
                num_questions += len(cat_q_scores)

        # Process Operational Questions (as part of qualitative weight now)
        ops_scores_list = []
        for category, questions in ETF_OPERATIONAL_QUESTIONS_GUIDANCE.items():
             ops_scores_list = [scores_dict.get(q['key'], 5) for q in questions]
             # Store operational scores separately if needed later, but include in normalization
             all_scores.extend(ops_scores_list)
             num_questions += len(ops_scores_list)
             # Add operational explanation under a general 'Operational' key if desired
             if ops_scores_list:
                  category_scores['operational_qual'] = np.mean(ops_scores_list)
                  category_explanations['operational_qual'] = f"Avg User Op Score: {category_scores['operational_qual']:.2f}"


        # Normalize based on total questions (Qual + Ops for this calculation)
        expected_questions = 9 + 3 # 9 Qual + 3 Ops
        if num_questions == 0:
            normalized_score = 0
        elif num_questions != expected_questions:
             print(f"Warning: Expected {expected_questions} qualitative/operational scores, found {num_questions}. Normalizing.")
             normalized_score = (sum(all_scores) / num_questions) # Normalize based on actual count
        else:
            normalized_score = (sum(all_scores) / expected_questions) # Normalize to 1-10 scale

        return {
            'normalized_score': normalized_score,
            'raw_total': sum(all_scores),
            'categories': category_scores, # Contains both qual categories and 'operational_qual'
            'explanations': category_explanations
        }

    def _calculate_valuation_score(self, data: Dict) -> Dict:
        """Calculates valuation score (1-10). [cite: 100-115]"""
        scores = {}
        exps = {}

        # 1. P/E
        pe = data.get('forward_pe', 0)
        peer_pe = data.get('peer_avg_pe', 0)
        if pe > 0 and peer_pe > 0:
            if pe < (peer_pe * 0.9): scores['pe'] = 10      # >10% cheaper
            elif pe <= (peer_pe * 1.2): scores['pe'] = 7    # +/- 10-20%
            else: scores['pe'] = 3                          # >20% premium
            exps['pe'] = f"P/E ({pe:.1f}) vs Peer ({peer_pe:.1f})"
        else: scores['pe'] = 3; exps['pe'] = "P/E=N/A"

        # 2. Premium/Discount
        nav = data.get('nav_price', 0)
        price = data.get('current_price', 0)
        if nav > 0 and price > 0:
            diff = abs(price - nav) / nav
            if diff <= 0.005: scores['nav_discount'] = 10 # +/- 0.5%
            elif diff <= 0.01: scores['nav_discount'] = 7  # +/- 0.5-1%
            else: scores['nav_discount'] = 3               # >1%
            exps['nav_discount'] = f"Price/NAV Diff ({diff:.2%})"
        else: scores['nav_discount'] = 3; exps['nav_discount'] = "Price/NAV=N/A"

        # 3. Yield + Growth
        div_yield = data.get('dividend_yield', 0)
        eps_growth = data.get('eps_growth', 0) # User input: Est. growth of holdings
        if div_yield < 0.02 and eps_growth > 0.15: scores['yield_growth'] = 10 # Low yield, high growth
        elif (0.01 <= div_yield <= 0.025) and (0.10 <= eps_growth <= 0.15): scores['yield_growth'] = 7 # Mid yield, mid growth
        elif div_yield > 0.04 and eps_growth < 0.10: scores['yield_growth'] = 3 # High yield, low growth (value trap?)
        else: scores['yield_growth'] = 5 # Other cases (e.g., low yield, low growth)
        exps['yield_growth'] = f"Yield ({div_yield:.1%}) + Growth ({eps_growth:.1%})"

        total_score = np.mean(list(scores.values()))
        return {'total_score': total_score, 'scores': scores, 'explanations': exps}

    # --- MODIFIED: Use PUSHED SCORES ---
    def _calculate_operational_score(self, pushed_scores: Dict) -> Dict:
        """Calculates operational factors score (1-10) using PUSHED scores."""
        # Get user scores directly from the pushed_scores dict
        liquidity_score = pushed_scores.get('liquidity', 5)
        tax_score = pushed_scores.get('tax', 5) # Key matches ETF_OPERATIONAL_QUESTIONS_GUIDANCE
        fit_score = pushed_scores.get('portfolio_fit', 5) # Key matches ETF_OPERATIONAL_QUESTIONS_GUIDANCE

        scores = {
            'liquidity': liquidity_score,
            'tax_efficiency': tax_score, # Use consistent naming with report
            'portfolio_fit': fit_score
        }
        total_score = np.mean(list(scores.values()))
        explanations = {
            'liquidity': f"User Score -> {liquidity_score}",
            'tax_efficiency': f"User Score -> {tax_score}",
            'portfolio_fit': f"User Score -> {fit_score}"
        }
        return {'total_score': total_score, 'scores': scores, 'explanations': explanations}


    def _calculate_final_score_and_recommendation(self, quant_total, qual_norm, val_total, ops_total) -> Dict:
         """Calculates the final weighted score and recommendation for ETF."""
         # Using suggested weights (adjust as needed)
         weights = {'quant': 0.30, 'qual': 0.30, 'val': 0.20, 'ops': 0.20}

         analysis_score_10 = (
             (quant_total * weights['quant']) +
             (qual_norm * weights['qual']) +  # qual_norm now includes ops implicitly
             (val_total * weights['val']) +
             (ops_total * weights['ops'])   # Use the dedicated ops score here
         )
         final_score = analysis_score_10 * 10 # Scale to 100

         # [cite_start]Using thresholds from PDF [cite: 169] - adjusted slightly
         if final_score >= 80: recommendation = "STRONG BUY"
         elif final_score >= 65: recommendation = "BUY / Core"
         elif final_score >= 50: recommendation = "HOLD / Satellite"
         else: recommendation = "AVOID"

         return {
             'final_score': final_score,
             'analysis_score_100': final_score, # For ETF, analysis score is final score (no credits)
             'recommendation': recommendation,
             'weights': weights
         }

    def _generate_report(self, quant, qual, val, ops, final) -> str:
         """Generates a formatted text report of the ETF evaluation."""
         report = f"""
{'='*60}
COMPREHENSIVE ETF EVALUATION: {self.ticker} ({self.info.get('longName', '')})
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINAL SCORE: {final['final_score']:.2f} / 100
RECOMMENDATION: {final['recommendation']}

{'='*60}
SCORE BREAKDOWN (Weights: Quant={final['weights']['quant']:.0%}, Qual={final['weights']['qual']:.0%}, Val={final['weights']['val']:.0%}, Ops={final['weights']['ops']:.0%})
{'='*60}
1. Quantitative........: {quant['total_score']:.2f}/10
   - Annualized Returns: {quant['scores']['returns']:.2f}/10 ({quant['explanations']['returns']})
   - Beta............: {quant['scores']['beta']:.2f}/10 ({quant['explanations']['beta']})
   - Sharpe Ratio....: {quant['scores']['sharpe']:.2f}/10 ({quant['explanations']['sharpe']})
   - Expense Ratio...: {quant['scores']['expense']:.2f}/10 ({quant['explanations']['expense']})
   - Tracking Error..: {quant['scores']['tracking_error']:.2f}/10 ({quant['explanations']['tracking_error']})

2. Qualitative........: {qual['normalized_score']:.2f}/10 (Raw Total: {qual['raw_total']:.1f})
"""
         # Display qualitative categories (excluding the combined operational one)
         for cat, score in qual['categories'].items():
              if cat != 'operational_qual': # Don't show ops here
                  exp = qual['explanations'].get(cat, '')
                  report += f"   - {cat.replace('_',' ').title():<18}: {score:.2f}/10 ({exp})\n"
         report += f"""
3. Valuation..........: {val['total_score']:.2f}/10
   - P/E...............: {val['scores']['pe']:.2f}/10 ({val['explanations']['pe']})
   - Premium/Discount..: {val['scores']['nav_discount']:.2f}/10 ({val['explanations']['nav_discount']})
   - Yield + Growth....: {val['scores']['yield_growth']:.2f}/10 ({val['explanations']['yield_growth']})

4. Operational........: {ops['total_score']:.2f}/10
   - Liquidity.........: {ops['scores']['liquidity']:.2f}/10 ({ops['explanations']['liquidity']})
   - Tax Efficiency....: {ops['scores']['tax_efficiency']:.2f}/10 ({ops['explanations']['tax_efficiency']})
   - Portfolio Fit.....: {ops['scores']['portfolio_fit']:.2f}/10 ({ops['explanations']['portfolio_fit']})
"""
         report += f"\n{'='*60}\nEND OF REPORT\n{'='*60}"
         return report