"""
Corporate Governance Qualitative Analyzer
Analyzes governance, business model, industry outlook, and R&D capabilities
Supports both Gemini and OpenAI APIs
"""

# Import dependencies with proper error handling
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

import yfinance as yf
import time
from typing import Dict, Optional


class GovernanceQualitativeAnalyzer:
    """
    Analyzes corporate governance, business model, industry outlook, and R&D capabilities
    Supports both Gemini and OpenAI APIs
    """
    
    def __init__(self, api_key: str, provider: str = "gemini"):
        """
        Initialize AI API for qualitative governance analysis
        
        Args:
            api_key: API key (Gemini or OpenAI)
            provider: "gemini" or "openai"
        
        Raises:
            ImportError: If required package is not installed
            ValueError: If provider is not supported
        """
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider == "gemini":
            if not GENAI_AVAILABLE:
                raise ImportError(
                    "google-generativeai package is not installed. "
                    "Install with: pip install google-generativeai"
                )
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "openai package is not installed. "
                    "Install with: pip install openai"
                )
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
            
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'gemini' or 'openai'"
            )
    
    def _format_number(self, value, is_currency: bool = False, is_percentage: bool = False) -> str:
        """Safely format numbers with proper handling of N/A values"""
        if value == 'N/A' or value is None:
            return 'N/A'
        
        try:
            if is_percentage:
                return f"{float(value) * 100:.2f}%"
            elif is_currency:
                return f"${float(value):,.0f}"
            elif isinstance(value, (int, float)):
                return f"{value:,}"
            else:
                return str(value)
        except (ValueError, TypeError):
            return 'N/A'
    
    def _get_risk_score_text(self, score) -> str:
        """Convert risk score to readable text with quality assessment"""
        if score == 'N/A' or score is None:
            return "N/A (not available)"
        
        try:
            score_int = int(score)
            if score_int < 5:
                quality = "Low Risk - Good"
            elif score_int <= 7:
                quality = "Moderate Risk"
            else:
                quality = "High Risk - Concerning"
            return f"{score_int}/10 ({quality})"
        except (ValueError, TypeError):
            return "N/A (invalid score)"
    
    def _extract_rd_info(self, stock) -> str:
        """Extract R&D information with proper error handling"""
        try:
            income_stmt = stock.financials
            
            if income_stmt is None or income_stmt.empty:
                return "N/A (financial data unavailable)"
            
            # Check for R&D in financials
            if 'Research And Development' in income_stmt.index:
                rd_expense = income_stmt.loc['Research And Development'].iloc[0]
                
                # Get total revenue for ratio calculation
                if 'Total Revenue' in income_stmt.index:
                    total_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                    if total_revenue and total_revenue > 0:
                        rd_ratio = (rd_expense / total_revenue) * 100
                        return f"${rd_expense:,.0f} ({rd_ratio:.2f}% of revenue)"
                
                return f"${rd_expense:,.0f}"
            else:
                return "N/A (not separately reported)"
                
        except (KeyError, IndexError, AttributeError, ValueError, TypeError) as e:
            return f"N/A (error retrieving data: {type(e).__name__})"
    
    def _get_additional_metrics(self, info: Dict) -> Dict:
        """Extract additional metrics for deeper analysis"""
        metrics = {}
        
        # Profitability metrics
        metrics['profit_margins'] = self._format_number(info.get('profitMargins'), is_percentage=True)
        metrics['operating_margins'] = self._format_number(info.get('operatingMargins'), is_percentage=True)
        metrics['roe'] = self._format_number(info.get('returnOnEquity'), is_percentage=True)
        
        # Debt metrics
        metrics['debt_to_equity'] = self._format_number(info.get('debtToEquity'))
        metrics['current_ratio'] = self._format_number(info.get('currentRatio'))
        
        # Growth metrics
        metrics['earnings_growth'] = self._format_number(info.get('earningsGrowth'), is_percentage=True)
        metrics['revenue_per_share'] = self._format_number(info.get('revenuePerShare'), is_currency=True)
        
        # Market metrics
        metrics['forward_pe'] = self._format_number(info.get('forwardPE'))
        metrics['peg_ratio'] = self._format_number(info.get('pegRatio'))
        metrics['price_to_book'] = self._format_number(info.get('priceToBook'))
        
        # Analyst recommendations
        metrics['target_mean_price'] = self._format_number(info.get('targetMeanPrice'), is_currency=True)
        metrics['recommendation'] = info.get('recommendationKey', 'N/A')
        
        return metrics
    
    def fetch_governance_analysis(self, ticker: str) -> Dict:
        """
        Fetch qualitative analysis focused on governance, business model, and innovation
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing qualitative analysis results
        """
        try:
            print(f"\nFetching company information for governance analysis of {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract basic company information
            company_name = info.get('longName', ticker)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            description = info.get('longBusinessSummary', 'N/A')
            website = info.get('website', 'N/A')
            country = info.get('country', 'N/A')
            
            # Format employee count
            employees = info.get('fullTimeEmployees')
            employees_display = self._format_number(employees)
            
            # Format financial metrics
            revenue_growth = info.get('revenueGrowth')
            revenue_growth_display = self._format_number(revenue_growth, is_percentage=True)
            
            market_cap = info.get('marketCap')
            market_cap_display = self._format_number(market_cap, is_currency=True)
            
            # Get R&D information
            rd_info = self._extract_rd_info(stock)
            
            # Extract governance risk scores
            audit_risk = info.get('auditRisk', 'N/A')
            board_risk = info.get('boardRisk', 'N/A')
            compensation_risk = info.get('compensationRisk', 'N/A')
            shareholder_rights_risk = info.get('shareHolderRightsRisk', 'N/A')
            overall_risk = info.get('overallRisk', 'N/A')
            
            # Format risk scores for display
            audit_risk_display = self._get_risk_score_text(audit_risk)
            board_risk_display = self._get_risk_score_text(board_risk)
            compensation_risk_display = self._get_risk_score_text(compensation_risk)
            shareholder_rights_risk_display = self._get_risk_score_text(shareholder_rights_risk)
            overall_risk_display = self._get_risk_score_text(overall_risk)
            
            # Get additional metrics
            additional_metrics = self._get_additional_metrics(info)
            
            # Build context for AI analysis
            context = f"""
=== COMPANY INFORMATION FOR QUALITATIVE ANALYSIS ===
Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}
Country: {country}
Employees: {employees_display}
Website: {website}

**Business Description:**
{description}

**Market Position:**
- Market Capitalization: {market_cap_display}
- Revenue Growth (YoY): {revenue_growth_display}
- Earnings Growth: {additional_metrics['earnings_growth']}
- Revenue Per Share: {additional_metrics['revenue_per_share']}

**Profitability Metrics:**
- Profit Margins: {additional_metrics['profit_margins']}
- Operating Margins: {additional_metrics['operating_margins']}
- Return on Equity (ROE): {additional_metrics['roe']}

**Financial Health:**
- Debt to Equity: {additional_metrics['debt_to_equity']}
- Current Ratio: {additional_metrics['current_ratio']}

**Valuation Metrics:**
- Forward P/E: {additional_metrics['forward_pe']}
- PEG Ratio: {additional_metrics['peg_ratio']}
- Price to Book: {additional_metrics['price_to_book']}

**Innovation Investment:**
- R&D Spending (Latest): {rd_info}

**Governance Risk Scores (1-10 scale, lower is better):**
- Overall Risk: {overall_risk_display}
- Audit Risk: {audit_risk_display}
- Board Risk: {board_risk_display}
- Compensation Risk: {compensation_risk_display}
- Shareholder Rights Risk: {shareholder_rights_risk_display}

**Analyst Perspective:**
- Recommendation: {additional_metrics['recommendation']}
- Target Mean Price: {additional_metrics['target_mean_price']}

Note: Risk scores range from 1-10 where 1 is lowest risk and 10 is highest risk.
Scores below 5 are generally good, 5-7 are moderate, above 7 are concerning.
"""
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(context, company_name, ticker, 
                                                  board_risk, compensation_risk, 
                                                  overall_risk, audit_risk, rd_info,
                                                  sector, industry, description)
            
            # Get AI response
            print(f"Requesting analysis from {self.provider.upper()} API...")
            if self.provider == "gemini":
                response_text = self._fetch_with_gemini(prompt)
            else:  # openai
                response_text = self._fetch_with_openai(prompt)
            
            return {
                'ticker': ticker,
                'company_name': company_name,
                'provider': self.provider,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': 'Qualitative Governance & Strategy Analysis',
                'raw_response': response_text,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in governance analysis: {e}")
            return {
                'ticker': ticker,
                'provider': self.provider,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': 'Qualitative Governance & Strategy Analysis',
                'error': str(e),
                'raw_response': '',
                'success': False
            }
    
    def _create_analysis_prompt(self, context: str, company_name: str, ticker: str,
                               board_risk, compensation_risk, overall_risk, 
                               audit_risk, rd_info: str, sector: str, industry: str,
                               description: str) -> str:
        """Create the analysis prompt with proper formatting"""
        
        prompt = f"""You are an expert qualitative analyst evaluating companies for investment quality.
You must provide thorough, evidence-based analysis while critically examining the limitations of your own reasoning.

{context}

Analyze the company: {company_name} ({ticker})

================================
SECTION 0: COMPANY OVERVIEW
================================

Before the detailed analysis, provide a comprehensive company summary that includes:

**Company Profile:**
- What does this company do? (2-3 sentences explaining their core business)
- Key products/services and their market significance
- Company's position in the industry landscape

**Quick Facts:**
- Sector: {sector}
- Industry: {industry}
- Market presence and scale

Write this overview in 4-6 sentences total. Make it informative and accessible.

================================
DETAILED ANALYSIS INSTRUCTIONS
================================

For EACH question below, you MUST follow this 3-step structure:

**Step 1 - Self-Critique (Data Limitations):**
Start with "DATA AVAILABLE:" and list what specific data points you have access to for this question.
Then write "LIMITATIONS:" and explicitly state what information is missing or uncertain that would make your analysis more reliable.

**Step 2 - Analysis:**
Provide 3-4 sentences of detailed analysis addressing the question. Include:
- Specific numbers, metrics, or facts from the data provided
- Industry context and comparisons where relevant
- Multiple perspectives or considerations
- Concrete examples or evidence

**Step 3 - Verdict:**
End with a clear verdict line: "[YES] - " OR "[NO] - " OR "[DEPENDS] - " followed by a one-sentence conclusion.

================================
QUESTIONS TO ANSWER
================================

### 1. Corporate Governance

**1.1 Is the board structure sound?**
[Follow 3-step structure. Board Risk: {board_risk}]

**1.2 Does management demonstrate integrity and a long-term vision?**
[Follow 3-step structure. Compensation Risk: {compensation_risk}, Overall Risk: {overall_risk}]

**1.3 Has the company been involved in fraud or accounting scandals?**
[Follow 3-step structure. Audit Risk: {audit_risk}. If no specific scandal information available, state this clearly and base analysis on audit risk score and industry reputation]

**Section Rating: X/5** (Strong=5/Good=4/Average=3/Weak=2/Poor=1)
[Provide 2-3 sentences justifying this rating based on all three answers above]

---

### 2. Business Model & Moat

**2.1 Do the company's products/services have uniqueness?**
[Follow 3-step structure. Consider: patents, proprietary technology, brand strength, differentiation factors]

**2.2 Does it benefit from network effects, brand value, patents, or high switching costs as a moat?**
[Follow 3-step structure. Identify and evaluate SPECIFIC moats: network effects, economies of scale, brand equity, regulatory barriers, patents, high switching costs, cost advantages]

**Section Rating: X/5** (Strong=5/Good=4/Average=3/Weak=2/Poor=1)
[Provide 2-3 sentences justifying this rating based on both answers above]

---

### 3. Industry Outlook & Competitiveness

**3.1 Is the industry in a growth phase?**
[Follow 3-step structure. Consider: market trends, technological disruption, regulatory changes, consumer behavior shifts]

**3.2 What is the company's ranking and competitive position in the sector?**
[Follow 3-step structure. Estimate market share, identify key competitors, assess competitive advantages]

**3.3 Does it have a global presence or regional dominance?**
[Follow 3-step structure. Consider: geographic diversification, international revenue %, market penetration in key regions]

**Section Rating: X/5** (Strong=5/Good=4/Average=3/Weak=2/Poor=1)
[Provide 2-3 sentences justifying this rating based on all three answers above]

---

### 4. R&D and Innovation Capability

**4.1 Is the company continuously investing in R&D?**
[Follow 3-step structure. R&D Data: {rd_info}. Compare to industry standards and competitors where possible]

**4.2 Does it generate innovative results that can translate into revenue?**
[Follow 3-step structure. Look for: patent filings, new product launches, market disruption, innovation track record]

**4.3 Can it consistently launch new products in a rapidly changing market?**
[Follow 3-step structure. Consider: product pipeline, time-to-market, adaptation to market changes, historical launch success]

**Section Rating: X/5** (Strong=5/Good=4/Average=3/Weak=2/Poor=1)
[Provide 2-3 sentences justifying this rating based on all three answers above]

---

### Overall Qualitative Verdict:

**Summary Assessment:**
[Write 3-4 sentences summarizing the company's overall qualitative strengths and weaknesses]

**Investment Perspective:**
[POSITIVE] - [2 sentences on why this is a strong qualitative investment]
OR
[NEUTRAL] - [2 sentences on why this is a mixed qualitative picture]
OR
[NEGATIVE] - [2 sentences on why this has qualitative concerns]

**Key Risks to Monitor:**
[Bullet point 2-3 specific risks that could impact the assessment]

================================
CRITICAL REQUIREMENTS
================================

1. Start with Section 0 (Company Overview) before any detailed analysis
2. For ALL 10 questions, use the 3-step structure: Data Limitations → Analysis → Verdict
3. Be specific with numbers and facts in your analysis (not generic statements)
4. In self-critique, honestly state what information you're missing
5. Each analysis paragraph should be 3-4 substantive sentences
6. Every verdict must start with [YES], [NO], or [DEPENDS]
7. Section ratings must be justified with 2-3 sentences
8. Overall verdict must include summary, perspective, and risks to monitor
9. Be intellectually honest - if data is limited, say so clearly
10. Prioritize depth and specificity over breadth

================================
EXAMPLE OF 3-STEP STRUCTURE:
================================

**1.1 Is the board structure sound?**

DATA AVAILABLE: Board Risk score of 3/10 (low risk), company operates in {industry} sector.
LIMITATIONS: No specific information about board composition, independence ratios, committee structures, or individual director qualifications. Cannot verify diversity metrics or meeting frequency.

The Board Risk score of 3/10 indicates strong governance practices well below the concerning threshold of 5. For a company of this scale in {industry}, this suggests established oversight mechanisms and proper separation of management and board functions. The low score typically correlates with independent directors, regular board evaluations, and effective committee structures, though these specifics aren't directly available in the data. Industry best practices would suggest this company likely maintains compliance with governance standards.

[YES] - The board structure appears sound based on the low risk score, though detailed composition data would strengthen this assessment.
"""
        
        return prompt
    
    def _fetch_with_gemini(self, prompt: str) -> str:
        """Fetch data using Gemini API with error handling"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,  # Slightly higher for more detailed analysis
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 8192,  # Increased for longer responses
                }
            )
            
            if not hasattr(response, 'text') or not response.text:
                raise ValueError("Empty response from Gemini API")
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _fetch_with_openai(self, prompt: str) -> str:
        """Fetch data using OpenAI API with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert qualitative analyst with a commitment to intellectual honesty. 
                        You provide thorough analysis while clearly acknowledging data limitations. 
                        You must follow the exact structure provided: Company Overview first, then detailed 3-step analysis for each question.
                        Be specific with numbers and evidence. Avoid generic statements."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Slightly higher for more detailed analysis
                max_tokens=8192  # Increased for longer responses
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI API")
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def format_governance_report(self, data: Dict) -> str:
        """Format the qualitative governance analysis into a readable report"""
        
        # Handle error cases
        if not data.get('success', True) or data.get('error'):
            error_msg = data.get('error', 'Unknown error')
            return f"{'='*80}\n[ERROR] Analysis failed for {data['ticker']}\n{'='*80}\n{error_msg}\n{'='*80}\n"
        
        # Determine provider display name
        provider_name = data.get('provider', 'AI').upper()
        if provider_name == 'GEMINI':
            provider_display = "Google Gemini 2.0 Flash"
        elif provider_name == 'OPENAI':
            provider_display = "OpenAI GPT-4o-mini"
        else:
            provider_display = provider_name
        
        # Build report
        report = f"{'='*80}\n"
        report += f"  [QUALITATIVE ANALYSIS] GOVERNANCE & STRATEGY\n"
        report += f"  Company: {data.get('company_name', data['ticker'])} ({data['ticker']})\n"
        report += f"{'='*80}\n"
        report += f"  AI Provider: {provider_display}\n"
        report += f"  Analysis Timestamp: {data.get('timestamp', 'N/A')}\n"
        report += f"{'='*80}\n\n"
        
        # Add analysis content
        if 'raw_response' in data and data['raw_response']:
            report += data['raw_response']
        else:
            report += "No analysis available.\n"
        
        # Add framework explanation
        report += f"\n\n{'='*80}\n"
        report += "ANALYSIS FRAMEWORK:\n"
        report += f"{'='*80}\n"
        report += "This enhanced qualitative analysis evaluates non-financial factors through a\n"
        report += "rigorous methodology that includes:\n\n"
        report += "  0. Company Overview - Executive summary of the business\n\n"
        report += "  1. Corporate Governance (3 questions)\n"
        report += "     - Board structure quality\n"
        report += "     - Management integrity and vision\n"
        report += "     - Scandal history and accounting quality\n\n"
        report += "  2. Business Model & Moat (2 questions)\n"
        report += "     - Product/service uniqueness\n"
        report += "     - Competitive advantages and barriers to entry\n\n"
        report += "  3. Industry Outlook & Competitiveness (3 questions)\n"
        report += "     - Industry growth phase\n"
        report += "     - Competitive ranking and position\n"
        report += "     - Geographic presence and market reach\n\n"
        report += "  4. R&D and Innovation Capability (2 questions)\n"
        report += "     - R&D investment levels\n"
        report += "     - Innovation effectiveness and product pipeline\n\n"
        report += "METHODOLOGY:\n"
        report += "Each answer follows a 3-step structure:\n"
        report += "  1. Data Limitations - What information is available and what's missing\n"
        report += "  2. Detailed Analysis - Evidence-based evaluation with specific metrics\n"
        report += "  3. Clear Verdict - [YES]/[NO]/[DEPENDS] with justification\n\n"
        report += "This self-critical approach ensures transparency about analytical confidence\n"
        report += "and helps identify areas where additional research may be beneficial.\n\n"
        report += f"Analysis powered by {provider_display}\n"
        report += f"{'='*80}\n"
        
        return report


# Helper functions
def get_available_providers() -> list:
    """Returns list of available AI providers"""
    providers = []
    if GENAI_AVAILABLE:
        providers.append("gemini")
    if OPENAI_AVAILABLE:
        providers.append("openai")
    return providers


def is_provider_available(provider: str) -> bool:
    """Check if a specific provider is available"""
    provider_lower = provider.lower()
    if provider_lower == "gemini":
        return GENAI_AVAILABLE
    elif provider_lower == "openai":
        return OPENAI_AVAILABLE
    return False


# Example usage
if __name__ == "__main__":
    print("Governance Qualitative Analyzer - Enhanced Version")
    print("="*50)
    print(f"Available providers: {get_available_providers()}")
    print("\nFeatures:")
    print("  - Company overview summary at the beginning")
    print("  - Self-critical analysis with data limitation acknowledgment")
    print("  - Detailed 3-4 sentence responses per question")
    print("  - Evidence-based reasoning with specific metrics")
    print("\nUsage:")
    print("  analyzer = GovernanceQualitativeAnalyzer(api_key='YOUR_KEY', provider='gemini')")
    print("  result = analyzer.fetch_governance_analysis('AAPL')")
    print("  report = analyzer.format_governance_report(result)")
    print("  print(report)")