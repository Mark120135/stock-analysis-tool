import pandas as pd


class StockDataProcessor:
    def calculate_technical_indicators(self, hist_df):
        df = hist_df.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['BB_Mid'] = df['SMA_20']
        df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(window=20).std()
        return df

    def get_yearly_financial_data(self, financial_data_dict, statement_type='income_stmt'):
        if statement_type in financial_data_dict:
            return financial_data_dict[statement_type]
        return pd.DataFrame()

    def calculate_free_cash_flow(self, cash_flow_df):
        if cash_flow_df.empty: return pd.Series(dtype=float)
        op_cash_flow_keys = ['Total Cash From Operating Activities', 'Operating Cash Flow']
        capex_keys = ['Capital Expenditures', 'Capital Expenditure']
        op_cash_flow = pd.Series(dtype=float)
        for key in op_cash_flow_keys:
            if key in cash_flow_df.index:
                op_cash_flow = cash_flow_df.loc[key]
                break
        capex = pd.Series(dtype=float)
        for key in capex_keys:
            if key in cash_flow_df.index:
                capex = cash_flow_df.loc[key]
                break
        if not op_cash_flow.empty and not capex.empty:
            common_index = op_cash_flow.index.intersection(capex.index)
            fcf = op_cash_flow.loc[common_index] + capex.loc[common_index]
            return fcf.sort_index(ascending=True)
        return pd.Series(dtype=float)

    def get_eps_from_financials(self, income_stmt_df):
        if not income_stmt_df.empty and 'Diluted EPS' in income_stmt_df.index:
            return income_stmt_df.loc['Diluted EPS'].sort_index(ascending=True)
        return pd.Series(dtype=float)

    def get_shares_outstanding(self, info_dict):
        return info_dict.get('sharesOutstanding')

    def get_market_cap(self, info_dict):
        return info_dict.get('marketCap')

    def get_total_debt(self, balance_sheet_df):
        """(Final fixed version) Get total debt - Bug fixed and uses exact names found from debug information"""
        if balance_sheet_df.empty: return pd.Series(dtype=float)

        # Method 1: Prioritize direct search for total debt items
        # Based on the debug list you provided, we've added exact names
        possible_keys = ['Total Liab', 'Total Liabilities', 'Total Liabilities Net Minority Interest', 'Total Debt']
        for key in possible_keys:
            if key in balance_sheet_df.index:
                print(f"Info: Direct item '{key}' found.")
                return balance_sheet_df.loc[key].sort_index(ascending=True)

        # Method 2: If direct items don't exist, calculate by summing components
        # Based on your debug list, we use exact component names
        current_liab_keys = ['Total Current Liabilities', 'Current Liabilities']
        non_current_liab_keys = ['Total Non Current Liabilities', 'Total Non Current Liabilities Net Minority Interest']

        current_liab_key_found = None
        for key in current_liab_keys:
            if key in balance_sheet_df.index:
                current_liab_key_found = key
                break

        non_current_liab_key_found = None
        # --- Bug fix section ---
        for key in non_current_liab_keys:
            # Fixed logic error here, previously was checking key in non_current_liab_keys
            if key in balance_sheet_df.index:
                non_current_liab_key_found = key
                break
        # --- End of bug fix ---

        if current_liab_key_found and non_current_liab_key_found:
            print(
                f"Info: Total debt item not found, calculating via '{current_liab_key_found}' + '{non_current_liab_key_found}'...")
            current_liabilities = balance_sheet_df.loc[current_liab_key_found]
            non_current_liabilities = balance_sheet_df.loc[non_current_liab_key_found]
            total_liabilities = current_liabilities + non_current_liabilities
            return total_liabilities.sort_index(ascending=True)

        print("Warning: After final attempts, still unable to find or calculate 'Total Debt' in balance sheet.")
        return pd.Series(dtype=float)

    def get_cash_and_equivalents(self, balance_sheet_df):
        if balance_sheet_df.empty: return pd.Series(dtype=float)
        possible_keys = ['Total Cash', 'Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']
        for key in possible_keys:
            if key in balance_sheet_df.index:
                return balance_sheet_df.loc[key].sort_index(ascending=True)
        print(f"Warning: Cash items not found in balance sheet.")
        return pd.Series(dtype=float)

    def get_total_stockholder_equity(self, balance_sheet_df):
        if balance_sheet_df.empty: return pd.Series(dtype=float)
        possible_keys = ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity', 'Common Stock Equity']
        for key in possible_keys:
            if key in balance_sheet_df.index:
                return balance_sheet_df.loc[key].sort_index(ascending=True)
        print(f"Warning: Stockholder equity items not found in balance sheet.")
        return pd.Series(dtype=float)

    def get_ebitda(self, annual_income_stmt):
        """
        Extract EBITDA from income statement
        """
        if annual_income_stmt.empty:
            return pd.Series(dtype=float)
        
        # EBITDA should be directly available in the income statement
        if 'EBITDA' in annual_income_stmt.index:
            return annual_income_stmt.loc['EBITDA'].sort_index(ascending=True)
        else:
            # Return empty series if not available
            return pd.Series(dtype=float)