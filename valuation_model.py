import numpy as np


class StockValuationModel:
    def __init__(self, risk_free_rate=0.04, market_return=0.09, corporate_tax_rate=0.21):
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.corporate_tax_rate = corporate_tax_rate

    def calculate_cost_of_equity(self, beta):
        """Calculate cost of equity using CAPM model (Re)"""
        return self.risk_free_rate + beta * (self.market_return - self.risk_free_rate)

    def calculate_wacc(self, market_cap, total_debt, cost_of_equity, cost_of_debt):
        """Calculate Weighted Average Cost of Capital (WACC)"""
        V = market_cap + total_debt
        if V == 0: return np.nan
        equity_weight = market_cap / V
        debt_weight = total_debt / V
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - self.corporate_tax_rate))
        return wacc

    def dcf_valuation(self, current_fcf, growth_rates_high, terminal_growth_rate,
                      wacc, shares_outstanding, total_debt, cash_equivalents):
        """Discounted Cash Flow (DCF) valuation model"""
        if wacc <= terminal_growth_rate:
            raise ValueError("WACC must be greater than terminal growth rate")

        projected_fcf_discounted = []
        fcf_t = current_fcf

        for i, growth_rate in enumerate(growth_rates_high):
            fcf_t *= (1 + growth_rate)
            projected_fcf_discounted.append(fcf_t / ((1 + wacc) ** (i + 1)))

        # Calculate terminal value and discount it
        last_fcf = fcf_t
        terminal_value = (last_fcf * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
        discounted_terminal_value = terminal_value / ((1 + wacc) ** len(growth_rates_high))

        enterprise_value = sum(projected_fcf_discounted) + discounted_terminal_value
        equity_value = enterprise_value - total_debt + cash_equivalents

        if shares_outstanding == 0: return np.nan
        per_share_value = equity_value / shares_outstanding
        return per_share_value

    def relative_valuation(self, target_eps, target_sales_per_share, target_book_value_per_share,
                           competitor_avg_pe=None, competitor_avg_ps=None, competitor_avg_pb=None):
        """Relative valuation model"""
        valuations = {}
        if competitor_avg_pe and target_eps:
            valuations['P/E Ratio Method'] = target_eps * competitor_avg_pe
        if competitor_avg_ps and target_sales_per_share:
            valuations['P/S Ratio Method'] = target_sales_per_share * competitor_avg_ps
        if competitor_avg_pb and target_book_value_per_share:
            valuations['P/B Ratio Method'] = target_book_value_per_share * competitor_avg_pb
        return valuations

    def benjamin_graham_valuation(self, eps, g, Y):
        """
        Calculate intrinsic value using the revised Benjamin Graham formula.
        V = (EPS * (8.5 + 2g) * 4.4) / Y
        :param eps: Earnings per share for the last 12 months
        :param g: Expected annual growth rate (e.g., input 15 for 15%)
        :param Y: Current high-grade corporate bond yield (e.g., input 4.5 for 4.5%)
        :return: Intrinsic value per share
        """
        if Y is None or Y == 0:
            return None
        # To prevent unrealistic valuations for ultra-high growth companies, 
        # Graham suggests limiting the growth rate g
        g = min(g, 20.0)  # Set growth rate cap at 20%
        value = (eps * (8.5 + 2 * g) * 4.4) / Y
        return value