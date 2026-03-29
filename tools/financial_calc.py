"""
FinSaarthi — Financial Mathematical Core
=========================================
Analytical engine for all calculations including XIRR, Tax, FIRE, and Portfolio Overlap.
As per M2 requirements: ALL numbers come from here, never from the LLM.

File: tools/financial_calc.py
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import numpy_financial as npf
import pandas as pd
from scipy.optimize import brentq

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial_calc")


class FinancialCalculator:
    """
    Core mathematical engine for FinSaarthi, optimized for Indian financial contexts.
    """

    # 1. --- XIRR CALCULATION (Using Brentq) ---

    @staticmethod
    def calculate_xirr(cashflows: List[float], dates: List[date]) -> float:
        """
        Calculate annualized XIRR using Brent's method.
        
        Args:
            cashflows: List of signed amounts (negative=investment, positive=current/out).
            dates: Corresponding transaction dates.
            
        Returns:
            float: Annualized XIRR as a decimal (0.1415 = 14.15%).
        """
        if len(cashflows) != len(dates) or len(cashflows) < 2:
            return 0.0

        # Ensure we have at least one negative and one positive value
        if all(c >= 0 for c in cashflows) or all(c <= 0 for c in cashflows):
            return 0.0

        def xnpv(rate: float) -> float:
            """Internal XNPV function: NPV = sum(C_i / (1+r)^((d_i - d_0)/365))"""
            d0 = dates[0]
            total = 0.0
            for c, d in zip(cashflows, dates):
                days = (d - d0).days
                total += c / ((1 + rate) ** (days / 365.0))
            return total

        try:
            # Brent's method requires a bracket [a, b] where f(a) and f(b) have opposite signs
            # We'll search between -0.999 (near death return) and 100 (10,000% return)
            # Find a bracket
            for b in [1, 10, 100]:
                if xnpv(-0.9) * xnpv(b) < 0:
                    return brentq(xnpv, -0.9, b, xtol=1e-6)
            return 0.0
        except (ValueError, RuntimeError) as e:
            logger.error(f"XIRR non-convergence: {e}")
            return 0.0

    # 2. --- SIP FOR GOAL ---

    @staticmethod
    def calculate_sip_for_goal(
        goal_amount_today: float,
        years: int,
        expected_return: float,
        inflation: float,
        current_savings: float = 0
    ) -> Dict[str, Any]:
        """
        Calculate required monthly SIP to meet a future goal, adjusted for inflation.
        """
        # Final future value needed in actual rupees (inflated)
        i_inf = inflation / 100.0
        future_goal_value = goal_amount_today * ((1 + i_inf) ** years)
        
        r_nom = expected_return / 100.0
        monthly_rate = r_nom / 12
        months = years * 12
        
        # FV of current savings
        fv_of_lumpsum = current_savings * ((1 + r_nom) ** years)
        gap = future_goal_value - fv_of_lumpsum
        
        if gap <= 0:
            monthly_sip = 0.0
        else:
            monthly_sip = npf.pmt(monthly_rate, months, 0, -gap)

        total_investment = (monthly_sip * months) + current_savings
        total_returns = future_goal_value - total_investment
        
        # Real return rate (inflation adjusted)
        real_return_rate = ((1 + r_nom) / (1 + i_inf)) - 1

        return {
            "monthly_sip": round(float(monthly_sip), 2),
            "future_goal_value": round(future_goal_value, 2),
            "real_return_rate": round(real_return_rate * 100, 2),
            "total_investment": round(total_investment, 2),
            "total_returns": round(total_returns, 2)
        }

    # 3. --- PORTFOLIO OVERLAP (Jaccard Similarity) ---

    @staticmethod
    def calculate_portfolio_overlap(fund_holdings: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Calculate stock overlap between mutual funds using Jaccard Similarity.
        """
        funds = list(fund_holdings.keys())
        matrix = {}
        highest_overlap = 0.0
        overlap_pair = ("None", "None")

        for i in range(len(funds)):
            f1 = funds[i]
            matrix[f1] = {}
            for j in range(len(funds)):
                f2 = funds[j]
                if f1 == f2:
                    matrix[f1][f2] = 100.0
                    continue
                
                s1 = set(fund_holdings[f1])
                s2 = set(fund_holdings[f2])
                
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                
                jaccard = (intersection / union) if union > 0 else 0
                percentage = round(jaccard * 100, 2)
                matrix[f1][f2] = percentage

                if i < j and percentage > highest_overlap:
                    highest_overlap = percentage
                    overlap_pair = (f1, f2)

        recommendation = "Maintain holdings — diversification is healthy."
        if highest_overlap > 50:
            recommendation = f"Critical Overlap: {overlap_pair[0]} and {overlap_pair[1]} share over 50% stocks. Consider switching one to a different category."
        elif highest_overlap > 30:
            recommendation = f"Moderate Overlap between {overlap_pair[0]} and {overlap_pair[1]}. Monitor for concentration risk."

        return {
            "overlap_matrix": matrix,
            "highest_overlap": highest_overlap,
            "highest_overlap_pair": overlap_pair,
            "recommendation": recommendation
        }

    # 4. --- EXPENSE DRAG ---

    @staticmethod
    def calculate_expense_drag(portfolio: List[Dict[str, Any]], years: int = 10) -> Dict[str, Any]:
        """
        Calculate compounding cost of Expense Ratios over N years.
        """
        annual_fees_total = 0.0
        total_value = 0.0
        drag_results = []
        r = 0.12 # Assumption for market return

        for item in portfolio:
            val = item['current_value']
            er = item['expense_ratio'] / 100.0
            total_value += val
            annual_fees_total += (val * er)
            
            fv_no_fees = val * ((1 + r) ** years)
            fv_with_fees = val * ((1 + (r - er)) ** years)
            drag = fv_no_fees - fv_with_fees
            
            drag_results.append({
                "fund_name": item['fund_name'],
                "drag": drag,
                "expense_ratio": item['expense_ratio']
            })

        worst_fund = max(drag_results, key=lambda x: x['expense_ratio']) if drag_results else None
        ten_year_drag = sum(d['drag'] for d in drag_results)
        
        # Savings if switched to index at 0.1%
        target_er = 0.001
        fv_optimized = total_value * ((1 + (r - target_er)) ** years)
        fv_current = sum(d['drag'] for d in drag_results) # simplified
        best_alternative_saving = fv_optimized - (sum(p['current_value'] for p in portfolio) * ((1 + (r - (worst_fund['expense_ratio']/100.0 if worst_fund else 0.01))) ** years)) # placeholder

        return {
            "annual_fees_total": round(annual_fees_total, 2),
            "ten_year_drag": round(ten_year_drag, 2),
            "worst_fund": worst_fund['fund_name'] if worst_fund else None,
            "best_alternative_saving": round(best_alternative_saving, 2)
        }

    # 5. --- TAX ENGINE (FY 2024-25) ---

    @staticmethod
    def calculate_hra_exemption(basic: float, hra_received: float, rent_paid: float, is_metro: bool) -> float:
        """Exact HRA Exemption formula: min of 3 conditions."""
        c1 = hra_received
        c2 = max(0.0, rent_paid - (0.1 * basic))
        c3 = (0.5 * basic) if is_metro else (0.4 * basic)
        return float(min(c1, c2, c3))

    @staticmethod
    def compare_tax_regimes(
        gross_salary: float,
        basic: float,
        hra_received: float,
        rent_paid: float,
        city_type: str,
        deductions_80c: float,
        deductions_80d: float,
        nps_80ccd: float,
        home_loan_interest: float,
        other_deductions: float
    ) -> Dict[str, Any]:
        
        is_metro = city_type.lower() == "metro"
        
        # 1. Old Regime
        hra_ex = FinancialCalculator.calculate_hra_exemption(basic, hra_received, rent_paid, is_metro)
        std_ded_old = 50000.0
        d80c = min(deductions_80c, 150000.0)
        d80d = min(deductions_80d, 25000.0) # Non-senior assumed
        dnps = min(nps_80ccd, 50000.0)
        
        taxable_old = gross_salary - std_ded_old - hra_ex - d80c - d80d - dnps - home_loan_interest - other_deductions
        taxable_old = max(0.0, taxable_old)
        
        def calc_old(inc):
            if inc <= 250000: return 0.0
            tax = 0.0
            if inc > 1000000: tax += (inc - 1000000) * 0.30; inc = 1000000
            if inc > 500000: tax += (inc - 500000) * 0.20; inc = 500000
            if inc > 250000: tax += (inc - 250000) * 0.05
            return tax

        base_old = calc_old(taxable_old)
        if taxable_old <= 500000: base_old = 0.0 # Rebate 87A
        tax_old = base_old * 1.04 # 4% Cess
        
        # 2. New Regime (FY 24-25 Budget Update)
        std_ded_new = 75000.0
        taxable_new = max(0.0, gross_salary - std_ded_new)
        
        def calc_new(inc):
            if inc <= 300000: return 0.0
            tax = 0.0
            if inc > 1500000: tax += (inc - 1500000) * 0.30; inc = 1500000
            if inc > 1200000: tax += (inc - 1200000) * 0.20; inc = 1200000
            if inc > 1000000: tax += (inc - 1000000) * 0.15; inc = 1000000
            if inc > 700000: tax += (inc - 700000) * 0.10; inc = 700000
            if inc > 300000: tax += (inc - 300000) * 0.05
            return tax
        
        base_new = calc_new(taxable_new)
        if taxable_new <= 700000: base_new = 0.0 
        tax_new = base_new * 1.04
        
        missed = []
        if deductions_80c < 150000: missed.append(f"80C: Gap of {150000 - deductions_80c}")
        if deductions_80d < 25000: missed.append(f"80D: Gap of {25000 - deductions_80d}")
        if nps_80ccd < 50000: missed.append(f"NPS: Gap of {50000 - nps_80ccd}")

        return {
            "old_tax": round(tax_old, 2),
            "new_tax": round(tax_new, 2),
            "old_net_salary": round(gross_salary - tax_old, 2),
            "new_net_salary": round(gross_salary - tax_new, 2),
            "recommended_regime": "NEW" if tax_new <= tax_old else "OLD",
            "savings_amount": round(abs(tax_new - tax_old), 2),
            "missed_deductions_list": missed
        }

    # 6. --- COUPLE OPTIMIZATION ---

    @staticmethod
    def calculate_couple_optimization(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
        """Test 4 scenarios for combined tax savings."""
        def get_total(d1, d2):
            res1 = FinancialCalculator.compare_tax_regimes(**d1)
            res2 = FinancialCalculator.compare_tax_regimes(**d2)
            return min(res1['old_tax'], res1['new_tax']) + min(res2['old_tax'], res2['new_tax'])

        # Scenario 1: Current
        s1_total = get_total(p1, p2)
        
        # Scenario 2: Shift 80C to higher earner
        hp1, hp2 = (p1.copy(), p2.copy()) if p1['gross_salary'] >= p2['gross_salary'] else (p2.copy(), p1.copy())
        joint_80c = hp1['deductions_80c'] + hp2['deductions_80c']
        hp1['deductions_80c'] = min(150000, joint_80c)
        hp2['deductions_80c'] = max(0, joint_80c - 150000)
        s2_total = get_total(hp1, hp2)

        # Scenario 3: Max NPS for both
        np1, np2 = p1.copy(), p2.copy()
        np1['nps_80ccd'] = 50000; np2['nps_80ccd'] = 50000
        s3_total = get_total(np1, np2)

        # Scenario 4: HRA shift
        sp1, sp2 = hp1.copy(), hp2.copy() # using hp copy
        combined_rent = p1['rent_paid'] + p2['rent_paid']
        sp1['rent_paid'] = combined_rent; sp2['rent_paid'] = 0
        s4_total = get_total(sp1, sp2)

        scenarios = [
            ("Current Status", s1_total),
            ("Shift 80C to Higher Earner", s2_total),
            ("Maximize NPS (50k each)", s3_total),
            ("Claim HRA via Higher Earner", s4_total)
        ]
        best = min(scenarios, key=lambda x: x[1])
        
        return {
            "best_scenario": best[0],
            "annual_savings": round(s1_total - best[1], 2),
            "total_tax_optimized": round(best[1], 2)
        }

    # 7. --- FIRE NUMBER ---

    @staticmethod
    def calculate_fire_number(monthly_expenses: float, inflation: float = 0.06, safe_withdrawal_rate: float = 0.04) -> Dict[str, Any]:
        annual_expenses = monthly_expenses * 12
        corpus = annual_expenses / safe_withdrawal_rate
        return {
            "fire_corpus": round(corpus, 2),
            "monthly_passive_income_at_retirement": round(corpus * safe_withdrawal_rate / 12, 2),
            "years_of_runway": round(1 / safe_withdrawal_rate, 1)
        }

if __name__ == "__main__":
    calc = FinancialCalculator()
    print("--- Testing with Rs. 12 Lakh Salary ---")
    res = calc.compare_tax_regimes(1200000, 500000, 200000, 180000, "metro", 150000, 25000, 50000, 0, 0)
    print(res)
