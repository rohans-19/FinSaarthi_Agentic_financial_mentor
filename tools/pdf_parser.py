"""
FinSaarthi — Financial PDF Parser
=================================
Handles extractive parsing of CAMS Consolidated Account Statements (CAS) 
and Form 16 (Income Tax) documents.

File: tools/pdf_parser.py
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_parser")


class CAMSParser:
    """
    Parser for CAMS-generated Consolidated Account Statements (CAS).
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.full_text = ""
        self._load_text()

    def _load_text(self):
        """Load text from PDF using pdfplumber with PyMuPDF fallback."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    self.full_text += (page.extract_text() or "") + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed, falling back to PyMuPDF: {e}")
            doc = fitz.open(self.pdf_path)
            for page in doc:
                self.full_text += page.get_text() + "\n"

    def extract_all_transactions(self) -> pd.DataFrame:
        """
        Extract all transaction rows from the statement.
        """
        transactions = []
        # Regex for CAMS transactions: DD-MMM-YYYY followed by description and numbers
        pattern = re.compile(
            r"(\d{2}-[a-zA-Z]{3}-\d{4})\s+(.*?)\s+([-+]?[\d,]+\.\d{2,})\s+([-+]?[\d,]+\.\d{3,})\s+([\d,]+\.\d{2,})\s+([\d,]+\.\d{3,})"
        )

        current_fund = "Unknown Fund"
        lines = self.full_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Identify fund name (Major houses listed as markers)
            if any(house in line.upper() for house in ["MIRAE", "PARAG", "HDFC", "SBI", "AXIS", "ICICI"]):
                if "TOTAL" not in line.upper() and len(line) > 10:
                    current_fund = line
            
            match = pattern.search(line)
            if match:
                date_str, desc, amount, units, nav, balance = match.groups()
                
                # Transaction type logic
                du = desc.upper()
                t_type = "SIP" if "SIP" in du else "Purchase" if "PURCHASE" in du else "Redemption" if "REDEEM" in du else "Switch" if "SWITCH" in du else "Other"

                transactions.append({
                    "date": datetime.strptime(date_str, "%d-%b-%Y"),
                    "fund_name": current_fund,
                    "transaction_type": t_type,
                    "description": desc.strip(),
                    "amount": float(amount.replace(',', '')),
                    "units": float(units.replace(',', '')),
                    "nav": float(nav.replace(',', '')),
                    "balance_units": float(balance.replace(',', ''))
                })

        df = pd.DataFrame(transactions)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def extract_current_holdings(self) -> pd.DataFrame:
        """Extract summary section containing folio and value."""
        holdings = []
        lines = self.full_text.split('\n')
        for line in lines:
            if "Folio:" in line and "Units:" in line:
                try:
                    fund_name = line.split("Folio:")[0].strip()
                    folio = re.search(r"Folio:\s*(\w+)", line).group(1)
                    units = re.search(r"Units:\s*([\d,]+\.\d+)", line).group(1).replace(',', '')
                    nav = re.search(r"NAV:\s*([\d,]+\.\d+)", line).group(1).replace(',', '')
                    val = re.search(r"Value:\s*([\d,]+\.\d+)", line).group(1).replace(',', '')
                    
                    holdings.append({
                        "fund_name": fund_name,
                        "folio_number": folio,
                        "units": float(units),
                        "current_nav": float(nav),
                        "current_value": float(val),
                        "scheme_type": self.get_fund_category(fund_name)
                    })
                except Exception: continue
        return pd.DataFrame(holdings)

    def get_fund_category(self, fund_name: str) -> str:
        n = fund_name.lower()
        if any(k in n for k in ["flexi", "multi cap", "large cap", "mid cap", "small cap", "equity"]): return "Equity"
        if any(k in n for k in ["debt", "liquid", "overnight", "gilt"]): return "Debt"
        if any(k in n for k in ["hybrid", "balanced", "aggressive"]): return "Hybrid"
        return "Other"

    def prepare_for_xirr(self) -> List[Dict[str, Any]]:
        df = self.extract_all_transactions()
        holdings = self.extract_current_holdings()
        prepared = []
        if df.empty: return []

        for fund in df['fund_name'].unique():
            fund_tx = df[df['fund_name'] == fund]
            cashflows = list(fund_tx['amount'] * -1)
            dates = list(fund_tx['date'].dt.date)
            
            fund_holding = holdings[holdings['fund_name'] == fund]
            if not fund_holding.empty:
                cashflows.append(float(fund_holding.iloc[0]['current_value']))
                dates.append(datetime.now().date())
            
            prepared.append({"fund_name": fund, "cashflows": cashflows, "dates": dates})
        return prepared


class Form16Parser:
    """Parser for Indian Income Tax Form 16 Part B."""
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.full_text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                self.full_text += (page.extract_text() or "") + "\n"

    def _extract_numeric(self, pattern: str) -> float:
        match = re.search(pattern, self.full_text, re.IGNORECASE | re.DOTALL)
        return float(match.group(1).replace(',', '')) if match else 0.0

    def extract_salary_details(self) -> Dict[str, Any]:
        return {
            "gross_salary": self._extract_numeric(r"Gross Salary.*?(\d+[\d,]*\.\d{2})"),
            "basic_salary": self._extract_numeric(r"Basic Salary.*?(\d+[\d,]*\.\d{2})"),
            "hra_received": self._extract_numeric(r"House Rent Allowance.*?(\d+[\d,]*\.\d{2})"),
            "tds_deducted": self._extract_numeric(r"Tax payable on total income.*?(\d+[\d,]*\.\d{2})"),
            "assessment_year": re.search(r"Assessment Year\s*(\d{4}-\d{2})", self.full_text).group(1) if re.search(r"Assessment Year", self.full_text) else "N/A"
        }

    def extract_deductions_claimed(self) -> Dict[str, Any]:
        return {
            "sec_80c": self._extract_numeric(r"Section 80C.*?(\d+[\d,]*\.\d{2})"),
            "sec_80d": self._extract_numeric(r"Section 80D.*?(\d+[\d,]*\.\d{2})"),
            "hra_exemption": self._extract_numeric(r"exemption under section 10\(13A\).*?(\d+[\d,]*\.\d{2})")
        }


def create_sample_cams_pdf(output_path: str) -> None:
    """Generate a realistic dummy CAMS PDF for hackathon demonstration."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("<b>CAMS Consolidated Account Statement</b>", styles['Title']))

    funds = [
        {"name": "Mirae Asset Large Cap Fund", "base_nav": 100.0},
        {"name": "Parag Parikh Flexi Cap Fund", "base_nav": 50.0},
        {"name": "HDFC Mid Cap Opportunities Fund", "base_nav": 120.0}
    ]

    for fund in funds:
        elements.append(Paragraph(f"<b>{fund['name']}</b>", styles['Heading2']))
        data = [["Date", "Description", "Amount", "Units", "NAV", "Balance"]]
        nav = fund['base_nav']; units = 0.0
        for i in range(24):
            date_str = (datetime(2022, 1, 1) + pd.DateOffset(months=i)).strftime('%d-%b-%Y')
            nav *= 1.01; sip = 5000.00; u = sip / nav; units += u
            data.append([date_str, "SIP Purchase", f"{sip:.2f}", f"{u:.3f}", f"{nav:.2f}", f"{units:.3f}"])
        
        t = Table(data)
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')]))
        elements.append(t)
        elements.append(Paragraph(f"{fund['name']} Folio: 91029384 Units: {units:.3f} NAV: {nav:.2f} Value: {units*nav:.2f}", styles['Normal']))

    doc.build(elements)
    logger.info(f"Sample CAMS PDF generated at {output_path}")

# Wrapper functions for api.py
def parse_cams_cas(path: str) -> Dict[str, Any]:
    p = CAMSParser(path)
    return {"transactions": p.extract_all_transactions().to_dict('records'), "holdings": p.extract_current_holdings().to_dict('records')}

def parse_form16(path: str) -> Dict[str, Any]:
    p = Form16Parser(path)
    return {**p.extract_salary_details(), **p.extract_deductions_claimed()}
