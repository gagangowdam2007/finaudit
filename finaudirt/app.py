#!/usr/bin/env python3
"""FinAudit-QuantTerminal Streamlit dashboard.

An institutional front end for ``quant_terminal_engine``: forensic earnings
screens, a dual-terminal-value Monte Carlo DCF, forecast and statement views
and a downloadable summary report. All calculations run in the engine, so
this module only handles inputs, presentation and export.

Run from this directory with::

    streamlit run app.py

Monetary values are in ₹ crore, except per-share values in ₹.
"""

from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy import stats

from quant_terminal_engine import (
    KAYNES_DCF,
    KAYNES_FY25,
    KAYNES_FY26,
    KAYNES_WACC,
    Alert,
    DCFInputs,
    DistributionStats,
    FinancialYear,
    ForensicAuditor,
    MonteCarloValuation,
    Severity,
    SimulationConfig,
    ValuationResult,
    WaccInputs,
)

APP_TITLE = ("FinAudit-QuantTerminal | Autonomous Forensic Screener & "
             "Monte Carlo Valuation Engine")
AUTHOR_TAG = "Built by Gagan Gowda M"
DISCLAIMER = ("Forensic scores are statistical screens, not findings of "
              "manipulation. Valuations depend on the stated assumptions. "
              "For research use; this is not investment advice.")

SURFACE = "#0f172a"
PANEL = "#1e293b"
BORDER = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#263449"
SERIES_GORDON = "#3987e5"
SERIES_EXIT = "#d95926"
SERIES_THIRD = "#199e70"
STATUS_COLOR = {
    Severity.RED: "#f87171",
    Severity.AMBER: "#fbbf24",
    Severity.GREEN: "#34d399",
}
STATUS_LABEL = {
    Severity.RED: "▲ RED FLAG",
    Severity.AMBER: "● WATCH",
    Severity.GREEN: "✓ CLEAN",
}

BENEISH = "Beneish M-Score"
ALTMAN = "Altman Z''-Score (EM)"
SLOAN = "Sloan accrual ratio"
CONVERSION = "CFO / EBITDA"

ALERT_EXPLANATIONS = {
    BENEISH: (
        "The eight-factor profile resembles companies later found to have "
        "manipulated earnings. Read which indices drive the score before "
        "concluding anything, because rapid or acquired growth also lifts it."
    ),
    SLOAN: (
        "A large share of reported profit is not matched by cash after "
        "operating and investing flows. This ratio includes growth capex, "
        "so compare it with TATA to isolate true accruals."
    ),
    CONVERSION: (
        "Operating cash flow is weak relative to EBITDA. Persistent "
        "shortfalls mean profit is being tied up in working capital or "
        "was never collected. Reconcile the working-capital lines to the "
        "cash flow statement."
    ),
    "Cash conversion cycle change": (
        "Each rupee of sales now takes longer to return as cash. Identify "
        "whether debtors, stock or supplier terms moved, and whether "
        "management has given a date for reversal."
    ),
    "Debtor-to-sales divergence": (
        "Receivables are growing much faster than revenue, a classic sign "
        "of aggressive recognition or stretched collections. Check ageing "
        "buckets, related-party balances, unbilled revenue and receivables "
        "that arrived through acquisitions."
    ),
    ALTMAN: (
        "Combines liquidity, retained profit, operating return and balance "
        "sheet cushion to gauge solvency stress."
    ),
    "Beneish DSRI": (
        "Receivables rose faster than sales, the single largest driver of "
        "the M-Score. Possible causes include channel stuffing, extended "
        "credit terms or a late-year acquisition."
    ),
    "Beneish GMI": (
        "Gross margin has fallen, which can create pressure to flatter "
        "reported earnings elsewhere."
    ),
    "Beneish AQI": (
        "More of the asset base sits in soft assets, which can indicate "
        "that costs are being capitalised rather than expensed."
    ),
    "Beneish SGI": (
        "Very high growth is not manipulation, but it raises the pressure "
        "to keep growth going and makes controls harder to maintain."
    ),
    "Beneish DEPI": (
        "The depreciation rate has slowed, which may reflect longer "
        "useful-life assumptions that lift profit."
    ),
    "Beneish TATA": (
        "Accounting profit is running ahead of operating cash, the most "
        "direct accrual signal in the M-Score."
    ),
}

STATEMENT_LINES = (
    ("Income statement", "Revenue from operations", "revenue"),
    ("Income statement", "Cost of goods sold", "cogs"),
    ("Income statement", "Employee and other expenses", "sga"),
    ("Income statement", "EBITDA", "ebitda"),
    ("Income statement", "Depreciation and amortisation", "depreciation"),
    ("Income statement", "EBIT", "ebit"),
    ("Income statement", "Profit after tax", "net_income"),
    ("Balance sheet", "Trade receivables", "receivables"),
    ("Balance sheet", "Inventory", "inventory"),
    ("Balance sheet", "Trade payables", "payables"),
    ("Balance sheet", "Total current assets", "current_assets"),
    ("Balance sheet", "Net PP&E including leases", "ppe"),
    ("Balance sheet", "Capital work-in-progress", "cwip"),
    ("Balance sheet", "Non-current investments", "investments"),
    ("Balance sheet", "Total assets", "total_assets"),
    ("Balance sheet", "Total current liabilities", "current_liabilities"),
    ("Balance sheet", "Long-term debt and leases", "long_term_debt"),
    ("Balance sheet", "Total liabilities", "total_liabilities"),
    ("Balance sheet", "Retained earnings", "retained_earnings"),
    ("Balance sheet", "Total equity", "book_equity"),
    ("Cash flow", "Cash from operations", "cfo"),
    ("Cash flow", "Cash from investing", "cfi"),
)
CUSTOM_FIELDS = tuple(
    (label, attribute) for _, label, attribute in STATEMENT_LINES
    if attribute != "ebit"
)


@dataclass(frozen=True)
class CaseProfile:
    """A company case with its data and default assumptions.

    Attributes:
        key: Stable identifier, used to namespace widget state.
        name: Company name.
        focus: Short description of the case.
        current: Latest fiscal year.
        prior: Preceding fiscal year.
        wacc: CAPM inputs.
        dcf: Valuation inputs.
        cagr_mean: Mean 5-year revenue CAGR.
        cagr_std: Standard deviation of revenue CAGR.
        peak_margin: Low, mode and high of peak EBIT margin.
        margin_fade: Low, mode and high of the margin fade.
        growth_range: Default terminal growth range.
        exit_range: Default EV/EBITDA exit range.
        history: Multi-year trend table.
        unit_economics: Optional retail unit-economics table.
        context: Case-specific reading notes for the forensic tab.
        provenance: Where the data and assumptions come from.
    """

    key: str
    name: str
    focus: str
    current: FinancialYear
    prior: FinancialYear
    wacc: WaccInputs
    dcf: DCFInputs
    cagr_mean: float
    cagr_std: float
    peak_margin: tuple[float, float, float]
    margin_fade: tuple[float, float, float]
    growth_range: tuple[float, float]
    exit_range: tuple[float, float]
    history: pd.DataFrame
    unit_economics: pd.DataFrame | None
    context: str
    provenance: str


@dataclass(frozen=True)
class RunSettings:
    """Inputs for one valuation run, resolved from the sidebar."""

    dcf: DCFInputs
    config: SimulationConfig


TRENT_CMP = 3_058.0
TRENT_SHARES_CR = 53.325
TRENT_TAX_RATE = 0.2517
TRENT_LEASE_PRINCIPAL_FY26 = 796.86
TRENT_CAPEX_FY26 = 1_762.85
TRENT_STORES_FY26 = 1_263
TRENT_STORES_FY25 = 1_013
TRENT_AREA_MN_SQFT_FY26 = 17.7
TRENT_OCCUPANCY = {"FY25": 1_516.88, "FY26": 1_563.94}
TRENT_EMPLOYEE = {"FY25": 1_308.45, "FY26": 1_355.49}

# Source: audited consolidated results filed 22 April 2026 (₹ crore).
# retained_earnings is an ESTIMATE: Other Equity less about ₹2,100 Cr of
# assumed non-distributable reserves. The filing does not split them.
TRENT_FY26 = FinancialYear(
    label="FY26", revenue=20_074.21, cogs=11_264.71, sga=5_136.11,
    depreciation=1_361.19, ebitda=3_673.39, net_income=1_721.33,
    cfo=2_667.62, cfi=-1_578.36, receivables=51.14, inventory=2_289.01,
    payables=1_210.66, current_assets=4_058.30, ppe=5_416.60, cwip=253.99,
    investments=737.38, total_assets=11_728.58, current_liabilities=3_210.99,
    long_term_debt=1_206.80, total_liabilities=4_620.62,
    retained_earnings=4_850.0, book_equity=7_107.96,
)
TRENT_FY25 = FinancialYear(
    label="FY25", revenue=17_134.61, cogs=9_689.14, sga=4_687.02,
    depreciation=895.18, ebitda=2_758.45, net_income=1_534.41,
    cfo=1_660.90, cfi=-923.67, receivables=62.95, inventory=2_045.05,
    payables=928.19, current_assets=3_673.45, ppe=3_794.99, cwip=179.50,
    investments=845.56, total_assets=9_419.64, current_liabilities=2_010.69,
    long_term_debt=1_601.00, total_liabilities=3_836.28,
    retained_earnings=3_326.0, book_equity=5_583.36,
)
# Debt includes Ind AS 116 leases, consistent with post-lease EBITDA.
TRENT_GROSS_DEBT = 499.89 + 1_206.80 + 854.56
TRENT_WACC = WaccInputs(
    risk_free=0.066,
    beta=1.0,
    equity_risk_premium=0.06,
    pre_tax_cost_of_debt=0.08,
    tax_rate=TRENT_TAX_RATE,
    market_cap=TRENT_CMP * TRENT_SHARES_CR,
    gross_debt=TRENT_GROSS_DEBT,
)
TRENT_DCF = DCFInputs(
    base_revenue=TRENT_FY26.revenue,
    base_ebit_margin=TRENT_FY26.ebit / TRENT_FY26.revenue,
    tax_rate=TRENT_TAX_RATE,
    incremental_nwc=0.06,
    incremental_fixed_capital=0.35,
    terminal_roic=0.22,
    net_debt=TRENT_GROSS_DEBT - 281.36 - 1.64,
    non_controlling_interest=123.28,
    non_operating_assets=737.38 + 647.53,
    shares_outstanding=TRENT_SHARES_CR,
    cmp=TRENT_CMP,
    da_to_revenue=TRENT_FY26.depreciation / TRENT_FY26.revenue,
)

KAYNES_HISTORY = pd.DataFrame({
    "Period": ["FY22", "FY23", "FY24", "FY25", "FY26"],
    "Revenue": [706.0, 1_126.0, 1_805.0, 2_721.75, 3_626.35],
    "EBITDA": [95.0, 170.0, 257.0, 410.70, 574.06],
    "PAT": [42.0, 95.0, 183.0, 293.43, 363.89],
    "CFO": [21.0, -42.0, 88.0, -82.32, -600.40],
    "Debtor days": [102.0, 74.0, 72.0, 77.05, 153.76],
    "Cash conversion cycle": [148.0, 162.0, 123.0, 102.32, 192.61],
})
TRENT_HISTORY = pd.DataFrame({
    "Period": ["FY22", "FY23", "FY24", "FY25", "FY26"],
    "Revenue": [4_498.0, 8_242.0, 12_375.0, 17_134.61, 20_074.21],
    "EBITDA": [595.0, 1_114.0, 1_971.0, 2_758.45, 3_673.39],
    "PAT": [35.0, 394.0, 1_477.0, 1_534.41, 1_721.33],
    "CFO": [58.0, 595.0, 1_349.0, 1_660.90, 2_667.62],
    "Debtor days": [1.0, 2.0, 2.0, 1.34, 0.93],
    "Cash conversion cycle": [74.0, 55.0, 45.0, 43.0, 36.0],
})


def trent_unit_economics() -> pd.DataFrame:
    """Return Trent's retail unit-economics table."""
    cur, pri = TRENT_FY26, TRENT_FY25
    per_store_cur = cur.revenue / TRENT_STORES_FY26
    per_store_pri = pri.revenue / TRENT_STORES_FY25
    net_new_stores = TRENT_STORES_FY26 - TRENT_STORES_FY25
    per_sqft = cur.revenue * 10 / TRENT_AREA_MN_SQFT_FY26
    occupancy = [TRENT_OCCUPANCY[y.label] / y.revenue for y in (pri, cur)]
    employee = [TRENT_EMPLOYEE[y.label] / y.revenue for y in (pri, cur)]
    rows = [
        ("Westside + Zudio stores, year-end",
         f"{TRENT_STORES_FY25:,}", f"{TRENT_STORES_FY26:,}"),
        ("Revenue per closing store (₹ Cr)",
         f"{per_store_pri:.2f}", f"{per_store_cur:.2f}"),
        ("Retail area, year-end (mn sq ft)", "n/a",
         f"{TRENT_AREA_MN_SQFT_FY26:.1f}"),
        ("Revenue per closing sq ft (₹)", "n/a", f"{per_sqft:,.0f}"),
        ("Gross margin", f"{pri.gross_margin:.1%}", f"{cur.gross_margin:.1%}"),
        ("Occupancy cost / revenue", f"{occupancy[0]:.1%}",
         f"{occupancy[1]:.1%}"),
        ("Employee cost / revenue", f"{employee[0]:.1%}",
         f"{employee[1]:.1%}"),
        ("EBITDA margin", f"{pri.ebitda / pri.revenue:.1%}",
         f"{cur.ebitda / cur.revenue:.1%}"),
        ("Inventory days", f"{pri.inventory_days:.0f}",
         f"{cur.inventory_days:.0f}"),
        ("Capex per net new store (₹ Cr)", "n/a",
         f"{TRENT_CAPEX_FY26 / net_new_stores:.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", pri.label, cur.label])


KAYNES_PROFILE = CaseProfile(
    key="kaynes",
    name="Kaynes Technology",
    focus="Forensic Red Flags + Faded DCF",
    current=KAYNES_FY26,
    prior=KAYNES_FY25,
    wacc=KAYNES_WACC,
    dcf=replace(KAYNES_DCF, da_to_revenue=0.03),
    cagr_mean=0.27,
    cagr_std=0.05,
    peak_margin=(0.125, 0.15, 0.18),
    margin_fade=(0.0, 0.015, 0.035),
    growth_range=(3.5, 6.5),
    exit_range=(12.0, 22.0),
    history=KAYNES_HISTORY,
    unit_economics=None,
    context=(
        "FY26 consolidates acquired businesses, and press reports attribute "
        "part of the receivables build-up to smart metering. Year-end "
        "acquired debtors against part-year revenue can overstate DSRI, but "
        "the M-Score stays above -1.78 even with DSRI at a normal level. "
        "Roughly half of the Sloan ratio comes from investing outflows."
    ),
    provenance=(
        "Audited consolidated results filed 13 May 2026; FY22-FY24 trend "
        "from Screener.in. CMP as supplied for the case study. Retained "
        "earnings are estimated from cumulative profit. WACC inputs, growth, "
        "margins, reinvestment and exit range are analyst assumptions."
    ),
)
TRENT_PROFILE = CaseProfile(
    key="trent",
    name="Trent Limited",
    focus="Retail Unit Economics & Compounding",
    current=TRENT_FY26,
    prior=TRENT_FY25,
    wacc=TRENT_WACC,
    dcf=TRENT_DCF,
    cagr_mean=0.20,
    cagr_std=0.04,
    peak_margin=(0.105, 0.125, 0.145),
    margin_fade=(0.0, 0.01, 0.025),
    growth_range=(3.5, 6.5),
    exit_range=(16.0, 28.0),
    history=TRENT_HISTORY,
    unit_economics=trent_unit_economics(),
    context=(
        "Under Ind AS 116, lease principal sits in financing, so CFO "
        "flatters cash conversion for a retailer. After FY26 lease principal "
        f"of ₹{TRENT_LEASE_PRINCIPAL_FY26:,.0f} Cr, CFO / EBITDA falls to "
        f"{(TRENT_FY26.cfo - TRENT_LEASE_PRINCIPAL_FY26) / TRENT_FY26.ebitda:.2f}x"  # noqa: E501
        ". Depreciation also includes right-of-use assets, which affects "
        "DEPI and the AQI asset base."
    ),
    provenance=(
        "Audited consolidated results filed 22 April 2026; FY22-FY24 trend "
        "from Screener.in; store data from press coverage of the FY26 "
        "update, with FY25 derived from reported net additions. CMP is the "
        "8 Nov 2026 close, on post-bonus shares. Retained earnings are "
        "estimated. WACC inputs, growth, margins, reinvestment and exit "
        "range are analyst assumptions."
    ),
)
PROFILES = {
    f"{KAYNES_PROFILE.name} ({KAYNES_PROFILE.focus})": KAYNES_PROFILE,
    f"{TRENT_PROFILE.name} ({TRENT_PROFILE.focus})": TRENT_PROFILE,
}
CUSTOM_OPTION = "Custom / Manual Input"


def inject_styles() -> None:
    """Apply the slate theme and component styles."""
    st.markdown(f"""
    <style>
    .stApp, [data-testid="stHeader"] {{
        background: {SURFACE}; color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background: {PANEL}; border-right: 1px solid {BORDER};
    }}
    .block-container {{ padding-top: 3.8rem; max-width: 1400px; }}
    .ft-title {{
        font-size: 1.55rem; font-weight: 650; color: {TEXT};
        line-height: 1.3; margin: 0;
    }}
    .ft-author {{ color: {MUTED}; font-size: .88rem; margin: .25rem 0 .9rem; }}
    .ft-card {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-top: 3px solid var(--status); border-radius: 8px;
        padding: 14px 16px; min-height: 168px; margin-bottom: 8px;
    }}
    .ft-card-label {{ color: {MUTED}; font-size: .8rem; }}
    .ft-card-value {{
        color: {TEXT}; font-size: 1.85rem; font-weight: 650;
        margin: 2px 0 4px;
    }}
    .ft-status {{ color: var(--status); font-size: .76rem; font-weight: 700; }}
    .ft-card-note {{ color: {MUTED}; font-size: .78rem; margin-top: 3px; }}
    .ft-banner {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-left: 4px solid var(--status); border-radius: 6px;
        padding: 12px 16px; margin-bottom: 10px;
    }}
    .ft-banner-head {{
        display: flex; flex-wrap: wrap; gap: 4px 12px;
        align-items: baseline; color: {TEXT}; font-weight: 650;
    }}
    .ft-banner-rule {{ color: {MUTED}; font-size: .78rem; font-weight: 400; }}
    .ft-banner-body {{
        color: {TEXT}; font-size: .9rem; margin-top: 5px; line-height: 1.45;
    }}
    </style>
    """, unsafe_allow_html=True)


def inr(value: float) -> str:
    """Format a per-share value in rupees."""
    return f"₹{value:,.0f}"


def crore(value: float) -> str:
    """Format a monetary value in rupee crore."""
    return f"₹{value:,.0f} Cr"


def render_header() -> None:
    """Render the application title and author tag."""
    st.markdown(
        f'<p class="ft-title">{html.escape(APP_TITLE)}</p>'
        f'<p class="ft-author">{html.escape(AUTHOR_TAG)}</p>',
        unsafe_allow_html=True,
    )


def custom_defaults_table() -> pd.DataFrame:
    """Return the statement entry grid, seeded with the Kaynes template."""
    return pd.DataFrame(
        {
            "Prior year": [
                float(getattr(KAYNES_FY25, attr)) for _, attr in CUSTOM_FIELDS
            ],
            "Current year": [
                float(getattr(KAYNES_FY26, attr)) for _, attr in CUSTOM_FIELDS
            ],
        },
        index=[label for label, _ in CUSTOM_FIELDS],
    )


def financial_year_from_column(label: str, column: pd.Series) -> FinancialYear:
    """Build a ``FinancialYear`` from one column of the entry grid."""
    if column.isna().any():
        missing = ", ".join(column[column.isna()].index)
        raise ValueError(f"{label}: enter a value for {missing}.")
    values = {
        attribute: float(column[field_label])
        for field_label, attribute in CUSTOM_FIELDS
    }
    return FinancialYear(label=label, **values)


def build_custom_profile() -> CaseProfile | None:
    """Render manual-entry controls and return the resulting profile.

    Returns:
        The custom profile, or ``None`` if the inputs are invalid.
    """
    with st.expander("Custom company inputs", expanded=True):
        st.warning(
            "This form starts from the Kaynes FY25 and FY26 figures as a "
            "template. Overwrite every field before relying on the output."
        )
        name_col, prior_col, current_col = st.columns([2, 1, 1])
        name = name_col.text_input("Company name", "Custom Company")
        prior_label = prior_col.text_input("Prior period", "FY25")
        current_label = current_col.text_input("Current period", "FY26")

        st.markdown("**Financial statements (₹ crore)**")
        edited = st.data_editor(
            custom_defaults_table(),
            key="custom_statements",
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.2f")
                for column in ("Prior year", "Current year")
            },
        )

        st.markdown("**Market and capital structure**")
        row_one = st.columns(4)
        cmp_price = row_one[0].number_input(
            "CMP (₹)", min_value=0.01, value=3_210.0)
        shares = row_one[1].number_input(
            "Shares (crore)", min_value=0.0001, value=6.7035, format="%.4f")
        gross_debt = row_one[2].number_input(
            "Gross debt incl. leases", min_value=0.0, value=912.87)
        cash = row_one[3].number_input(
            "Cash and bank", min_value=0.0, value=798.59)
        row_two = st.columns(5)
        minority = row_two[0].number_input(
            "Minority interest", min_value=0.0, value=14.14)
        non_operating = row_two[1].number_input(
            "Non-operating assets", min_value=0.0, value=257.05)
        risk_free = row_two[2].number_input(
            "Risk-free (%)", min_value=0.0, max_value=20.0, value=6.6)
        beta = row_two[3].number_input(
            "Beta", min_value=0.0, max_value=4.0, value=1.25)
        premium = row_two[4].number_input(
            "Equity premium (%)", min_value=0.0, max_value=20.0, value=6.0)
        row_three = st.columns(2)
        debt_cost = row_three[0].number_input(
            "Pre-tax debt cost (%)", min_value=0.0, max_value=30.0, value=9.5)
        tax_rate = row_three[1].number_input(
            "Tax rate (%)", min_value=0.0, max_value=60.0, value=25.17)

    try:
        prior = financial_year_from_column(prior_label, edited["Prior year"])
        current = financial_year_from_column(
            current_label, edited["Current year"])
    except ValueError as error:
        st.error(f"Check the inputs. {error}")
        return None
    if prior.ebitda == 0 or current.ebitda == 0:
        st.error("Check the inputs. EBITDA cannot be exactly zero.")
        return None

    tax = tax_rate / 100
    wacc = WaccInputs(
        risk_free=risk_free / 100,
        beta=beta,
        equity_risk_premium=premium / 100,
        pre_tax_cost_of_debt=debt_cost / 100,
        tax_rate=tax,
        market_cap=cmp_price * shares,
        gross_debt=gross_debt,
    )
    dcf = DCFInputs(
        base_revenue=current.revenue,
        base_ebit_margin=current.ebit / current.revenue,
        tax_rate=tax,
        incremental_nwc=0.20,
        incremental_fixed_capital=0.30,
        terminal_roic=0.18,
        net_debt=gross_debt - cash,
        non_controlling_interest=minority,
        non_operating_assets=non_operating,
        shares_outstanding=shares,
        cmp=cmp_price,
        da_to_revenue=current.depreciation / current.revenue,
    )
    base_margin = dcf.base_ebit_margin
    history = pd.DataFrame({
        "Period": [prior.label, current.label],
        "Revenue": [prior.revenue, current.revenue],
        "EBITDA": [prior.ebitda, current.ebitda],
        "PAT": [prior.net_income, current.net_income],
        "CFO": [prior.cfo, current.cfo],
        "Debtor days": [prior.debtor_days, current.debtor_days],
        "Cash conversion cycle": [
            prior.cash_conversion_cycle, current.cash_conversion_cycle],
    })
    return CaseProfile(
        key="custom",
        name=name,
        focus="Manual input",
        current=current,
        prior=prior,
        wacc=wacc,
        dcf=dcf,
        cagr_mean=0.15,
        cagr_std=0.04,
        peak_margin=(base_margin - 0.02, base_margin, base_margin + 0.03),
        margin_fade=(0.0, 0.01, 0.03),
        growth_range=(3.5, 6.5),
        exit_range=(10.0, 20.0),
        history=history,
        unit_economics=None,
        context="Results depend entirely on the figures entered above.",
        provenance="User-entered figures and assumptions.",
    )


def shifted_triangle(default: tuple[float, float, float],
                     mode: float) -> tuple[float, float, float]:
    """Move a triangular distribution so that its mode equals ``mode``."""
    shift = mode - default[1]
    low = max(default[0] + shift, 0.0)
    return low, max(mode, low), default[2] + shift


def slider_default(value: float, low: float, high: float,
                   digits: int = 1) -> float:
    """Round a default value and keep it inside a slider's bounds."""
    return min(max(round(value, digits), low), high)


def render_sidebar_controls(profile: CaseProfile) -> RunSettings | None:
    """Render the valuation sidebar and resolve the run settings.

    Args:
        profile: The selected case.

    Returns:
        Settings for the run, or ``None`` if the ranges are invalid.
    """
    sidebar = st.sidebar
    key = profile.key
    capm_wacc = profile.wacc.wacc * 100

    sidebar.header("Simulation")
    runs = sidebar.slider(
        "Simulation runs", 1_000, 10_000, 10_000, step=500, key=f"{key}_runs")

    low_default = slider_default(capm_wacc - 3, 6.0, 24.0, 2)
    high_default = slider_default(capm_wacc + 3, 7.0, 25.0, 2)
    wacc_low, wacc_high = sidebar.slider(
        "WACC range (%)", 6.0, 25.0, (low_default, high_default), step=0.01,
        format="%.2f", key=f"{key}_wacc",
        help="Normal distribution centred on the midpoint, with the range "
             "spanning ±3 standard deviations.",
    )
    sidebar.caption(f"CAPM WACC for this case: {capm_wacc:.2f}%")

    growth_low, growth_high = sidebar.slider(
        "Terminal growth g (%)", 0.0, 9.0, profile.growth_range, step=0.1,
        key=f"{key}_growth",
        help="Triangular distribution with its mode at the midpoint.",
    )
    exit_low, exit_high = sidebar.slider(
        "Exit EV/EBITDA multiple (x)", 2.0, 60.0, profile.exit_range,
        step=0.5, key=f"{key}_exit",
        help="Triangular distribution with its mode at the midpoint.",
    )

    with sidebar.expander("Operating assumptions"):
        cagr_mean = st.slider(
            "Revenue CAGR mean (%)", 0.0, 60.0,
            round(profile.cagr_mean * 100, 1), step=0.5, key=f"{key}_cagr")
        cagr_std = st.slider(
            "Revenue CAGR std dev (%)", 0.5, 15.0,
            round(profile.cagr_std * 100, 1), step=0.5, key=f"{key}_cagr_sd")
        peak_mode = st.slider(
            "Peak EBIT margin, mode (%)", 1.0, 50.0,
            slider_default(profile.peak_margin[1] * 100, 1.0, 50.0),
            step=0.1, key=f"{key}_peak")
        fade_mode = st.slider(
            "Margin fade by year 10, mode (pp)", 0.0, 10.0,
            round(profile.margin_fade[1] * 100, 1), step=0.1,
            key=f"{key}_fade")
        nwc = st.slider(
            "Working capital per ₹1 added revenue", 0.0, 1.0,
            profile.dcf.incremental_nwc, step=0.01, key=f"{key}_nwc")
        fixed = st.slider(
            "Net capex per ₹1 added revenue", 0.0, 1.5,
            profile.dcf.incremental_fixed_capital, step=0.01,
            key=f"{key}_fixed")
        roic = st.slider(
            "Terminal ROIC (%)", 8.0, 50.0,
            round(profile.dcf.terminal_roic * 100, 1), step=0.5,
            key=f"{key}_roic")
        depreciation = st.slider(
            "Depreciation / revenue (%)", 0.0, 15.0,
            slider_default(profile.dcf.da_to_revenue * 100, 0.0, 15.0, 2),
            step=0.05, key=f"{key}_da")
        seed = st.number_input(
            "Random seed", min_value=0, value=42, step=1, key=f"{key}_seed")

    if wacc_high <= wacc_low or growth_high <= growth_low \
            or exit_high <= exit_low:
        st.error("Each range slider needs a high value above its low value.")
        return None

    floor = growth_high + 3.0
    if (wacc_low + wacc_high) / 2 <= floor:
        st.error(
            f"The WACC midpoint must exceed {floor:.1f}%, which is the "
            "highest terminal growth plus a 3-point minimum spread. Raise "
            "WACC or lower terminal growth."
        )
        return None
    if roic <= growth_high:
        st.error("Terminal ROIC must exceed the highest terminal growth.")
        return None
    if wacc_low < floor:
        sidebar.caption(
            f"WACC paths below {floor:.1f}% are excluded to keep Gordon "
            "growth defined."
        )

    peak = peak_mode / 100
    fade = fade_mode / 100
    config = SimulationConfig(
        n_paths=runs,
        seed=int(seed),
        wacc_range=(wacc_low / 100, wacc_high / 100),
        terminal_growth=(
            growth_low / 100,
            (growth_low + growth_high) / 200,
            growth_high / 100,
        ),
        cagr_mean=cagr_mean / 100,
        cagr_std=cagr_std / 100,
        peak_margin=shifted_triangle(profile.peak_margin, peak),
        margin_fade=shifted_triangle(profile.margin_fade, fade),
        exit_multiple=(exit_low, (exit_low + exit_high) / 2, exit_high),
        exit_basis="EBITDA",
    )
    dcf = replace(
        profile.dcf,
        incremental_nwc=nwc,
        incremental_fixed_capital=fixed,
        terminal_roic=roic / 100,
        da_to_revenue=depreciation / 100,
    )
    return RunSettings(dcf=dcf, config=config)


@st.cache_data(show_spinner="Running Monte Carlo paths...", max_entries=48)
def run_valuation(
    name: str,
    dcf: DCFInputs,
    wacc: WaccInputs,
    config: SimulationConfig,
) -> tuple[ValuationResult, pd.DataFrame]:
    """Run and cache the simulation and central-case forecast."""
    model = MonteCarloValuation(name, dcf, wacc, config)
    return model.run(), model.central_forecast()


def alert_for(alerts: list[Alert], test: str) -> Alert:
    """Return the alert with the given test name."""
    return next(alert for alert in alerts if alert.test == test)


def metric_card(title: str, alert: Alert, note: str) -> str:
    """Return the HTML for a forensic metric card."""
    color = STATUS_COLOR[alert.severity]
    return (
        f'<div class="ft-card" style="--status:{color}">'
        f'<div class="ft-card-label">{html.escape(title)}</div>'
        f'<div class="ft-card-value">{html.escape(alert.reading)}</div>'
        f'<div class="ft-status">{STATUS_LABEL[alert.severity]}</div>'
        f'<div class="ft-card-note">{html.escape(note)}</div>'
        "</div>"
    )


def banner(alert: Alert) -> str:
    """Return the HTML for a forensic warning banner."""
    color = STATUS_COLOR[alert.severity]
    explanation = ALERT_EXPLANATIONS.get(alert.test, alert.comment)
    return (
        f'<div class="ft-banner" style="--status:{color}">'
        '<div class="ft-banner-head">'
        f'<span class="ft-status">{STATUS_LABEL[alert.severity]}</span>'
        f"<span>{html.escape(alert.test)}: "
        f"{html.escape(alert.reading)}</span>"
        f'<span class="ft-banner-rule">{html.escape(alert.rule)}</span>'
        "</div>"
        f'<div class="ft-banner-body">{html.escape(explanation)}</div>'
        "</div>"
    )


def render_forensic_tab(profile: CaseProfile, auditor: ForensicAuditor,
                        alerts: list[Alert]) -> None:
    """Render metric cards, warning banners and the case notes."""
    red_count = sum(a.severity is Severity.RED for a in alerts)
    amber_count = sum(a.severity is Severity.AMBER for a in alerts)
    st.subheader(
        f"{profile.name}: {red_count} red flags and {amber_count} watch "
        f"items, {profile.current.label} vs {profile.prior.label}"
    )

    columns = st.columns(4)
    cards = (
        (BENEISH, "Beneish M-Score", "Above -1.78 fits the manipulator group"),
        (ALTMAN, "Altman Z''-Score (EM)",
         "Safe above 5.85, distress below 4.35"),
        (SLOAN, "Sloan Accrual Ratio", "Includes investing cash flows"),
        (CONVERSION, "CFO / EBITDA", "Cash realised per rupee of EBITDA"),
    )
    for column, (test, title, note) in zip(columns, cards):
        column.markdown(metric_card(title, alert_for(alerts, test), note),
                        unsafe_allow_html=True)

    cur = profile.current
    with columns[0].popover("Index breakdown", width="stretch"):
        st.dataframe(auditor.beneish_table().round(3), width="stretch")
    with columns[1].popover("Ratio breakdown", width="stretch"):
        table = pd.Series(auditor.altman_components(), name="Value")
        st.dataframe(table.round(3).to_frame(), width="stretch")
    with columns[2].popover("Calculation", width="stretch"):
        st.write(f"Net income: {crore(cur.net_income)}")
        st.write(f"Less CFO: {crore(cur.cfo)}")
        st.write(f"Less CFI: {crore(cur.cfi)}")
        st.write(f"Total assets: {crore(cur.total_assets)}")
    with columns[3].popover("Cash conversion detail", width="stretch"):
        st.dataframe(auditor.cash_conversion_quality().round(2),
                     width="stretch")
        divergence = auditor.debtor_sales_divergence()
        st.write(f"Receivable growth: {divergence['receivable_growth']:.1%}")
        st.write(f"Sales growth: {divergence['sales_growth']:.1%}")

    st.markdown("#### Warning banners")
    warnings = [a for a in alerts if a.severity is not Severity.GREEN]
    if warnings:
        st.markdown("".join(banner(alert) for alert in warnings),
                    unsafe_allow_html=True)
    else:
        st.success("No screen breached its amber or red threshold.")
    clean = [a for a in alerts if a.severity is Severity.GREEN]
    if clean:
        with st.expander(f"Clean screens ({len(clean)})"):
            st.markdown("".join(banner(alert) for alert in clean),
                        unsafe_allow_html=True)

    st.info(profile.context)
    st.caption(DISCLAIMER)


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex colour to a Plotly rgba string."""
    red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def add_distribution_panel(fig: go.Figure, dist: DistributionStats,
                           name: str, color: str, edges: np.ndarray,
                           column: int) -> float:
    """Draw one histogram and density curve, returning the peak density."""
    values = dist.values
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts / (len(values) * widths)
    fig.add_bar(
        x=(edges[:-1] + edges[1:]) / 2, y=density, width=widths * 0.88,
        name=name, legendgroup=name,
        marker={"color": hex_to_rgba(color, 0.42), "line": {"width": 0}},
        customdata=np.column_stack((edges[:-1], edges[1:],
                                    counts / len(values))),
        hovertemplate=(f"<b>{name}</b><br>"
                       "₹%{customdata[0]:,.0f} to ₹%{customdata[1]:,.0f}<br>"
                       "%{customdata[2]:.1%} of paths<extra></extra>"),
        row=1, col=column,
    )
    peak = float(density.max()) if len(density) else 0.0
    visible = values[(values >= edges[0]) & (values <= edges[-1])]
    if len(visible) > 1 and np.ptp(visible) > 0:
        grid = np.linspace(visible.min(), visible.max(), 300)
        curve = stats.gaussian_kde(values)(grid)
        fig.add_scatter(
            x=grid, y=curve, mode="lines", name=name, legendgroup=name,
            showlegend=False, line={"color": color, "width": 2},
            hovertemplate=f"₹%{{x:,.0f}}<extra>{name}</extra>",
            row=1, col=column,
        )
        peak = max(peak, float(curve.max()))
    return peak


def add_reference_lines(fig: go.Figure, dist: DistributionStats,
                        cmp_price: float, column: int, top: float,
                        min_gap: float) -> None:
    """Draw P5, median, P95 and CMP lines with staggered labels."""
    markers = sorted([
        (dist.percentiles[5], "P5", MUTED, "dot"),
        (dist.percentiles[50], "Median", MUTED, "dash"),
        (dist.percentiles[95], "P95", MUTED, "dot"),
        (cmp_price, "CMP", TEXT, "solid"),
    ])
    previous, level = -np.inf, 0
    for value, label, color, dash in markers:
        level = level + 1 if value - previous < min_gap else 0
        previous = value
        label_y = top * (1.06 + 0.15 * level)
        fig.add_shape(
            type="line", x0=value, x1=value, y0=0, y1=label_y,
            line={"color": color, "width": 1.5, "dash": dash},
            row=1, col=column,
        )
        fig.add_annotation(
            x=value, y=label_y, yanchor="bottom", showarrow=False,
            text=f"<b>{label}</b> {inr(value)}",
            font={"color": TEXT, "size": 11},
            row=1, col=column,
        )


def distribution_figure(result: ValuationResult) -> go.Figure:
    """Build side-by-side Gordon and exit-multiple distribution panels."""
    gordon, exit_dist = result.gordon, result.exit_multiple
    pooled = np.concatenate((gordon.values, exit_dist.values))
    low_cut, high_cut = np.percentile(pooled, [0.1, 99.9])
    if high_cut <= low_cut:
        high_cut = low_cut + 1.0
    low, high = min(low_cut, result.cmp), max(high_cut, result.cmp)
    padding = (high - low) * 0.04
    x_range = [max(low - padding, 0), high + padding]
    edges = np.linspace(low_cut, high_cut, 76)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
        subplot_titles=("Gordon growth terminal value",
                        "Exit EV/EBITDA terminal value"),
    )
    top = max(
        add_distribution_panel(fig, gordon, "Gordon growth", SERIES_GORDON,
                               edges, 1),
        add_distribution_panel(fig, exit_dist, "Exit multiple", SERIES_EXIT,
                               edges, 2),
    )
    min_gap = (x_range[1] - x_range[0]) * 0.17
    add_reference_lines(fig, gordon, result.cmp, 1, top, min_gap)
    add_reference_lines(fig, exit_dist, result.cmp, 2, top, min_gap)

    fig.update_xaxes(
        range=x_range, tickformat=",.0f", showgrid=False, linecolor=BORDER,
        title_text="Value per share (₹)", color=MUTED,
    )
    fig.update_yaxes(
        range=[0, top * 1.72], showticklabels=False, gridcolor=GRID,
        zeroline=False,
    )
    fig.update_yaxes(title_text="Probability density", color=MUTED,
                     row=1, col=1)
    fig.update_annotations(font_color=TEXT)
    fig.update_layout(
        height=480, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"color": TEXT}, bargap=0, hovermode="closest",
        showlegend=True, margin={"t": 70, "l": 50, "r": 20, "b": 55},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.13},
    )
    return fig


def percentile_table(result: ValuationResult) -> pd.DataFrame:
    """Return the two valuation distributions side by side."""
    rows = {}
    for name, dist in (("Gordon growth", result.gordon),
                       ("Exit multiple", result.exit_multiple)):
        column = {}
        for level, value in dist.percentiles.items():
            label = "Median" if level == 50 else f"P{level}"
            column[label] = f"{inr(value)} ({value / result.cmp - 1:+.1%})"
        column["Mean"] = inr(dist.mean_value)
        column["Central case"] = inr(dist.base_case)
        column["Terminal value share of EV"] = (
            f"{dist.median_terminal_share:.1%}")
        rows[name] = column
    return pd.DataFrame(rows)


def render_risk_metrics(name: str, dist: DistributionStats) -> None:
    """Render median, downside probability and VaR for one method."""
    st.markdown(f"**{name}**")
    columns = st.columns(3)
    columns[0].metric("Median value", inr(dist.percentiles[50]))
    columns[1].metric("Downside probability",
                      f"{dist.probability_of_downside:.1%}")
    columns[2].metric("VaR (95%)", inr(dist.var_95),
                      f"{dist.var_95_pct:.1%} of CMP", delta_color="off",
                      delta_arrow="off")


def render_valuation_tab(profile: CaseProfile, settings: RunSettings,
                         result: ValuationResult) -> None:
    """Render the Monte Carlo tab."""
    gordon, exit_dist = result.gordon, result.exit_multiple
    st.subheader(
        f"{profile.name}: {gordon.probability_of_downside:.1%} of Gordon "
        f"and {exit_dist.probability_of_downside:.1%} of exit-multiple paths "
        f"sit below the {inr(result.cmp)} CMP"
    )
    left, right = st.columns(2)
    with left:
        render_risk_metrics("Gordon growth", gordon)
    with right:
        render_risk_metrics("Exit multiple", exit_dist)

    st.plotly_chart(distribution_figure(result), width="stretch", theme=None,
                    config={"displaylogo": False})
    st.caption(
        f"{len(gordon.values):,} paths. Both panels share one value axis. "
        "VaR is the shortfall of the 5th percentile value against CMP, "
        "not a market-price VaR."
    )

    table_col, check_col = st.columns([3, 2])
    with table_col:
        st.markdown("#### Value per share vs CMP")
        st.dataframe(percentile_table(result), width="stretch")
    with check_col:
        st.markdown("#### Terminal value cross-check")
        config = settings.config
        cross_check = pd.DataFrame({
            "Measure": [
                "Central exit EV/EBITDA",
                "EV/EBITDA implied by Gordon",
                "Exit EV/EBITDA needed for CMP",
                "Growth implied by exit multiple",
                "Central terminal growth",
            ],
            "Value": [
                f"{config.exit_multiple[1]:.1f}x",
                f"{result.median_gordon_multiple:.1f}x",
                f"{result.breakeven_multiple:.1f}x",
                f"{result.median_implied_growth:.1%}",
                f"{config.terminal_growth[1]:.1%}",
            ],
        })
        st.dataframe(cross_check, width="stretch", hide_index=True)
        if result.median_implied_growth > config.terminal_growth[2]:
            st.warning(
                "The exit multiple implies perpetual growth above your "
                "terminal range. Treat it as a market-sentiment value "
                "rather than an intrinsic one."
            )


def forecast_periods(label: str, count: int) -> list[str]:
    """Return forecast period names following the last actual period."""
    match = re.fullmatch(r"FY(\d{2})", label.strip(), flags=re.IGNORECASE)
    if match:
        start = int(match.group(1))
        return [f"FY{start + offset}" for offset in range(1, count + 1)]
    return [f"Year {offset}" for offset in range(1, count + 1)]


def bridge_figure(profile: CaseProfile, settings: RunSettings,
                  forecast: pd.DataFrame, dist: DistributionStats,
                  method: str) -> go.Figure:
    """Build the enterprise-to-equity value waterfall for the central case."""
    dcf = settings.dcf
    stage_pv = forecast.groupby("Stage")["PV of FCFF"].sum()
    net_claims = (dcf.net_debt + dcf.non_controlling_interest
                  - dcf.non_operating_assets)
    equity = dist.base_case * dcf.shares_outstanding
    terminal = equity + net_claims - stage_pv.sum()
    steps = [
        ("PV explicit<br>FCFF", stage_pv["Explicit"], "relative"),
        ("PV fade<br>FCFF", stage_pv["Fade"], "relative"),
        (f"PV terminal<br>{method}", terminal, "relative"),
        ("Enterprise<br>value", 0.0, "total"),
        ("Net debt", -dcf.net_debt, "relative"),
        ("Minority<br>interest", -dcf.non_controlling_interest, "relative"),
        ("Non-operating<br>assets", dcf.non_operating_assets, "relative"),
        ("Equity<br>value", 0.0, "total"),
    ]
    enterprise = stage_pv.sum() + terminal
    display = [
        stage_pv["Explicit"], stage_pv["Fade"], terminal, enterprise,
        -dcf.net_debt, -dcf.non_controlling_interest,
        dcf.non_operating_assets, equity,
    ]
    fig = go.Figure(go.Waterfall(
        x=[step[0] for step in steps],
        y=[step[1] for step in steps],
        measure=[step[2] for step in steps],
        text=[f"{value:,.0f}" for value in display],
        textposition="outside",
        textfont={"color": TEXT},
        connector={"line": {"color": BORDER}},
        increasing={"marker": {"color": SERIES_GORDON}},
        decreasing={"marker": {"color": SERIES_EXIT}},
        totals={"marker": {"color": MUTED}},
        hovertemplate="%{x}<br>₹%{text} Cr<extra></extra>",
    ))
    fig.update_layout(
        height=420, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"color": TEXT}, showlegend=False,
        margin={"t": 30, "l": 50, "r": 20, "b": 50},
        yaxis={"title": "₹ crore", "gridcolor": GRID, "color": MUTED,
               "zeroline": False},
        xaxis={"color": MUTED, "tickangle": 0},
    )
    return fig


def trend_figure(periods: list[str], revenue: list[float],
                 lines: dict[str, list[float]]) -> go.Figure:
    """Build a revenue panel beside a panel of profit and cash lines."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=("Revenue (₹ Cr)",
                                        "Earnings and cash (₹ Cr)"))
    fig.add_bar(
        x=periods, y=revenue, name="Revenue", marker_color=SERIES_GORDON,
        hovertemplate="%{x}<br>₹%{y:,.0f} Cr<extra>Revenue</extra>",
        showlegend=False, row=1, col=1,
    )
    palette = (SERIES_GORDON, SERIES_EXIT, SERIES_THIRD)
    for (name, values), color in zip(lines.items(), palette):
        fig.add_scatter(
            x=periods, y=values, mode="lines+markers", name=name,
            line={"color": color, "width": 2}, marker={"size": 8},
            hovertemplate=f"%{{x}}<br>₹%{{y:,.0f}} Cr<extra>{name}</extra>",
            row=1, col=2,
        )
    fig.update_xaxes(color=MUTED, showgrid=False, linecolor=BORDER,
                     type="category")
    fig.update_yaxes(color=MUTED, gridcolor=GRID, zeroline=True,
                     zerolinecolor=BORDER, tickformat=",.0f")
    fig.update_annotations(font_color=TEXT)
    fig.update_layout(
        height=360, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"color": TEXT}, margin={"t": 60, "l": 50, "r": 20, "b": 40},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.2},
        hovermode="closest", bargap=0.45,
    )
    return fig


def statements_table(profile: CaseProfile) -> pd.DataFrame:
    """Return the two-year statement lines with year-on-year change."""
    rows = []
    for section, label, attribute in STATEMENT_LINES:
        prior = float(getattr(profile.prior, attribute))
        current = float(getattr(profile.current, attribute))
        change = (current / prior - 1) * 100 if prior > 0 else None
        rows.append({
            "Statement": section,
            "Line item": label,
            profile.prior.label: prior,
            profile.current.label: current,
            "Change %": change,
        })
    return pd.DataFrame(rows)


def cagr_text(series: pd.Series) -> str:
    """Return a CAGR across a series, or n/m with a non-positive endpoint."""
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0 or end <= 0 or len(series) < 2:
        return "n/m"
    return f"{(end / start) ** (1 / (len(series) - 1)) - 1:.1%}"


def render_statements_tab(profile: CaseProfile, settings: RunSettings,
                          result: ValuationResult,
                          forecast: pd.DataFrame) -> None:
    """Render forecast, waterfall, statements and historical views."""
    periods = forecast_periods(profile.current.label, len(forecast))
    display = forecast.drop(columns=["Year"]).round({
        column: 0 for column in ("Revenue", "EBITDA", "EBIT", "NOPAT",
                                 "Reinvestment", "FCFF", "PV of FCFF")
    })
    display.insert(0, "Period", periods)
    display[["Growth", "EBIT margin"]] *= 100

    st.subheader("Central-case forecast")
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Growth": st.column_config.NumberColumn(format="%.1f%%"),
            "EBIT margin": st.column_config.NumberColumn(format="%.1f%%"),
            "Discount factor": st.column_config.NumberColumn(format="%.3f"),
            **{
                column: st.column_config.NumberColumn(format="localized")
                for column in ("Revenue", "EBITDA", "EBIT", "NOPAT",
                               "Reinvestment", "FCFF", "PV of FCFF")
            },
        },
    )
    st.caption("₹ crore. Drivers held at the central values from the sidebar.")
    st.plotly_chart(
        trend_figure(periods, forecast["Revenue"].tolist(), {
            "EBITDA": forecast["EBITDA"].tolist(),
            "FCFF": forecast["FCFF"].tolist(),
        }),
        width="stretch", theme=None, config={"displaylogo": False},
    )

    st.subheader("Enterprise-to-equity waterfall")
    method = st.radio(
        "Terminal value method", ["Gordon growth", "Exit multiple"],
        horizontal=True, key=f"{profile.key}_waterfall",
    )
    dist = result.gordon if method == "Gordon growth" else result.exit_multiple
    st.plotly_chart(
        bridge_figure(profile, settings, forecast, dist, method),
        width="stretch", theme=None, config={"displaylogo": False},
    )
    st.caption(
        f"Equity value divided by {settings.dcf.shares_outstanding:,.2f} "
        f"crore shares gives {inr(dist.base_case)} per share."
    )

    st.subheader("Historical trends")
    history = profile.history
    span = len(history) - 1
    columns = st.columns(3)
    columns[0].metric(f"Revenue CAGR ({span}y)", cagr_text(history["Revenue"]))
    columns[1].metric(f"EBITDA CAGR ({span}y)", cagr_text(history["EBITDA"]))
    columns[2].metric(f"PAT CAGR ({span}y)", cagr_text(history["PAT"]))
    st.plotly_chart(
        trend_figure(history["Period"].tolist(), history["Revenue"].tolist(), {
            "EBITDA": history["EBITDA"].tolist(),
            "PAT": history["PAT"].tolist(),
            "CFO": history["CFO"].tolist(),
        }),
        width="stretch", theme=None, config={"displaylogo": False},
    )
    with st.expander("Historical data table"):
        st.dataframe(history.round(2), width="stretch", hide_index=True)

    if profile.unit_economics is not None:
        st.subheader("Retail unit economics")
        st.dataframe(profile.unit_economics, width="stretch", hide_index=True)
        st.caption(
            "Store metrics use year-end counts, so they understate "
            "productivity in a fast rollout. The FY22 base is depressed by "
            "the pandemic, which inflates the multi-year CAGRs."
        )

    st.subheader("Financial statements")
    st.dataframe(
        statements_table(profile),
        width="stretch",
        hide_index=True,
        column_config={
            profile.prior.label: st.column_config.NumberColumn(
                format="localized"),
            profile.current.label: st.column_config.NumberColumn(
                format="localized"),
            "Change %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


def summary_rows(profile: CaseProfile, auditor: ForensicAuditor,
                 alerts: list[Alert], settings: RunSettings,
                 result: ValuationResult) -> pd.DataFrame:
    """Assemble the report as a Section, Metric, Value table."""
    rows: list[tuple[str, str, object]] = [
        ("Case", "Company", profile.name),
        ("Case", "Periods",
         f"{profile.current.label} vs {profile.prior.label}"),
        ("Case", "Report date", date.today().isoformat()),
        ("Case", "CMP (Rs)", round(result.cmp, 2)),
        ("Forensic scores", "Beneish M-Score",
         round(auditor.beneish_m_score(), 3)),
        ("Forensic scores", "Altman Z''-Score (EM)",
         round(auditor.altman_z_score(), 3)),
        ("Forensic scores", "Sloan accrual ratio (%)",
         round(auditor.sloan_accrual_ratio() * 100, 2)),
        ("Forensic scores", "CFO / EBITDA (x)",
         round(profile.current.cfo / profile.current.ebitda, 3)),
        ("Forensic scores", "Cash conversion cycle (days)",
         round(profile.current.cash_conversion_cycle, 1)),
        ("Forensic scores", "Debtor days",
         round(profile.current.debtor_days, 1)),
    ]
    for alert in alerts:
        outcome = f"{alert.severity.name} | {alert.reading} | {alert.rule}"
        rows.append(("Alerts", alert.test, outcome))
    for name, dist in (("Gordon growth", result.gordon),
                       ("Exit multiple", result.exit_multiple)):
        section = f"Simulation: {name}"
        for level, value in dist.percentiles.items():
            label = "Median" if level == 50 else f"P{level}"
            rows.append((section, f"{label} value per share (Rs)",
                         round(value, 2)))
        rows.extend([
            (section, "Mean value per share (Rs)", round(dist.mean_value, 2)),
            (section, "Central case value per share (Rs)",
             round(dist.base_case, 2)),
            (section, "Probability of downside vs CMP (%)",
             round(dist.probability_of_downside * 100, 2)),
            (section, "VaR 95% (Rs per share)", round(dist.var_95, 2)),
            (section, "VaR 95% (% of CMP)", round(dist.var_95_pct * 100, 2)),
        ])
    config = settings.config
    assumptions = "Assumptions"
    rows.extend([
        (assumptions, "Simulation runs", config.n_paths),
        (assumptions, "WACC range (%)",
         f"{config.wacc_range[0]:.2%} to {config.wacc_range[1]:.2%}"),
        (assumptions, "Terminal growth range",
         f"{config.terminal_growth[0]:.2%} to "
         f"{config.terminal_growth[2]:.2%}"),
        (assumptions, "Exit EV/EBITDA range (x)",
         f"{config.exit_multiple[0]:.1f} to {config.exit_multiple[2]:.1f}"),
        (assumptions, "Revenue CAGR mean (%)",
         round(config.cagr_mean * 100, 2)),
        (assumptions, "Implied EV/EBITDA from Gordon (x)",
         round(result.median_gordon_multiple, 2)),
        (assumptions, "Exit EV/EBITDA needed for CMP (x)",
         round(result.breakeven_multiple, 2)),
        (assumptions, "Growth implied by exit multiple (%)",
         round(result.median_implied_growth * 100, 2)),
        (assumptions, "Data provenance", profile.provenance),
    ])
    return pd.DataFrame(rows, columns=["Section", "Metric", "Value"])


def build_csv(summary: pd.DataFrame) -> bytes:
    """Return the summary as UTF-8 CSV bytes that open cleanly in Excel."""
    return summary.to_csv(index=False).encode("utf-8-sig")


def build_pdf(profile: CaseProfile, summary: pd.DataFrame) -> bytes:
    """Return the summary as a formatted one-document PDF snapshot."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title=f"{profile.name} summary report", author=AUTHOR_TAG[9:],
    )
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 8.5
    body.leading = 11

    def cell(value: object) -> Paragraph:
        return Paragraph(html.escape(str(value)), body)

    story = [
        Paragraph("FinAudit-QuantTerminal | Summary Report", styles["Title"]),
        Paragraph(html.escape(f"{profile.name}. {AUTHOR_TAG}."), body),
        Spacer(1, 6),
    ]
    for section, group in summary.groupby("Section", sort=False):
        story.append(Paragraph(html.escape(str(section)), styles["Heading3"]))
        table = Table(
            [[cell(metric), cell(value)]
             for metric, value in zip(group["Metric"], group["Value"])],
            colWidths=[72 * mm, 106 * mm],
        )
        table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.extend([table, Spacer(1, 4)])
    story.append(Paragraph(html.escape(DISCLAIMER), body))
    document.build(story)
    return buffer.getvalue()


def render_export(profile: CaseProfile, auditor: ForensicAuditor,
                  alerts: list[Alert], settings: RunSettings,
                  result: ValuationResult) -> None:
    """Render the report format selector and download button."""
    sidebar = st.sidebar
    sidebar.header("Export")
    report_format = sidebar.radio(
        "Report format", ["PDF", "CSV"], horizontal=True, key="report_format")
    summary = summary_rows(profile, auditor, alerts, settings, result)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", profile.name).strip("_").lower()
    if report_format == "PDF":
        data = build_pdf(profile, summary)
        mime, suffix = "application/pdf", "pdf"
    else:
        data = build_csv(summary)
        mime, suffix = "text/csv", "csv"
    sidebar.download_button(
        "Download Summary Report",
        data=data,
        file_name=f"{stem or 'company'}_summary_report.{suffix}",
        mime=mime,
        width="stretch",
        on_click="ignore",
    )


def main() -> None:
    """Build the dashboard for the selected case."""
    st.set_page_config(
        page_title="FinAudit-QuantTerminal",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_header()

    st.sidebar.header("Case study")
    choice = st.sidebar.selectbox(
        "Ticker / case", [*PROFILES, CUSTOM_OPTION], key="case_choice")
    profile = (build_custom_profile() if choice == CUSTOM_OPTION
               else PROFILES[choice])
    if profile is None:
        st.stop()
    st.sidebar.caption(profile.provenance)

    settings = render_sidebar_controls(profile)
    if settings is None:
        st.stop()

    auditor = ForensicAuditor(profile.name, profile.current, profile.prior)
    try:
        alerts = auditor.alerts()
        result, forecast = run_valuation(
            profile.name, settings.dcf, profile.wacc, settings.config)
    except (ValueError, ZeroDivisionError) as error:
        st.error(f"These inputs cannot be evaluated: {error}")
        st.stop()

    forensic_tab, valuation_tab, statements_tab = st.tabs([
        "Forensic Audit & Red Flags",
        "Monte Carlo Valuation",
        "Financial Statements & Waterfall",
    ])
    with forensic_tab:
        render_forensic_tab(profile, auditor, alerts)
    with valuation_tab:
        render_valuation_tab(profile, settings, result)
    with statements_tab:
        render_statements_tab(profile, settings, result, forecast)

    render_export(profile, auditor, alerts, settings, result)


if __name__ == "__main__":
    main()
