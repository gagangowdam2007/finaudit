#!/usr/bin/env python3
"""FinAudit-QuantTerminal: forensic audit and probabilistic DCF engine.

The engine automates two desk workflows for Indian listed equities:

``ForensicAuditor``
    Earnings-quality screens: Beneish M-Score, Sloan accrual ratio, cash
    conversion quality and the Altman Z''-Score for emerging markets.

``MonteCarloValuation``
    A 2-stage faded DCF (5-year explicit forecast and 5-year fade to
    terminal growth) simulated over 10,000 paths, with terminal value
    computed by both Gordon growth and an exit EV/EBIT multiple.

Running this module executes the Kaynes Technology FY26 validation case.
All monetary inputs are in ₹ crore, except per-share values in ₹.

Usage::

    python quant_terminal_engine.py
    python quant_terminal_engine.py --paths 50000 --seed 7 --no-show

Forensic scores are statistical screens. A flag is a reason to read the
notes to the accounts; it is not a finding of manipulation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

DAYS_IN_YEAR = 365
EXPLICIT_YEARS = 5
FADE_YEARS = 5
TOTAL_YEARS = EXPLICIT_YEARS + FADE_YEARS
PERCENTILES = (5, 25, 50, 75, 95)

BENEISH_INTERCEPT = -4.84
BENEISH_COEFFICIENTS = {
    "DSRI": 0.920,
    "GMI": 0.528,
    "AQI": 0.404,
    "SGI": 0.892,
    "DEPI": 0.115,
    "SGAI": -0.172,
    "TATA": 4.679,
    "LVGI": -0.327,
}
# Group means from Beneish (1999). SGAI and LVGI carry negative weights,
# so a high reading lowers the M-Score and is not treated as a red flag.
BENEISH_NON_MANIPULATOR_MEAN = {
    "DSRI": 1.031,
    "GMI": 1.014,
    "AQI": 1.039,
    "SGI": 1.134,
    "DEPI": 1.001,
    "SGAI": 1.054,
    "TATA": 0.018,
    "LVGI": 1.037,
}
BENEISH_MANIPULATOR_MEAN = {
    "DSRI": 1.465,
    "GMI": 1.193,
    "AQI": 1.254,
    "SGI": 1.607,
    "DEPI": 1.077,
    "SGAI": 1.041,
    "TATA": 0.031,
    "LVGI": 1.111,
}
BENEISH_COMPONENT_MEANING = {
    "DSRI": "Receivables growing faster than sales",
    "GMI": "Gross margin deterioration",
    "AQI": "Costs being capitalised into soft assets",
    "SGI": "High growth, which raises pressure to manage earnings",
    "DEPI": "Depreciation rate slowing",
    "TATA": "Profit not backed by operating cash",
}
M_SCORE_RED = -1.78
M_SCORE_AMBER = -2.22

SLOAN_AMBER = 0.10
SLOAN_RED = 0.25
CFO_EBITDA_AMBER = 0.70
CCC_CHANGE_AMBER = 10.0
CCC_CHANGE_RED = 30.0
DEBTOR_DIVERGENCE_AMBER = 0.10
DEBTOR_DIVERGENCE_RED = 0.25
Z_SCORE_SAFE = 5.85
Z_SCORE_DISTRESS = 4.35

COLOR_SERIES = "#2a78d6"
COLOR_SERIES_EXIT = "#eb6834"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e8e7e3"
COLOR_DOWNSIDE_BAND = "#b9b7ae"


class Severity(Enum):
    """Alert level, ordered by the rank stored in the enum value."""

    RED = 0
    AMBER = 1
    GREEN = 2


@dataclass(frozen=True)
class Alert:
    """A single forensic screening result.

    Attributes:
        severity: Traffic-light level of the result.
        test: Name of the screen.
        reading: Formatted metric value.
        rule: Threshold that produced the severity.
        comment: What the reading implies and what to check.
    """

    severity: Severity
    test: str
    reading: str
    rule: str
    comment: str


@dataclass(frozen=True)
class FinancialYear:
    """One fiscal year of consolidated financials, in ₹ crore.

    Attributes:
        label: Period name, for example ``"FY26"``.
        revenue: Revenue from operations.
        cogs: Materials consumed plus changes in inventories.
        sga: Employee cost plus other expenses. Ind AS has no SG&A line.
        depreciation: Depreciation and amortisation.
        ebitda: Operating EBITDA, excluding other income.
        net_income: Profit after tax.
        cfo: Net cash from operating activities.
        cfi: Net cash from investing activities.
        receivables: Trade receivables.
        inventory: Inventories.
        payables: Trade payables.
        current_assets: Total current assets.
        ppe: Net property, plant and equipment, including right-of-use
            assets and excluding capital work-in-progress.
        cwip: Capital work-in-progress on tangible assets.
        investments: Financial investments, treated as non-soft assets.
        total_assets: Total assets.
        current_liabilities: Total current liabilities.
        long_term_debt: Non-current borrowings and lease liabilities.
        total_liabilities: Total liabilities, excluding equity.
        retained_earnings: Accumulated profits, excluding share premium.
        book_equity: Total equity.
    """

    label: str
    revenue: float
    cogs: float
    sga: float
    depreciation: float
    ebitda: float
    net_income: float
    cfo: float
    cfi: float
    receivables: float
    inventory: float
    payables: float
    current_assets: float
    ppe: float
    cwip: float
    investments: float
    total_assets: float
    current_liabilities: float
    long_term_debt: float
    total_liabilities: float
    retained_earnings: float
    book_equity: float

    def __post_init__(self) -> None:
        """Reject inputs that would make the ratios undefined."""
        for name in ("revenue", "cogs", "sga", "total_assets",
                     "total_liabilities", "receivables"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{self.label}: {name} must be positive.")
        if self.revenue <= self.cogs:
            raise ValueError(f"{self.label}: gross profit must be positive.")

    @property
    def ebit(self) -> float:
        """Return operating profit: EBITDA less depreciation."""
        return self.ebitda - self.depreciation

    @property
    def gross_margin(self) -> float:
        """Return gross profit as a share of revenue."""
        return (self.revenue - self.cogs) / self.revenue

    @property
    def working_capital(self) -> float:
        """Return current assets less current liabilities."""
        return self.current_assets - self.current_liabilities

    @property
    def debtor_days(self) -> float:
        """Return days of sales outstanding, on closing receivables."""
        return self.receivables / self.revenue * DAYS_IN_YEAR

    @property
    def inventory_days(self) -> float:
        """Return days of inventory, on closing stock and COGS."""
        return self.inventory / self.cogs * DAYS_IN_YEAR

    @property
    def payable_days(self) -> float:
        """Return days of payables, on closing payables and COGS."""
        return self.payables / self.cogs * DAYS_IN_YEAR

    @property
    def cash_conversion_cycle(self) -> float:
        """Return debtor days plus inventory days less payable days."""
        return self.debtor_days + self.inventory_days - self.payable_days

    @property
    def soft_asset_share(self) -> float:
        """Return the share of assets outside CA, PP&E and investments."""
        hard_assets = (self.current_assets + self.ppe + self.cwip
                       + self.investments)
        return 1 - hard_assets / self.total_assets


class ForensicAuditor:
    """Screen two consecutive years of financials for earnings quality.

    Args:
        company: Company name, used in reports.
        current: Latest fiscal year.
        prior: The immediately preceding fiscal year.
    """

    def __init__(self, company: str, current: FinancialYear,
                 prior: FinancialYear) -> None:
        self.company = company
        self.current = current
        self.prior = prior

    def beneish_components(self) -> dict[str, float]:
        """Compute the eight Beneish (1999) indices.

        TATA uses the cash-flow form ``(Net income - CFO) / Total assets``.

        Returns:
            Mapping of component name to its value.
        """
        cur, pri = self.current, self.prior
        dep_rate_cur = cur.depreciation / (cur.depreciation + cur.ppe)
        dep_rate_pri = pri.depreciation / (pri.depreciation + pri.ppe)
        leverage_cur = (cur.current_liabilities
                        + cur.long_term_debt) / cur.total_assets
        leverage_pri = (pri.current_liabilities
                        + pri.long_term_debt) / pri.total_assets
        return {
            "DSRI": (cur.receivables / cur.revenue)
            / (pri.receivables / pri.revenue),
            "GMI": pri.gross_margin / cur.gross_margin,
            "AQI": cur.soft_asset_share / pri.soft_asset_share,
            "SGI": cur.revenue / pri.revenue,
            "DEPI": dep_rate_pri / dep_rate_cur,
            "SGAI": (cur.sga / cur.revenue) / (pri.sga / pri.revenue),
            "TATA": (cur.net_income - cur.cfo) / cur.total_assets,
            "LVGI": leverage_cur / leverage_pri,
        }

    def beneish_m_score(self) -> float:
        """Return the 8-variable Beneish M-Score.

        Scores above -1.78 are classified as likely manipulators in the
        original paper; -2.22 is a stricter, widely used practitioner cut.
        """
        components = self.beneish_components()
        return BENEISH_INTERCEPT + sum(
            weight * components[name]
            for name, weight in BENEISH_COEFFICIENTS.items()
        )

    def beneish_table(self) -> pd.DataFrame:
        """Return each index with its M-Score contribution and benchmarks."""
        rows = []
        for name, value in self.beneish_components().items():
            weight = BENEISH_COEFFICIENTS[name]
            rows.append({
                "Index": name,
                "Value": value,
                "Weight": weight,
                "Contribution": weight * value,
                "Non-manip. mean": BENEISH_NON_MANIPULATOR_MEAN[name],
                "Manip. mean": BENEISH_MANIPULATOR_MEAN[name],
            })
        return pd.DataFrame(rows).set_index("Index")

    def sloan_accrual_ratio(self) -> float:
        """Return ``(Net income - CFO - CFI) / Total assets``.

        CFI includes growth capex and treasury movements, so an
        equity-funded capex cycle raises this ratio without any accounting
        issue. Read it alongside TATA, which excludes investing flows.
        """
        cur = self.current
        return (cur.net_income - cur.cfo - cur.cfi) / cur.total_assets

    def cash_conversion_quality(self) -> pd.DataFrame:
        """Return CFO/EBITDA and working-capital days for both years."""
        rows = {}
        for year in (self.prior, self.current):
            rows[year.label] = {
                "CFO / EBITDA": year.cfo / year.ebitda,
                "Debtor days": year.debtor_days,
                "Inventory days": year.inventory_days,
                "Payable days": year.payable_days,
                "Cash conversion cycle": year.cash_conversion_cycle,
            }
        return pd.DataFrame(rows)

    def debtor_sales_divergence(self) -> dict[str, float]:
        """Return receivable growth, sales growth and their difference."""
        cur, pri = self.current, self.prior
        receivable_growth = cur.receivables / pri.receivables - 1
        sales_growth = cur.revenue / pri.revenue - 1
        return {
            "receivable_growth": receivable_growth,
            "sales_growth": sales_growth,
            "divergence": receivable_growth - sales_growth,
        }

    def altman_components(self) -> dict[str, float]:
        """Return the four ratios used by the Z''-Score."""
        cur = self.current
        return {
            "X1 Working capital / Assets":
                cur.working_capital / cur.total_assets,
            "X2 Retained earnings / Assets":
                cur.retained_earnings / cur.total_assets,
            "X3 EBIT / Assets": cur.ebit / cur.total_assets,
            "X4 Book equity / Liabilities":
                cur.book_equity / cur.total_liabilities,
        }

    def altman_z_score(self) -> float:
        """Return Altman's emerging-market Z''-Score.

        ``Z'' = 3.25 + 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4``. Above 5.85
        is the safe zone, below 4.35 is distress, and between is grey.
        """
        x1, x2, x3, x4 = self.altman_components().values()
        return 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    @staticmethod
    def _grade_high(value: float, amber: float, red: float) -> Severity:
        """Grade a metric where a higher reading is worse."""
        if value > red:
            return Severity.RED
        return Severity.AMBER if value > amber else Severity.GREEN

    def _beneish_alerts(self) -> list[Alert]:
        """Build the headline M-Score alert and component warnings."""
        score = self.beneish_m_score()
        alerts = [Alert(
            self._grade_high(score, M_SCORE_AMBER, M_SCORE_RED),
            "Beneish M-Score",
            f"{score:.2f}",
            f"Amber > {M_SCORE_AMBER}; Red > {M_SCORE_RED}",
            "Statistical profile compared with known manipulators.",
        )]
        for name, value in self.beneish_components().items():
            benchmark = BENEISH_MANIPULATOR_MEAN[name]
            if name in BENEISH_COMPONENT_MEANING and value > benchmark:
                alerts.append(Alert(
                    Severity.AMBER,
                    f"Beneish {name}",
                    f"{value:.3f}",
                    f"Above manipulator mean of {benchmark}",
                    BENEISH_COMPONENT_MEANING[name] + ".",
                ))
        return alerts

    def _cash_alerts(self) -> list[Alert]:
        """Build the accrual, cash-conversion and receivable alerts."""
        cur, pri = self.current, self.prior
        sloan = self.sloan_accrual_ratio()
        alerts = [Alert(
            self._grade_high(sloan, SLOAN_AMBER, SLOAN_RED),
            "Sloan accrual ratio",
            f"{sloan:.1%}",
            f"Amber > {SLOAN_AMBER:.0%}; Red > {SLOAN_RED:.0%}",
            "Includes investing outflows; split capex from true accruals.",
        )]

        if cur.ebitda <= 0:
            conversion = "n/m"
            level = Severity.RED
        else:
            ratio = cur.cfo / cur.ebitda
            conversion = f"{ratio:.2f}x"
            if ratio < 0:
                level = Severity.RED
            elif ratio < CFO_EBITDA_AMBER:
                level = Severity.AMBER
            else:
                level = Severity.GREEN
        alerts.append(Alert(
            level, "CFO / EBITDA", conversion,
            f"Amber < {CFO_EBITDA_AMBER}x; Red < 0x",
            "EBITDA is not converting into operating cash.",
        ))

        ccc_change = cur.cash_conversion_cycle - pri.cash_conversion_cycle
        alerts.append(Alert(
            self._grade_high(ccc_change, CCC_CHANGE_AMBER, CCC_CHANGE_RED),
            "Cash conversion cycle change",
            f"{ccc_change:+.0f} days to {cur.cash_conversion_cycle:.0f}",
            f"Amber > +{CCC_CHANGE_AMBER:.0f}d; Red > +{CCC_CHANGE_RED:.0f}d",
            "Working capital is absorbing more cash per rupee of sales.",
        ))

        divergence = self.debtor_sales_divergence()
        alerts.append(Alert(
            self._grade_high(divergence["divergence"],
                             DEBTOR_DIVERGENCE_AMBER, DEBTOR_DIVERGENCE_RED),
            "Debtor-to-sales divergence",
            f"{divergence['divergence'] * 100:+.1f} pp",
            f"Amber > {DEBTOR_DIVERGENCE_AMBER * 100:.0f} pp; "
            f"Red > {DEBTOR_DIVERGENCE_RED * 100:.0f} pp",
            "Check receivable ageing, related parties and acquired books.",
        ))
        return alerts

    def _solvency_alert(self) -> Alert:
        """Build the Altman Z''-Score alert."""
        score = self.altman_z_score()
        if score < Z_SCORE_DISTRESS:
            level, zone = Severity.RED, "Distress zone."
        elif score < Z_SCORE_SAFE:
            level, zone = Severity.AMBER, "Grey zone."
        else:
            level, zone = Severity.GREEN, "Safe zone."
        return Alert(
            level, "Altman Z''-Score (EM)", f"{score:.2f}",
            f"Safe > {Z_SCORE_SAFE}; Distress < {Z_SCORE_DISTRESS}", zone,
        )

    def alerts(self) -> list[Alert]:
        """Return every screening result, most severe first."""
        results = self._beneish_alerts() + self._cash_alerts()
        results.append(self._solvency_alert())
        return sorted(results, key=lambda alert: alert.severity.value)

    def print_report(self) -> None:
        """Print the complete forensic audit to standard output."""
        cur, pri = self.current, self.prior
        _print_header(
            f"FORENSIC AUDIT: {self.company} ({cur.label} vs {pri.label})"
        )

        print("\n[1] Beneish M-Score")
        print(self.beneish_table().round(3).to_string())
        print(f"Intercept: {BENEISH_INTERCEPT}")
        print(f"M-Score:   {self.beneish_m_score():.3f}")

        print("\n[2] Sloan accrual ratio")
        print(f"({cur.net_income:,.1f} - ({cur.cfo:,.1f}) - ({cur.cfi:,.1f}))"
              f" / {cur.total_assets:,.1f} = "
              f"{self.sloan_accrual_ratio():.1%}")

        print("\n[3] Cash conversion quality")
        print(self.cash_conversion_quality().round(2).to_string())
        divergence = self.debtor_sales_divergence()
        print(f"Receivable growth: {divergence['receivable_growth']:.1%}")
        print(f"Sales growth:      {divergence['sales_growth']:.1%}")
        print(f"Divergence:        {divergence['divergence'] * 100:+.1f} pp")

        print("\n[4] Altman Z''-Score (emerging markets)")
        for name, value in self.altman_components().items():
            print(f"{name:<31}{value:>8.3f}")
        print(f"{'Z-Score':<31}{self.altman_z_score():>8.3f}")

        print("\n[5] Alerts")
        for alert in self.alerts():
            print(f"[{alert.severity.name:<5}] {alert.test:<30}"
                  f"{alert.reading:<20}{alert.rule}")
            print(f"        {alert.comment}")
        counts = pd.Series(
            [alert.severity.name for alert in self.alerts()]
        ).value_counts()
        print("\nSummary: " + ", ".join(
            f"{counts.get(level.name, 0)} {level.name}" for level in Severity
        ))


@dataclass(frozen=True)
class WaccInputs:
    """CAPM and capital-structure inputs for the discount rate.

    Attributes:
        risk_free: Risk-free rate.
        beta: Levered equity beta.
        equity_risk_premium: Equity risk premium.
        pre_tax_cost_of_debt: Marginal pre-tax borrowing cost.
        tax_rate: Marginal tax rate.
        market_cap: Equity market value in ₹ crore.
        gross_debt: Borrowings and leases in ₹ crore.
    """

    risk_free: float
    beta: float
    equity_risk_premium: float
    pre_tax_cost_of_debt: float
    tax_rate: float
    market_cap: float
    gross_debt: float

    @property
    def cost_of_equity(self) -> float:
        """Return the CAPM cost of equity."""
        return self.risk_free + self.beta * self.equity_risk_premium

    @property
    def wacc(self) -> float:
        """Return the market-value weighted average cost of capital."""
        capital = self.market_cap + self.gross_debt
        equity_weight = self.market_cap / capital
        after_tax_debt = self.pre_tax_cost_of_debt * (1 - self.tax_rate)
        return (equity_weight * self.cost_of_equity
                + (1 - equity_weight) * after_tax_debt)


@dataclass(frozen=True)
class DCFInputs:
    """Deterministic inputs for the faded DCF, in ₹ crore unless noted.

    Attributes:
        base_revenue: Latest annual revenue.
        base_ebit_margin: Latest EBIT margin.
        tax_rate: Cash tax rate on EBIT.
        incremental_nwc: Net working capital per ₹1 of added revenue.
        incremental_fixed_capital: Net capex per ₹1 of added revenue.
        terminal_roic: Return on new capital in perpetuity.
        net_debt: Borrowings and leases less cash and bank balances.
        non_controlling_interest: Minority interest at book value.
        non_operating_assets: Financial investments at book value.
        shares_outstanding: Shares outstanding in crore.
        cmp: Current market price in ₹.
        da_to_revenue: Depreciation as a share of revenue, used to derive
            forecast EBITDA.
    """

    base_revenue: float
    base_ebit_margin: float
    tax_rate: float
    incremental_nwc: float
    incremental_fixed_capital: float
    terminal_roic: float
    net_debt: float
    non_controlling_interest: float
    non_operating_assets: float
    shares_outstanding: float
    cmp: float
    da_to_revenue: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    """Distribution settings for the Monte Carlo run.

    WACC and revenue CAGR are normal, and truncated so that no path can
    produce an undefined Gordon growth value. Terminal growth, peak EBIT
    margin, the margin fade and the exit multiple are triangular.

    Attributes:
        n_paths: Number of simulated paths.
        seed: Random seed, for reproducible output.
        wacc_std: Standard deviation of WACC.
        terminal_growth: Low, mode and high of terminal growth.
        cagr_mean: Mean 5-year revenue CAGR.
        cagr_std: Standard deviation of revenue CAGR.
        peak_margin: Low, mode and high of the year-5 EBIT margin.
        margin_fade: Low, mode and high of margin lost by year 10.
        min_spread: Minimum WACC less terminal growth on any path.
        max_sigma: Truncation width, in standard deviations.
        exit_multiple: Low, mode and high of the multiple applied to
            year-10 earnings.
        exit_basis: Earnings measure for the exit multiple, ``"EBIT"`` or
            ``"EBITDA"``.
        wacc_range: Optional (low, high) replacing the CAPM-centred WACC.
            The mean becomes the midpoint and the range spans
            ``2 * max_sigma`` standard deviations.
    """

    n_paths: int = 10_000
    seed: int = 42
    wacc_std: float = 0.01
    terminal_growth: tuple[float, float, float] = (0.035, 0.05, 0.065)
    cagr_mean: float = 0.27
    cagr_std: float = 0.05
    peak_margin: tuple[float, float, float] = (0.125, 0.15, 0.18)
    margin_fade: tuple[float, float, float] = (0.0, 0.015, 0.035)
    min_spread: float = 0.03
    max_sigma: float = 3.0
    exit_multiple: tuple[float, float, float] = (14.0, 20.0, 28.0)
    exit_basis: str = "EBIT"
    wacc_range: tuple[float, float] | None = None


class OperatingPaths(NamedTuple):
    """Per-path, per-year operating forecast, as (paths, years) arrays."""

    growth: np.ndarray
    revenue: np.ndarray
    margin: np.ndarray
    ebit: np.ndarray
    ebitda: np.ndarray
    nopat: np.ndarray
    reinvestment: np.ndarray
    free_cash_flow: np.ndarray
    discount: np.ndarray


class ProjectedPaths(NamedTuple):
    """Per-path DCF output for both terminal-value methods.

    Attributes:
        gordon_value: Value per share using Gordon growth.
        gordon_terminal_share: PV of Gordon terminal value as a share of EV.
        exit_value: Value per share using the exit multiple.
        exit_terminal_share: PV of exit terminal value as a share of EV.
        gordon_implied_multiple: Year-10 exit multiple, on the configured
            earnings basis, implied by the Gordon terminal value.
        implied_growth: Perpetual growth implied by the exit multiple.
        breakeven_multiple: Exit multiple at which value equals CMP.
    """

    gordon_value: np.ndarray
    gordon_terminal_share: np.ndarray
    exit_value: np.ndarray
    exit_terminal_share: np.ndarray
    gordon_implied_multiple: np.ndarray
    implied_growth: np.ndarray
    breakeven_multiple: np.ndarray


@dataclass(frozen=True)
class DistributionStats:
    """Summary of a simulated value-per-share distribution.

    Attributes:
        values: Simulated value per share for every path.
        percentiles: Value per share at each percentile in ``PERCENTILES``.
        base_case: Value per share with every driver at its central value.
        mean_value: Mean simulated value per share.
        probability_of_downside: Share of paths valued below CMP.
        var_95: CMP less the 5th percentile value, floored at zero.
        var_95_pct: ``var_95`` as a share of CMP.
        median_terminal_share: Median terminal-value share of EV.
    """

    values: np.ndarray = field(repr=False)
    percentiles: dict[int, float]
    base_case: float
    mean_value: float
    probability_of_downside: float
    var_95: float
    var_95_pct: float
    median_terminal_share: float


@dataclass(frozen=True)
class ValuationResult:
    """Output of a Monte Carlo valuation run.

    Attributes:
        cmp: Current market price.
        gordon: Distribution using the Gordon growth terminal value.
        exit_multiple: Distribution using the exit-multiple terminal value.
        median_gordon_multiple: Median year-10 exit multiple implied by
            Gordon, on the configured earnings basis.
        median_implied_growth: Median perpetual growth implied by the exit
            multiple.
        breakeven_multiple: Exit multiple that justifies CMP in the central
            case.
        drivers: The simulated inputs for each path.
    """

    cmp: float
    gordon: DistributionStats
    exit_multiple: DistributionStats
    median_gordon_multiple: float
    median_implied_growth: float
    breakeven_multiple: float
    drivers: pd.DataFrame = field(repr=False)


class MonteCarloValuation:
    """Simulate a 2-stage faded DCF and summarise value per share.

    Stage 1 (years 1-5) compounds revenue at the sampled CAGR while EBIT
    margin moves linearly from the base to the sampled peak. Stage 2
    (years 6-10) fades growth linearly to terminal growth and loses the
    sampled margin fade. The primary terminal value uses the value-driver
    formula, so perpetual growth is paid for by reinvestment:
    ``NOPAT(11) * (1 - g / ROIC) / (WACC - g)``.

    The cross-check applies a sampled multiple to year-10 EBIT or EBITDA
    on the same paths. Each method is translated into the other's terms:
    the multiple that Gordon implies, and the perpetual growth that the
    exit multiple implies. A wide gap means the two sets of assumptions
    cannot both hold.

    Cash flows are discounted with an end-year convention from the
    latest balance-sheet date. The percentile "VaR" is a valuation
    shortfall against CMP and not a market-price VaR.

    Args:
        name: Company name, used in reports and the chart.
        dcf: Deterministic valuation inputs.
        wacc_inputs: CAPM inputs that set the mean WACC.
        config: Distribution settings.

    Raises:
        ValueError: If the settings could produce an invalid terminal value.
    """

    def __init__(self, name: str, dcf: DCFInputs, wacc_inputs: WaccInputs,
                 config: SimulationConfig | None = None) -> None:
        self.name = name
        self.dcf = dcf
        self.wacc_inputs = wacc_inputs
        self.config = config or SimulationConfig()
        cfg = self.config
        self.wacc_floor = cfg.terminal_growth[2] + cfg.min_spread
        if cfg.wacc_range is None:
            self.mean_wacc = wacc_inputs.wacc
            self.wacc_std = cfg.wacc_std
            self.wacc_low = self.wacc_floor
            self.wacc_high = self.mean_wacc + cfg.max_sigma * cfg.wacc_std
        else:
            low, high = cfg.wacc_range
            if high <= low:
                raise ValueError("WACC range must have high above low.")
            self.mean_wacc = (low + high) / 2
            self.wacc_std = (high - low) / (2 * cfg.max_sigma)
            self.wacc_low = max(low, self.wacc_floor)
            self.wacc_high = high
        if self.mean_wacc <= self.wacc_floor:
            raise ValueError(
                f"Mean WACC {self.mean_wacc:.2%} must exceed the floor "
                f"{self.wacc_floor:.2%}."
            )
        if dcf.terminal_roic <= cfg.terminal_growth[2]:
            raise ValueError("Terminal ROIC must exceed maximum growth.")
        if cfg.exit_multiple[0] <= 0:
            raise ValueError("Exit multiple must be positive.")
        if cfg.exit_basis not in ("EBIT", "EBITDA"):
            raise ValueError("Exit basis must be EBIT or EBITDA.")

    @staticmethod
    def _triangular(bounds: tuple[float, float, float], size: int,
                    rng: np.random.Generator) -> np.ndarray:
        """Draw from a triangular distribution given (low, mode, high)."""
        low, mode, high = bounds
        return stats.triang.rvs(
            c=(mode - low) / (high - low),
            loc=low,
            scale=high - low,
            size=size,
            random_state=rng,
        )

    @staticmethod
    def _truncated_normal(mean: float, std: float, low: float, high: float,
                          size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw from a normal distribution truncated to [low, high]."""
        return stats.truncnorm.rvs(
            a=(low - mean) / std,
            b=(high - mean) / std,
            loc=mean,
            scale=std,
            size=size,
            random_state=rng,
        )

    def _draw_drivers(self) -> pd.DataFrame:
        """Sample every stochastic driver, one independent row per path."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        return pd.DataFrame({
            "wacc": self._truncated_normal(
                self.mean_wacc, self.wacc_std, self.wacc_low, self.wacc_high,
                cfg.n_paths, rng),
            "terminal_growth": self._triangular(
                cfg.terminal_growth, cfg.n_paths, rng),
            "revenue_cagr": self._truncated_normal(
                cfg.cagr_mean, cfg.cagr_std,
                cfg.cagr_mean - cfg.max_sigma * cfg.cagr_std,
                cfg.cagr_mean + cfg.max_sigma * cfg.cagr_std,
                cfg.n_paths, rng),
            "peak_margin": self._triangular(
                cfg.peak_margin, cfg.n_paths, rng),
            "margin_fade": self._triangular(
                cfg.margin_fade, cfg.n_paths, rng),
            # Drawn last so that adding it leaves the other draws unchanged.
            "exit_multiple": self._triangular(
                cfg.exit_multiple, cfg.n_paths, rng),
        })

    def _implied_growth(self, ev_to_ebit: np.ndarray,
                        wacc: np.ndarray) -> np.ndarray:
        """Solve the perpetual growth that a trailing EV/EBIT implies.

        With ``T = ev_to_ebit / (1 - tax)``, the value-driver formula gives
        ``T * (WACC - g) = (1 + g) * (1 - g / ROIC)``, a quadratic in g
        whose smaller root is the economically meaningful one.

        Args:
            ev_to_ebit: Terminal value as a multiple of year-10 EBIT.
            wacc: Discount rate, with the same shape as ``ev_to_ebit``.

        Returns:
            Implied growth, or NaN where no real solution exists.
        """
        tv_to_nopat = ev_to_ebit / (1 - self.dcf.tax_rate)
        quad_a = 1 / self.dcf.terminal_roic
        quad_b = tv_to_nopat + 1 - quad_a
        quad_c = tv_to_nopat * wacc - 1
        discriminant = quad_b ** 2 - 4 * quad_a * quad_c
        root = np.sqrt(np.where(discriminant >= 0, discriminant, np.nan))
        return (quad_b - root) / (2 * quad_a)

    def _operating_paths(self, drivers: pd.DataFrame) -> OperatingPaths:
        """Forecast the operating line items for every path and year."""
        dcf = self.dcf
        wacc = drivers["wacc"].to_numpy()[:, np.newaxis]
        growth_t = drivers["terminal_growth"].to_numpy()[:, np.newaxis]
        cagr = drivers["revenue_cagr"].to_numpy()[:, np.newaxis]
        peak = drivers["peak_margin"].to_numpy()[:, np.newaxis]
        fade = drivers["margin_fade"].to_numpy()[:, np.newaxis]

        years = np.arange(1, TOTAL_YEARS + 1)
        ramp = np.minimum(years / EXPLICIT_YEARS, 1.0)
        fade_weight = np.maximum(years - EXPLICIT_YEARS, 0) / FADE_YEARS

        growth = cagr * (1 - fade_weight) + growth_t * fade_weight
        revenue = dcf.base_revenue * np.cumprod(1 + growth, axis=1)
        opening_revenue = np.column_stack((
            np.full(len(revenue), dcf.base_revenue), revenue[:, :-1],
        ))
        margin = (dcf.base_ebit_margin * (1 - ramp) + peak * ramp
                  - fade * fade_weight)

        ebit = revenue * margin
        nopat = ebit * (1 - dcf.tax_rate)
        capital_intensity = dcf.incremental_nwc + dcf.incremental_fixed_capital
        reinvestment = (revenue - opening_revenue) * capital_intensity
        return OperatingPaths(
            growth=growth,
            revenue=revenue,
            margin=margin,
            ebit=ebit,
            ebitda=ebit + dcf.da_to_revenue * revenue,
            nopat=nopat,
            reinvestment=reinvestment,
            free_cash_flow=nopat - reinvestment,
            discount=(1 + wacc) ** years,
        )

    def _project(self, drivers: pd.DataFrame) -> ProjectedPaths:
        """Run the DCF across all paths with NumPy array operations.

        Args:
            drivers: One row per path, with the six sampled drivers.

        Returns:
            Per-path values and diagnostics for both terminal methods.
        """
        dcf = self.dcf
        wacc = drivers["wacc"].to_numpy()[:, np.newaxis]
        growth_t = drivers["terminal_growth"].to_numpy()[:, np.newaxis]
        multiple = drivers["exit_multiple"].to_numpy()[:, np.newaxis]

        ops = self._operating_paths(drivers)
        ebit = ops.ebit[:, -1:]
        use_ebit = self.config.exit_basis == "EBIT"
        basis = ebit if use_ebit else ops.ebitda[:, -1:]
        final_discount = ops.discount[:, -1:]
        pv_explicit = (ops.free_cash_flow / ops.discount).sum(axis=1)

        gordon_tv = (
            ops.nopat[:, -1:] * (1 + growth_t)
            * (1 - growth_t / dcf.terminal_roic) / (wacc - growth_t)
        )
        exit_tv = multiple * np.maximum(basis, 0)
        pv_gordon = (gordon_tv / final_discount).ravel()
        pv_exit = (exit_tv / final_discount).ravel()

        net_claims = (dcf.net_debt + dcf.non_controlling_interest
                      - dcf.non_operating_assets)
        gordon_ev = pv_explicit + pv_gordon
        exit_ev = pv_explicit + pv_exit
        required_ev = dcf.cmp * dcf.shares_outstanding + net_claims
        breakeven = ((required_ev - pv_explicit) * final_discount.ravel()
                     / basis.ravel())
        # Limited liability: equity cannot be worth less than zero.
        return ProjectedPaths(
            gordon_value=(np.maximum(gordon_ev - net_claims, 0)
                          / dcf.shares_outstanding),
            gordon_terminal_share=pv_gordon / gordon_ev,
            exit_value=(np.maximum(exit_ev - net_claims, 0)
                        / dcf.shares_outstanding),
            exit_terminal_share=pv_exit / exit_ev,
            gordon_implied_multiple=(gordon_tv / basis).ravel(),
            implied_growth=self._implied_growth(
                multiple * basis / ebit, wacc).ravel(),
            breakeven_multiple=breakeven,
        )

    def _central_drivers(self) -> pd.DataFrame:
        """Return a single-row driver table at the central values."""
        cfg = self.config
        return pd.DataFrame({
            "wacc": [self.mean_wacc],
            "terminal_growth": [cfg.terminal_growth[1]],
            "revenue_cagr": [cfg.cagr_mean],
            "peak_margin": [cfg.peak_margin[1]],
            "margin_fade": [cfg.margin_fade[1]],
            "exit_multiple": [cfg.exit_multiple[1]],
        })

    def _central_case(self) -> ProjectedPaths:
        """Project a single path with each driver at its central value."""
        return self._project(self._central_drivers())

    def central_forecast(self) -> pd.DataFrame:
        """Return the year-by-year forecast for the central case.

        Returns:
            One row per forecast year, in ₹ crore except the ratios.
        """
        ops = self._operating_paths(self._central_drivers())
        return pd.DataFrame({
            "Year": np.arange(1, TOTAL_YEARS + 1),
            "Stage": ["Explicit"] * EXPLICIT_YEARS + ["Fade"] * FADE_YEARS,
            "Growth": ops.growth[0],
            "Revenue": ops.revenue[0],
            "EBITDA": ops.ebitda[0],
            "EBIT": ops.ebit[0],
            "EBIT margin": ops.margin[0],
            "NOPAT": ops.nopat[0],
            "Reinvestment": ops.reinvestment[0],
            "FCFF": ops.free_cash_flow[0],
            "Discount factor": 1 / ops.discount[0],
            "PV of FCFF": ops.free_cash_flow[0] / ops.discount[0],
        })

    def _summarise(self, values: np.ndarray, terminal_share: np.ndarray,
                   base_case: float) -> DistributionStats:
        """Compute the distribution statistics for one terminal method."""
        cmp_price = self.dcf.cmp
        var_95 = max(cmp_price - float(np.percentile(values, 5)), 0.0)
        return DistributionStats(
            values=values,
            percentiles={
                level: float(np.percentile(values, level))
                for level in PERCENTILES
            },
            base_case=base_case,
            mean_value=float(values.mean()),
            probability_of_downside=float((values < cmp_price).mean()),
            var_95=var_95,
            var_95_pct=var_95 / cmp_price,
            median_terminal_share=float(np.median(terminal_share)),
        )

    def run(self) -> ValuationResult:
        """Run the simulation and summarise both terminal methods."""
        drivers = self._draw_drivers()
        paths = self._project(drivers)
        central = self._central_case()
        return ValuationResult(
            cmp=self.dcf.cmp,
            gordon=self._summarise(
                paths.gordon_value, paths.gordon_terminal_share,
                float(central.gordon_value[0])),
            exit_multiple=self._summarise(
                paths.exit_value, paths.exit_terminal_share,
                float(central.exit_value[0])),
            median_gordon_multiple=float(
                np.median(paths.gordon_implied_multiple)),
            median_implied_growth=float(np.nanmedian(paths.implied_growth)),
            breakeven_multiple=float(central.breakeven_multiple[0]),
            drivers=drivers,
        )

    def _comparison_table(self, result: ValuationResult) -> pd.DataFrame:
        """Format the value-per-share output of both methods side by side."""
        columns = {}
        for name, dist in (("Gordon growth", result.gordon),
                           ("Exit multiple", result.exit_multiple)):
            column = {}
            for level, value in dist.percentiles.items():
                label = "Median" if level == 50 else f"P{level}"
                column[label] = (f"₹{value:,.0f} "
                                 f"({value / result.cmp - 1:+.1%})")
            column.update({
                "Mean": f"₹{dist.mean_value:,.0f}",
                "Deterministic base case": f"₹{dist.base_case:,.0f}",
                "Probability of downside":
                    f"{dist.probability_of_downside:.1%}",
                "Value-at-Risk (95%)":
                    f"₹{dist.var_95:,.0f} ({dist.var_95_pct:.1%})",
                "Median terminal share of EV":
                    f"{dist.median_terminal_share:.1%}",
            })
            columns[name] = column
        return pd.DataFrame(columns)

    def print_report(self, result: ValuationResult) -> None:
        """Print WACC build-up, driver ranges and valuation output."""
        _print_header(f"MONTE CARLO DCF: {self.name}")
        inputs = self.wacc_inputs
        print(f"\nCost of equity (CAPM): {inputs.cost_of_equity:.2%}")
        print(f"Mean WACC:             {inputs.wacc:.2%}")
        print(f"Paths:                 {len(result.gordon.values):,}")

        rows = ["mean", "std", "5%", "50%", "95%"]
        summary = result.drivers.describe(
            percentiles=[0.05, 0.5, 0.95]
        ).loc[rows].T
        scale = pd.Series(100.0, index=summary.index)
        scale["exit_multiple"] = 1.0
        summary = summary.mul(scale, axis=0)
        basis = f"EV/{self.config.exit_basis}"
        print(f"\nSampled drivers (% except exit_multiple, in x {basis})")
        print(summary.round(2).to_string())

        print(f"\nValue per share (% vs CMP of ₹{result.cmp:,.0f})")
        print(self._comparison_table(result).to_string())

        growth_high = self.config.terminal_growth[2]
        print("\nTerminal value cross-check")
        print(f"Central exit multiple:          "
              f"{self.config.exit_multiple[1]:.1f}x {basis}")
        print(f"{basis} implied by Gordon:      "
              f"{result.median_gordon_multiple:.1f}x")
        print(f"Growth implied by exit multiple: "
              f"{result.median_implied_growth:.1%} perpetual, vs "
              f"{self.config.terminal_growth[1]:.1%} assumed")
        print(f"Exit {basis} needed for CMP:    "
              f"{result.breakeven_multiple:.1f}x (central case)")
        if result.median_implied_growth > growth_high:
            print(f"WARNING: the exit multiple implies perpetual growth above "
                  f"the {growth_high:.1%} ceiling.\nTreat it as a "
                  "market-sentiment value, not an intrinsic one.")

    @staticmethod
    def _add_distribution(fig: go.Figure, dist: DistributionStats, name: str,
                          color: str, edges: np.ndarray) -> float:
        """Add a histogram and density curve for one terminal method.

        Args:
            fig: Figure to draw on.
            dist: Distribution to plot.
            name: Legend label.
            color: Series colour.
            edges: Histogram bin edges, shared across methods.

        Returns:
            The highest density drawn.
        """
        values = dist.values
        counts, _ = np.histogram(values, bins=edges)
        widths = np.diff(edges)
        density = counts / (len(values) * widths)
        visible = values[(values >= edges[0]) & (values <= edges[-1])]
        kde_x = np.linspace(visible.min(), visible.max(), 400)
        kde_y = stats.gaussian_kde(values)(kde_x)

        fig.add_bar(
            x=(edges[:-1] + edges[1:]) / 2, y=density, width=widths * 0.88,
            name=name, legendgroup=name, marker_color=color,
            marker_opacity=0.45, marker_line_width=0,
            customdata=np.column_stack((edges[:-1], edges[1:],
                                        counts / len(values))),
            hovertemplate=(f"<b>{name}</b><br>"
                           "₹%{customdata[0]:,.0f} to "
                           "₹%{customdata[1]:,.0f}<br>"
                           "%{customdata[2]:.1%} of paths<extra></extra>"),
        )
        fig.add_scatter(
            x=kde_x, y=kde_y, mode="lines", name=name, legendgroup=name,
            showlegend=False, line={"color": color, "width": 2},
            hovertemplate=f"₹%{{x:,.0f}}<extra>{name}</extra>",
        )
        return float(max(density.max(), kde_y.max()))

    def plot(self, result: ValuationResult) -> go.Figure:
        """Build the value-per-share distribution chart for both methods.

        Args:
            result: Output of :meth:`run`.

        Returns:
            A Plotly figure with histograms, KDEs and reference lines.
        """
        gordon, exit_dist = result.gordon, result.exit_multiple
        pooled = np.concatenate((gordon.values, exit_dist.values))
        low_cut, high_cut = np.percentile(pooled, [0.1, 99.9])
        low, high = min(low_cut, result.cmp), max(high_cut, result.cmp)
        padding = (high - low) * 0.04
        x_range = (max(low - padding, 0), high + padding)
        edges = np.linspace(low_cut, high_cut, 101)

        fig = go.Figure()
        fig.add_vrect(x0=x_range[0], x1=result.cmp, layer="below",
                      line_width=0, fillcolor=COLOR_DOWNSIDE_BAND,
                      opacity=0.14)
        top = max(
            self._add_distribution(fig, gordon, "Gordon growth",
                                   COLOR_SERIES, edges),
            self._add_distribution(fig, exit_dist, "Exit multiple",
                                   COLOR_SERIES_EXIT, edges),
        )

        markers = sorted([
            (gordon.percentiles[50], "Gordon median",
             COLOR_TEXT_SECONDARY, "dash"),
            (exit_dist.percentiles[50], "Exit median",
             COLOR_TEXT_SECONDARY, "dash"),
            (result.cmp, "CMP", COLOR_TEXT, "solid"),
        ])
        min_label_gap = (x_range[1] - x_range[0]) * 0.12
        previous, level = -np.inf, 0
        for value, label, color, dash in markers:
            level = 1 - level if value - previous < min_label_gap else 0
            previous = value
            label_y = top * (1.1 + 0.16 * level)
            fig.add_shape(
                type="line", x0=value, x1=value, y0=0, y1=label_y,
                line={"color": color, "width": 2, "dash": dash},
            )
            fig.add_annotation(
                x=value, y=label_y, yanchor="bottom", showarrow=False,
                text=f"<b>{label}</b><br>₹{value:,.0f}",
                font={"color": COLOR_TEXT, "size": 12},
            )

        fig.update_layout(
            title={
                "text": (
                    f"{self.name}: paths valued below CMP, "
                    f"{gordon.probability_of_downside:.1%} on Gordon growth "
                    f"and {exit_dist.probability_of_downside:.1%} on exit "
                    "multiple<br><sup>P5 to P95: Gordon "
                    f"₹{gordon.percentiles[5]:,.0f} to "
                    f"₹{gordon.percentiles[95]:,.0f}, exit "
                    f"₹{exit_dist.percentiles[5]:,.0f} to "
                    f"₹{exit_dist.percentiles[95]:,.0f}. "
                    "Grey band marks downside versus CMP.</sup>"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            xaxis={"title": "Value per share (₹)", "range": x_range,
                   "tickformat": ",.0f", "showgrid": False,
                   "linecolor": COLOR_GRID},
            yaxis={"title": "Probability density", "showticklabels": False,
                   "range": [0, top * 1.55], "gridcolor": COLOR_GRID,
                   "zeroline": False},
            legend={"orientation": "h", "x": 1, "xanchor": "right",
                    "y": 1.02, "yanchor": "bottom"},
            margin={"t": 120, "l": 60, "r": 40, "b": 70},
            font={"family": "Segoe UI, Arial, sans-serif",
                  "color": COLOR_TEXT},
            paper_bgcolor=COLOR_SURFACE,
            plot_bgcolor=COLOR_SURFACE,
            barmode="overlay",
            bargap=0,
            hovermode="closest",
        )
        return fig


def _print_header(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


KAYNES_CMP = 3_210.0
KAYNES_SHARES_CR = 6.7035
KAYNES_TAX_RATE = 0.2517

# Source: audited consolidated results filed 13 May 2026 (₹ million / 10).
# retained_earnings is an ESTIMATE from cumulative PAT, because the results
# filing reports only total Other Equity, most of which is share premium.
KAYNES_FY26 = FinancialYear(
    label="FY26", revenue=3_626.35, cogs=2_371.89, sga=680.41,
    depreciation=107.07, ebitda=574.06, net_income=363.89, cfo=-600.40,
    cfi=-917.22, receivables=1_527.62, inventory=1_103.23, payables=850.76,
    current_assets=3_847.29, ppe=1_196.08, cwip=372.41, investments=257.05,
    total_assets=6_894.02, current_liabilities=1_727.26,
    long_term_debt=369.01, total_liabilities=2_131.53,
    retained_earnings=1_000.0, book_equity=4_762.49,
)
KAYNES_FY25 = FinancialYear(
    label="FY25", revenue=2_721.75, cogs=1_899.71, sga=411.34,
    depreciation=44.74, ebitda=410.70, net_income=293.43, cfo=-82.32,
    cfi=-354.66, receivables=574.58, inventory=814.42, payables=682.92,
    current_assets=2_683.24, ppe=697.97, cwip=300.22, investments=132.38,
    total_assets=4_641.22, current_liabilities=1_682.11,
    long_term_debt=91.80, total_liabilities=1_796.99,
    retained_earnings=636.0, book_equity=2_844.23,
)

# Analyst assumptions, not reported data. Change these to run a view.
KAYNES_WACC = WaccInputs(
    risk_free=0.066,
    beta=1.25,
    equity_risk_premium=0.06,
    pre_tax_cost_of_debt=0.095,
    tax_rate=KAYNES_TAX_RATE,
    market_cap=KAYNES_CMP * KAYNES_SHARES_CR,
    gross_debt=912.87,
)
KAYNES_DCF = DCFInputs(
    base_revenue=KAYNES_FY26.revenue,
    base_ebit_margin=KAYNES_FY26.ebit / KAYNES_FY26.revenue,
    tax_rate=KAYNES_TAX_RATE,
    incremental_nwc=0.30,
    incremental_fixed_capital=0.45,
    terminal_roic=0.18,
    net_debt=912.87 - 109.43 - 689.16,
    non_controlling_interest=14.14,
    non_operating_assets=KAYNES_FY26.investments,
    shares_outstanding=KAYNES_SHARES_CR,
    cmp=KAYNES_CMP,
)


def validate_kaynes_inputs() -> None:
    """Confirm the audited inputs reproduce the mandated test figures.

    Raises:
        AssertionError: If an input no longer rounds to the mandated value.
    """
    actual = KAYNES_FY26
    checks = (
        ("Revenue (₹ Cr)", actual.revenue, 3_626),
        ("EBITDA (₹ Cr)", actual.ebitda, 574),
        ("CFO (₹ Cr)", actual.cfo, -600),
        ("Debtor days", actual.debtor_days, 154),
        ("CMP (₹)", KAYNES_CMP, 3_210),
    )
    _print_header("VALIDATION CASE: KAYNES TECHNOLOGY FY26 INPUTS")
    for name, value, expected in checks:
        assert round(value) == expected, f"{name}: {value} != {expected}"
        print(f"[PASS] {name:<16}{value:>12,.2f}   mandated {expected:,}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=10_000,
                        help="number of Monte Carlo paths (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("kaynes_valuation.html"),
        help="path of the chart HTML file",
    )
    parser.add_argument("--no-show", action="store_true",
                        help="save the chart without opening a browser")
    return parser.parse_args()


def main() -> None:
    """Execute the Kaynes Technology validation case end to end."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    validate_kaynes_inputs()
    ForensicAuditor(
        "Kaynes Technology", KAYNES_FY26, KAYNES_FY25,
    ).print_report()

    valuation = MonteCarloValuation(
        "Kaynes Technology", KAYNES_DCF, KAYNES_WACC,
        SimulationConfig(n_paths=args.paths, seed=args.seed),
    )
    result = valuation.run()
    valuation.print_report(result)

    figure = valuation.plot(result)
    figure.write_html(args.output, include_plotlyjs=True)
    print(f"\nChart saved to: {args.output}")
    if not args.no_show:
        figure.show()


if __name__ == "__main__":
    main()
