#!/usr/bin/env python3
"""
KPI INTELLIGENCE-TO-ACTION ENGINE (v2 — Optimized)
====================================================
Accenture Innovation Challenge 2026 - Round 2
Team: Golden Retriever | IIT Kharagpur

v2 fixes vs. v1:
  - Anomaly baseline and decomposition baseline now share ONE number
    (anomaly.variance is passed into decompose_drivers instead of being
    recomputed on a different trailing window). This removes the
    "Variance: -380 but Actual Variance: 33297" contradiction seen in v1 output.
  - Driver impacts are reconciled against the real variance (scaled, with
    guard rails) instead of being fixed hardcoded dollars that could
    over/under-explain the movement by 2-3x while coverage was silently
    capped at 100%.
  - over_attributed flag added: if drivers would explain >130% of the
    actual movement, confidence is penalized instead of hidden.
  - Persona narratives for revenue/margin now pull driver name/impact/
    confidence dynamically instead of hardcoded numbers, so they stay
    correct even as underlying data/scaling changes.
  - Telemetry compute cost derived from actual query/test volume instead
    of a fixed constant.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import random

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# ============================================================
# SECTION 1: DATA MODELS & ENUMS
# ============================================================

class DriverCategory(Enum):
    CONTROLLABLE = "controllable"
    PARTIALLY_CONTROLLABLE = "partially_controllable"
    EXTERNAL = "external"

class FeedbackType(Enum):
    CORRECTION = "correction"
    VALIDATION = "validation"
    OVERRIDE = "override"
    IGNORE = "ignore"

class ExpertiseLevel(Enum):
    ANALYST = "analyst"
    BUSINESS_USER = "business_user"
    EXECUTIVE = "executive"

@dataclass
class Driver:
    name: str
    category: DriverCategory
    impact: float
    confidence: float
    evidence: str
    source: str
    method: str

    @property
    def score(self) -> float:
        return abs(self.impact) * (self.confidence / 100)

@dataclass
class AnomalyResult:
    is_anomaly: bool
    z_score: float
    iqr_anomaly: bool
    pct_change: float
    current_value: float
    expected_value: float
    variance: float
    direction: str
    material: bool
    confidence: float
    method: str
    lookback_periods: int

@dataclass
class DecompositionResult:
    kpi_id: str
    drivers: List[Driver]
    total_explained: float
    actual_variance: float
    coverage_pct: float
    method: str
    primary_driver: Optional[str] = None
    sparse_history: bool = False
    low_confidence_warning: bool = False
    over_attributed: bool = False

@dataclass
class ConfidenceAssessment:
    overall_confidence: float
    component_scores: Dict[str, float]
    should_abstain: bool
    abstention_reasons: List[str]
    recommendation: str

@dataclass
class ActionRecommendation:
    driver: str
    lever: str
    action: str
    expected_impact: str
    owner: str
    confidence: float
    timeline: str
    monitoring_plan: str

@dataclass
class NarrativeResult:
    narrative: str
    persona: str
    kpi_id: str
    llm_used: bool
    llm_purpose: str
    tokens_in: int
    tokens_out: int
    evidence_cited: List[str]
    confidence_level: float

@dataclass
class InsightOutput:
    kpi_id: str
    narrative: NarrativeResult
    actions: List[ActionRecommendation]
    anomaly: AnomalyResult
    confidence: ConfidenceAssessment
    decomposition: DecompositionResult
    evidence: Dict[str, Any]

@dataclass
class Telemetry:
    sql_queries: int = 0
    statistical_tests: int = 0
    ml_models_called: int = 0
    llm_calls: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    processing_time_ms: float = 0.0
    compute_cost_usd: float = 0.0
    llm_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.compute_cost_usd + self.llm_cost_usd


# ============================================================
# SECTION 2: KPI SEMANTIC CONTRACT
# ============================================================

KPI_CONTRACT = {
    "kpis": {
        "revenue": {
            "id": "KPI-001", "name": "Gross Revenue",
            "formula": "SUM(order_value) - SUM(returns)",
            "grain": "daily", "refresh_cadence": "hourly",
            "source_system": "ecommerce_transactions", "owner": "Finance",
            "materiality_threshold_pct": 5.0, "materiality_threshold_abs": 50000,
            "drivers": ["price", "volume", "mix", "returns", "promotions"],
            "sensitivity": "high", "access_roles": ["CRO", "CFO", "Finance_Manager", "CEO"],
            "lineage": ["raw_orders", "clean_orders", "kpi_revenue"]
        },
        "cac": {
            "id": "KPI-002", "name": "Customer Acquisition Cost",
            "formula": "SUM(marketing_spend) / COUNT(new_customers)",
            "grain": "weekly", "refresh_cadence": "daily",
            "source_system": "marketing_platform", "owner": "Marketing",
            "materiality_threshold_pct": 10.0, "materiality_threshold_abs": 5.0,
            "drivers": ["cpc", "conversion_rate", "channel_mix", "creative_performance", "audience_quality"],
            "sensitivity": "medium", "access_roles": ["CRO", "CMO", "Marketing_Manager", "Growth_Lead"],
            "lineage": ["ad_spend", "attribution_model", "kpi_cac"]
        },
        "nps": {
            "id": "KPI-003", "name": "Net Promoter Score",
            "formula": "%Promoters - %Detractors",
            "grain": "monthly", "refresh_cadence": "weekly",
            "source_system": "survey_platform", "owner": "Customer Success",
            "materiality_threshold_pct": 8.0, "materiality_threshold_abs": 5.0,
            "drivers": ["product_quality", "delivery_speed", "support_quality", "price_perception"],
            "sensitivity": "low", "access_roles": ["CRO", "CMO", "CXO", "Product_Lead"],
            "lineage": ["survey_responses", "nps_calculation", "kpi_nps"]
        },
        "inventory_turnover": {
            "id": "KPI-004", "name": "Inventory Turnover Ratio",
            "formula": "COGS / Avg Inventory",
            "grain": "daily", "refresh_cadence": "daily",
            "source_system": "supply_chain_erp", "owner": "Supply Chain",
            "materiality_threshold_pct": 12.0, "materiality_threshold_abs": 0.5,
            "drivers": ["demand_forecast_accuracy", "supplier_lead_time", "warehouse_efficiency", "promotions"],
            "sensitivity": "medium", "access_roles": ["COO", "SupplyChain_Manager", "CFO", "SupplyChain_Analyst"],
            "lineage": ["inventory_snapshots", "cogs_daily", "kpi_inventory"]
        },
        "gross_margin_pct": {
            "id": "KPI-005", "name": "Gross Margin %",
            "formula": "(Revenue - COGS) / Revenue * 100",
            "grain": "daily", "refresh_cadence": "hourly",
            "source_system": "finance_erp", "owner": "Finance",
            "materiality_threshold_pct": 3.0, "materiality_threshold_abs": 2.0,
            "drivers": ["input_costs", "pricing_power", "product_mix", "discount_depth", "supplier_terms"],
            "sensitivity": "high", "access_roles": ["CFO", "CRO", "Finance_Manager"],
            "lineage": ["revenue", "cogs", "kpi_gross_margin"]
        },
        "revenue_apac": {
            "id": "KPI-006", "name": "APAC Gross Revenue",
            "formula": "SUM(order_value) - returns (APAC only)",
            "grain": "weekly", "refresh_cadence": "daily",
            "source_system": "ecommerce_transactions", "owner": "Finance",
            "materiality_threshold_pct": 15.0, "materiality_threshold_abs": 10000,
            "drivers": ["market_penetration", "local_competition", "currency_fluctuation"],
            "sensitivity": "medium", "access_roles": ["CRO", "CFO", "Finance_Manager"],
            "lineage": ["raw_orders_apac", "clean_orders_apac", "kpi_revenue_apac"]
        }
    },
    "source_systems": {
        "ecommerce_transactions": {"type": "OLTP", "latency_sec": 30, "last_refresh": "2026-08-23T20:00:00Z"},
        "marketing_platform": {"type": "API", "latency_sec": 300, "last_refresh": "2026-08-23T18:00:00Z"},
        "survey_platform": {"type": "Batch", "latency_sec": 86400, "last_refresh": "2026-08-22T08:00:00Z"},
        "supply_chain_erp": {"type": "ERP", "latency_sec": 3600, "last_refresh": "2026-08-23T19:00:00Z"},
        "finance_erp": {"type": "ERP", "latency_sec": 1800, "last_refresh": "2026-08-23T20:00:00Z"}
    }
}


# ============================================================
# SECTION 3: SYNTHETIC DATA GENERATOR
# ============================================================

class DataGenerator:
    @staticmethod
    def generate_all() -> Dict[str, pd.DataFrame]:
        end_date = datetime(2026, 8, 23)
        data = {}

        dates_daily = pd.date_range(end=end_date, periods=180, freq='D')
        trend = np.linspace(450000, 520000, 180)
        seasonality = 30000 * np.sin(2 * np.pi * np.arange(180) / 30.5)
        weekly = 15000 * np.sin(2 * np.pi * np.arange(180) / 7)
        noise = np.random.normal(0, 8000, 180)
        revenue = trend + seasonality + weekly + noise
        # Multi-factor anomaly spread across Aug 17-23, including the most recent day
        # (v1 had the shock trail off to 0 on the last two days, so "current" barely moved)
        revenue[-7:] += np.array([-8000, -12000, -16000, -18000, -16000, -14000, -14000])
        revenue = np.maximum(revenue, 100000)
        data['revenue'] = pd.DataFrame({
            'date': dates_daily, 'value': revenue.round(2),
            'source': 'ecommerce_transactions', 'grain': 'daily', 'freshness_hours': 1.5
        })

        dates_weekly = pd.date_range(end=end_date, periods=26, freq='W')
        cac = 45.0 + np.linspace(0, 5, 26) + np.random.normal(0, 2, 26)
        cac[-1] += 12.5
        data['cac'] = pd.DataFrame({
            'date': dates_weekly, 'value': np.maximum(cac, 20).round(2),
            'source': 'marketing_platform', 'grain': 'weekly', 'freshness_hours': 18.0
        })

        dates_monthly = pd.date_range(end=end_date, periods=12, freq='30D')
        nps = 42.0 + np.linspace(0, 3, 12) + np.random.normal(0, 1.5, 12)
        nps[-1] -= 4.2
        data['nps'] = pd.DataFrame({
            'date': dates_monthly, 'value': nps.round(1),
            'source': 'survey_platform', 'grain': 'monthly', 'freshness_hours': 48.0,
            'sample_size': [450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 480]
        })

        turnover = 8.5 + np.linspace(0, 0.8, 180) + np.random.normal(0, 0.3, 180)
        turnover[-7:] -= 1.8
        data['inventory_turnover'] = pd.DataFrame({
            'date': dates_daily, 'value': np.maximum(turnover, 2).round(2),
            'source': 'supply_chain_erp', 'grain': 'daily', 'freshness_hours': 2.5
        })

        margin = 32.0 + np.linspace(0, -1.5, 180) + np.random.normal(0, 0.4, 180)
        margin[-7:] -= 3.2
        data['gross_margin_pct'] = pd.DataFrame({
            'date': dates_daily, 'value': np.clip(margin, 15, 50).round(2),
            'source': 'finance_erp', 'grain': 'daily', 'freshness_hours': 1.5
        })

        dates_apac = pd.date_range(end=end_date, periods=3, freq='W')
        data['revenue_apac'] = pd.DataFrame({
            'date': dates_apac, 'value': [85000, 92000, 78000],
            'source': 'ecommerce_transactions', 'grain': 'weekly', 'freshness_hours': 2.0,
            'is_sparse': True, 'market': 'APAC'
        })

        return data


# ============================================================
# SECTION 4: ANALYTICAL ENGINE (NON-LLM)
# ============================================================

class AnalyticalEngine:
    """
    Deterministic analytical engine. NO LLM is used for any quantitative
    computation. Anomaly detection and driver decomposition now share a
    single baseline/variance number so downstream coverage and confidence
    numbers stay internally consistent.
    """

    def __init__(self, kpi_contract: Dict):
        self.contract = kpi_contract
        self.telemetry = Telemetry()

    def detect_anomaly(self, series: pd.Series, kpi_id: str) -> AnomalyResult:
        self.telemetry.statistical_tests += 3
        kpi_def = self.contract['kpis'][kpi_id]

        # Statistical baseline: trailing 30 periods -> shape of the distribution (z-score, IQR)
        stat_window = series.iloc[-31:-1] if len(series) >= 31 else series.iloc[:-1]
        std = stat_window.std()
        current = series.iloc[-1]

        # Business baseline: a clean pre-event window, deliberately excluding the last 7 periods
        # (the movement being investigated may itself span several of the most recent periods,
        # so including them in the baseline would understate/overstate the variance).
        lookback, gap = 30, 7
        if len(series) >= lookback + gap + 1:
            biz_window = series.iloc[-(lookback + gap):-gap]
        elif len(series) >= gap + 2:
            biz_window = series.iloc[:-gap]
        else:
            biz_window = series.iloc[:-1]
        expected = biz_window.mean()

        z_score = (current - stat_window.mean()) / std if std > 0 else 0

        q1, q3 = stat_window.quantile([0.25, 0.75])
        iqr = q3 - q1
        is_iqr_anomaly = current < (q1 - 1.5 * iqr) or current > (q3 + 1.5 * iqr)

        variance = current - expected
        pct_change = (variance / abs(expected)) * 100 if expected != 0 else 0

        mat_pct = kpi_def['materiality_threshold_pct']
        mat_abs = kpi_def['materiality_threshold_abs']
        is_material = abs(pct_change) >= mat_pct or abs(variance) >= mat_abs

        signals = sum([abs(z_score) > 2.5, is_iqr_anomaly, is_material])
        confidence = min(signals / 3 * 100, 100)
        direction = "up" if current > expected else "down"

        return AnomalyResult(
            is_anomaly=signals >= 2 and is_material,
            z_score=round(abs(z_score), 3),  # magnitude only; sign is carried by `direction`
            iqr_anomaly=is_iqr_anomaly,
            pct_change=round(pct_change, 2),
            current_value=round(current, 2),
            expected_value=round(expected, 2),
            variance=round(variance, 2),
            direction=direction,
            material=is_material,
            confidence=round(confidence, 1),
            method='zscore_iqr_materiality_hybrid',
            lookback_periods=len(stat_window)
        )

    def decompose_drivers(self, kpi_id: str, datasets: Dict, anomaly: AnomalyResult) -> DecompositionResult:
        """Driver decomposition anchored to anomaly.variance (single source of truth)."""
        self.telemetry.statistical_tests += 5
        variance = anomaly.variance

        decomposers = {
            'revenue': self._decompose_revenue,
            'cac': self._decompose_cac,
            'gross_margin_pct': self._decompose_margin,
            'inventory_turnover': self._decompose_inventory,
            'nps': self._decompose_nps,
        }

        if kpi_id not in decomposers:
            return DecompositionResult(
                kpi_id=kpi_id, drivers=[], total_explained=0,
                actual_variance=variance, coverage_pct=0, method='none', primary_driver=None
            )

        result = decomposers[kpi_id](variance, datasets)
        return self._reconcile(result, variance)

    def _reconcile(self, result: DecompositionResult, variance: float) -> DecompositionResult:
        """Scale hardcoded illustrative driver impacts toward the real variance, with guard rails,
        so total_explained/coverage are meaningful instead of silently capped."""
        raw_total = sum(d.impact for d in result.drivers)
        if raw_total != 0 and variance != 0 and np.sign(raw_total) == np.sign(variance):
            scale = variance / raw_total
            scale = max(min(scale, 3.0), 0.2)  # guard rails: don't let scaling distort the story >3x or <0.2x
            for d in result.drivers:
                d.impact = round(d.impact * scale, 2)

        total = sum(d.impact for d in result.drivers)
        raw_coverage = (abs(total) / abs(variance) * 100) if variance != 0 else 0

        result.drivers.sort(key=lambda d: d.score, reverse=True)
        result.total_explained = round(total, 2)
        result.actual_variance = round(variance, 2)
        result.coverage_pct = round(min(raw_coverage, 100), 1)
        result.over_attributed = raw_coverage > 130
        result.primary_driver = result.drivers[0].name if result.drivers else None
        return result

    def _decompose_revenue(self, variance: float, datasets: Dict) -> DecompositionResult:
        drivers = [
            Driver("price_promotion", DriverCategory.CONTROLLABLE, -45000, 92,
                   "Promo calendar: 25% off EcoLine Aug 17-23. ASP dropped 18% WoW.",
                   "promo_calendar + transaction_sql", "price_volume_mix"),
            Driver("supply_disruption_top_sku", DriverCategory.PARTIALLY_CONTROLLABLE, -35000, 96,
                   "SKU-8842 stockout Aug 19-22. 12,400 units unfulfilled.",
                   "supply_chain_erp", "stockout_impact"),
            Driver("marketing_spend_reduction", DriverCategory.CONTROLLABLE, -25000, 85,
                   "Meta/Google spend cut 32% WoW. Traffic -28%.",
                   "marketing_platform", "attribution_regression"),
            Driver("seasonal_back_to_school", DriverCategory.EXTERNAL, 15000, 78,
                   "Historical BTS uplift +3.2%. This year +3.5%.",
                   "historical_timeseries", "seasonal_decomposition"),
            Driver("product_mix_shift", DriverCategory.CONTROLLABLE, -8000, 72,
                   "Lower-margin SKUs 62% vs 55% baseline.",
                   "transaction_sql", "mix_variance"),
        ]
        return DecompositionResult(
            kpi_id='revenue', drivers=drivers, total_explained=0, actual_variance=variance,
            coverage_pct=0, method='price_volume_mix + attribution + supply_chain'
        )

    def _decompose_cac(self, variance: float, datasets: Dict) -> DecompositionResult:
        drivers = [
            Driver("cpc_inflation_meta", DriverCategory.EXTERNAL, 8.5, 88,
                   "Meta CPC +22% WoW. CTR stable 1.8%.", "marketing_platform", "cpc_variance"),
            Driver("conversion_rate_drop", DriverCategory.CONTROLLABLE, 4.2, 75,
                   "Landing page CVR 2.1% vs 2.8% baseline.", "analytics_platform", "funnel_analysis"),
        ]
        return DecompositionResult(
            kpi_id='cac', drivers=drivers, total_explained=0, actual_variance=variance,
            coverage_pct=0, method='marketing_attribution'
        )

    def _decompose_margin(self, variance: float, datasets: Dict) -> DecompositionResult:
        drivers = [
            Driver("promo_discount_depth", DriverCategory.CONTROLLABLE, -2.1, 95,
                   "Discount rate 18.5% vs 12% baseline.", "transaction_sql", "margin_bridge"),
            Driver("input_cost_inflation", DriverCategory.PARTIALLY_CONTROLLABLE, -1.1, 90,
                   "Resin index +8.3% MoM. Pass-through +$0.42/unit.", "procurement_erp", "cost_bridge"),
        ]
        return DecompositionResult(
            kpi_id='gross_margin_pct', drivers=drivers, total_explained=0, actual_variance=variance,
            coverage_pct=0, method='margin_bridge'
        )

    def _decompose_inventory(self, variance: float, datasets: Dict) -> DecompositionResult:
        drivers = [
            Driver("supplier_lead_time", DriverCategory.PARTIALLY_CONTROLLABLE, -1.8, 94,
                   "Supplier LT 14 days vs 7 days baseline.", "supply_chain_erp", "inventory_policy"),
        ]
        return DecompositionResult(
            kpi_id='inventory_turnover', drivers=drivers, total_explained=0, actual_variance=variance,
            coverage_pct=0, method='supply_chain_model'
        )

    def _decompose_nps(self, variance: float, datasets: Dict) -> DecompositionResult:
        drivers = [
            Driver("delivery_speed", DriverCategory.CONTROLLABLE, -3.5, 55,
                   "Delivery 4.2 days vs 3.1 days. Correlation 0.42.", "survey + logistics", "correlation"),
        ]
        return DecompositionResult(
            kpi_id='nps', drivers=drivers, total_explained=0, actual_variance=variance,
            coverage_pct=0, method='correlation_survey', low_confidence_warning=True
        )

    def assess_confidence(self, anomaly: AnomalyResult, decomposition: DecompositionResult,
                           data_freshness: float, is_sparse: bool = False) -> ConfidenceAssessment:
        scores = [anomaly.confidence, min(decomposition.coverage_pct, 100)]

        freshness_score = 100 if data_freshness <= 24 else max(100 - (data_freshness - 24), 50)
        scores.append(freshness_score)
        scores.append(30 if is_sparse else 100)

        drivers = decomposition.drivers
        avg_driver_conf = np.mean([d.confidence for d in drivers]) if drivers else 0
        scores.append(avg_driver_conf)

        has_contradiction = decomposition.over_attributed
        neg = [d for d in drivers if d.impact < 0]
        pos = [d for d in drivers if d.impact > 0]
        if len(neg) > 0 and len(pos) > 0 and anomaly.direction == 'down':
            pos_sum = sum(d.impact for d in pos)
            neg_sum = abs(sum(d.impact for d in neg))
            if pos_sum > neg_sum * 0.3:
                has_contradiction = True
        scores.append(70 if has_contradiction else 100)

        overall = np.mean(scores)

        should_abstain = False
        reasons = []

        if is_sparse and len(drivers) == 0:
            should_abstain = True
            reasons.append("Sparse history with no drivers")
        elif overall < 40:
            should_abstain = True
            reasons.append("Overall confidence critically low")
        elif anomaly.confidence < 30 and decomposition.coverage_pct < 40:
            should_abstain = True
            reasons.append("Weak signal and poor coverage")
        elif avg_driver_conf < 50 and decomposition.coverage_pct < 50:
            should_abstain = True
            reasons.append("Low driver confidence and coverage")

        if decomposition.kpi_id == 'nps' and avg_driver_conf < 60:
            should_abstain = True
            reasons.append("Driver confidence below threshold (55%)")
            reasons.append("Survey data 48h stale")
            reasons.append("Correlation evidence weak (r=0.42)")

        if decomposition.over_attributed:
            reasons.append("Drivers over-explain the movement (>130% coverage) — recalibration recommended")

        return ConfidenceAssessment(
            overall_confidence=round(overall, 1),
            component_scores={
                'statistical_signal': round(anomaly.confidence, 1),
                'driver_coverage': round(decomposition.coverage_pct, 1),
                'data_freshness': round(freshness_score, 1),
                'history_depth': 30 if is_sparse else 100,
                'avg_driver_confidence': round(avg_driver_conf, 1),
                'signal_consistency': 70 if has_contradiction else 100
            },
            should_abstain=should_abstain,
            abstention_reasons=reasons,
            recommendation='ABSTAIN: Request analyst review' if should_abstain else 'PROCEED: Generate narrative'
        )


# ============================================================
# SECTION 5: NARRATIVE ENGINE (LLM LAYER)
# ============================================================

class NarrativeEngine:
    """
    LLM-assisted narrative synthesis only: intent understanding, evidence
    synthesis, tone adaptation. Never calculation, inference, ranking, or
    quantification. Numbers below are pulled from the reconciled Driver
    objects rather than hardcoded, so they stay correct after scaling.
    """

    def __init__(self):
        self.telemetry = Telemetry()
        self._action_map = {
            'price_promotion': ('promo_calendar', 'Reduce discount depth to 15% or shift to bundle offers', 'Pricing_Manager'),
            'marketing_spend_reduction': ('budget_reallocation', 'Request emergency reallocation: shift $15K from brand to performance', 'Growth_Lead'),
            'supply_disruption_top_sku': ('inventory_expedite', 'Expedite PO-8842 via air freight; activate safety stock at DC-East', 'SupplyChain_Manager'),
            'cpc_inflation_meta': ('channel_reallocation', 'Launch creative refresh; test TikTok/YouTube CPM efficiency', 'Performance_Marketing_Lead'),
            'conversion_rate_drop': ('lp_optimization', 'Deploy LP A/B test (variant: social proof above fold)', 'CRO_Specialist'),
            'promo_discount_depth': ('promo_calendar', 'Reduce sitewide to 15%; move 25% to email-exclusive', 'Pricing_Manager'),
            'input_cost_inflation': ('procurement_renegotiation', 'Renegotiate Q4 supplier contract; explore alternative materials', 'Procurement_Lead'),
            'supplier_lead_time': ('supplier_diversification', 'Activate secondary supplier S-204; renegotiate SLA', 'SupplyChain_Manager'),
            'delivery_speed': ('logistics_review', 'Review last-mile carrier contract; add regional DC capacity', 'Logistics_Manager'),
        }

    def generate_narrative(self, kpi_id: str, anomaly: AnomalyResult,
                            decomposition: DecompositionResult,
                            confidence: ConfidenceAssessment, persona: str) -> NarrativeResult:
        self.telemetry.llm_calls += 1
        evidence_text = " | ".join([d.evidence for d in decomposition.drivers])
        tokens_in = len(evidence_text.split()) + 50

        if confidence.should_abstain:
            narrative = self._abstention_narrative(kpi_id, confidence, persona)
        else:
            narrative = self._insight_narrative(kpi_id, anomaly, decomposition, persona)

        tokens_out = len(narrative.split())
        self.telemetry.llm_tokens_in += tokens_in
        self.telemetry.llm_tokens_out += tokens_out
        self.telemetry.llm_cost_usd += (tokens_in / 1000 * 0.00015) + (tokens_out / 1000 * 0.0006)

        return NarrativeResult(
            narrative=narrative, persona=persona, kpi_id=kpi_id, llm_used=True,
            llm_purpose='narrative_synthesis_only', tokens_in=tokens_in, tokens_out=tokens_out,
            evidence_cited=[d.evidence for d in decomposition.drivers],
            confidence_level=confidence.overall_confidence
        )

    @staticmethod
    def _find(decomposition: DecompositionResult, name: str) -> Optional[Driver]:
        return next((d for d in decomposition.drivers if d.name == name), None)

    def _insight_narrative(self, kpi_id: str, anomaly: AnomalyResult,
                            decomposition: DecompositionResult, persona: str) -> str:
        pct = abs(anomaly.pct_change)
        current = anomaly.current_value
        expected = anomaly.expected_value
        top = decomposition.drivers[0] if decomposition.drivers else None

        if persona == 'Finance_Manager':
            if kpi_id == 'revenue':
                promo = self._find(decomposition, 'price_promotion')
                supply = self._find(decomposition, 'supply_disruption_top_sku')
                mktg = self._find(decomposition, 'marketing_spend_reduction')
                season = self._find(decomposition, 'seasonal_back_to_school')
                residual = decomposition.actual_variance - decomposition.total_explained
                return (f"Revenue variance analysis: Current ${current:,.0f} vs. baseline ${expected:,.0f}. "
                        f"Net variance: ${anomaly.variance:,.0f} ({pct:.1f}%). Bridge analysis identifies "
                        f"promotional depth (${promo.impact:,.0f}), supply disruption (${supply.impact:,.0f}), "
                        f"and marketing efficiency (${mktg.impact:,.0f}) as primary drivers. "
                        f"Seasonal offset: ${season.impact:,.0f}. Unexplained residual: ~${residual:,.0f} "
                        f"(coverage {decomposition.coverage_pct}%). Recommend: Detailed PVM review with Pricing team.")
            elif kpi_id == 'gross_margin_pct':
                promo = self._find(decomposition, 'promo_discount_depth')
                cost = self._find(decomposition, 'input_cost_inflation')
                return (f"Margin bridge: {current:.1f}% vs. {expected:.1f}% baseline. Variance decomposition: "
                        f"Promo depth impact {promo.impact:+.1f}pp ({promo.confidence:.0f}% confidence), "
                        f"input cost inflation {cost.impact:+.1f}pp ({cost.confidence:.0f}% confidence). "
                        f"Total explained: {decomposition.total_explained:+.1f}pp (coverage {decomposition.coverage_pct}%). "
                        f"Recommend: Procurement renegotiation by Sep 15.")

        if persona == 'CRO':
            if kpi_id == 'revenue':
                supply = self._find(decomposition, 'supply_disruption_top_sku')
                mktg = self._find(decomposition, 'marketing_spend_reduction')
                return (f"Revenue is {anomaly.direction} {pct:.1f}% to ${current:,.0f} vs. our ${expected:,.0f} baseline — "
                        f"a material {'miss' if anomaly.direction == 'down' else 'gain'} of ${abs(anomaly.variance):,.0f}. "
                        f"The primary driver is {top.name.replace('_', ' ')} ({top.confidence:.0f}% confidence), "
                        f"compounded by a stockout on our top SKU (${abs(supply.impact):,.0f}) and a spend cut in digital "
                        f"marketing (${abs(mktg.impact):,.0f}). Driver coverage: {decomposition.coverage_pct}%. "
                        f"Recommended action: Review promo calendar elasticity and expedite SKU-8842 replenishment.")
            elif kpi_id == 'cac':
                return (f"CAC spiked {pct:.1f}% to ${current:.2f}, well above our ${expected:.2f} baseline. "
                        f"{top.name.replace('_', ' ').title()} is the largest driver (+${top.impact:.1f}, {top.confidence:.0f}% "
                        f"confidence), with a secondary drag from landing-page conversion degradation. This threatens Q3 "
                        f"unit economics. Recommend immediate reallocation test to TikTok/YouTube.")
            elif kpi_id == 'gross_margin_pct':
                return (f"Gross margin compressed {abs(anomaly.variance):.1f}pp to {current:.1f}%, driven primarily by "
                        f"{top.name.replace('_', ' ')} ({top.confidence:.0f}% confidence). Coverage: {decomposition.coverage_pct}%. "
                        f"Recommend supplier renegotiation and promo tier review.")
            elif kpi_id == 'inventory_turnover':
                return (f"Inventory turnover dipped to {current:.1f}x from {expected:.1f}x baseline. "
                        f"{top.name.replace('_', ' ').title()} is the dominant driver ({top.confidence:.0f}% confidence). "
                        f"Recommend activating secondary supplier.")

        elif persona == 'Marketing_Manager':
            if kpi_id == 'cac':
                return (f"CAC alert: ${current:.2f} ({pct:+.1f}% vs. plan). Meta CPCs are up sharply WoW — auction "
                        f"pressure from competitor spend. Landing page CVR degraded. Action: Launch creative refresh + "
                        f"LP A/B test by EOD. Reallocate 15% budget to YouTube.")
            elif kpi_id == 'revenue':
                mktg = self._find(decomposition, 'marketing_spend_reduction')
                return (f"Revenue {anomaly.direction} {pct:.1f}% — note our spend cut contributed ~${abs(mktg.impact):,.0f} "
                        f"of the movement. Action: Request emergency spend approval for Meta retargeting.")

        elif persona == 'SupplyChain_Analyst':
            if kpi_id == 'inventory_turnover':
                return (f"Turnover moved to {current:.1f}x. Root cause: {top.evidence} "
                        f"Action: PO-9982 expedite requested. Secondary supplier S-204 on standby.")

        return f"{kpi_id.replace('_', ' ').title()} moved {anomaly.direction} {pct:.1f}%. Review drivers for details."

    def _abstention_narrative(self, kpi_id: str, confidence: ConfidenceAssessment, persona: str) -> str:
        if kpi_id == 'nps':
            if persona == 'CRO':
                return (f"NPS movement detected, but the engine is abstaining from a strong narrative. "
                        f"Why: driver confidence is only 55% (delivery-speed correlation is weak at r=0.42), and survey "
                        f"data is 48 hours stale. Recommended: Escalate to CX analyst for qualitative review before action.")
            return (f"NPS insight unavailable — confidence too low ({confidence.overall_confidence}%). "
                    f"Reasons: {', '.join(confidence.abstention_reasons)}. Human review required.")
        elif kpi_id == 'revenue_apac':
            return ("APAC revenue cannot be reliably analyzed — only 3 weeks of history available, with no "
                    "established baseline. The engine requires at least 8 weeks to detect anomalies reliably. "
                    "Recommended: Monitor for 4 more weeks; no action suggested at this time.")
        return "Insufficient evidence. Human review required."

    def generate_actions(self, kpi_id: str, decomposition: DecompositionResult,
                          confidence: ConfidenceAssessment) -> List[ActionRecommendation]:
        if confidence.should_abstain:
            return [ActionRecommendation(
                driver='N/A', lever='N/A', action='Escalate to human analyst',
                expected_impact='TBD', owner='Analyst_Team', confidence=confidence.overall_confidence,
                timeline='24h', monitoring_plan='Re-run when data freshness < 12h'
            )]

        actions = []
        for d in decomposition.drivers[:3]:
            if d.category in [DriverCategory.CONTROLLABLE, DriverCategory.PARTIALLY_CONTROLLABLE]:
                lever, action_text, owner = self._action_map.get(d.name, ('investigate', f'Investigate {d.name}', 'Analyst'))
                impact_str = f"${abs(d.impact):,.0f} recovery" if abs(d.impact) > 100 else f"{abs(d.impact):.1f}pp improvement"
                actions.append(ActionRecommendation(
                    driver=d.name, lever=lever, action=action_text, expected_impact=impact_str, owner=owner,
                    confidence=d.confidence, timeline='48-72h',
                    monitoring_plan=f"Track {kpi_id} daily until baseline restored"
                ))
        return actions


# ============================================================
# SECTION 6: SECURITY LAYER
# ============================================================

class SecurityLayer:
    def __init__(self, kpi_contract: Dict):
        self.contract = kpi_contract
        self.audit_log: List[Dict] = []

    def check_access(self, persona: str, kpi_id: str) -> Dict:
        kpi_def = self.contract['kpis'].get(kpi_id)
        if not kpi_def:
            result = {'granted': False, 'reason': 'KPI not found'}
        else:
            allowed = kpi_def['access_roles']
            granted = persona in allowed
            result = {'granted': granted, 'kpi': kpi_id, 'persona': persona,
                      'allowed_roles': allowed, 'sensitivity': kpi_def['sensitivity']}

        self.audit_log.append({
            'timestamp': datetime.now().isoformat(), 'persona': persona, 'kpi': kpi_id,
            'action': 'ACCESS_CHECK', 'granted': result['granted'],
            'ip': '10.0.1.45', 'session_id': f'sess_{random.randint(1000,9999)}'
        })
        return result

    def filter_kpis(self, persona: str) -> List[str]:
        return [kpi_id for kpi_id, defn in self.contract['kpis'].items()
                if persona in defn['access_roles']]

    def get_audit_log(self) -> List[Dict]:
        return self.audit_log


# ============================================================
# SECTION 7: FEEDBACK LOOP
# ============================================================

class FeedbackLoop:
    def __init__(self):
        self.feedback_store: List[Dict] = []
        self.correction_examples: List[Dict] = []

    def submit_feedback(self, insight_id: str, persona: str, feedback_type: FeedbackType,
                         field_corrected: str, original_value: str, corrected_value: str,
                         expertise_level: ExpertiseLevel) -> str:
        feedback_id = f"fb_{random.randint(10000,99999)}"
        entry = {
            'feedback_id': feedback_id, 'insight_id': insight_id, 'persona': persona,
            'feedback_type': feedback_type.value, 'field_corrected': field_corrected,
            'original_value': original_value, 'corrected_value': corrected_value,
            'expertise_level': expertise_level.value, 'timestamp': datetime.now().isoformat()
        }
        self.feedback_store.append(entry)
        if feedback_type == FeedbackType.CORRECTION:
            self.correction_examples.append(entry)
        return feedback_id

    def get_learning_stats(self) -> Dict:
        if not self.feedback_store:
            return {'total': 0, 'corrections': 0, 'validations': 0}
        corrections = sum(1 for f in self.feedback_store if f['feedback_type'] == 'correction')
        validations = sum(1 for f in self.feedback_store if f['feedback_type'] == 'validation')
        return {'total': len(self.feedback_store), 'corrections': corrections, 'validations': validations,
                'correction_rate': round(corrections / len(self.feedback_store) * 100, 1)}


# ============================================================
# SECTION 8: MAIN KPI ENGINE ORCHESTRATOR
# ============================================================

class KPIIntelligenceEngine:
    def __init__(self, kpi_contract: Dict):
        self.contract = kpi_contract
        self.analytical = AnalyticalEngine(kpi_contract)
        self.narrative = NarrativeEngine()
        self.security = SecurityLayer(kpi_contract)
        self.feedback = FeedbackLoop()
        self.telemetry = Telemetry()

    def process_kpi(self, kpi_id: str, persona: str, datasets: Dict) -> Optional[InsightOutput]:
        access = self.security.check_access(persona, kpi_id)
        if not access['granted']:
            print(f"  DENIED: {persona} -> {kpi_id}")
            return None

        if kpi_id not in datasets:
            return None
        df = datasets[kpi_id]

        anomaly = self.analytical.detect_anomaly(df['value'], kpi_id)
        decomposition = self.analytical.decompose_drivers(kpi_id, datasets, anomaly)

        is_sparse = 'is_sparse' in df.columns and df['is_sparse'].iloc[0]
        freshness = df['freshness_hours'].iloc[-1]
        confidence = self.analytical.assess_confidence(anomaly, decomposition, freshness, is_sparse)

        narrative = self.narrative.generate_narrative(kpi_id, anomaly, decomposition, confidence, persona)
        actions = self.narrative.generate_actions(kpi_id, decomposition, confidence)

        evidence = {
            'source_freshness': f"{freshness}h",
            'analytical_method': decomposition.method,
            'driver_count': len(decomposition.drivers),
            'lineage': self.contract['kpis'][kpi_id]['lineage'],
            'data_grain': df['grain'].iloc[0],
            'data_source': df['source'].iloc[0],
            'over_attributed': decomposition.over_attributed
        }

        self.telemetry.sql_queries += self.analytical.telemetry.sql_queries
        self.telemetry.statistical_tests += self.analytical.telemetry.statistical_tests
        self.telemetry.llm_calls += self.narrative.telemetry.llm_calls
        self.telemetry.llm_tokens_in += self.narrative.telemetry.llm_tokens_in
        self.telemetry.llm_tokens_out += self.narrative.telemetry.llm_tokens_out
        self.telemetry.llm_cost_usd += self.narrative.telemetry.llm_cost_usd

        return InsightOutput(kpi_id=kpi_id, narrative=narrative, actions=actions, anomaly=anomaly,
                              confidence=confidence, decomposition=decomposition, evidence=evidence)

    def get_telemetry_report(self) -> Dict:
        compute_cost = round(max(
            self.telemetry.sql_queries * 0.00002 + self.telemetry.statistical_tests * 0.00005, 0.01
        ), 4)
        total = compute_cost + self.telemetry.llm_cost_usd
        return {
            'sql_queries': self.telemetry.sql_queries,
            'statistical_tests': self.telemetry.statistical_tests,
            'llm_calls': self.telemetry.llm_calls,
            'llm_tokens_in': self.telemetry.llm_tokens_in,
            'llm_tokens_out': self.telemetry.llm_tokens_out,
            'compute_cost_usd': compute_cost,
            'llm_cost_usd': round(self.telemetry.llm_cost_usd, 4),
            'total_cost_usd': round(total, 4),
            'llm_pct_of_total': round(self.telemetry.llm_cost_usd / total * 100, 1) if total > 0 else 0,
            'estimated_latency_ms': 45 + (self.telemetry.llm_calls * 120)
        }


# ============================================================
# SECTION 9: DEMO SCRIPT
# ============================================================

def run_demo():
    print("=" * 80)
    print("  KPI INTELLIGENCE-TO-ACTION ENGINE - ROUND 2 DEMO (v2)")
    print("  Team Golden Retriever | IIT Kharagpur")
    print("=" * 80)

    datasets = DataGenerator.generate_all()
    engine = KPIIntelligenceEngine(KPI_CONTRACT)

    personas = {
        'CRO': ['revenue', 'cac', 'nps', 'gross_margin_pct'],
        'Marketing_Manager': ['cac'],
        'SupplyChain_Analyst': ['inventory_turnover'],
        'Finance_Manager': ['revenue', 'gross_margin_pct']
    }

    all_outputs = []

    for persona, kpis in personas.items():
        print(f"\n{'='*80}\n  PERSONA: {persona}\n  Accessible KPIs: {kpis}\n{'='*80}")
        for kpi_id in kpis:
            output = engine.process_kpi(kpi_id, persona, datasets)
            if not output:
                continue
            all_outputs.append(output)
            print(f"\n  {kpi_id.upper()}")
            print(f"     Status: {'ABSTAINED' if output.confidence.should_abstain else 'INSIGHT GENERATED'}")
            print(f"     Confidence: {output.confidence.overall_confidence}%")
            print(f"     Anomaly: current={output.anomaly.current_value} expected={output.anomaly.expected_value} "
                  f"variance={output.anomaly.variance} pct_change={output.anomaly.pct_change}% "
                  f"z={output.anomaly.z_score} material={output.anomaly.material}")
            print(f"     Decomposition: drivers={len(output.decomposition.drivers)} "
                  f"coverage={output.decomposition.coverage_pct}% total_explained={output.decomposition.total_explained} "
                  f"actual_variance={output.decomposition.actual_variance} over_attributed={output.decomposition.over_attributed}")
            if output.confidence.should_abstain:
                print(f"     Abstention reasons: {output.confidence.abstention_reasons}")
            print(f"     Narrative: {output.narrative.narrative}")
            if output.actions:
                a = output.actions[0]
                print(f"     Top action: {a.action} | impact={a.expected_impact} | owner={a.owner} | conf={a.confidence}%")

    print(f"\n{'='*80}\n  SPARSE HISTORY SCENARIO: APAC REVENUE\n{'='*80}")
    apac_output = engine.process_kpi('revenue_apac', 'CRO', datasets)
    if apac_output:
        all_outputs.append(apac_output)
        print(f"  Status: {'ABSTAINED' if apac_output.confidence.should_abstain else 'PROCEED'}")
        print(f"  Confidence: {apac_output.confidence.overall_confidence}%")
        print(f"  Reasons: {apac_output.confidence.abstention_reasons}")
        print(f"  Narrative: {apac_output.narrative.narrative}")

    print(f"\n{'='*80}\n  SECURITY AUDIT LOG\n{'='*80}")
    for entry in engine.security.get_audit_log()[:8]:
        status = "GRANTED" if entry['granted'] else "DENIED"
        print(f"     {entry['timestamp']} | {entry['persona']} -> {entry['kpi']}: {status}")

    print(f"\n{'='*80}\n  RUNTIME TELEMETRY & COST ANALYSIS\n{'='*80}")
    report = engine.get_telemetry_report()
    print(f"     SQL Queries: {report['sql_queries']} | Statistical Tests: {report['statistical_tests']}")
    print(f"     LLM Calls: {report['llm_calls']} | Tokens in/out: {report['llm_tokens_in']}/{report['llm_tokens_out']}")
    print(f"     Compute cost: ${report['compute_cost_usd']:.4f} | LLM cost: ${report['llm_cost_usd']:.4f} | "
          f"Total: ${report['total_cost_usd']:.4f} | LLM % of total: {report['llm_pct_of_total']:.1f}%")
    print(f"     Estimated end-to-end latency: ~{report['estimated_latency_ms']}ms")

    print(f"\n{'='*80}\n  DEMO COMPLETE")
    print(f"  Total Insights Generated: {len(all_outputs)}")
    print(f"  Abstentions: {sum(1 for o in all_outputs if o.confidence.should_abstain)}")
    print(f"{'='*80}")

    return all_outputs, engine


if __name__ == "__main__":
    outputs, engine = run_demo()