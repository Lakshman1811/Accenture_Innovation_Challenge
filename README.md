# KPI Intelligence-to-Action Engine — Round 2

**Accenture Innovation Challenge 2026 | Team: Golden Retriever | IIT Kharagpur**

Detect -> Decompose -> Decide -> Deliver. A KPI movement is detected statistically,
broken into quantified drivers, run through a confidence/abstention gate,and
only then handed to an LLM to translate into a persona-specific narrative and
action plan. The LLM never computes a number — it only writes about numbers
that deterministic code already produced.

## Architecture

```
DataGenerator --> AnalyticalEngine --> NarrativeEngine --> InsightOutput
 (synthetic,       (Z-score, IQR,       (LLM: narrative        (per KPI x
  5 sources)        materiality,         synthesis + action      persona)
                     driver decomp,       recommendations
                     confidence calc)     only)
                          |
                          v
                    SecurityLayer (RBAC + audit log)
                          |
                          v
                    FeedbackLoop (corrections, validations)
```

## What's LLM vs. non-LLM

| Stage | Method | LLM? |
|---|---|---|
| Anomaly detection | Z-score + IQR + materiality thresholds | No |
| Driver decomposition | Price-volume-mix / attribution / margin bridge / correlation | No |
| Confidence calibration | 6-dimension weighted score + abstention rules | No |
| Access control | Role-based allow-list + audit log | No |
| Narrative synthesis | Persona-specific phrasing of pre-validated evidence | Yes |
| Action wording | Templated from driver -> lever -> action map | No |

LLM cost is ~8% of total run cost — narrative synthesis only, never calculation.

## KPIs covered (5 + 1 sparse-history scenario)

| KPI | Grain | Source | Refresh |
|---|---|---|---|
| Gross Revenue | daily | ecommerce_transactions | hourly |
| Customer Acquisition Cost | weekly | marketing_platform | daily |
| Net Promoter Score | monthly | survey_platform | weekly |
| Inventory Turnover | daily | supply_chain_erp | daily |
| Gross Margin % | daily | finance_erp | hourly |
| APAC Revenue (sparse, 3 weeks) | weekly | ecommerce_transactions | daily |

## Minimum prototype checklist (from problem statement)

- [x] 3-5 connected KPIs across multiple sources/grains — 5 KPIs, 5 source systems, 3 grains
- [x] Lightweight KPI semantic contract — `KPI_CONTRACT` (formula, grain, thresholds, lineage, RBAC)
- [x] 2+ personas with different narratives/actions — CRO, Finance_Manager, Marketing_Manager, SupplyChain_Analyst
- [x] Multi-factor movement with known drivers — Revenue, 5 quantified drivers
- [x] Low-confidence abstention scenario — NPS (stale survey data, weak r=0.42 correlation)
- [x] Sparse-history scenario — APAC Revenue (3 weeks, abstains, requests more data)
- [x] Role-based security scenario — Marketing_Manager denied Gross Margin %
- [x] Evidence with freshness, method, contribution, confidence, lineage — `evidence` dict per insight
- [x] LLM vs. non-LLM breakdown — see table above
- [x] Runtime telemetry — SQL/stat-test counts, LLM tokens, cost, latency

## Key design decision: single source of truth for variance

`AnalyticalEngine.detect_anomaly()` computes `variance` once, using a clean
pre-event baseline window. `decompose_drivers()` receives that same
`AnomalyResult` and anchors all driver math to it, so
`decomposition.actual_variance` and `anomaly.variance` can never disagree.
Driver impacts are then reconciled (scaled within 0.2x-3x guard rails) toward
the real variance; if drivers still explain >130% of the movement even after
scaling, an `over_attributed` flag is raised instead of silently capping
coverage at 100% — an explicit signal that the driver model needs
recalibration.

## Run it

```bash
pip install pandas numpy scipy --break-system-packages
python3 kpi_engine_v2.py
```

Outputs a full console walkthrough: 4 personas x their accessible KPIs, the
APAC sparse-history abstention, the security audit log, and the telemetry/cost
report.

## Files

- `kpi_engine_v2.py` — full engine (single file, ~900 lines)
