# UK Recession & Inflation Forecast (May 2026)

Monte Carlo macro forecast of UK recession probability and CPI inflation for
the 8 quarters from 2026 Q2 through 2028 Q1, produced by the Multi-Agent
Reasoning Council.

## Headlines (probability-weighted, baseline 0.45 / adverse 0.25 /
stagflation 0.10 / soft-landing 0.20)

| Metric | Value |
|---|---:|
| P(technical recession in 2026) | ~52% |
| P(CPI > 4% any quarter, next 8q) | ~35% |
| Median Bank Rate end-2028 | ~4.4% |
| Peak unemployment | ~5.3% |

Central case alone: **P(recession 2026) = 43.6%** (vs prediction market 45.5%);
CPI back to target by 2027Q2 median.

## Layout

```
forecasts/uk_2026/
├── README.md                              # this file
├── data/uk_macro_inputs.json              # snapshot of starting state
├── model/monte_carlo.py                   # the v2 model (~280 lines)
├── outputs/                               # generated artefacts (CSV/JSON/PNG)
│   ├── scenario_summary.csv               # 1-line summary per scenario
│   ├── results_summary.json               # full per-scenario quantiles
│   ├── paths_summary.json                 # P5/P50/P95/mean by variable
│   ├── compare_cpi.png                    # cross-scenario CPI fan
│   ├── compare_gdp.png                    # cross-scenario GDP fan
│   ├── compare_bankrate.png               # cross-scenario Bank Rate
│   ├── recession_prob.png                 # recession probability by quarter
│   └── {cpi,gdp,unemp,bankrate}_{scenario}.png  # per-scenario fans
└── report/UK_Economic_Forecast_May2026.md # narrative report
```

## How to reproduce

```bash
cd forecasts/uk_2026
python3 -m pip install --quiet numpy scipy pandas matplotlib statsmodels
python3 model/monte_carlo.py
```

Seed `20260512` is hard-coded; 25,000 paths × 4 scenarios runs in ~3 seconds.

## Method (one paragraph)

Five-variable semi-structural macro model: hybrid Phillips curve with an
expectations anchor; output IS curve driven by lagged growth, real-rate gap,
inflation gap, slack and a PMI-confidence term; quarterly Okun for the labour
market; smoothed Taylor rule with `φ_π = 1.5`, `φ_g = 0.5`; AR(1) oil-shock
proxy with Student-t (df = 4) innovations. Shock cross-correlations are
calibrated to capture supply-vs-demand co-movement. Recession is defined as
two consecutive quarters of negative QoQ GDP. Calibration parameters are
informed by BoE, OBR, NIESR and IMF UK Article IV transmission estimates.

## Status

v2 — after an adversarial Critic-subagent review that flagged three priority
calibration fixes (Okun units, flat post-2010 Phillips curve, stronger
monetary transmission). v3 roadmap in §7.3 of the report.
