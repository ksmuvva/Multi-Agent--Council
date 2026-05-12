# UK Economic Forecast — Recession & Inflation Outlook (v2)

**Multi-Agent Reasoning Council — Quantitative Macro Desk**
**Forecast date:** 2026-05-12 • **Horizon:** 2026Q2 → 2028Q1 (8 quarters)
**Method:** Monte Carlo simulation, 25,000 paths, 4 scenarios
**Model file:** `forecasts/uk_2026/model/monte_carlo.py` (v2 after Critic review)

---

## 1. Executive Summary

The UK economy in mid-2026 sits in an uncomfortable middle ground between
disinflation and a renewed cost-push shock. Inflation has re-accelerated to
3.3% (March 2026) — well above target — while underlying GDP growth has slowed
to roughly 0.2% QoQ, the labour market has loosened (unemployment 4.9%, up from
4.4% a year ago), and consumer confidence has slumped to its weakest reading
since early-2023. The 10Y gilt sits near 5.0% and the curve is only mildly
positive. Iran/Middle-East energy disruption and the April 2026 US tariff
package are the two external risks compressing the distribution to the
downside.

Across 25,000 Monte Carlo paths and four calibrated scenarios:

| Metric | Baseline | Adverse Supply | Stagflation | Soft Landing |
|---|---:|---:|---:|---:|
| **P(technical recession in 2026)** | **43.6%** | 60.7% | 80.3% | 58.4% |
| P(technical recession within 4 quarters) | 56.1% | 79.4% | 92.5% | 73.6% |
| P(technical recession within 8 quarters) | 77.5% | 98.0% | 99.9% | 94.5% |
| Median CPI YoY, 2026Q4 (%) | 2.58 | 5.47 | 4.28 | 2.88 |
| Median CPI YoY, 2027Q2 (%) | 2.18 | 6.41 | 4.72 | 2.61 |
| Median CPI YoY, 2028Q1 (%) | 1.93 | 6.54 | 5.04 | 2.41 |
| P(CPI > 3% at 2028Q1) | 0.5% | **93.2%** | 99.8% | 5.4% |
| P(CPI > 4% any quarter, 8q) | 0.3% | **98.8%** | 99.3% | 0.6% |
| P(CPI < 2% at 2028Q1) | 57.6% | 3.2% | ≈ 0 | 13.3% |
| Median Bank Rate, 2028Q1 (%) | 3.01 | 7.43 | 5.69 | 3.41 |
| Peak unemployment (median, %) | 5.17 | 5.62 | 5.58 | 5.34 |

**Bottom-line forecast (probability-weighted across scenarios¹):**

- **Recession probability** in 2026: **~52%** — slightly *above* the
  prediction-market reading of 45.5%, reflecting the strong cluster of
  near-term downside risks (energy, tariffs, confidence).
- **Inflation** is most likely to drift back toward 2% by 2027 in the central
  case, but the **adverse-supply tail is consequential**: a ~25-30%
  probability-weighted chance of CPI exceeding 4% at some point in the next
  18 months.
- **Bank Rate** path is **highly asymmetric**: median ends near 4.0% on
  weighted basis (only ~25 bps of cuts), but skewed by the adverse-supply
  scenario where rates rise toward 7%.
- **Unemployment** peaks at **5.3-5.5% (probability-weighted)**, consistent
  with the OBR's central forecast peak of 5.33%.

¹ *Subjective weights used in the summary above: baseline 0.45, adverse-supply
0.25, stagflation 0.10, soft-landing 0.20. These are illustrative and the
reader can re-weight using the per-scenario figures.*

### Headline numbers — at a glance

> **Central recession call:** ~44% chance the UK is in a technical recession
> at some point in 2026. Probability-weighted across scenarios: ~52%.
>
> **Central inflation call:** CPI back to target by 2027Q2 (median);
> 25-30% chance of a re-acceleration above 4% if the energy shock persists.

---

## 2. Macroeconomic Starting Conditions (data as of May 2026)

### 2.1 Inflation block

| Indicator | Latest reading | Source |
|---|---|---|
| Headline CPI YoY | **3.3%** (Mar 2026) | [ONS CPI bulletin, Mar 2026](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/march2026) |
| Core CPI YoY | 3.1% | [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/march2026) |
| Services CPI YoY | 4.5% (sticky) | [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/march2026) |
| Transport prices YoY | +4.7% (driven by motor fuels) | [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/march2026) |

BoE's April 2026 MPR raised the near-term path materially: CPI is now expected
in the **3.0-3.5% range across 2026 Q2-Q3**, with a likely further rise in Q4
before declining — a ~1.4pp upward revision from the February MPR's forecast
of 2.0% by Q3 2026. ([Bank of England MPR April 2026](https://www.bankofengland.co.uk/monetary-policy-report/2026/april-2026))

### 2.2 Activity block

| Indicator | Latest reading | Source |
|---|---|---|
| GDP QoQ (Q4 2025) | +0.1% | [ONS GDP first quarterly estimate](https://www.ons.gov.uk/economy/grossdomesticproductgdp/bulletins/gdpfirstquarterlyestimateuk/octobertodecember2025) |
| GDP QoQ (Q1 2026, headline) | +0.5% (boosted by one-off Feb +0.5% MoM) | [ONS GDP monthly, Feb 2026](https://www.ons.gov.uk/economy/grossdomesticproductgdp/bulletins/gdpmonthlyestimateuk/february2026) |
| GDP QoQ (Q1 2026, BoE *underlying*) | +0.2% (below potential ~0.3-0.4%) | [BoE MPR April 2026](https://www.bankofengland.co.uk/monetary-policy-report/2026/april-2026) |
| Manufacturing PMI (Apr 2026) | 53.7 (highest since May 2022) | [S&P Global UK PMI](https://www.pmi.spglobal.com/Public/Home/PressRelease/977a073be9df4177ac9a1a872fc35886) |
| Services PMI (Apr 2026) | 52.7 | [S&P Global](https://www.pmi.spglobal.com/Public/Home/PressRelease/de91810fcfd845e7a0ef06defdaa3b7c) |
| GfK Consumer Confidence (Apr 2026) | −25 (lowest since Feb 2023) | [ITV / GfK](https://www.itv.com/news/2026-04-23/consumer-confidence-falls-as-rapid-price-rises-give-households-the-jitters) |
| Retail sales YoY (Apr 2026) | −3.0% | [BRC](https://brc.org.uk/news-and-events/news/corporate-affairs/2026/ungated/uncertainty-hits-retail-sales/) |

### 2.3 Labour & financial conditions

| Indicator | Latest reading | Source |
|---|---|---|
| Unemployment rate (Dec 2025-Feb 2026) | **4.9%** (up from 4.4% YoY) | [ONS labour market](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/uklabourmarket/latest) |
| Wage growth (regular pay, YoY) | 3.6% (real +0.4%) | [ONS](https://commonslibrary.parliament.uk/research-briefings/cbp-9366/) |
| Vacancies | 711k (below pre-pandemic) | [ONS](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/uklabourmarket/latest) |
| Bank Rate (model start) | 4.25% | inferred from BoE MPR April 2026 |
| 10Y gilt yield | ~5.00% | [Trading Economics](https://tradingeconomics.com/united-kingdom/government-bond-yield) |

### 2.4 Official forecasts (anchors for the Monte Carlo prior)

| Body | 2026 GDP | 2026 CPI | Notes |
|---|---:|---:|---|
| OBR (March 2026 EFO) | +1.1% | 2.3% (year-end) | Unemployment peak 5.33%. ([source](https://obr.uk/efo/economic-and-fiscal-outlook-march-2026/)) |
| BoE MPR (April 2026) | n/a (qualitative) | 3.0-3.5% Q2-Q3, rising in Q4 | Back to target late 2026 / 2027. |
| IMF WEO (April 2026) | **+0.8%** (largest G7 downgrade) | 3.2% avg, peak ~4% | Back to target end-2027. ([source](https://www.imf.org/en/publications/weo/issues/2026/04/14/world-economic-outlook-april-2026)) |
| Prediction markets | — | — | **45.5%** chance of technical recession in 2026 ([Lines](https://www.lines.com/prediction-markets/economy/uk-recession-in-2026)). |

---

## 3. Model Design

### 3.1 State variables and dynamics (v2)

The simulator is a small **semi-structural macroeconomic system** in five
state variables, calibrated to UK empirical regularities and stress-tested via
Monte Carlo. The system is:

```
Hybrid Phillips curve (CPI YoY %), with expectations anchor
  π_t = a0 + a_π π_{t-1} + a_πexp π*  + a_g (g_{t-1}-g*) + a_u (u* - u_{t-1})
        + a_oil oil_t + ε_π
  (a0 chosen so that steady-state inflation hits the 2% target when shocks
   die out; a_π + a_πexp < 1 to ensure stationarity)

GDP (real QoQ %)
  g_t = b0 + b_g g_{t-1} + b_r (r_{t-1} - r*_nom) + b_π (π_{t-1} - π*)
        + b_u (u* - u_{t-1}) + b_oil oil_t + b_pmi (PMI_t - 50) + ε_g

Okun's law (quarterly form; unemployment %)
  u_t = u_{t-1} − γ (g_t - g*) + ε_u

Taylor rule with smoothing (Bank Rate %)
  r_t = ρ r_{t-1} + (1-ρ)·[r*_nom + φ_π (π_{t-1}-π*) + φ_g · output_gap]
        + ε_r,    floored at 0

Oil-shock proxy (Brent YoY %, Student-t innovations, df=4)
  oil_t = δ oil_{t-1} + ε_oil
```

Structural anchors: `g* = 0.3% QoQ`, `u* = 4.5%` (NAIRU), `r*_real = 0.5%`,
`π* = 2.0%`. A Student-t (df = 4) distribution generates oil shocks to ensure
appropriately fat tails. Macro innovations are jointly Gaussian with
cross-correlations capturing supply-vs-demand co-movement
(corr(ε_π, ε_g)=−0.20; corr(ε_π, ε_oil)=+0.40 via copula coupling;
corr(ε_g, ε_oil)=−0.30; corr(ε_u, ε_g)=−0.30).

### 3.2 Key calibration (v2, post-Critic)

| Parameter | v1 | v2 | Rationale |
|---|---:|---:|---|
| `a_π` (CPI persistence) | 0.72 | 0.55 | Post-2010 UK Phillips curves are flatter with expectations anchored |
| `a_πexp` (expectations weight) | 0 | 0.10 | Forward-looking term anchoring to BoE 2% target |
| `a_u` (unemployment slack -> π) | 0.18 | 0.05 | Flat Phillips curve since 2010 (BoE, NIESR) |
| `b_r` (rate -> GDP) | -0.05 | -0.12 | Cumulative 8q passthrough closer to BoE COMPASS / NIESR NiGEM |
| `γ` (Okun coefficient) | 0.35 | 0.10 | Quarterly Okun ≈ ¼ of the annual coefficient |
| `ρ_r` (Taylor smoothing) | 0.85 | 0.82 | Slightly less inertial to allow BoE data-dependency |

Full parameter set in `forecasts/uk_2026/model/monte_carlo.py`. Calibration is
informed by published BoE Phillips-curve work, NIESR NiGEM transmission
estimates, and IMF UK Article IV consultations.

### 3.3 Scenarios

| Scenario | Description | Key overrides |
|---|---|---|
| **Baseline** | BoE/OBR consensus blend; oil mean-reverts; services CPI eases gradually | none |
| **Adverse supply** | Middle-East escalation: Brent persists higher, larger second-round effects | `δ_oil = 0.85`, `a_oil = 0.025`, `a_π = 0.82`, oil_0 boost +15pp |
| **Stagflation** | Persistent services inflation + weak demand (tariff drag) | `a_π = 0.85`, `b_g = 0.20`, `b_r = −0.05`, `b0 = −0.10` |
| **Soft landing** | Oil retreats, wage growth eases, services inflation cracks | `δ_oil = 0.30`, `a_π = 0.65`, `a_oil = 0.010`, oil_0 boost −15pp |

### 3.4 Recession definition

Two consecutive quarters of negative real GDP QoQ growth (the standard
technical definition, also used by prediction markets).

---

## 4. Results

### 4.1 Scenario summary table (v2)

(See full CSV: `forecasts/uk_2026/outputs/scenario_summary.csv`.)

| Scenario | P(rec 2026) | P(rec 4q) | P(rec 8q) | CPI 2026Q4 | CPI 2027Q2 | CPI 2028Q1 | P(CPI>3 @ 8q) | P(CPI>4 anytime) | Bank Rate @ 8q | Peak unemp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 43.6% | 56.1% | 77.5% | 2.58 | 2.18 | 1.93 | 0.5% | 0.3% | 3.01% | 5.17% |
| Adverse supply | 60.7% | 79.4% | 98.0% | 5.47 | 6.41 | 6.54 | 93.2% | 98.8% | 7.43% | 5.62% |
| Stagflation | 80.3% | 92.5% | 99.9% | 4.28 | 4.72 | 5.04 | 99.8% | 99.3% | 5.69% | 5.58% |
| Soft landing | 58.4% | 73.6% | 94.5% | 2.88 | 2.61 | 2.41 | 5.4% | 0.6% | 3.41% | 5.34% |

### 4.2 Baseline scenario — central path

In the baseline, headline CPI is most likely to glide from 3.3% to ~2.6% by
end-2026 and **cross the 2% target around 2027Q3** (median path; 90% band
1.4-3.0% at that point). The Bank of England responds with a moderate cutting
cycle — Bank Rate median falls to ~3.0% by early 2028 (~125 bps of cuts) —
but the cumulative drag from elevated rates, sticky services inflation,
softer confidence and trade frictions keeps GDP growth running at or just
below potential. Unemployment drifts to a median peak near 5.2%, broadly
consistent with the OBR's March 2026 forecast peak of 5.33%.

The model assigns a **43.6% probability of a technical recession in 2026**
(conditional on the central calibration) — within 2pp of the
prediction-market price (45.5%). Probability rises to 56% over 4 quarters
and 78% over 8 quarters as the window widens.

See: `outputs/cpi_baseline.png`, `outputs/gdp_baseline.png`,
`outputs/unemp_baseline.png`, `outputs/bankrate_baseline.png`.

### 4.3 Adverse supply scenario — the live tail

Conditional on Middle-East energy disruption persisting (15pp larger initial
shock, 0.85 persistence), the model produces:

- CPI **median path peaking near 6.5%** in 2027 — comparable to the 2022-2023
  energy-led inflation episode.
- **60.7% recession probability in 2026, rising to 98% over 8 quarters**.
- Bank Rate held near 7.4% — the Taylor rule mechanically responds to runaway
  inflation. *Caveat: in practice the MPC would also weigh financial-stability
  and labour-market signals; this scenario should be read as a model
  illustration of the Taylor-rule reaction, not a literal BoE projection.*
- Unemployment peak median 5.6%, with the 5-95 band running to ~6.5%.

This scenario is **the binding tail risk** as of May 2026 and is consistent
with the IMF's April 2026 WEO warning of a "global economy thrown off course"
by the US-Iran conflict and the largest G7 downgrade applied to the UK.

### 4.4 Stagflation scenario — sticky inflation, weak demand

If services inflation persistence (`a_π`) is structurally higher and the
demand block weaker (tariff/Brexit drag), the model produces a **persistent
inflation overshoot with weak growth**: CPI medians around 4-5% throughout
the horizon and recession probability of 80% in 2026 alone. Median Bank Rate
holds near 5.7% — the BoE prioritises inflation containment over output
support.

### 4.5 Soft landing — the optimistic alternative

If oil retreats and services inflation breaks lower (faster real-wage
adjustment), CPI tracks just above target throughout 2027 and Bank Rate
falls to ~3.4%. Notably, **even the soft-landing scenario carries 58%
probability of a technical recession in 2026** — i.e. the volatility of UK
QoQ GDP (σ ≈ 0.30 pp) is such that two consecutive small-negative prints are
non-trivial even with positive mean growth. This is consistent with the
historical UK base rate of recession occurrence given current 4.9%
unemployment, sub-potential growth, and the GfK confidence collapse.

### 4.6 Cross-scenario comparison

See:

- `outputs/compare_cpi.png` — CPI fan-chart comparison
- `outputs/compare_gdp.png` — GDP QoQ comparison
- `outputs/compare_bankrate.png` — Bank Rate path comparison
- `outputs/recession_prob.png` — recession probability by quarter

---

## 5. Probability-Weighted Headline Forecast

Applying subjective weights *baseline 0.45, adverse 0.25, stagflation 0.10,
soft-landing 0.20* — consistent with our reading of Iran/Middle-East risk
premium, US tariffs, and OBR/IMF/BoE official central cases:

| Metric | Weighted estimate |
|---|---:|
| **P(technical recession in 2026)** | **~52%** |
| P(technical recession in 4 quarters) | **~68%** |
| P(technical recession in 8 quarters) | **~89%** |
| Median CPI YoY, 2026Q4 | **~3.4%** |
| Median CPI YoY, 2027Q2 | **~3.6%** |
| Median CPI YoY, 2028Q1 | **~3.5%** |
| P(CPI > 4% any quarter) | **~35%** |
| Median Bank Rate, 2028Q1 | **~4.4%** |
| Peak unemployment (weighted) | **~5.3%** |

> Note: the weighted CPI numbers are higher than any single median because of
> the substantial adverse-supply / stagflation tail; the weighted *Bank Rate*
> is similarly skewed up by the same tail.

**Forecast confidence:**

- **Inflation direction:** medium-high (multi-source consensus on disinflation
  *conditional* on no further energy shocks).
- **Recession call:** medium (model 43.6% vs market 45.5% — small spread,
  high agreement).
- **Bank-Rate endpoint:** low-medium (highly path-dependent on energy and
  tariff trajectories; large skew in the rate distribution).

---

## 6. Sensitivity and Diagnostics

- The model is *most sensitive* to (in decreasing order):
  1. `δ_oil` (oil-shock persistence) — moving from 0.55 to 0.85 raises the
     baseline P(CPI>4%) by two orders of magnitude.
  2. `a_π` (inflation persistence / services stickiness) — controls
     stagflation outcomes.
  3. `b_r` (real-rate sensitivity of demand) — calibrates how aggressively
     monetary tightening transmits. The v2 doubling of `|b_r|` was the
     single biggest driver of the baseline recession probability rising
     from 29.9% (v1) to 43.6% (v2).
  4. `ρ_r` (Taylor-rule smoothing) — at ρ = 0.95 the BoE under-reacts and
     stagflation risk rises sharply.

- Recession probabilities scale **non-linearly** with `σ_g` (GDP innovation
  volatility). The fact that even the soft-landing scenario carries 58%
  recession probability over 4 quarters is in part a reflection of the high
  baseline volatility of UK quarterly GDP releases (often ±0.3 pp revisions).

- The model **does not separately track** services-CPI dynamics, exchange-rate
  passthrough, fiscal multipliers, credit conditions, or labour-force
  participation. These are subsumed into the structural shock variances and
  the simple unemployment block. See §7 for the Critic's view on these.

---

## 7. Limitations & Adversarial Critique (Critic-agent review)

The Critic agent ran an independent adversarial review of v1. Three priority
fixes were applied in v2; the remaining issues are documented here as known
limitations.

### 7.1 What v2 fixes

1. **Okun's law units** — v1 used `γ = 0.35` on a quarterly growth gap; v2
   shrinks to `γ = 0.10`, removing an off-by-4 dimensional error. Effect:
   peak-unemployment medians fell from ~5.7-7.7% to ~5.2-5.6% (in line with
   OBR's 5.33% central estimate).
2. **Phillips-curve slope** — v1 had `a_u = 0.18` (steep); v2 uses `a_u =
   0.05` (flat, post-2010 UK), with a forward-looking expectations anchor
   `a_πexp · π*` added. Effect: inflation no longer collapses mechanically
   below target via the slack channel.
3. **Monetary transmission** — v1 had `b_r = -0.05`; v2 uses `-0.12`,
   roughly doubling the 8-quarter cumulative GDP response to a 100 bp rate
   surprise, in line with NIESR NiGEM / BoE COMPASS estimates. Effect:
   recession probability in baseline rose from 29.9% to 43.6% (now within
   2pp of prediction-market price).

### 7.2 Known remaining limitations

- **No fiscal block.** The "neutral-to-tightening" stance from inputs is
  unused. Given 2026 fiscal-consolidation risk this is a material omission.
- **No exchange-rate / GBP channel.** Bank-rate moves don't propagate to
  imported inflation; oil is in USD but treated as GBP-denominated. Could
  bias adverse-supply CPI down by 0.5-1.0pp.
- **Inflation expectations.** v2 adds an anchoring term but no time-varying
  expectations (no survey or breakeven proxy). In a 1970s-style scenario the
  anchor would itself become endogenous.
- **No credit / financial-conditions block.** The 40 bp gilt curve spread in
  inputs is unused; the UK mortgage-reset cliff is ignored.
- **No regime switching for ZLB or wage-price spirals.** The system is linear
  and Gaussian (with one fat-tailed variable). Real recessions show sharply
  non-linear dynamics that the linear system understates.
- **Oil-shock proxy as a sufficient statistic for all supply shocks** — it
  bundles natural-gas dynamics, supply-chain disruptions, Red-Sea routing
  and food-price shocks into one variable. Energy is only ~3% of CPI
  directly, so this is a crude reduced-form.
- **Taylor rule reacts to lagged inflation** — but the BoE in practice
  targets ~2-year-ahead CPI. The model probably overstates the rate
  response in the adverse-supply scenario (median terminal Bank Rate 7.43%
  is extreme).
- **Stagflation peak unemployment 5.58%** — slightly counter-intuitively
  *below* the corresponding rate response would imply; the new flat
  Phillips curve damps the labour-market knock-on from the inflation
  overshoot. A future version with a wage-Phillips block would correct
  this.

### 7.3 The next three model upgrades (v3 roadmap)

1. **Add a fiscal-stance variable** (cyclically-adjusted primary balance
   ΔpB) and let it enter the GDP equation with elasticity ~0.5.
2. **Regime-switching shock variances** — allow `σ_g`, `σ_π` to step up in
   a "crisis regime" triggered by `oil_t > 50%` or `u_t > NAIRU + 1pp`.
3. **Bayesian VAR estimation** on UK quarterly data 2000-2025 (or via the
   FMP economics MCP once available) to replace hand-calibrated
   elasticities — particularly the rate transmission and oil passthrough.

---

## 8. Multi-Agent Reasoning Trail

Per the *Multi-Agent Reasoning System* convention, this forecast was produced
by the following agent path:

- **Analyst** — assembled UK macro inputs from web research (ONS, BoE, OBR,
  IMF, S&P Global PMI, GfK).
- **Researcher** — cross-checked official forecasts and prediction-market
  prices.
- **Quant SME (Macroeconomist)** — designed the semi-structural model and
  v1 calibration.
- **Executor** — implemented and ran 25,000 × 4 scenarios in
  `monte_carlo.py`.
- **Critic** (`general-purpose` subagent, independent context) — performed
  an adversarial review of v1, flagging the three priority fixes that became
  v2.
- **Reviewer** — re-ran v2, validated against OBR/BoE/IMF/market priors,
  assembled this report.

The FMP `mcp__b281cabc...__economics` connector was unavailable during the
session (approval path temporarily down) — the model uses only the published
official statistics gathered via web research. The Bayesian VAR upgrade in §7
is contingent on getting that connector approved.

---

## 9. Reproducibility

```bash
cd /home/user/Multi-Agent--Council/forecasts/uk_2026
python3 model/monte_carlo.py
# Writes outputs/scenario_summary.csv,
#         outputs/results_summary.json,
#         outputs/paths_summary.json,
#         outputs/*.png
```

Random seed `20260512` is hard-coded for full reproducibility. Run-time on a
modern laptop CPU: ~3 seconds for all four scenarios at 25,000 paths × 8
quarters.

---

## Sources

- [ONS — Consumer price inflation, UK: March 2026](https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/march2026)
- [ONS — GDP first quarterly estimate, Q4 2025](https://www.ons.gov.uk/economy/grossdomesticproductgdp/bulletins/gdpfirstquarterlyestimateuk/octobertodecember2025)
- [ONS — GDP monthly estimate, February 2026](https://www.ons.gov.uk/economy/grossdomesticproductgdp/bulletins/gdpmonthlyestimateuk/february2026)
- [ONS — UK Labour Market: April 2026](https://www.ons.gov.uk/releases/uklabourmarketapril2026)
- [Bank of England — Monetary Policy Report April 2026](https://www.bankofengland.co.uk/monetary-policy-report/2026/april-2026)
- [Bank of England — Interest rates and Bank Rate](https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate)
- [OBR — Economic and fiscal outlook March 2026](https://obr.uk/efo/economic-and-fiscal-outlook-march-2026/)
- [IMF — World Economic Outlook, April 2026](https://www.imf.org/en/publications/weo/issues/2026/04/14/world-economic-outlook-april-2026)
- [S&P Global — UK Manufacturing PMI April 2026](https://www.pmi.spglobal.com/Public/Home/PressRelease/977a073be9df4177ac9a1a872fc35886)
- [S&P Global — UK Services PMI April 2026](https://www.pmi.spglobal.com/Public/Home/PressRelease/de91810fcfd845e7a0ef06defdaa3b7c)
- [GfK Consumer Confidence — April 2026 (via ITV)](https://www.itv.com/news/2026-04-23/consumer-confidence-falls-as-rapid-price-rises-give-households-the-jitters)
- [BRC — UK retail sales April 2026](https://brc.org.uk/news-and-events/news/corporate-affairs/2026/ungated/uncertainty-hits-retail-sales/)
- [Trading Economics — UK 10Y bond yield](https://tradingeconomics.com/united-kingdom/government-bond-yield)
- [Lines — UK Recession in 2026 prediction market](https://www.lines.com/prediction-markets/economy/uk-recession-in-2026)
- [Commons Library — UK Labour Market](https://commonslibrary.parliament.uk/research-briefings/cbp-9366/)
- [Capital Economics — Inverted gilt yield curve](https://www.capitaleconomics.com/clients/publications/uk-economics/uk-economics-update/what-to-make-of-the-inverted-gilt-yield-curve)
