"""
UK Recession & Inflation Monte Carlo Forecasting Model
======================================================

A small-scale, transparent macroeconomic Monte Carlo simulator built around a
calibrated, semi-structural system of equations:

    Phillips curve (CPI YoY %):
        pi_t  = a0 + a_pi*pi_{t-1} + a_g*(g_{t-1}-g*) + a_u*(u* - u_{t-1})
                + a_oil*oil_t + eps_pi

    GDP (QoQ %):
        g_t   = b0 + b_g*g_{t-1} + b_r*(r_{t-1} - r*_nom) + b_pi*(pi_{t-1}-pi*)
                + b_u*(u* - u_{t-1}) + b_oil*oil_t + b_pmi*(pmi_t - 50) + eps_g

    Okun's law (unemployment %):
        u_t   = u_{t-1} - gamma*(g_t - g*) + eps_u

    Taylor rule with smoothing (Bank Rate %):
        r_t   = rho*r_{t-1} + (1-rho)*[(r*_real + pi*) + phi_pi*(pi_{t-1}-pi*)
                + phi_g * output_gap_proxy] + eps_r,  floored at 0

    Oil-price shock proxy (Brent YoY %), Student-t innovations:
        oil_t = delta*oil_{t-1} + eps_oil

Structural shocks are jointly Gaussian (except oil ~ Student-t) with calibrated
cross-correlations capturing supply-vs-demand co-movement.

Recession definition: 2 consecutive quarters of QoQ GDP growth < 0 (technical).
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "uk_macro_inputs.json"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """v2 calibration after Critic review.

    Key changes vs v1:
      - a_pi reduced 0.72 -> 0.55 (post-2010 UK Phillips curve is flatter once
        an inflation-expectations anchor is included)
      - a0 raised 0.45 -> 0.90 (steady-state anchor at the 2% target)
      - a_u reduced 0.18 -> 0.05 (post-2010 slack-inflation channel is flat)
      - pi_exp_anchor added (forward-looking expectations weight)
      - gamma reduced 0.35 -> 0.10 (quarterly Okun; v1 was using annual-scale)
      - b_r increased in magnitude -0.05 -> -0.12 (cumulative 8q rate-pass-
        through to GDP closer to BoE COMPASS / IMF estimates)
      - sigma_u 0.08 -> 0.06 (unemployment is sluggish)
    """
    # Phillips curve (CPI YoY %); a0 chosen so that with a_pi=0.55,
    # a_pi_exp=0.10, pi*=2.0, oil=0 the steady-state inflation = 2% exactly.
    a0: float = 0.70
    a_pi: float = 0.55
    a_pi_exp: float = 0.10  # anchoring on 2% expectations
    a_g: float = 0.06
    a_u: float = 0.05
    a_oil: float = 0.012
    sigma_pi: float = 0.22

    # GDP equation (QoQ %)
    b0: float = 0.22
    b_g: float = 0.32
    b_r: float = -0.12  # stronger monetary transmission
    b_pi: float = -0.025
    b_u: float = 0.12
    b_oil: float = -0.0025
    b_pmi: float = 0.04
    sigma_g: float = 0.30

    # Okun's law (quarterly): de-trended quarterly growth gap -> unemp change
    gamma: float = 0.10
    sigma_u: float = 0.06

    # Taylor rule with smoothing
    rho_r: float = 0.82
    phi_pi: float = 1.50
    phi_g: float = 0.50
    sigma_r: float = 0.18

    # Oil shock proxy
    delta_oil: float = 0.55
    sigma_oil: float = 10.0
    oil_df: int = 4

    # Structural anchors
    g_star: float = 0.30
    u_star: float = 4.50
    r_star_real: float = 0.50
    pi_star: float = 2.00

    # Shock cross-correlations (cost-push co-movement)
    corr_pi_g: float = -0.20
    corr_pi_oil: float = 0.40
    corr_g_oil: float = -0.30
    corr_u_g: float = -0.30


@dataclass
class Scenario:
    name: str
    desc: str
    overrides: Dict[str, float] = field(default_factory=dict)
    oil_initial_boost: float = 0.0  # additive shock to oil_0


SCENARIOS: List[Scenario] = [
    Scenario(
        name="baseline",
        desc="Central path: BoE/OBR consensus blend; oil mean-reverts; sticky services",
    ),
    Scenario(
        name="adverse_supply",
        desc="Middle-East escalation: oil persists, larger second-round inflation",
        overrides={"delta_oil": 0.85, "a_oil": 0.025, "a_pi": 0.82, "sigma_oil": 16.0},
        oil_initial_boost=15.0,
    ),
    Scenario(
        name="stagflation",
        desc="Persistent services inflation + weak demand (tariff drag)",
        overrides={"a_pi": 0.85, "b_g": 0.20, "b_r": -0.05, "b0": -0.10},
    ),
    Scenario(
        name="soft_landing",
        desc="Oil retreats, wage growth eases, services inflation cracks",
        overrides={"delta_oil": 0.30, "a_pi": 0.65, "a_oil": 0.010, "b0": 0.12},
        oil_initial_boost=-15.0,
    ),
]


# ---------------------------------------------------------------------------
# Initial state from data file
# ---------------------------------------------------------------------------

def load_initial_state() -> Dict[str, float]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    cs = data["current_state"]
    rf = data["risk_factors"]
    return {
        "pi": cs["cpi_yoy_pct"],
        "g": cs["gdp_qoq_pct_underlying"],
        "u": cs["unemployment_rate_pct"],
        "r": cs["bank_rate_pct"],
        "oil": rf["oil_shock_iran_conflict"]["brent_yoy_pct_assumed"],
        "pmi": cs["pmi_composite"],
    }


# ---------------------------------------------------------------------------
# Shock generation
# ---------------------------------------------------------------------------

def build_shock_corr_matrix(c: Calibration) -> np.ndarray:
    """Return 4x4 correlation matrix for (eps_pi, eps_g, eps_u, eps_r).

    Oil shocks are drawn separately from a heavy-tailed Student-t and then
    correlated to the macro block via a copula-like coupling.
    """
    C = np.eye(4)
    C[0, 1] = C[1, 0] = c.corr_pi_g
    C[1, 2] = C[2, 1] = c.corr_u_g
    return C


def draw_shocks(c: Calibration, n_sim: int, horizon: int, rng: np.random.Generator):
    """Draw correlated macro shocks (Gaussian) and heavy-tailed oil shocks.

    Returns array shape (horizon, n_sim, 5) with columns
        [eps_pi, eps_g, eps_u, eps_r, eps_oil].
    """
    corr = build_shock_corr_matrix(c)
    L = np.linalg.cholesky(corr)
    sigmas = np.array([c.sigma_pi, c.sigma_g, c.sigma_u, c.sigma_r])

    z = rng.standard_normal((horizon, n_sim, 4))
    macro = z @ L.T * sigmas

    # Oil: scaled Student-t for fat tails
    t_draws = stats.t(df=c.oil_df).rvs(size=(horizon, n_sim), random_state=rng)
    # Normalise so variance ~ 1
    t_draws = t_draws * np.sqrt((c.oil_df - 2) / c.oil_df)
    oil = t_draws * c.sigma_oil

    # Couple oil to macro shocks (copula-style): adjust pi and g shocks
    # so that simulated correlation tracks the calibrated targets.
    pi_extra = c.corr_pi_oil * (oil / c.sigma_oil) * c.sigma_pi
    g_extra = c.corr_g_oil * (oil / c.sigma_oil) * c.sigma_g
    macro[:, :, 0] = np.sqrt(max(1 - c.corr_pi_oil ** 2, 0)) * macro[:, :, 0] + pi_extra
    macro[:, :, 1] = np.sqrt(max(1 - c.corr_g_oil ** 2, 0)) * macro[:, :, 1] + g_extra

    return np.concatenate([macro, oil[:, :, None]], axis=2)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate(scenario: Scenario, n_sim: int = 20000, horizon: int = 8,
             seed: int = 20260512) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = Calibration()
    for k, v in scenario.overrides.items():
        setattr(c, k, v)

    state0 = load_initial_state()
    pi = np.full(n_sim, state0["pi"])
    g = np.full(n_sim, state0["g"])
    u = np.full(n_sim, state0["u"])
    r = np.full(n_sim, state0["r"])
    oil = np.full(n_sim, state0["oil"] + scenario.oil_initial_boost)
    pmi = np.full(n_sim, state0["pmi"])

    pi_path = np.empty((horizon + 1, n_sim))
    g_path = np.empty((horizon + 1, n_sim))
    u_path = np.empty((horizon + 1, n_sim))
    r_path = np.empty((horizon + 1, n_sim))
    oil_path = np.empty((horizon + 1, n_sim))

    pi_path[0] = pi
    g_path[0] = g
    u_path[0] = u
    r_path[0] = r
    oil_path[0] = oil

    shocks = draw_shocks(c, n_sim, horizon, rng)

    r_star_nominal = c.r_star_real + c.pi_star

    for t in range(horizon):
        e_pi, e_g, e_u, e_r, e_oil = (shocks[t, :, k] for k in range(5))

        # Oil first (informs others)
        oil_new = c.delta_oil * oil + e_oil

        # Inflation: hybrid Phillips curve with expectations anchor at pi*
        pi_new = (c.a0
                  + c.a_pi * pi
                  + c.a_pi_exp * c.pi_star
                  + c.a_g * (g - c.g_star)
                  + c.a_u * (c.u_star - u)
                  + c.a_oil * oil_new
                  + e_pi)

        # GDP (uses lagged states & contemporaneous oil)
        output_gap_proxy = c.u_star - u  # positive when slack is low
        g_new = (c.b0
                 + c.b_g * g
                 + c.b_r * (r - r_star_nominal)
                 + c.b_pi * (pi - c.pi_star)
                 + c.b_u * output_gap_proxy
                 + c.b_oil * oil_new
                 + c.b_pmi * (pmi - 50.0)
                 + e_g)

        # PMI mean reverts toward 51 with light noise tracking GDP
        pmi = 0.6 * pmi + 0.4 * (51.0 + 8.0 * (g_new - c.g_star)) + 0.5 * rng.standard_normal(n_sim)

        # Unemployment (Okun)
        u_new = u - c.gamma * (g_new - c.g_star) + e_u
        u_new = np.clip(u_new, 2.0, 12.0)

        # Taylor with smoothing
        annualised_gap = 4.0 * (g - c.g_star)
        r_target = r_star_nominal + c.phi_pi * (pi - c.pi_star) + c.phi_g * annualised_gap * 0.25
        r_new = c.rho_r * r + (1 - c.rho_r) * r_target + e_r
        r_new = np.clip(r_new, 0.0, 12.0)

        pi, g, u, r, oil = pi_new, g_new, u_new, r_new, oil_new
        pi_path[t + 1] = pi
        g_path[t + 1] = g
        u_path[t + 1] = u
        r_path[t + 1] = r
        oil_path[t + 1] = oil

    return {
        "pi": pi_path,
        "g": g_path,
        "u": u_path,
        "r": r_path,
        "oil": oil_path,
        "calibration": asdict(c),
        "scenario": {"name": scenario.name, "desc": scenario.desc},
    }


# ---------------------------------------------------------------------------
# Diagnostics & summary statistics
# ---------------------------------------------------------------------------

def recession_probability(g_path: np.ndarray, window: int = None) -> Tuple[float, np.ndarray]:
    """Probability of a *technical* recession (two consecutive negative QoQ
    GDP prints) anywhere in the simulated horizon (or first `window` quarters)."""
    g = g_path[1:]  # drop initial state
    if window is not None:
        g = g[:window]
    neg = g < 0
    pair = neg[:-1] & neg[1:]
    any_recession = pair.any(axis=0)
    by_quarter = pair.mean(axis=1)
    return float(any_recession.mean()), by_quarter


def summarise(paths: Dict[str, np.ndarray]) -> Dict:
    pi, g, u, r, oil = paths["pi"], paths["g"], paths["u"], paths["r"], paths["oil"]
    rec_prob_8q, rec_by_q = recession_probability(g)
    rec_prob_4q, _ = recession_probability(g, window=4)
    rec_prob_3q, _ = recession_probability(g, window=3)
    rec_prob_in_2026, _ = recession_probability(g, window=3)  # 2026 Q2-Q4

    def pct_band(x):
        return {
            "p05": np.percentile(x, 5, axis=1).tolist(),
            "p25": np.percentile(x, 25, axis=1).tolist(),
            "p50": np.percentile(x, 50, axis=1).tolist(),
            "p75": np.percentile(x, 75, axis=1).tolist(),
            "p95": np.percentile(x, 95, axis=1).tolist(),
            "mean": x.mean(axis=1).tolist(),
        }

    # Hit-rate flags for inflation
    pi_end = pi[-1]
    pi_max = pi.max(axis=0)
    bank_rate_end = r[-1]

    return {
        "scenario": paths["scenario"],
        "recession_prob_8q": rec_prob_8q,
        "recession_prob_4q": rec_prob_4q,
        "recession_prob_3q": rec_prob_3q,
        "recession_prob_in_2026": rec_prob_in_2026,
        "recession_prob_by_q": rec_by_q.tolist(),
        "inflation": pct_band(pi),
        "gdp_qoq": pct_band(g),
        "unemployment": pct_band(u),
        "bank_rate": pct_band(r),
        "oil_yoy": pct_band(oil),
        "prob_pi_gt_2_at_horizon": float((pi_end > 2.0).mean()),
        "prob_pi_gt_3_at_horizon": float((pi_end > 3.0).mean()),
        "prob_pi_gt_4_at_any_point": float((pi_max > 4.0).mean()),
        "prob_pi_lt_2_at_horizon": float((pi_end < 2.0).mean()),
        "prob_r_gt_5_at_horizon": float((bank_rate_end > 5.0).mean()),
        "prob_r_lt_3_at_horizon": float((bank_rate_end < 3.0).mean()),
        "mean_pi_path": pi.mean(axis=1).tolist(),
        "mean_g_path": g.mean(axis=1).tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fan(paths: Dict[str, np.ndarray], var: str, ylabel: str, title: str,
             outfile: Path, hline: float = None):
    x = np.arange(paths[var].shape[0])
    p = paths[var]
    p05, p25, p50, p75, p95 = (np.percentile(p, q, axis=1) for q in (5, 25, 50, 75, 95))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, p05, p95, color="#1f77b4", alpha=0.15, label="90% band")
    ax.fill_between(x, p25, p75, color="#1f77b4", alpha=0.35, label="50% band")
    ax.plot(x, p50, color="#1f77b4", lw=2, label="Median")
    ax.plot(x, p.mean(axis=1), color="#d62728", lw=1.2, ls="--", label="Mean")
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.8, ls=":", label=f"Reference {hline}")
    ax.set_xlabel("Quarters from 2026Q2")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def plot_recession_prob(scenario_results: Dict[str, Dict], outfile: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, summary in scenario_results.items():
        ax.plot(range(1, len(summary["recession_prob_by_q"]) + 1),
                100 * np.array(summary["recession_prob_by_q"]),
                marker="o", label=f"{name} (8q={100*summary['recession_prob_8q']:.1f}%)")
    ax.set_xlabel("Quarter-pair end (quarters ahead)")
    ax.set_ylabel("Prob. of technical recession ending in quarter (%)")
    ax.set_title("UK technical-recession probability by quarter, by scenario")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def plot_scenario_compare(all_paths: Dict[str, Dict], var: str, ylabel: str,
                          title: str, outfile: Path, hline: float = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"baseline": "#1f77b4", "adverse_supply": "#d62728",
              "stagflation": "#9467bd", "soft_landing": "#2ca02c"}
    for name, paths in all_paths.items():
        p = paths[var]
        x = np.arange(p.shape[0])
        med = np.percentile(p, 50, axis=1)
        p05 = np.percentile(p, 5, axis=1)
        p95 = np.percentile(p, 95, axis=1)
        c = colors.get(name, None)
        ax.plot(x, med, lw=2, label=f"{name}", color=c)
        ax.fill_between(x, p05, p95, alpha=0.10, color=c)
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.8, ls=":", label=f"Reference {hline}")
    ax.set_xlabel("Quarters from 2026Q2")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    n_sim = 25000
    horizon = 8
    seed = 20260512

    all_paths = {}
    all_summaries = {}

    for sc in SCENARIOS:
        print(f"Simulating scenario: {sc.name} ...")
        paths = simulate(sc, n_sim=n_sim, horizon=horizon, seed=seed)
        summary = summarise(paths)
        all_paths[sc.name] = paths
        all_summaries[sc.name] = summary

        plot_fan(paths, "pi", "CPI YoY (%)",
                 f"UK CPI inflation - {sc.name}", OUT_DIR / f"cpi_{sc.name}.png",
                 hline=2.0)
        plot_fan(paths, "g", "Real GDP QoQ (%)",
                 f"UK real GDP QoQ - {sc.name}", OUT_DIR / f"gdp_{sc.name}.png",
                 hline=0.0)
        plot_fan(paths, "u", "Unemployment rate (%)",
                 f"UK unemployment - {sc.name}", OUT_DIR / f"unemp_{sc.name}.png",
                 hline=4.5)
        plot_fan(paths, "r", "Bank Rate (%)",
                 f"UK Bank Rate - {sc.name}", OUT_DIR / f"bankrate_{sc.name}.png",
                 hline=2.5)

    plot_recession_prob(all_summaries, OUT_DIR / "recession_prob.png")
    plot_scenario_compare(all_paths, "pi", "CPI YoY (%)",
                          "Scenario comparison - CPI inflation (median & 90% band)",
                          OUT_DIR / "compare_cpi.png", hline=2.0)
    plot_scenario_compare(all_paths, "g", "Real GDP QoQ (%)",
                          "Scenario comparison - Real GDP QoQ (median & 90% band)",
                          OUT_DIR / "compare_gdp.png", hline=0.0)
    plot_scenario_compare(all_paths, "r", "Bank Rate (%)",
                          "Scenario comparison - Bank Rate (median & 90% band)",
                          OUT_DIR / "compare_bankrate.png", hline=2.5)

    # Cross-scenario recession & inflation summary table
    summary_table = []
    for name, s in all_summaries.items():
        summary_table.append({
            "scenario": name,
            "P(recession in 2026) %": round(100 * s["recession_prob_in_2026"], 1),
            "P(recession 4q) %": round(100 * s["recession_prob_4q"], 1),
            "P(recession 8q) %": round(100 * s["recession_prob_8q"], 1),
            "Median CPI 2026Q4 %": round(s["inflation"]["p50"][2], 2),
            "Median CPI 2027Q2 %": round(s["inflation"]["p50"][4], 2),
            "Median CPI 2028Q2 %": round(s["inflation"]["p50"][8], 2),
            "P(CPI>3 at 8q) %": round(100 * s["prob_pi_gt_3_at_horizon"], 1),
            "P(CPI>4 anytime) %": round(100 * s["prob_pi_gt_4_at_any_point"], 1),
            "P(CPI<2 at 8q) %": round(100 * s["prob_pi_lt_2_at_horizon"], 1),
            "Median GDP path mean QoQ %": round(np.mean(s["mean_g_path"]), 2),
            "Median Bank Rate 8q %": round(s["bank_rate"]["p50"][8], 2),
            "Peak unemployment median %": round(max(s["unemployment"]["p50"]), 2),
        })
    df = pd.DataFrame(summary_table)
    df.to_csv(OUT_DIR / "scenario_summary.csv", index=False)
    print("\n=== Scenario summary ===")
    print(df.to_string(index=False))

    # Persist detailed JSON
    with open(OUT_DIR / "results_summary.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "calibration"}
                   for k, v in all_summaries.items()}, f, indent=2, default=float)

    # Persist a thin paths dump (P5/P50/P95 only, to keep file small)
    thin = {}
    for name, paths in all_paths.items():
        thin[name] = {}
        for var in ("pi", "g", "u", "r", "oil"):
            arr = paths[var]
            thin[name][var] = {
                "p05": np.percentile(arr, 5, axis=1).tolist(),
                "p50": np.percentile(arr, 50, axis=1).tolist(),
                "p95": np.percentile(arr, 95, axis=1).tolist(),
                "mean": arr.mean(axis=1).tolist(),
            }
    with open(OUT_DIR / "paths_summary.json", "w") as f:
        json.dump(thin, f, indent=2)

    print(f"\nWrote outputs to {OUT_DIR}")
    return all_summaries


if __name__ == "__main__":
    main()
