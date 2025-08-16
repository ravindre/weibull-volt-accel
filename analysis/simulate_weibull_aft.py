#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, time, gzip, warnings, statistics, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# Optional system/memory
try:
    import psutil
    HAVE_PSUTIL = True
except Exception:
    HAVE_PSUTIL = False
import tracemalloc

# Modeling
from lifelines import WeibullAFTFitter

# Storage
import duckdb

# ===================== CONFIG =====================
# Scenario grid
GAMMAS = [10, 15, 20, 25, 30]
BETAS = [0.5, 0.8, 2.0]
ANCHOR_ETAS_1_6V = [150, 200]
VOLTAGES = [1.6, 1.4, 1.3]
STRESS_STRATEGIES = {"16x_increments": None, "50x_increments": None}
MAX_DURATIONS = [200, 300]

# Per-run sample size
SAMPLE_SIZE_PER_LEG = 610

# Monte Carlo control
NUM_MONTE_CARLO_RUNS_MAX = 1000    # upper bound if early stopping doesn't trigger
MIN_RUNS_BEFORE_CHECK = 30         # don't stop before we have enough runs

# Early stopping thresholds (meet either to stop)
ABS_CI_TARGET = 0.5                # stop when 95% CI half-width < 0.5 gamma units
REL_CI_TARGET = 0.05               # or 5% of true gamma

# Penalizer sweep and model selection
PENALIZERS = [0.001, 0.003, 0.005]
CHOOSE_BY = "aic"                  # "aic" | "bic" | "holdout"
HOLDOUT_FRACTION = 0.2             # only used if CHOOSE_BY="holdout"

# Parallel runtime
USE_MULTIPROCESSING = True
MAX_PROCESSES = None               # None -> cpu_count(); else clamp
RUNS_CHUNK_SIZE = 64               # batch runs to reduce IPC overhead

# Output
OUTPUT_DIR = "simulation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DB_PATH = os.path.join(OUTPUT_DIR, "results.duckdb")

# Plots
GENERATE_DASHBOARD = True
PLOT_DPI = 140
LIMIT_SCATTER_ROWS = 250_000
TOPN_SLOWEST = 20

# Seed
GLOBAL_SEED = 42
# ==================================================

def generate_readout_points(strategy_name, max_duration):
    points = [1, 2, 4, 8, 16, 32]
    current_time = 32
    increment = 16 if strategy_name == "16x_increments" else 50
    while current_time + increment <= max_duration:
        current_time += increment
        points.append(current_time)
    if max_duration not in points:
        points.append(max_duration)
    return sorted(set(points))


def extract_parameters_safely(aft):
    gamma_hat = np.nan
    beta_hat = np.nan
    try:
        params_ = getattr(aft, "params_", None)
        if params_ is not None:
            if isinstance(params_, pd.Series):
                if ('lambda_', 'voltage') in params_.index:
                    gamma_hat = -params_[('lambda_', 'voltage')]
                if ('rho_', 'Intercept') in params_.index:
                    beta_hat = np.exp(params_[('rho_', 'Intercept')])
            elif isinstance(params_, pd.DataFrame):
                if ('lambda_', 'voltage') in params_.index:
                    gamma_hat = -params_.loc[('lambda_', 'voltage'), 'coef']
                if ('rho_', 'Intercept') in params_.index:
                    beta_hat = np.exp(params_.loc[('rho_', 'Intercept'), 'coef'])
        if np.isnan(gamma_hat):
            summ = getattr(aft, "summary", None)
            if summ is not None and 'voltage' in summ.index:
                gamma_hat = -summ.loc['voltage', 'coef']
        if np.isnan(beta_hat) and hasattr(aft, 'rho_'):
            try:
                beta_hat = 1.0 / aft.rho_
            except Exception:
                pass
    except Exception:
        pass
    return gamma_hat, beta_hat


def split_holdout(df, frac=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = np.arange(len(df))
    rng.shuffle(idx)
    k = int(len(idx) * frac)
    test_idx = idx[:k]
    train_idx = idx[k:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def aft_loglik_on(df, aft):
    try:
        if not {'time', 'event', 'voltage'}.issubset(df.columns):
            return np.nan
        params_ = aft.params_
        if isinstance(params_, pd.Series):
            b0 = float(params_[('lambda_', 'Intercept')])
            b1 = float(params_[('lambda_', 'voltage')])
            log_rho = float(params_[('rho_', 'Intercept')])
        else:
            b0 = float(params_.loc[('lambda_', 'Intercept'), 'coef'])
            b1 = float(params_.loc[('lambda_', 'voltage'), 'coef'])
            log_rho = float(params_.loc[('rho_', 'Intercept'), 'coef'])
        rho = np.exp(log_rho)
        eta = b0 + b1 * df['voltage'].values
        lam = np.exp(eta)
        t = df['time'].values
        e = df['event'].values.astype(bool)
        z = (t / lam) ** rho
        log_pdf = np.log(rho) - np.log(lam) + (rho - 1.0) * (np.log(t) - np.log(lam)) - z
        log_surv = -z
        ll = np.sum(np.where(e, log_pdf, log_surv))
        return float(ll)
    except Exception:
        return np.nan


def fit_with_penalizers(experiment_data, penalizers, choose_by="aic", holdout_frac=0.2, rng=None):
    best = None
    for pen in penalizers:
        aft = WeibullAFTFitter(penalizer=pen)
        try:
            if choose_by == "holdout":
                train_df, test_df = split_holdout(experiment_data, frac=holdout_frac, rng=rng)
                aft.fit(train_df, 'time', event_col='event', formula='voltage')
                score = aft_loglik_on(test_df, aft)
                criterion = -score  # lower is better
            else:
                aft.fit(experiment_data, 'time', event_col='event', formula='voltage')
                if choose_by == "aic":
                    criterion = float(aft.AIC_) if hasattr(aft, "AIC_") else float(aft.log_likelihood_) * -2
                elif choose_by == "bic":
                    if hasattr(aft, "BIC_"):
                        criterion = float(aft.BIC_)
                    else:
                        k = len(aft.params_) if hasattr(aft, "params_") else 2
                        n = len(experiment_data)
                        ll = float(aft.log_likelihood_) if hasattr(aft, "log_likelihood_") else 0.0
                        criterion = k * math.log(max(n, 2)) - 2 * ll
                else:
                    criterion = float(aft.AIC_) if hasattr(aft, "AIC_") else float(aft.log_likelihood_) * -2
        except Exception:
            continue
        if best is None or criterion < best["criterion"]:
            gamma_hat, beta_hat = extract_parameters_safely(aft)
            best = {
                "penalizer": pen,
                "aft": aft,
                "criterion": criterion,
                "gamma_hat": gamma_hat,
                "beta_hat": beta_hat
            }
    return best


def worker_run(args):
    """
    Returns a dict with timing, memory, chosen penalizer, and estimates.
    """
    tracemalloc.start()
    p = psutil.Process(os.getpid()) if HAVE_PSUTIL else None
    rss0 = p.memory_info().rss if p else None

    t0 = time.perf_counter()

    (true_gamma, true_beta, true_etas, readout_points, max_duration, sample_size_per_leg,
     voltages, run_seed, penalizers, choose_by, holdout_frac) = args

    rng = np.random.default_rng(run_seed)

    # Generate and censor
    gen_start = time.perf_counter()
    all_legs = []
    num_fails = {V: 0 for V in voltages}
    for V in voltages:
        eta = true_etas[V]
        U = rng.random(sample_size_per_leg)
        tfail = eta * (-np.log(U)) ** (1.0 / true_beta)
        observed_times, event = [], []
        for t in tfail:
            if t > max_duration:
                observed_times.append(max_duration); event.append(0)
            else:
                idx = np.searchsorted(readout_points, t)
                if idx == len(readout_points):
                    idx -= 1
                observed_times.append(readout_points[idx]); event.append(1)
        df_leg = pd.DataFrame({"time": observed_times, "event": event, "voltage": V})
        all_legs.append(df_leg)
        num_fails[V] = int(np.sum(event))
    gen_end = time.perf_counter()

    # Concat
    cat_start = time.perf_counter()
    experiment_data = pd.concat(all_legs, ignore_index=True)
    cat_end = time.perf_counter()

    if experiment_data['voltage'].nunique() < 2 or experiment_data['event'].sum() < 5:
        py_curr, py_peak = tracemalloc.get_traced_memory()
        rss1 = p.memory_info().rss if p else None
        tracemalloc.stop()
        return {
            "success": False, "gamma_hat": None, "beta_hat": None, "penalizer": None,
            "fails16": num_fails.get(1.6, 0), "fails14": num_fails.get(1.4, 0), "fails13": num_fails.get(1.3, 0),
            "t_gen": gen_end - gen_start, "t_cat": cat_end - cat_start, "t_fit": 0.0, "t_extract": 0.0,
            "t_total": time.perf_counter() - t0, "py_peak_mb": py_peak/1e6, "rss_peak_mb": (rss1/1e6) if rss1 else None
        }

    # Fit with penalizer sweep / selection
    fit_start = time.perf_counter()
    best = fit_with_penalizers(experiment_data, penalizers, choose_by=choose_by, holdout_frac=holdout_frac, rng=rng)
    fit_end = time.perf_counter()

    if best is None or np.isnan(best["gamma_hat"]) or np.isnan(best["beta_hat"]):
        py_curr, py_peak = tracemalloc.get_traced_memory()
        rss1 = p.memory_info().rss if p else None
        tracemalloc.stop()
        return {
            "success": False, "gamma_hat": None, "beta_hat": None, "penalizer": None,
            "fails16": num_fails.get(1.6, 0), "fails14": num_fails.get(1.4, 0), "fails13": num_fails.get(1.3, 0),
            "t_gen": gen_end - gen_start, "t_cat": cat_end - cat_start, "t_fit": fit_end - fit_start,
            "t_extract": 0.0, "t_total": time.perf_counter() - t0,
            "py_peak_mb": py_peak/1e6, "rss_peak_mb": (rss1/1e6) if rss1 else None
        }

    # Extract already done in best
    ext_start = time.perf_counter()
    gamma_hat = best["gamma_hat"]; beta_hat = best["beta_hat"]
    ext_end = time.perf_counter()

    py_curr, py_peak = tracemalloc.get_traced_memory()
    rss1 = p.memory_info().rss if p else None
    tracemalloc.stop()

    return {
        "success": True,
        "gamma_hat": float(gamma_hat), "beta_hat": float(beta_hat),
        "penalizer": best["penalizer"],
        "fails16": num_fails.get(1.6, 0), "fails14": num_fails.get(1.4, 0), "fails13": num_fails.get(1.3, 0),
        "t_gen": gen_end - gen_start, "t_cat": cat_end - cat_start, "t_fit": fit_end - fit_start,
        "t_extract": ext_end - ext_start, "t_total": time.perf_counter() - t0,
        "py_peak_mb": py_peak/1e6, "rss_peak_mb": (rss1/1e6) if rss1 else None
    }


def setup_duckdb(db_path):
    con = duckdb.connect(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS run_results (
      true_gamma DOUBLE,
      true_beta DOUBLE,
      anchor_eta_1_6v DOUBLE,
      strategy VARCHAR,
      max_duration DOUBLE,
      run_id BIGINT,
      estimated_gamma DOUBLE,
      estimated_beta DOUBLE,
      chosen_penalizer DOUBLE,
      num_fails_1_6v BIGINT,
      num_fails_1_4v BIGINT,
      num_fails_1_3v BIGINT,
      t_gen DOUBLE, t_cat DOUBLE, t_fit DOUBLE, t_extract DOUBLE, t_total DOUBLE,
      py_peak_mb DOUBLE, rss_peak_mb DOUBLE
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS scenario_summaries (
      true_gamma DOUBLE,
      true_beta DOUBLE,
      anchor_eta_1_6v DOUBLE,
      strategy VARCHAR,
      max_duration DOUBLE,
      n_runs BIGINT,
      n_success BIGINT,
      mean_gamma DOUBLE, median_gamma DOUBLE, std_gamma DOUBLE, bias_gamma DOUBLE,
      mean_beta DOUBLE, median_beta DOUBLE, std_beta DOUBLE, bias_beta DOUBLE,
      median_fails_1_6v DOUBLE, median_fails_1_4v DOUBLE, median_fails_1_3v DOUBLE,
      mean_py_peak_mb DOUBLE, max_py_peak_mb DOUBLE, mean_rss_peak_mb DOUBLE, max_rss_peak_mb DOUBLE,
      t_gen_sum DOUBLE, t_cat_sum DOUBLE, t_fit_sum DOUBLE, t_extract_sum DOUBLE, t_total_sum DOUBLE,
      scenario_time DOUBLE,
      stopped_early BOOLEAN, ci_halfwidth DOUBLE, ci_method VARCHAR
    );
    """)
    return con


def ci_halfwidth(values, alpha=0.05):
    n = len(values)
    if n < 2:
        return float('nan'), float('nan'), float('inf')
    mu = np.mean(values)
    sd = np.std(values, ddof=1)
    z = 1.96  # normal approx
    hw = z * sd / math.sqrt(n)
    return mu, sd, hw


def run_scenarios():
    con = setup_duckdb(DB_PATH)
    from multiprocessing import Pool, cpu_count

    total_scenarios = len(GAMMAS) * len(BETAS) * len(ANCHOR_ETAS_1_6V) * len(MAX_DURATIONS) * len(STRESS_STRATEGIES)
    scen_idx = 0

    for true_gamma in GAMMAS:
        for true_beta in BETAS:
            for anchor_eta in ANCHOR_ETAS_1_6V:
                true_etas = {
                    1.6: anchor_eta,
                    1.4: anchor_eta * np.exp(-true_gamma * (1.4 - 1.6)),
                    1.3: anchor_eta * np.exp(-true_gamma * (1.3 - 1.6))
                }
                for strategy_name in STRESS_STRATEGIES:
                    for max_duration in MAX_DURATIONS:
                        scen_idx += 1

                        readout_points = generate_readout_points(strategy_name, max_duration)
                        scen_t0 = time.perf_counter()

                        est_gammas, est_betas = [], []
                        f16s, f14s, f13s = [], [], []
                        t_gen_sum = t_cat_sum = t_fit_sum = t_ex_sum = t_tot_sum = 0.0
                        py_peaks, rss_peaks = [], []
                        success_count, run_id = 0, 0

                        stopped_early = False
                        ci_hw_used = None
                        ci_method = None

                        n_runs_target = NUM_MONTE_CARLO_RUNS_MAX
                        seeds = [int(1e9 * np.random.rand() + i * 9973) for i in range(n_runs_target)]
                        arg_base = (true_gamma, true_beta, true_etas, readout_points, max_duration,
                                    SAMPLE_SIZE_PER_LEG, VOLTAGES)

                        def maybe_stop():
                            nonlocal stopped_early, ci_hw_used, ci_method
                            if len(est_gammas) < MIN_RUNS_BEFORE_CHECK:
                                return False
                            mu, sd, hw = ci_halfwidth(est_gammas, alpha=0.05)
                            if hw <= ABS_CI_TARGET:
                                stopped_early = True; ci_hw_used = hw; ci_method = "abs"
                                return True
                            rel_target = REL_CI_TARGET * true_gamma
                            if hw <= rel_target:
                                stopped_early = True; ci_hw_used = hw; ci_method = "rel"
                                return True
                            return False

                        num_chunks = math.ceil(n_runs_target / RUNS_CHUNK_SIZE)
                        for ci in range(num_chunks):
                            if stopped_early:
                                break
                            s = ci * RUNS_CHUNK_SIZE
                            e = min((ci + 1) * RUNS_CHUNK_SIZE, n_runs_target)
                            chunk_args = [(*arg_base, seeds[i], PENALIZERS, CHOOSE_BY, HOLDOUT_FRACTION)
                                          for i in range(s, e)]
                            results = []
                            if USE_MULTIPROCESSING:
                                nprocs = cpu_count() if MAX_PROCESSES is None else min(cpu_count(), MAX_PROCESSES)
                                with Pool(processes=nprocs) as pool:
                                    for r in pool.imap_unordered(worker_run, chunk_args, chunksize=4):
                                        results.append(r)
                            else:
                                for a in chunk_args:
                                    results.append(worker_run(a))

                            for r in results:
                                run_id += 1
                                t_gen_sum += r["t_gen"]; t_cat_sum += r["t_cat"]; t_fit_sum += r["t_fit"]
                                t_ex_sum += r["t_extract"]; t_tot_sum += r["t_total"]
                                if r["py_peak_mb"] is not None: py_peaks.append(r["py_peak_mb"])
                                if r["rss_peak_mb"] is not None: rss_peaks.append(r["rss_peak_mb"])
                                if r["success"]:
                                    success_count += 1
                                    est_gammas.append(r["gamma_hat"]); est_betas.append(r["beta_hat"])
                                    f16s.append(r["fails16"]); f14s.append(r["fails14"]); f13s.append(r["fails13"])
                                # stream insert
                                con.execute("""
                                  INSERT INTO run_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                """, (
                                    float(true_gamma), float(true_beta), float(anchor_eta),
                                    strategy_name, float(max_duration), int(run_id),
                                    float(r["gamma_hat"]) if r["gamma_hat"] is not None else None,
                                    float(r["beta_hat"]) if r["beta_hat"] is not None else None,
                                    float(r["penalizer"]) if r["penalizer"] is not None else None,
                                    int(r["fails16"]), int(r["fails14"]), int(r["fails13"]),
                                    float(r["t_gen"]), float(r["t_cat"]), float(r["t_fit"]),
                                    float(r["t_extract"]), float(r["t_total"]),
                                    float(r["py_peak_mb"]) if r["py_peak_mb"] is not None else None,
                                    float(r["rss_peak_mb"]) if r["rss_peak_mb"] is not None else None
                                ))

                            if maybe_stop():
                                break

                        scen_t1 = time.perf_counter()
                        scenario_time = scen_t1 - scen_t0

                        if est_gammas:
                            mean_g = float(np.mean(est_gammas)); med_g = float(np.median(est_gammas)); std_g = float(np.std(est_gammas))
                            mean_b = float(np.mean(est_betas)); med_b = float(np.median(est_betas)); std_b = float(np.std(est_betas))
                            med_f16 = float(np.median(f16s)); med_f14 = float(np.median(f14s)); med_f13 = float(np.median(f13s))
                            bias_g = mean_g - float(true_gamma)
                            bias_b = mean_b - float(true_beta)
                        else:
                            mean_g = med_g = std_g = bias_g = np.nan
                            mean_b = med_b = std_b = bias_b = np.nan
                            med_f16 = med_f14 = med_f13 = np.nan

                        mean_py_peak = float(np.mean(py_peaks)) if py_peaks else 0.0
                        max_py_peak = float(np.max(py_peaks)) if py_peaks else 0.0
                        mean_rss_peak = float(np.mean(rss_peaks)) if rss_peaks else 0.0
                        max_rss_peak = float(np.max(rss_peaks)) if rss_peaks else 0.0

                        # Corrected: 31 placeholders for 31 columns
                        con.execute("""
                          INSERT INTO scenario_summaries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            float(true_gamma), float(true_beta), float(anchor_eta), strategy_name, float(max_duration),
                            int(run_id), int(success_count),
                            mean_g, med_g, std_g, bias_g,
                            mean_b, med_b, std_b, bias_b,
                            med_f16, med_f14, med_f13,
                            mean_py_peak, max_py_peak, mean_rss_peak, max_rss_peak,
                            float(t_gen_sum), float(t_cat_sum), float(t_fit_sum), float(t_ex_sum), float(t_tot_sum),
                            float(scenario_time),
                            bool(stopped_early),
                            float(ci_hw_used) if ci_hw_used is not None else None,
                            ci_method
                        ))

                        print(f"[{scen_idx}/{total_scenarios}] scen={scenario_time:.2f}s runs={run_id} succ={success_count} "
                              f"stopped={stopped_early} ci_hw={ci_hw_used if ci_hw_used else 'NA'} "
                              f"t_fit={t_fit_sum:.2f}s peak_py={mean_py_peak:.1f}/{max_py_peak:.1f}MB")

    con.close()
    return DB_PATH


def export_csvs(db_path, out_dir):
    # FIXED: Remove read_only=True
    con = duckdb.connect(db_path)
    runs_csv = os.path.join(out_dir, "all_run_results.csv.gz")
    summary_csv = os.path.join(out_dir, "simulation_summary.csv")
    con.execute(f"COPY (SELECT * FROM run_results) TO '{runs_csv}' (FORMAT CSV, HEADER, COMPRESSION GZIP);")
    con.execute(f"COPY (SELECT * FROM scenario_summaries) TO '{summary_csv}' (FORMAT CSV, HEADER);")
    con.close()
    return summary_csv, runs_csv


def dashboard(db_path, out_dir, dpi=140, limit_scatter=250_000, topN=20):
    # FIXED: Remove read_only=True
    con = duckdb.connect(db_path)
    s = con.execute("SELECT * FROM scenario_summaries").df()
    r = con.execute("SELECT * FROM run_results").df()
    con.close()

    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(s["t_fit_sum"], bins=30, ax=ax1, color="#7fbf7f")
    ax1.set_title("t_fit_sum per scenario"); ax1.set_xlabel("seconds"); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(s["scenario_time"], bins=30, ax=ax2, color="#b0c4de")
    ax2.set_title("scenario_time"); ax2.set_xlabel("seconds"); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(1, len(s)+1)
    ax3.plot(x, s["mean_py_peak_mb"], "o-", label="mean_py_peak", color="#2ca25f", alpha=0.9)
    ax3.plot(x, s["max_py_peak_mb"], "o-", label="max_py_peak", color="#006d2c", alpha=0.9)
    if "mean_rss_peak_mb" in s and "max_rss_peak_mb" in s:
        ax3.plot(x, s["mean_rss_peak_mb"], "s--", label="mean_rss_peak", color="#3182bd", alpha=0.8)
        ax3.plot(x, s["max_rss_peak_mb"], "s--", label="max_rss_peak", color="#08519c", alpha=0.8)
    ax3.set_title("Worker peak memory"); ax3.set_xlabel("scenario index"); ax3.set_ylabel("MB")
    ax3.grid(True, alpha=0.3); ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[1, 0:2])
    s2 = s.copy()
    s2["scenario_id"] = np.arange(len(s2)) + 1
    s2 = s2.sort_values("scenario_time", ascending=False).head(min(topN, len(s2)))
    comp = s2[["scenario_id","t_fit_sum","t_gen_sum","t_cat_sum","t_extract_sum"]].set_index("scenario_id")
    comp.columns = ["fit","gen","cat","extract"]
    comp.plot(kind="bar", stacked=True, ax=ax4, color=["#6baed6","#74c476","#fd8d3c","#9e9ac8"], width=0.9, legend=True)
    ax4.set_title(f"Top {min(topN,len(s2))} slowest scenarios: time breakdown"); ax4.set_ylabel("seconds"); ax4.grid(True, axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    if len(s) > 0:
        row = s.sort_values("n_runs", ascending=False).iloc[0]
        tg, tb, ae, st, md = row["true_gamma"], row["true_beta"], row["anchor_eta_1_6v"], row["strategy"], row["max_duration"]
        sub = r[(r["true_gamma"]==tg)&(r["true_beta"]==tb)&(r["anchor_eta_1_6v"]==ae)&(r["strategy"]==st)&(r["max_duration"]==md)]
        sub = sub.sort_values("run_id")
        gammas = sub["estimated_gamma"].dropna().values
        hw_vals, x_idx = [], []
        for k in range(5, len(gammas)+1):
            _, _, hw = ci_halfwidth(gammas[:k]); hw_vals.append(hw); x_idx.append(k)
        ax5.plot(x_idx, hw_vals, "-o", ms=3, color="#cc6")
        ax5.axhline(ABS_CI_TARGET, color="r", linestyle="--", label="ABS target")
        ax5.axhline(REL_CI_TARGET*tg, color="m", linestyle="--", label="REL target")
        ax5.set_title("CI half-width vs runs (example)"); ax5.set_xlabel("runs"); ax5.set_ylabel("95% CI half-width")
        ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 0])
    dfp = r.dropna(subset=["estimated_gamma","true_gamma"])
    if len(dfp) > limit_scatter:
        dfp = dfp.sample(limit_scatter, random_state=42)
        label = f"Gamma accuracy (sampled {len(dfp):,})"
    else:
        label = "Gamma accuracy"
    sns.scatterplot(data=dfp, x="true_gamma", y="estimated_gamma", hue="strategy", s=8, alpha=0.5, ax=ax6)
    mn, mx = dfp["true_gamma"].min(), dfp["true_gamma"].max()
    ax6.plot([mn, mx], [mn, mx], "r--", alpha=0.8); ax6.set_title(label); ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 1])
    dfb = r.dropna(subset=["estimated_beta","true_beta"])
    if len(dfb) > limit_scatter:
        dfb = dfb.sample(limit_scatter, random_state=42)
        label2 = f"Beta accuracy (sampled {len(dfb):,})"
    else:
        label2 = "Beta accuracy"
    sns.scatterplot(data=dfb, x="true_beta", y="estimated_beta", hue="strategy", s=8, alpha=0.5, ax=ax7)
    mn, mx = dfb["true_beta"].min(), dfb["true_beta"].max()
    ax7.plot([mn, mx], [mn, mx], "r--", alpha=0.8); ax7.set_title(label2); ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 2])
    grp_cols = ["true_gamma","true_beta","anchor_eta_1_6v","strategy","max_duration"]
    rows_per = r.groupby(grp_cols).size().reset_index(name="rows")
    sns.violinplot(data=rows_per, y="rows", ax=ax8, color="#c6b4d1", inner="box", cut=0)
    ax8.set_title("Rows per scenario (disk proxy)"); ax8.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Weibull AFT Simulation – Performance Dashboard", fontsize=14)
    out = os.path.join(out_dir, "performance_dashboard.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("Starting adaptive Monte Carlo with penalizer sweep and DuckDB streaming")
    np.random.seed(GLOBAL_SEED); random.seed(GLOBAL_SEED)

    t0 = time.perf_counter()
    db_path = run_scenarios()
    t1 = time.perf_counter()
    print(f"Completed; wall time {t1 - t0:.2f}s, DB at {db_path}")

    print("Exporting CSV artifacts...")
    summary_csv, runs_csv = export_csvs(db_path, OUTPUT_DIR)
    print(f"Wrote: {runs_csv}, {summary_csv}")

    if GENERATE_DASHBOARD:
        print("Creating consolidated dashboard...")
        dash_path = dashboard(db_path, OUTPUT_DIR, dpi=PLOT_DPI, limit_scatter=LIMIT_SCATTER_ROWS, topN=TOPN_SLOWEST)
        print(f"Dashboard: {dash_path}")


if __name__ == "__main__":
    main()
