from __future__ import annotations

import os
import sys
import time
import warnings


def _setup_r_windows() -> None:
    """Registra o diretório bin/x64 do R no Windows antes de importar torch e rpy2."""
    if sys.platform != "win32":
        return
    r_base = r"C:\Program Files\R"
    if not os.path.isdir(r_base):
        return
    versions = sorted(os.listdir(r_base), reverse=True)
    for ver in versions:
        r_home = os.path.join(r_base, ver)
        r_bin = os.path.join(r_home, "bin", "x64")
        if os.path.isdir(r_bin):
            os.environ.setdefault("R_HOME", r_home)
            os.environ["PATH"] = r_bin + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(r_bin)
            break


_setup_r_windows()

import numpy as np
import pandas as pd
import torch

import anvala as av

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _sync_if_cuda(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _try_init_r():
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        has_lavaan = bool(ro.r('requireNamespace("lavaan", quietly = TRUE)')[0])
        has_eganet = bool(ro.r('requireNamespace("EGAnet", quietly = TRUE)')[0])
        return ro, pandas2ri, localconverter, has_lavaan, has_eganet
    except Exception:
        return None, None, None, False, False


def run_sem_bootstrap_benchmark(
    b_list: list[int] = [250, 500, 1000, 2000, 5000],
    n_obs: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara o tempo de execução do bootstrap SEM (mediação) entre lavaan (R), anvala CPU e anvala GPU."""
    ro, pandas2ri, localconverter, has_lavaan, _ = _try_init_r()
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_obs)
    m = 0.55 * x + rng.normal(size=n_obs)
    y = 0.45 * m + 0.20 * x + rng.normal(size=n_obs)
    df = pd.DataFrame({"X": x, "M": m, "Y": y})

    mod_py = """
        M ~ a*X
        Y ~ b*M + c*X
        indirect := a*b
        total    := c + a*b
    """
    fit_py = av.sem(mod_py, data=df, estimator="ML")

    if torch.cuda.is_available():
        _ = fit_py.bootstrap_ci(n_boot=20, method="bca", seed=seed, device="cuda")
        torch.cuda.synchronize()

    if has_lavaan:
        ro.r("library(lavaan)")
        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["df_r"] = ro.conversion.py2rpy(df)
        ro.r("""
            mod_r <- '
                M ~ a*X
                Y ~ b*M + c*X
                indirect := a*b
                total    := c + a*b
            '
        """)

    rows = []
    for b_rep in b_list:
        print(f"[SEM Bootstrap | {gpu_label}] Rodando B = {b_rep}...")

        if has_lavaan:
            t0 = time.perf_counter()
            ro.r(f'fit_r <- sem(mod_r, data = df_r, se = "bootstrap", bootstrap = {b_rep})')
            t_r = time.perf_counter() - t0
        else:
            t_r = np.nan

        t0 = time.perf_counter()
        _ = fit_py.bootstrap_ci(n_boot=b_rep, method="bca", seed=seed, device="cpu")
        t_cpu = time.perf_counter() - t0

        if torch.cuda.is_available():
            _sync_if_cuda("cuda")
            t0 = time.perf_counter()
            _ = fit_py.bootstrap_ci(n_boot=b_rep, method="bca", seed=seed, device="cuda")
            _sync_if_cuda("cuda")
            t_gpu = time.perf_counter() - t0
        else:
            t_gpu = np.nan

        rows.append({
            "hardware": gpu_label,
            "task": "SEM_Mediation_Bootstrap_BCa",
            "N": n_obs,
            "B": b_rep,
            "R_lavaan_sec": round(t_r, 4) if np.isfinite(t_r) else np.nan,
            "anvala_CPU_sec": round(t_cpu, 4),
            "anvala_GPU_sec": round(t_gpu, 4) if np.isfinite(t_gpu) else np.nan,
            "speedup_GPU_vs_R": round(t_r / t_gpu, 2) if np.isfinite(t_r) and np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
            "speedup_GPU_vs_CPU": round(t_cpu / t_gpu, 2) if np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def run_bootega_benchmark(
    b_list: list[int] = [100, 250, 500, 1000],
    n_obs: int = 400,
    n_items: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara o tempo de execução do bootEGA (24 itens, 4 fatores) entre EGAnet (R), anvala CPU e anvala GPU."""
    ro, pandas2ri, localconverter, _, has_eganet = _try_init_r()
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"

    rng = np.random.default_rng(seed)
    n_factors = 4
    items_per_f = n_items // n_factors
    cov_f = np.full((n_factors, n_factors), 0.30)
    np.fill_diagonal(cov_f, 1.0)
    f = rng.multivariate_normal(mean=np.zeros(n_factors), cov=cov_f, size=n_obs)

    lam = np.zeros((n_items, n_factors))
    for k in range(n_factors):
        lam[k * items_per_f : (k + 1) * items_per_f, k] = 0.75
    x_mat = f @ lam.T + rng.normal(scale=0.65, size=(n_obs, n_items))
    df = pd.DataFrame(x_mat, columns=[f"I{i + 1}" for i in range(n_items)])

    if torch.cuda.is_available():
        _ = av.boot_ega(df, n_boot=10, boot_type="nonparametric", corr="pearson", seed=seed, device="cuda")
        torch.cuda.synchronize()

    if has_eganet:
        ro.r("library(EGAnet)")
        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["df_ega_r"] = ro.conversion.py2rpy(df)

    rows = []
    for b_rep in b_list:
        print(f"[bootEGA {n_items} itens | {gpu_label}] Rodando B = {b_rep}...")

        if has_eganet:
            t0 = time.perf_counter()
            ro.r(f'res_r <- bootEGA(df_ega_r, iter = {b_rep}, type = "resampling", corr = "cor_auto", ncores = 4, progress = FALSE)')
            t_r = time.perf_counter() - t0
        else:
            t_r = np.nan

        t0 = time.perf_counter()
        _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, n_jobs=4, device="cpu")
        t_cpu = time.perf_counter() - t0

        if torch.cuda.is_available():
            _sync_if_cuda("cuda")
            t0 = time.perf_counter()
            _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, device="cuda")
            _sync_if_cuda("cuda")
            t_gpu = time.perf_counter() - t0
        else:
            t_gpu = np.nan

        rows.append({
            "hardware": gpu_label,
            "task": f"bootEGA_{n_items}_items",
            "N": n_obs,
            "B": b_rep,
            "R_EGAnet_sec": round(t_r, 4) if np.isfinite(t_r) else np.nan,
            "anvala_CPU_sec": round(t_cpu, 4),
            "anvala_GPU_sec": round(t_gpu, 4) if np.isfinite(t_gpu) else np.nan,
            "speedup_GPU_vs_R": round(t_r / t_gpu, 2) if np.isfinite(t_r) and np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
            "speedup_GPU_vs_CPU": round(t_cpu / t_gpu, 2) if np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("simulations/results", exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"
    tag = gpu_name.replace(" ", "_").replace("/", "-")
    print(f"Dispositivo detectado: {gpu_name}\n")

    df_sem = run_sem_bootstrap_benchmark()
    sem_path = f"simulations/results/benchmark_sem_{tag}.csv"
    df_sem.to_csv(sem_path, index=False)
    print(f"\n=== Resultados Benchmark SEM Bootstrap ({sem_path}) ===")
    print(df_sem.to_string(index=False))

    print("\n" + "=" * 60 + "\n")

    df_ega = run_bootega_benchmark()
    ega_path = f"simulations/results/benchmark_bootega_{tag}.csv"
    df_ega.to_csv(ega_path, index=False)
    print(f"\n=== Resultados Benchmark bootEGA ({ega_path}) ===")
    print(df_ega.to_string(index=False))