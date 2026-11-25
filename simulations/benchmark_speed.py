from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import torch

import anvala as av

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _find_rscript() -> str | None:
    """Localiza o executável Rscript no Windows ou Linux/Colab."""
    if sys.platform == "win32":
        cand = r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe"
        if os.path.isfile(cand):
            return cand
        r_base = r"C:\Program Files\R"
        if os.path.isdir(r_base):
            for ver in sorted(os.listdir(r_base), reverse=True):
                exe = os.path.join(r_base, ver, "bin", "x64", "Rscript.exe")
                if os.path.isfile(exe):
                    return exe
    return shutil.which("Rscript")


def _run_r_timed(r_code: str, rscript_exe: str | None) -> float:
    """Executa um script R temporário e extrai o tempo interno medido via proc.time()."""
    if not rscript_exe:
        return np.nan
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as tmp:
        tmp.write(r_code)
        r_path = tmp.name
    try:
        proc = subprocess.run(
            [rscript_exe, "--vanilla", r_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        if proc.returncode != 0:
            return np.nan
        m = re.search(r"ELAPSED:([0-9.]+)", proc.stdout)
        return float(m.group(1)) if m else np.nan
    except Exception:
        return np.nan
    finally:
        if os.path.exists(r_path):
            os.remove(r_path)


def _sync_if_cuda(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_sem_bootstrap_benchmark(
    b_list: list[int] = [250, 500, 1000, 2000, 5000],
    n_obs: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara o tempo do bootstrap SEM (mediação) entre lavaan (R), anvala CPU e anvala GPU."""
    rscript_exe = _find_rscript()
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

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)
        csv_path = tmp_csv.name.replace("\\", "/")

    rows = []
    try:
        for b_rep in b_list:
            print(f"[SEM Bootstrap | {gpu_label}] Rodando B = {b_rep}...", flush=True)

            r_code = f"""
            if (!requireNamespace("lavaan", quietly = TRUE)) quit(status = 1)
            suppressPackageStartupMessages(library(lavaan))
            df <- read.csv("{csv_path}")
            mod <- '
                M ~ a*X
                Y ~ b*M + c*X
                indirect := a*b
                total    := c + a*b
            '
            invisible(sem(mod, data = df, se = "standard"))
            t0 <- proc.time()[["elapsed"]]
            fit <- sem(mod, data = df, se = "bootstrap", bootstrap = {b_rep})
            elapsed <- proc.time()[["elapsed"]] - t0
            cat(sprintf("ELAPSED:%.6f\\n", elapsed))
            """
            t_r = _run_r_timed(r_code, rscript_exe)

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
                "speedup_CPU_vs_R": round(t_r / t_cpu, 2) if np.isfinite(t_r) and t_cpu > 0 else np.nan,
                "speedup_GPU_vs_R": round(t_r / t_gpu, 2) if np.isfinite(t_r) and np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
            })
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    return pd.DataFrame(rows)


def run_bootega_benchmark(
    b_list: list[int] = [100, 250, 500, 1000],
    n_obs: int = 400,
    n_items: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara o tempo do bootEGA (12 itens, 3 fatores) entre EGAnet (R), anvala CPU e anvala GPU."""
    rscript_exe = _find_rscript()
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"

    rng = np.random.default_rng(seed)
    n_factors = 3
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

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)
        csv_path = tmp_csv.name.replace("\\", "/")

    rows = []
    try:
        for b_rep in b_list:
            print(f"[bootEGA {n_items} itens | {gpu_label}] Rodando B = {b_rep}...", flush=True)

            r_code = f"""
            if (!requireNamespace("EGAnet", quietly = TRUE)) quit(status = 1)
            suppressPackageStartupMessages(library(EGAnet))
            df <- read.csv("{csv_path}")
            t0 <- proc.time()[["elapsed"]]
            res <- bootEGA(df, iter = {b_rep}, type = "resampling", corr = "cor_auto", ncores = 1, progress = FALSE)
            elapsed <- proc.time()[["elapsed"]] - t0
            cat(sprintf("ELAPSED:%.6f\\n", elapsed))
            """
            t_r = _run_r_timed(r_code, rscript_exe)

            t0 = time.perf_counter()
            _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, n_jobs=1, device="cpu")
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
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("simulations/results", exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"
    tag = gpu_name.replace(" ", "_").replace("/", "-")
    print(f"Dispositivo detectado: {gpu_name} | Rscript: {_find_rscript()}\n", flush=True)

    df_sem = run_sem_bootstrap_benchmark()
    sem_path = f"simulations/results/benchmark_sem_{tag}.csv"
    df_sem.to_csv(sem_path, index=False)
    print(f"\n=== Resultados Benchmark SEM Bootstrap ({sem_path}) ===")
    print(df_sem.to_string(index=False))

    print("\n" + "=" * 60 + "\n", flush=True)

    df_ega = run_bootega_benchmark()
    ega_path = f"simulations/results/benchmark_bootega_{tag}.csv"
    df_ega.to_csv(ega_path, index=False)
    print(f"\n=== Resultados Benchmark bootEGA ({ega_path}) ===")
    print(df_ega.to_string(index=False))