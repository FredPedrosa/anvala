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

N_PHYSICAL_CORES = 6


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


def _run_r_timed(r_code: str, rscript_exe: str | None, timeout: int = 1800) -> float:
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
            timeout=timeout,
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


def _lavaan_parallel_args(n_cores: int) -> str:
    if n_cores <= 1:
        return 'parallel = "no"'
    if sys.platform == "win32":
        return f'parallel = "snow", ncpus = {n_cores}'
    return f'parallel = "multicore", ncpus = {n_cores}'


def _build_latent_mediation_data(
    n_obs: int,
    n_items: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    """Gera dados e sintaxe lavaan para mediação latente de 3 fatores (F1 -> F2 -> F3)."""
    rng = np.random.default_rng(seed)
    m_per_f = n_items // 3

    f1 = rng.normal(size=n_obs)
    f2 = 0.50 * f1 + rng.normal(scale=np.sqrt(1.0 - 0.50**2), size=n_obs)
    f3 = 0.45 * f2 + 0.20 * f1 + rng.normal(scale=0.75, size=n_obs)
    factors = np.column_stack([f1, f2, f3])

    lam = np.zeros((n_items, 3))
    for k in range(3):
        lam[k * m_per_f : (k + 1) * m_per_f, k] = 0.80

    x_mat = factors @ lam.T + rng.normal(scale=0.60, size=(n_obs, n_items))
    cols = [f"x{i + 1}" for i in range(n_items)]
    df = pd.DataFrame(x_mat, columns=cols)

    f1_items = " + ".join(cols[0:m_per_f])
    f2_items = " + ".join(cols[m_per_f : 2 * m_per_f])
    f3_items = " + ".join(cols[2 * m_per_f : 3 * m_per_f])

    syntax = f"""
        F1 =~ {f1_items}
        F2 =~ {f2_items}
        F3 =~ {f3_items}
        F2 ~ a*F1
        F3 ~ b*F2 + c*F1
        indirect := a*b
        total    := c + a*b
    """
    return df, syntax


def run_sem_bootstrap_benchmark(
    b_list: list[int] = [250, 500, 1000, 2000, 5000],
    n_obs: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Estudo 1A: Escalabilidade em B para mediação observada (p = 3) vs. lavaan (1 e 6 cores)."""
    rscript_exe = _find_rscript()
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"
    torch_threads = torch.get_num_threads()

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
            print(f"[SEM Path p=3 | {gpu_label}] B = {b_rep}...", flush=True)

            t_r_cores = {}
            for nc in [1, N_PHYSICAL_CORES]:
                par_str = _lavaan_parallel_args(nc)
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
                fit <- sem(mod, data = df, se = "bootstrap", bootstrap = {b_rep}, {par_str})
                elapsed <- proc.time()[["elapsed"]] - t0
                cat(sprintf("ELAPSED:%.6f\\n", elapsed))
                """
                t_r_cores[nc] = _run_r_timed(r_code, rscript_exe)

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

            t_r1 = t_r_cores[1]
            t_r6 = t_r_cores[N_PHYSICAL_CORES]
            rows.append({
                "hardware": gpu_label,
                "torch_threads": torch_threads,
                "task": "SEM_Path_Mediation_p3",
                "p": 3,
                "N": n_obs,
                "B": b_rep,
                "R_lavaan_1core_sec": round(t_r1, 4) if np.isfinite(t_r1) else np.nan,
                "R_lavaan_6core_sec": round(t_r6, 4) if np.isfinite(t_r6) else np.nan,
                "anvala_CPU_sec": round(t_cpu, 4),
                "anvala_GPU_sec": round(t_gpu, 4) if np.isfinite(t_gpu) else np.nan,
                "speedup_CPU_vs_R1": round(t_r1 / t_cpu, 2) if np.isfinite(t_r1) and t_cpu > 0 else np.nan,
                "speedup_GPU_vs_R1": round(t_r1 / t_gpu, 2) if np.isfinite(t_r1) and np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_CPU": round(t_cpu / t_gpu, 2) if np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
            })
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    return pd.DataFrame(rows)


def run_sem_dimensionality_benchmark(
    p_list: list[int] = [6, 15, 30, 60],
    n_obs: int = 400,
    b_rep: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estudo 1B: Curva de cruzamento CPU vs. GPU em mediação latente (3 fatores, p = 6..60)."""
    rscript_exe = _find_rscript()
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"
    torch_threads = torch.get_num_threads()

    rows = []
    for p_items in p_list:
        print(f"[SEM Latent Crossover | {gpu_label}] p = {p_items}, B = {b_rep}...", flush=True)
        df, syntax = _build_latent_mediation_data(n_obs=n_obs, n_items=p_items, seed=seed + p_items)
        fit_py = av.sem(syntax, data=df, estimator="ML")

        if torch.cuda.is_available():
            _ = fit_py.bootstrap_ci(n_boot=10, method="bca", seed=seed, device="cuda")
            torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp_csv:
            df.to_csv(tmp_csv.name, index=False)
            csv_path = tmp_csv.name.replace("\\", "/")

        try:
            t_r_cores = {}
            for nc in [1, N_PHYSICAL_CORES]:
                par_str = _lavaan_parallel_args(nc)
                r_code = f"""
                if (!requireNamespace("lavaan", quietly = TRUE)) quit(status = 1)
                suppressPackageStartupMessages(library(lavaan))
                df <- read.csv("{csv_path}")
                mod <- '{syntax}'
                invisible(sem(mod, data = df, se = "standard"))
                t0 <- proc.time()[["elapsed"]]
                fit <- sem(mod, data = df, se = "bootstrap", bootstrap = {b_rep}, {par_str})
                elapsed <- proc.time()[["elapsed"]] - t0
                cat(sprintf("ELAPSED:%.6f\\n", elapsed))
                """
                t_r_cores[nc] = _run_r_timed(r_code, rscript_exe)

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

            t_r1 = t_r_cores[1]
            t_r6 = t_r_cores[N_PHYSICAL_CORES]
            rows.append({
                "hardware": gpu_label,
                "torch_threads": torch_threads,
                "task": "SEM_Latent_Mediation_3F",
                "p": p_items,
                "N": n_obs,
                "B": b_rep,
                "R_lavaan_1core_sec": round(t_r1, 4) if np.isfinite(t_r1) else np.nan,
                "R_lavaan_6core_sec": round(t_r6, 4) if np.isfinite(t_r6) else np.nan,
                "anvala_CPU_sec": round(t_cpu, 4),
                "anvala_GPU_sec": round(t_gpu, 4) if np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_R1": round(t_r1 / t_gpu, 2) if np.isfinite(t_r1) and np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_R6": round(t_r6 / t_gpu, 2) if np.isfinite(t_r6) and np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_CPU": round(t_cpu / t_gpu, 2) if np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
            })
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    return pd.DataFrame(rows)


def _build_network_data(n_obs: int, n_items: int, seed: int) -> pd.DataFrame:
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
    return pd.DataFrame(x_mat, columns=[f"I{i + 1}" for i in range(n_items)])


def run_bootega_benchmark(
    b_list: list[int] = [100, 250, 500, 1000],
    n_obs: int = 400,
    n_items: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Estudo 1C: Escalabilidade em B para bootEGA (p = 12) vs. EGAnet (1 e 6 cores)."""
    rscript_exe = _find_rscript()
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"

    df = _build_network_data(n_obs=n_obs, n_items=n_items, seed=seed)

    if torch.cuda.is_available():
        _ = av.boot_ega(df, n_boot=10, boot_type="nonparametric", corr="pearson", seed=seed, device="cuda")
        torch.cuda.synchronize()

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)
        csv_path = tmp_csv.name.replace("\\", "/")

    rows = []
    try:
        for b_rep in b_list:
            print(f"[bootEGA p={n_items} | {gpu_label}] B = {b_rep}...", flush=True)

            t_r_cores = {}
            for nc in [1, N_PHYSICAL_CORES]:
                r_code = f"""
                if (!requireNamespace("EGAnet", quietly = TRUE)) quit(status = 1)
                suppressPackageStartupMessages(library(EGAnet))
                df <- read.csv("{csv_path}")
                t0 <- proc.time()[["elapsed"]]
                res <- bootEGA(df, iter = {b_rep}, type = "resampling", corr = "pearson", ncores = {nc})
                elapsed <- proc.time()[["elapsed"]] - t0
                cat(sprintf("ELAPSED:%.6f\\n", elapsed))
                """
                t_r_cores[nc] = _run_r_timed(r_code, rscript_exe)

            t0 = time.perf_counter()
            _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, n_jobs=1, device="cpu")
            t_cpu1 = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, n_jobs=N_PHYSICAL_CORES, device="cpu")
            t_cpu6 = time.perf_counter() - t0

            if torch.cuda.is_available():
                _sync_if_cuda("cuda")
                t0 = time.perf_counter()
                _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, device="cuda")
                _sync_if_cuda("cuda")
                t_gpu = time.perf_counter() - t0
            else:
                t_gpu = np.nan

            t_r1 = t_r_cores[1]
            t_r6 = t_r_cores[N_PHYSICAL_CORES]
            rows.append({
                "hardware": gpu_label,
                "task": f"bootEGA_{n_items}_items",
                "p": n_items,
                "N": n_obs,
                "B": b_rep,
                "R_EGAnet_1core_sec": round(t_r1, 4) if np.isfinite(t_r1) else np.nan,
                "R_EGAnet_6core_sec": round(t_r6, 4) if np.isfinite(t_r6) else np.nan,
                "anvala_CPU_1core_sec": round(t_cpu1, 4),
                "anvala_CPU_6core_sec": round(t_cpu6, 4),
                "anvala_GPU_sec": round(t_gpu, 4) if np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_R1": round(t_r1 / t_gpu, 2) if np.isfinite(t_r1) and np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_R6": round(t_r6 / t_gpu, 2) if np.isfinite(t_r6) and np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_CPU1": round(t_cpu1 / t_gpu, 2) if np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
                "speedup_GPU_vs_CPU6": round(t_cpu6 / t_gpu, 2) if np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
            })
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    return pd.DataFrame(rows)


def run_bootega_dimensionality_benchmark(
    p_list: list[int] = [12, 24, 48],
    n_obs: int = 400,
    b_rep: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Estudo 1D: Escalabilidade cúbica O(p^3) do bootEGA em p = 12, 24, 48 (B = 500)."""
    rscript_exe = _find_rscript()
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"

    rows = []
    for p_items in p_list:
        print(f"[bootEGA Dimensionality | {gpu_label}] p = {p_items}, B = {b_rep}...", flush=True)
        df = _build_network_data(n_obs=n_obs, n_items=p_items, seed=seed + p_items)

        if torch.cuda.is_available():
            _ = av.boot_ega(df, n_boot=10, boot_type="nonparametric", corr="pearson", seed=seed, device="cuda")
            torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp_csv:
            df.to_csv(tmp_csv.name, index=False)
            csv_path = tmp_csv.name.replace("\\", "/")

        try:
            t_r_cores = {}
            for nc in [1, N_PHYSICAL_CORES]:
                r_code = f"""
                if (!requireNamespace("EGAnet", quietly = TRUE)) quit(status = 1)
                suppressPackageStartupMessages(library(EGAnet))
                df <- read.csv("{csv_path}")
                t0 <- proc.time()[["elapsed"]]
                res <- bootEGA(df, iter = {b_rep}, type = "resampling", corr = "pearson", ncores = {nc})
                elapsed <- proc.time()[["elapsed"]] - t0
                cat(sprintf("ELAPSED:%.6f\\n", elapsed))
                """
                t_r_cores[nc] = _run_r_timed(r_code, rscript_exe)

            t0 = time.perf_counter()
            _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, n_jobs=N_PHYSICAL_CORES, device="cpu")
            t_cpu6 = time.perf_counter() - t0

            if torch.cuda.is_available():
                _sync_if_cuda("cuda")
                t0 = time.perf_counter()
                _ = av.boot_ega(df, n_boot=b_rep, boot_type="nonparametric", corr="pearson", seed=seed, device="cuda")
                _sync_if_cuda("cuda")
                t_gpu = time.perf_counter() - t0
            else:
                t_gpu = np.nan

            t_r1 = t_r_cores[1]
            t_r6 = t_r_cores[N_PHYSICAL_CORES]
            rows.append({
                "hardware": gpu_label,
                "task": "bootEGA_Dimensionality",
                "p": p_items,
                "N": n_obs,
                "B": b_rep,
                "R_EGAnet_1core_sec": round(t_r1, 4) if np.isfinite(t_r1) else np.nan,
                "R_EGAnet_6core_sec": round(t_r6, 4) if np.isfinite(t_r6) else np.nan,
                "anvala_CPU_6core_sec": round(t_cpu6, 4),
                "anvala_GPU_sec": round(t_gpu, 4) if np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_R1": round(t_r1 / t_gpu, 2) if np.isfinite(t_r1) and np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_R6": round(t_r6 / t_gpu, 2) if np.isfinite(t_r6) and np.isfinite(t_gpu) else np.nan,
                "speedup_GPU_vs_CPU6": round(t_cpu6 / t_gpu, 2) if np.isfinite(t_gpu) and t_gpu > 0 else np.nan,
            })
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("simulations/results", exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU_only"
    tag = gpu_name.replace(" ", "_").replace("/", "-")
    print(
        f"Dispositivo: {gpu_name} | CPU threads (PyTorch): {torch.get_num_threads()} "
        f"| Rscript: {_find_rscript()}\n",
        flush=True,
    )

    df_sem = run_sem_bootstrap_benchmark()
    df_sem.to_csv(f"simulations/results/benchmark_sem_{tag}.csv", index=False)
    print("\n=== Estudo 1A: SEM Path Mediation (p = 3) ===")
    print(df_sem.to_string(index=False))

    print("\n" + "=" * 60 + "\n", flush=True)

    df_sem_dim = run_sem_dimensionality_benchmark()
    df_sem_dim.to_csv(f"simulations/results/benchmark_sem_dim_{tag}.csv", index=False)
    print("\n=== Estudo 1B: SEM Latent Mediation Crossover (p = 6..60) ===")
    print(df_sem_dim.to_string(index=False))

    print("\n" + "=" * 60 + "\n", flush=True)

    df_ega = run_bootega_benchmark()
    df_ega.to_csv(f"simulations/results/benchmark_bootega_{tag}.csv", index=False)
    print("\n=== Estudo 1C: bootEGA B-Scaling (p = 12) ===")
    print(df_ega.to_string(index=False))

    print("\n" + "=" * 60 + "\n", flush=True)

    df_ega_dim = run_bootega_dimensionality_benchmark()
    df_ega_dim.to_csv(f"simulations/results/benchmark_bootega_dim_{tag}.csv", index=False)
    print("\n=== Estudo 1D: bootEGA Dimensionality O(p^3) (p = 12..48) ===")
    print(df_ega_dim.to_string(index=False))