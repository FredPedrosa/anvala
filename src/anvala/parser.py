from dataclasses import dataclass
import re
import numpy as np


@dataclass
class ParamSpec:
    lhs: str
    op: str
    rhs: str
    label: str = ""
    fixed: bool = False
    value: float = 0.0
    free_idx: int = -1


@dataclass
class ModelSpec:
    ov_names: list[str]
    lv_names: list[str]
    params: list[ParamSpec]
    n_free: int
    lambda_map: list[tuple[int, int, int, float]]
    beta_map: list[tuple[int, int, int, float]]
    psi_map: list[tuple[int, int, int, float]]
    theta_map: list[tuple[int, int, int, float]]
    theta_init: np.ndarray
    ordered: bool = False


def parse_lavaan_syntax(
    syntax: str,
    observed_vars: list[str],
    std_lv: bool = False,
    orthogonal: bool = False,
    sample_var: np.ndarray | None = None,
    ordered: bool = False,
) -> ModelSpec:
    """Analisa sintaxe estilo lavaan e constrói os mapas de parâmetros livres e fixos."""
    clean_lines = []
    for raw_line in syntax.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            clean_lines.extend([part.strip() for part in line.split(";") if part.strip()])

    raw_specs: list[ParamSpec] = []
    lv_order: list[str] = []
    ov_set = set(observed_vars)

    op_pattern = re.compile(r"^(.+?)\s*(=~|~~|~)\s*(.+)$")

    for line in clean_lines:
        match = op_pattern.match(line)
        if not match:
            raise ValueError(f"Linha inválida na sintaxe do modelo: '{line}'")
        lhs, op, rhs_str = match.group(1).strip(), match.group(2), match.group(3).strip()

        if op == "=~" and lhs not in lv_order and lhs not in ov_set:
            lv_order.append(lhs)

        terms = [t.strip() for t in rhs_str.split("+") if t.strip()]
        for term in terms:
            label = ""
            fixed = False
            value = np.nan

            if "*" in term:
                mod, var_name = [x.strip() for x in term.split("*", 1)]
                if mod.upper() == "NA":
                    fixed = False
                else:
                    try:
                        value = float(mod)
                        fixed = True
                    except ValueError:
                        label = mod
                        fixed = False
            else:
                var_name = term

            if op == "=~" and lhs not in lv_order and lhs not in ov_set:
                lv_order.append(lhs)
            if op in ("~", "~~"):
                for candidate in (lhs, var_name):
                    if candidate not in ov_set and candidate not in lv_order:
                        lv_order.append(candidate)

            raw_specs.append(
                ParamSpec(lhs=lhs, op=op, rhs=var_name, label=label, fixed=fixed, value=value)
            )

    ov_names = list(observed_vars)
    lv_names = lv_order
    ov_idx = {name: i for i, name in enumerate(ov_names)}
    lv_idx = {name: i for i, name in enumerate(lv_names)}

    endogenous_lvs = {s.lhs for s in raw_specs if s.op == "~" and s.lhs in lv_idx}

    first_indicator_seen: set[str] = set()
    for s in raw_specs:
        if s.op == "=~" and s.lhs in lv_idx:
            if s.lhs not in first_indicator_seen:
                first_indicator_seen.add(s.lhs)
                if not std_lv and not s.fixed and s.label == "" and np.isnan(s.value):
                    s.fixed = True
                    s.value = 1.0
            if not s.fixed and np.isnan(s.value):
                s.value = 0.7

    explicit_pairs = {(s.lhs, s.op, s.rhs) for s in raw_specs}
    explicit_pairs |= {(s.rhs, s.op, s.lhs) for s in raw_specs if s.op == "~~"}

    # 1. Variâncias residuais dos itens observados (fixas/implícitas sob ordered=True delta)
    for i, ov in enumerate(ov_names):
        if (ov, "~~", ov) not in explicit_pairs:
            if ordered:
                raw_specs.append(ParamSpec(lhs=ov, op="~~", rhs=ov, fixed=True, value=0.5))
            else:
                init_v = float(sample_var[i] * 0.5) if sample_var is not None else 0.5
                raw_specs.append(ParamSpec(lhs=ov, op="~~", rhs=ov, fixed=False, value=init_v))

    # 2. Variâncias dos fatores latentes (Psi_kk)
    for lv in lv_names:
        if (lv, "~~", lv) not in explicit_pairs:
            if std_lv and lv not in endogenous_lvs:
                raw_specs.append(ParamSpec(lhs=lv, op="~~", rhs=lv, fixed=True, value=1.0))
            else:
                raw_specs.append(ParamSpec(lhs=lv, op="~~", rhs=lv, fixed=False, value=0.5))

    # 3. Covariâncias entre fatores latentes exógenos (Psi_kl)
    if not orthogonal:
        exo_lvs = [lv for lv in lv_names if lv not in endogenous_lvs]
        for idx_a in range(len(exo_lvs)):
            for idx_b in range(idx_a + 1, len(exo_lvs)):
                lv_a, lv_b = exo_lvs[idx_a], exo_lvs[idx_b]
                if (lv_a, "~~", lv_b) not in explicit_pairs:
                    raw_specs.append(ParamSpec(lhs=lv_a, op="~~", rhs=lv_b, fixed=False, value=0.1))

    label_to_free_idx: dict[str, int] = {}
    free_counter = 0
    for s in raw_specs:
        if s.fixed:
            s.free_idx = -1
        else:
            if np.isnan(s.value):
                s.value = 0.3 if s.op != "~~" else 0.1
            if s.label:
                if s.label not in label_to_free_idx:
                    label_to_free_idx[s.label] = free_counter
                    free_counter += 1
                s.free_idx = label_to_free_idx[s.label]
            else:
                s.free_idx = free_counter
                free_counter += 1

    lambda_map, beta_map, psi_map, theta_map = [], [], [], []
    theta_init = np.zeros(free_counter, dtype=np.float64)

    for s in raw_specs:
        if s.free_idx >= 0:
            theta_init[s.free_idx] = s.value

        if s.op == "=~":
            lambda_map.append((ov_idx[s.rhs], lv_idx[s.lhs], s.free_idx, s.value))
        elif s.op == "~":
            beta_map.append((lv_idx[s.lhs], lv_idx[s.rhs], s.free_idx, s.value))
        elif s.op == "~~":
            if s.lhs in ov_idx and s.rhs in ov_idx:
                theta_map.append((ov_idx[s.lhs], ov_idx[s.rhs], s.free_idx, s.value))
            elif s.lhs in lv_idx and s.rhs in lv_idx:
                psi_map.append((lv_idx[s.lhs], lv_idx[s.rhs], s.free_idx, s.value))

    return ModelSpec(
        ov_names=ov_names,
        lv_names=lv_names,
        params=raw_specs,
        n_free=free_counter,
        lambda_map=lambda_map,
        beta_map=beta_map,
        psi_map=psi_map,
        theta_map=theta_map,
        theta_init=theta_init,
        ordered=ordered,
    )