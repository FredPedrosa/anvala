from anvala.bifactor import bifactor_indices
from anvala.cfa import AnvalaFit, cfa
from anvala.efa import DimTestsResult, EFAFit, dimtests, efa, kmo_bartlett
from anvala.ega import (
    BootEGAResult,
    EGAResult,
    GlassoResult,
    UVAResult,
    boot_ega,
    compute_net_loads,
    compute_tefi,
    ebic_glasso,
    ega,
    uva,
)
from anvala.polychoric import PolychoricResult, polychoric
from anvala.reliability import ReliabilityResult, cronbach_alpha, htmt, reliability

__all__ = [
    "cfa",
    "AnvalaFit",
    "efa",
    "EFAFit",
    "dimtests",
    "DimTestsResult",
    "kmo_bartlett",
    "reliability",
    "ReliabilityResult",
    "htmt",
    "cronbach_alpha",
    "bifactor_indices",
    "polychoric",
    "PolychoricResult",
    "ebic_glasso",
    "GlassoResult",
    "ega",
    "EGAResult",
    "uva",
    "UVAResult",
    "boot_ega",
    "BootEGAResult",
    "compute_net_loads",
    "compute_tefi",
]

from anvala.cfa import AnvalaFit
from anvala.diagnostics import modindices, residuals
from anvala.sem import SEMFit, sem

AnvalaFit.residuals = lambda self, type="raw": residuals(self, type=type)
AnvalaFit.modindices = lambda self, sort=True, minimum_value=0.0, op=None: modindices(self, sort=sort, minimum_value=minimum_value, op=op)

from anvala.multigroup import (
    InvarianceResult,
    MultiGroupFit,
    cfa_multigroup,
    compare_models,
    measurement_invariance,
)

from anvala.predict import predict

AnvalaFit.predict = predict
SEMFit.predict = predict
MultiGroupFit.predict = predict

from anvala.bootstrap import BootstrapResult, bootstrap_ci

AnvalaFit.bootstrap_ci = bootstrap_ci
SEMFit.bootstrap_ci = bootstrap_ci