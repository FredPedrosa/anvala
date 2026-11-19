import os
import pytest

# Configura o R 4.6.1 no Windows antes de importar o rpy2
r_home = r"C:\Program Files\R\R-4.6.1"
r_bin = os.path.join(r_home, "bin", "x64")
os.environ["R_HOME"] = r_home
os.environ["PATH"] = r_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["LANGUAGE"] = "en"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r_bin)

import rpy2.robjects as ro
from rpy2.robjects.packages import importr


@pytest.fixture(scope="session")
def r_env():
    """Mantém uma sessão única do R ativa durante toda a bateria de testes."""
    lavaan = importr("lavaan")
    bifactor_calc = importr("BifactorIndicesCalculator")
    return {"ro": ro, "lavaan": lavaan, "bifactor": bifactor_calc}