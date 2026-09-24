from .dataset import Dataset
from .import_utils import load_class, load_from_config, update

loaded = False
try:
    from .distributions_pt import (
        IndependentBeta,
        IndependentCauchy,
        IndependentChi2,
        IndependentExponential,
        IndependentFisherSnedecor,
        IndependentGamma,
        IndependentGumbel,
        IndependentHalfCauchy,
        IndependentHalfNormal,
        IndependentLaplace,
        IndependentLogNormal,
        IndependentNormal,
        IndependentPareto,
        IndependentStudentT,
        IndependentTruncatedNormal,
        IndependentVonMises,
        IndependentWeibull,
        LowRankMultivariateNormal,
        MultivariateNormal,
        Uniform,
    )
    from .ndes_pt import LampeEnsemble, LampeNPE, load_nde_lampe, load_nde_sbi
    loaded = True
except ImportError:
    pass

try:
    from .distributions_tf import (
        IndependentNormal,
        IndependentTruncatedNormal,
        MultivariateTruncatedNormal,
        Uniform,
    )
    from .ndes_tf import load_nde_pydelfi
    loaded = True
except ImportError:
    pass
if not loaded:
    raise ImportError("Neither Pytorch nor Tensorflow installed. "
                      "Cannot import distributions or ndes.")
