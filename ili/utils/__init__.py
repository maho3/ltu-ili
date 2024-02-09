from .import_utils import load_class, load_from_config, update
from .dataset import Dataset

loaded = False
try:
    from .distributions_pt import (
        Uniform, IndependentNormal, IndependentBeta, IndependentCauchy,
        IndependentChi2, IndependentExponential, IndependentFisherSnedecor,
        IndependentGamma, IndependentGumbel, IndependentHalfCauchy,
        IndependentHalfNormal, IndependentLaplace,
        IndependentLogNormal, IndependentPareto, IndependentStudentT,
        IndependentVonMises, IndependentWeibull,
        MultivariateNormal, LowRankMultivariateNormal,
        IndependentTruncatedNormal
    )
    from .ndes_pt import load_nde_sbi, load_nde_lampe, LampeNPE, LampeEnsemble
    loaded = True
except ImportError:
    pass

try:
    from .distributions_tf import (
        Uniform, IndependentNormal,
        MultivariateTruncatedNormal, IndependentTruncatedNormal
    )
    from .ndes_tf import load_nde_pydelfi
    loaded = True
except ImportError:
    pass
if not loaded:
    raise ImportError("Neither Pytorch nor Tensorflow installed. "
                      "Cannot import distributions or ndes.")
