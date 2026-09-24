from .dataset import Dataset as Dataset
from .import_utils import load_class as load_class
from .import_utils import load_from_config as load_from_config
from .import_utils import update as update

loaded = False
try:
    from .distributions_pt import IndependentBeta as IndependentBeta
    from .distributions_pt import IndependentCauchy as IndependentCauchy
    from .distributions_pt import IndependentChi2 as IndependentChi2
    from .distributions_pt import (
        IndependentExponential as IndependentExponential,
    )
    from .distributions_pt import (
        IndependentFisherSnedecor as IndependentFisherSnedecor,
    )
    from .distributions_pt import IndependentGamma as IndependentGamma
    from .distributions_pt import IndependentGumbel as IndependentGumbel
    from .distributions_pt import (
        IndependentHalfCauchy as IndependentHalfCauchy,
    )
    from .distributions_pt import (
        IndependentHalfNormal as IndependentHalfNormal,
    )
    from .distributions_pt import IndependentLaplace as IndependentLaplace
    from .distributions_pt import IndependentLogNormal as IndependentLogNormal
    from .distributions_pt import IndependentNormal as IndependentNormal
    from .distributions_pt import IndependentPareto as IndependentPareto
    from .distributions_pt import IndependentStudentT as IndependentStudentT
    from .distributions_pt import (
        IndependentTruncatedNormal as IndependentTruncatedNormal,
    )
    from .distributions_pt import IndependentVonMises as IndependentVonMises
    from .distributions_pt import IndependentWeibull as IndependentWeibull
    from .distributions_pt import (
        LowRankMultivariateNormal as LowRankMultivariateNormal,
    )
    from .distributions_pt import MultivariateNormal as MultivariateNormal
    from .distributions_pt import Uniform as Uniform
    from .ndes_pt import LampeEnsemble as LampeEnsemble
    from .ndes_pt import LampeNPE as LampeNPE
    from .ndes_pt import load_nde_lampe as load_nde_lampe
    from .ndes_pt import load_nde_sbi as load_nde_sbi
    loaded = True
except ImportError:
    pass

try:
    from .distributions_tf import IndependentNormal as IndependentNormal
    from .distributions_tf import (
        IndependentTruncatedNormal as IndependentTruncatedNormal,
    )
    from .distributions_tf import (
        MultivariateTruncatedNormal as MultivariateTruncatedNormal,
    )
    from .distributions_tf import Uniform as Uniform
    from .ndes_tf import load_nde_pydelfi as load_nde_pydelfi
    loaded = True
except ImportError:
    pass
if not loaded:
    raise ImportError("Neither Pytorch nor Tensorflow installed. "
                      "Cannot import distributions or ndes.")
