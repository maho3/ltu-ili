try:
    from .runner_sbi import *
except ModuleNotFoundError:
    from .pydelfi_wrappers import *
    from .runner_pydelfi import *
