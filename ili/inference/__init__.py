try:
    from .runner_lampe import LampeRunner as LampeRunner
    from .runner_sbi import ABCRunner as ABCRunner
    from .runner_sbi import SBIRunner as SBIRunner
    from .runner_sbi import SBIRunnerSequential as SBIRunnerSequential
except ModuleNotFoundError:
    from .runner_pydelfi import DelfiRunner as DelfiRunner
from .runner import InferenceRunner as InferenceRunner
