"""
Module for loading data into the ltu-ili pipeline.
"""

import yaml
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import json
import pandas as pd
from summarizer.dataset import Dataset

try:
    from sbi.simulators.simutils import simulate_in_batches
except ModuleNotFoundError:
    pass


class _BaseLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """

    @classmethod
    def from_config(
            cls, config_path: Path, stage: str = None) -> "_BaseLoader":
        """Create a data loader from a yaml config file

        Args:
            config_path (Path): path to config file.
            stage (str, optional): Data split to load (train, val, or test)
        Returns:
            BaseLoader: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        if stage:
            config['stage'] = stage
        return cls(**config)


class NumpyLoader(_BaseLoader):

    def __init__(self, x, theta) -> None:
        self.x = x
        self.theta = theta
        if len(self.x) != len(self.theta):
            raise Exception(
                "Stored summaries and parameters are not of same length.")

    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        return len(self.x)

    def get_all_data(self) -> np.array:
        """Returns all the loaded summaries

        Returns:
            np.array: summaries
        """
        return self.x

    def get_all_parameters(self):
        """Returns all the loaded parameters

        Returns:
            np.array: parameters
        """
        return self.theta


class StaticNumpyLoader(NumpyLoader):
    """Class to load single numpy files of summaries and parameters

    Args:
        in_dir (str): path to the location of stored data
        x_file (str): filename of the stored summaries
        theta_file (str): filename of the stored parameters
    """

    def __init__(self, in_dir: str, x_file: str, theta_file: str):
        self.in_dir = Path(in_dir)
        self.x_path = self.in_dir / x_file
        self.theta_path = self.in_dir / theta_file

        x = np.load(self.x_path)
        theta = np.load(self.theta_path)

        super().__init__(x=x, theta=theta)


class SummarizerDatasetLoader(_BaseLoader):
    """Class to load netCF files of summaries and a csv of parameters
    Basically a wrapper for ili-summarizer's Dataset, with added
    functionality for loading parameters


    Args:
        stage (str): whether to load train, test or val data
        data_dir (str): path to data directory
        summary_root_file (str): root of summary files
        param_file (str): parameter file name
        train_test_split_file (str): file name where train, test, val
            split idx are stored
        param_names (List[str]): parameters to fit

    Raises:
        Exception: won't work when summaries and parameters don't have
            same length
    """

    def __init__(
        self,
        stage: str,
        data_dir: str,
        summary_root_file: str,
        param_file: str,
        train_test_split_file: str,
        param_names: List[str],
    ):
        self.data_dir = Path(data_dir)
        self.nodes = self.get_nodes_for_stage(
            stage=stage, train_test_split_file=train_test_split_file
        )
        self.data = Dataset(
            nodes=self.nodes,
            path_to_data=self.data_dir,
            root_file=summary_root_file,
        )
        self.theta = self.load_parameters(
            param_file=param_file,
            nodes=self.nodes,
            param_names=param_names,
        )
        if len(self.data) != len(self.theta):
            raise Exception(
                "Stored summaries and parameters are not of same length.")

    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        return len(self.nodes)

    def get_all_data(self) -> np.array:
        """Returns all the loaded summaries

        Returns:
            np.array: summaries
        """
        return self.data.load().reshape((len(self), -1))

    def get_all_parameters(self):
        """Returns all the loaded parameters

        Returns:
            np.array: parameters
        """
        return self.theta

    def get_nodes_for_stage(
            self, stage: str,
            train_test_split_file: str) -> List[int]:
        """Get nodes for a given stage (train, test or val)

        Args:
            stage (str): either train, test or val
            train_test_split_file (str): file where node idx for each stage
                are stored

        Returns:
            List[int]: list of idx for stage
        """
        with open(self.data_dir / train_test_split_file) as f:
            train_test_split = json.load(f)
        return train_test_split[stage]

    def load_parameters(
        self, param_file: str, nodes: List[int], param_names: List[str]
    ) -> np.array:
        """Get parameters for nodes

        Args:
            param_file (str): where to find parameters of latin hypercube
            nodes (List[int]): list of nodes to read
            param_names (List[str]): parameters to use

        Returns:
            np.array: array of parameters
        """
        theta = pd.read_csv(
            self.data_dir / param_file, sep=" ", skipinitialspace=True
        ).iloc[nodes]
        return theta[param_names].values


class SBISimulator(_BaseLoader):
    """
    Class to run simulations of summaries and parameters and save
    results to numpy files. Only works for sbi backend.

    Args:
        in_dir (str): path to the location of stored data
        xobs_file (str): filename used for observed x values
        thetaobs_file (str): filename used for observed parameters
        out_dir (str): path to the location where to save  data
        x_file (str): filename to use to store summaries
        theta_file (str): filename to use to store parameters
        num_simulations (int): number of simulations to run at each call
        simulator (callable): function taking the parameters as an
            argument and returns data
    """

    def __init__(
            self,
            in_dir: str,
            xobs_file: str,
            thetaobs_file: str,
            out_dir: str,
            x_file: str,
            theta_file: str,
            num_simulations: int,
            simulator: Optional[callable] = None,
    ):
        self.in_dir = Path(in_dir)
        self.xobs_path = self.in_dir / xobs_file
        self.thetaobs_path = self.in_dir / thetaobs_file
        self.out_dir = Path(out_dir)
        self.x_path = self.out_dir / x_file
        self.theta_path = self.out_dir / theta_file
        self.num_simulations = num_simulations
        self.simulator = simulator

        self.xobs = np.load(self.xobs_path)
        self.thetaobs = np.load(self.thetaobs_path)

        self.theta = None
        self.x = None

    def __len__(self) -> int:
        """Returns the total number of data points produced when called

        Returns:
            int: length of dataset
        """
        return self.num_simulations

    def set_simulator(self, simulator: callable):
        """Set the simulator to be used in the inference

        Args:
            simulator (callable): function taking the parameters as an
                argument and returns data
        """
        self.simulator = simulator

    def simulate(self, proposal: Any) -> Tuple[np.array, np.array]:
        """Run simulations give a proposal and returns ($\theta, x$) pairs
        obtained from sampling the proposal and simulating.

        Args:
            proposal (Any): Distribution to sample paramaters from

        Returns:
            Tuple[np.array, np.array]: Sampled parameters $\theta$ and
                simulation-outputs $x$.
        """
        theta = proposal.sample((self.num_simulations,)).cpu()
        x = simulate_in_batches(self.simulator, theta)
        theta, x = theta.numpy(), x.numpy()
        if self.theta is None or self.x is None:
            self.theta, self.x = theta, x
        else:
            self.theta = np.concatenate((self.theta, theta))
            self.x = np.concatenate((self.x, x))
        np.save(self.theta_path, self.theta)
        np.save(self.x_path, self.x)
        return theta, x

    def get_obs_data(self) -> np.array:
        """Returns the observed summaries

        Returns:
            np.array: summaries
        """
        return self.xobs

    def get_obs_parameters(self):
        """Returns the observed parameters

        Returns:
            np.array: parameters
        """
        return self.thetaobs

    def get_all_data(self) -> np.array:
        """Returns all the loaded summaries

        Returns:
            np.array: summaries
        """
        return self.x

    def get_all_parameters(self):
        """Returns all the loaded parameters

        Returns:
            np.array: parameters
        """
        return self.theta


# TODO: Add loaders which load dynamically from many files, so
# that everything doesn't need to be stored in memory

# TODO: Add loaders which load from initialization
