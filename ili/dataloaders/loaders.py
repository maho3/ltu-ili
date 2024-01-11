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
import logging
import os
from ili.utils import Dataset

try:
    from sbi.simulators.simutils import simulate_in_batches
except ModuleNotFoundError:
    pass


class _BaseLoader(ABC):
    @classmethod
    def from_config(
        cls,
        config_path: Path,
        stage: str = None
    ) -> "_BaseLoader":
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

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        return NotImplemented

    @abstractmethod
    def get_all_data(self) -> Any:
        """Returns all the loaded data

        Returns:
            Any: data
        """
        return NotImplemented

    @abstractmethod
    def get_all_parameters(self) -> Any:
        """Returns all the loaded parameters

        Returns:
            Any: parameters
        """
        return NotImplemented

    @abstractmethod
    def get_obs_data(self) -> Any:
        """Returns the observed data

        Returns:
            Any: data
        """
        return NotImplemented

    @abstractmethod
    def get_fid_parameters(self) -> Any:
        """Returns the fiducial parameters which we expect the
        observed data to resemble

        Returns:
            Any: parameters
        """
        return NotImplemented


class NumpyLoader(_BaseLoader):
    """A class for loading in-memory data using numpy arrays.

    Args:
        x (np.array): Array of training data of
            shape (Ndata, \*data.shape)
        theta (np.array): Array of training parameters of
            shape (Ndata, \*parameters.shape)
        xobs (Optional[np.array]): Array of observed data of
            shape (\*data.shape). Defaults to None.
        thetafid (Optional[np.array]): Array of fiducial
            parameters of shape (\*parameters.shape). Defaults to None.
    """

    def __init__(
        self,
        x: np.array,
        theta: np.array,
        xobs: Optional[np.array] = None,
        thetafid: Optional[np.array] = None
    ) -> None:
        self.x = x
        self.theta = theta
        if len(self.x) != len(self.theta):
            raise Exception(
                "Stored data and parameters are not of same length.")
        self.xobs = xobs
        self.thetafid = thetafid

    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        if self.x is None:
            return 0
        return len(self.x)

    def get_all_data(self) -> np.array:
        """Returns all the loaded data for training

        Returns:
            np.array: data
        """
        return self.x

    def get_all_parameters(self):
        """Returns all the loaded parameters for training

        Returns:
            np.array: parameters
        """
        return self.theta

    def get_obs_data(self) -> np.array:
        """Returns the observed data

        Returns:
            np.array: data
        """
        return self.xobs

    def get_fid_parameters(self):
        """Returns the fiducial parameters which we expect the
        observed data to resemble

        Returns:
            np.array: parameters
        """
        return self.thetafid


class StaticNumpyLoader(NumpyLoader):
    """Loads single numpy files of data and parameters from disk

    Args:
        in_dir (str): path to the location of stored data
        x_file (str): filename of the stored training data
        theta_file (str): filename of the stored training parameters
        xobs_file (Optional[str]): filename used for observed x values
        thetafid_file (Optional[str]): filename used for fiducial parameters
    """

    def __init__(
        self,
        in_dir: str,
        x_file: str,
        theta_file: str,
        xobs_file: Optional[str] = None,
        thetafid_file: Optional[str] = None
    ) -> None:
        self.in_dir = Path(in_dir)
        self.x_path = self.in_dir / x_file
        self.theta_path = self.in_dir / theta_file

        # Load stored data (if specified)
        x = np.load(self.x_path)
        theta = np.load(self.theta_path)
        if xobs_file is None:
            self.xobs_path = None
            xobs = None
        else:
            self.xobs_path = self.in_dir / xobs_file
            xobs = np.load(self.xobs_path)
        if thetafid_file is None:
            self.thetafid_path = None
            thetafid = None
        else:
            self.thetafid_path = self.in_dir / thetafid_file
            thetafid = np.load(self.thetafid_path)

        super().__init__(x=x, theta=theta, xobs=xobs, thetafid=thetafid)


class SBISimulator(NumpyLoader):
    """
    Class to run simulations of data and parameters and save
    results to numpy files. Only works for sbi backend.

    Args:
        in_dir (str): path to the location of stored data
        xobs_file (str): filename used for observed x values
        num_simulations (int): number of simulations to run at each call
        simulator (callable): function taking the parameters as an
            argument and returns data
        save_simulated (Optional[bool]): whether to save simulated data.
            Concatenates to previous data if True. Defaults to False.
        x_file (Optional[str]): filename of the stored first-round
            training data
        theta_file (Optional[str]): filename of the stored first-round
            training parameters
        thetafid_file (Optional[str]): filename used for fiducial parameters
    """

    def __init__(
        self,
        in_dir: str,
        xobs_file: str,
        num_simulations: int,
        simulator: Optional[callable] = None,
        save_simulated: Optional[bool] = False,
        x_file: Optional[str] = None,
        theta_file: Optional[str] = None,
        thetafid_file: Optional[str] = None,
    ):
        self.in_dir = Path(in_dir)
        self.xobs_path = self.in_dir / xobs_file
        self.num_simulations = num_simulations
        self.simulator = simulator
        self.save_simulated = save_simulated

        # If save_simulated, check that x_file and theta_file are specified
        if save_simulated and (x_file is None or theta_file is None):
            raise Exception(
                "If save_simulated is True, x_file and theta_file must be "
                "specified."
            )

        # Load stored data (if specified)
        xobs = np.load(self.xobs_path)
        x = np.array([])
        theta = np.array([])
        thetafid = None
        if x_file is None:
            self.x_path = None
        else:
            self.x_path = self.in_dir / x_file
            if self.x_path.is_file():
                x = np.load(self.x_path)
        if theta_file is None:
            self.theta_path = None
        else:
            self.theta_path = self.in_dir / theta_file
            if self.theta_path.is_file():
                theta = np.load(self.theta_path)
        if thetafid_file is None:
            self.thetafid_path = None
        else:
            self.thetafid_path = self.in_dir / thetafid_file
            thetafid = np.load(self.thetafid_path)

        super().__init__(x=x, theta=theta, xobs=xobs, thetafid=thetafid)

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
        print(self.x.shape, x.shape)

        # Save simulated data (concatenates to previous data)
        if len(self) == 0:
            self.theta, self.x = theta, x
        else:
            self.theta = np.concatenate((self.theta, theta))
            self.x = np.concatenate((self.x, x))
        if self.save_simulated:
            np.save(self.theta_path, self.theta)
            np.save(self.x_path, self.x)

        return theta, x


class SummarizerDatasetLoader(NumpyLoader):
    """Class to load netCF files of data and a csv of parameters
    Basically a wrapper for ili-summarizer's Dataset, with added
    functionality for loading parameters

    Args:
        in_dir (str): path to data directory
        stage (str): whether to load train, test or val data
        x_root (str): root of data files
        theta_file (str): parameter file name
        train_test_split_file (str): file name where train, test, val
            split idx are stored
        param_names (List[str]): parameters to fit
        xobs_file (Optional[str]): filename used for observed x values
        thetafid_file (Optional[str]): filename used for fiducial parameters

    Raises:
        Exception: won't work when data and parameters don't have
            same length
    """

    def __init__(
        self,
        in_dir: str,
        stage: str,
        x_root: str,
        theta_file: str,
        train_test_split_file: str,
        param_names: List[str],
        xobs_file: Optional[str] = None,
        thetafid_file: Optional[str] = None
    ):
        self.in_dir = Path(in_dir)
        self.nodes = self.get_nodes_for_stage(
            stage=stage, train_test_split_file=train_test_split_file
        )
        self.x = Dataset(
            nodes=self.nodes,
            path_to_data=self.in_dir,
            root_file=x_root,
        )
        self.theta = self.load_parameters(
            param_file=theta_file,
            nodes=self.nodes,
            param_names=param_names,
        )
        if len(self.x) != len(self.theta):
            raise Exception(
                "Stored data and parameters are not of same length.")

        if xobs_file is None:
            self.xobs_path = None
            self.xobs = None
        else:
            self.xobs_path = self.in_dir / xobs_file
            self.xobs = np.load(self.xobs_path)
        if thetafid_file is None:
            self.thetafid_path = None
            self.thetafid = None
        else:
            self.thetafid_path = self.in_dir / thetafid_file
            self.thetafid = np.load(self.thetafid_path)

    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        return len(self.nodes)

    def get_all_data(self) -> np.array:
        """Returns all the loaded data

        Returns:
            np.array: data
        """
        return self.x.load().reshape((len(self), -1))

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
        with open(self.in_dir / train_test_split_file) as f:
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
            self.in_dir / param_file, sep=" ", skipinitialspace=True
        ).iloc[nodes]
        return theta[param_names].values

# TODO: Add loaders which load dynamically from many files, so
# that everything doesn't need to be stored in memory
