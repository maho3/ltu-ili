from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from summarizer.dataset import Dataset
from typing import List
import pandas as pd

class BaseLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """

class StaticNumpyLoader(BaseLoader):
    def __init__(
        self,
        in_dir: str,
        x_file: str,
        theta_file: str
    ):
        """Class to load single numpy files of summaries and parameters

        Args:
            in_dir (str): path to the location of stored data
            x_file (str): filename of the stored summaries
            theta_file (str): filename of the stored parameters
        """
        self.in_dir = Path(in_dir)
        self.x_path = self.in_dir / x_file
        self.theta_path = self.in_dir / theta_file

        self.x = np.load(self.x_path)
        self.theta = np.load(self.theta_path)

        if len(self.x) != len(self.theta):
            raise Exception('Stored summaries and parameters are not of same length.')

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

class SummarizerDatasetLoader(BaseLoader):
    def __init__(
        self,
        num_nodes: str,
        in_dir: str,
        root_file: str,
        param_file: str,
        param_names: List[str]
    ):
        """Class to load netCF files of summaries and a csv of parameters
        Basically a wrapper for ili-summarizer's Dataset, with added parameter loading

        Args:
            in_dir (str): path to the location of stored data
        """
        self.num_nodes = num_nodes
        self.in_dir = Path(in_dir)

        self.dat = Dataset(
            nodes=range(self.num_nodes),
            path_to_data=self.in_dir,
            root_file=root_file,
        )

        self.theta = pd.read_csv(
            self.in_dir / param_file,
            sep=' ',
            skipinitialspace=True
        )
        self.theta = self.theta[param_names]

        if self.num_nodes != len(self.theta):
            raise Exception('Stored summaries and parameters are not of same length.')

    def __len__(self) -> int:
        """Returns the total number of data points in the dataset

        Returns:
            int: length of dataset
        """
        return self.num_nodes

    def get_all_data(self) -> np.array:
        """Returns all the loaded summaries

        Returns:
            np.array: summaries
        """
        return self.dat.load().reshape((self.num_nodes,-1))

    def get_all_parameters(self):
        """Returns all the loaded parameters

        Returns:
            np.array: parameters
        """
        return self.theta.values

# TODO: Add loaders which load dynamically from many files
