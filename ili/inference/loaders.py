from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

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

# TODO: Add loaders which load dynamically from many files
