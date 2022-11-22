import numpy as np
from pathlib import Path

class StaticNumpyLoader():
    def __init__(
        self,
        in_dir,
        x_file,
        theta_file
    ):
        """Class to load single numpy files of summaries and parameters.

        """
        self.in_dir = Path(in_dir)
        self.x_path = self.in_dir / x_file
        self.theta_path = self.in_dir / theta_file

        self.x = np.load(self.x_path)
        self.theta = np.load(self.theta_path)

        if len(self.x) != len(self.theta):
            raise Exception('Stored data and parameters are not of same length.')

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i <= len(self):
            return self[i]
        else:
            raise StopIteration

    def __getitem__(self, i):
        return self.x[i], self.theta[i]

    def get_all_data(self):
        return self.x

    def get_all_parameters(self):
        return self.theta

# TODO: Add loaders which load dynamically from many files
