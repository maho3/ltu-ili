"""
Module to read summaries from disk following the ili-summarizer Dataset
convention.

Forked code from https://github.com/florpi/ili-summarizer/blob/main/summarizer/dataset.py
"""

from pathlib import Path
from typing import List, Dict
import xarray as xr
import numpy as np


class Dataset:
    def __init__(
        self,
        nodes: List[int],
        path_to_data: Path,
        root_file: str,
        islice_filters: Dict = None,
        slice_filters: Dict = None,
        select_filters: Dict = None,
    ):
        """Read dataset of summaries

        Args:
            nodes (List[int]): list of nodes to read 
            path_to_data (Path): path to where summaries are stored
            root_file (str): root file for summaries to be read
            islice_filters (Dict): dictionary of filters to slice on the summaries' coordinates indices
            slice_filters (Dict): dictionary of filters to slice on the summaries' coordinates
            select_filters (Dict): dictionary of filters to select on the summaries' coordinates
        """
        self.nodes = nodes
        self.path_to_data = path_to_data
        self.root_file = root_file
        if islice_filters is not None:
            self.islice_filters = self.transform_filters_to_islices(
                islice_filters)
        else:
            self.islice_filters = None
        if slice_filters is not None:
            self.slice_filters = self.transform_filters_to_slices(
                slice_filters)
        else:
            self.slice_filters = None
        self.select_filters = select_filters
        self.summaries = self.load()

    def __len__(self,) -> int:
        return len(self.nodes)

    def transform_filters_to_slices(self, filters: Dict) -> Dict:
        """Transform a dictionary of filters into slices that select from min to max

        Args:
            filters (Dict): dictionary of filters. Example:
                filters = {'r': (10,100)} , will select the summary statistics for 10 < r < 100

        Returns:
            Dict: dictionary of filters with slices
        """
        for filter, (min, max) in filters.items():
            filters[filter] = slice(min, max)
        return filters

    def transform_filters_to_islices(self, filters: Dict) -> Dict:
        """Transform a dictionary of filters into slices that select from min index to max index in step

        Args:
            filters (Dict): dictionary of filters. Example:
                filters = {'r': (2,-1 ,2)} , will select the summary statistics for indices r[2] to r[-1] in steps of 2

        Returns:
            Dict: dictionary of filters with slices
        """
        for filter, (min, max, step) in filters.items():
            filters[filter] = slice(min, max, step)
        return filters

    def load_summary(self, node: int,) -> xr.DataArray:
        """Load the summary for a particular node

        Args:
            node (int): node to load

        Returns:
            xr.DataArray: data array with coordinates and summary value
        """
        summary = xr.open_dataarray(
            self.path_to_data / f'{self.root_file}_node{node}.nc')
        if self.islice_filters:
            summary = summary.isel(**self.islice_filters)
        if self.slice_filters:
            summary = summary.sel(**self.slice_filters)
        if self.select_filters:
            summary = summary.sel(**self.select_filters)
        return summary

    def load(self,) -> np.array:
        """Load all summaries as a numpy array

        Returns:
            np.array: array of summaries
        """
        summaries = []
        for node in self.nodes:
            summaries.append(
                np.array(self.load_summary(node=node))
            )
        return np.array(summaries)
