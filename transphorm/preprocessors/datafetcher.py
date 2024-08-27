from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, NewType, Protocol

import h5py
import numpy as np
import polars as pl
import yaml

GuppyOuputPath = NewType("GuppyOuputPath", Path)


# set up logger for data fetcher
logger = logging.getLogger("DataFetcher")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# read data functions and strategy used by fetchers


def get_ts_cols(df):
    ts_list = ["CueA", "ShkA", "AvdA", "EspA", "CueB", "ShkB", "AvdB", "EspB"]
    columns = df.columns
    return np.intersect1d(ts_list, columns).tolist()


def compute_hot_encoded_ts(
    df: pl.DataFrame, col: str, tolerance: float = 0.00045
) -> np.array:
    # fetch time array
    time = df["time"].to_numpy()
    # filter col timestamps and create np array
    col_timestamps = df.filter(pl.col(col) > 0)[col].to_numpy()
    # create matrix of tolerance values
    diff = time[:, np.newaxis] - col_timestamps
    encoded_ts = np.any(np.abs(diff) <= tolerance, axis=1).astype(int)
    return encoded_ts


def create_encoded_ts_df(df):
    og_ts = get_ts_cols(df)
    new_cols = [f"{ts}_encoded" for ts in og_ts]
    encoded_ts_cols = {
        new_ts: compute_hot_encoded_ts(df, ts) for new_ts, ts in zip(new_cols, og_ts)
    }
    encoded_ts_cols["time"] = df["time"]
    timestamps = pl.DataFrame(encoded_ts_cols)
    return df.join(timestamps, on="time")


def filter_metadata_keys_for_keywords(*keywords: str, metadata: dict) -> List[str]:
    """filters dictionary for keyowords in the keys and returns a flat list of the values of the filtered keys"""
    filtered_data = {
        k: v for k, v in metadata.items() if any(kw in k for kw in keywords)
    }
    filtered_values = list(filtered_data.values())
    # returns flat list
    return [i for item in filtered_values for i in item]


def read_hdf5_event_timestamps_data(path: Path) -> np.array:
    """reads the timestamps from the hdf5 file and returns them as a numpy array"""
    with h5py.File(path, "r") as f:
        raw_timestamps = np.array(f.get("timestamps"))
        timestamps = raw_timestamps
        real_timestamps = timestamps - timestamps[0]
    return real_timestamps


def read_hdf5_fp_signal_data(path: Path) -> np.array:
    """reads the data from the hdf5 file and returns it as a numpy array"""
    with h5py.File(path, "r") as f:
        data = np.array(f.get("data"))
    return data


def get_max_value_length(data: Dict[str, np.array]) -> int:
    """takes in a dictionary with np.arrays as values and returns the length of the longest array"""
    return max(v.shape[0] for v in data.values())


def pad_dict_arrays(
    data_dict: Dict[str, np.array], pad_val=-100
) -> Dict[str, np.array]:
    """formats the dictionary of data to have the same length"""
    max_length = get_max_value_length(data_dict)
    padded_dict = {
        k: np.pad(v, (0, max_length - v.shape[0]), constant_values=-100)
        for k, v in data_dict.items()
    }
    return padded_dict


#! this should probably be moved to the main analysis module
guppy_read_strategies = {
    "recording": read_hdf5_fp_signal_data,
    "event": read_hdf5_event_timestamps_data,
}


class DataFetcher(Protocol): ...


def data_fetcher_factory(fetcher, path):
    return fetcher(path)


class GuppyDataFetcher:
    """
    A class used to extract data from directories outputted by Guppy.

    This class provides methods to load metadata from a 'metadata.yaml' file,
    and to manage time series data associated with the Guppy output.

    Attributes
    ----------
        output_path (Path): The path to the Guppy output directory.
        read_strategy (GuppyReadStrategies): The strategy to use for reading data.
        _metadata (dict): The metadata loaded from 'metadata.yaml'.
        _timeseries_dict (dict): A dictionary of time series data.
        _timeseries_dataframe (DataFrame): A DataFrame representation of the time series data.
        _timeseries_paths (list): A list of paths to the time series data files.
        max_length (int): The maximum length of the time series data.

    Methods
    -------
        _load_metadata(self) -> Dict | None:
            Load the metadata from the 'metadata.yaml' file.
        _instatntiate_path_objs(self, metadata: dict) -> Dict[str, Any]:
            Instantiate Path objects from the metadata.
        _get_strategy(self, stem: str) -> Callable:
            Get the read strategy based on the file stem.
        load_data_from_path(self, file_path: Path) -> Dict[str, np.array]:
            Load data from a single file in the Guppy output directory.

    """

    def __init__(self, path_to_metadata: Path, read_strategy=guppy_read_strategies):
        self.path_to_metadata = path_to_metadata
        self.read_strategy = read_strategy
        self._metadata = None
        self._timeseries_dict = None
        self._timeseries_dataframe = None
        self._timeseries_paths = None

    def _instatntiate_path_objs(self, metadata: dict) -> Dict[str, Any]:
        """finds keys with the word "path" in it and iterats its value to return a list of Path objects"""
        for key in metadata.keys():
            if "path" in key:
                metadata[key] = [Path(p) for p in metadata[key]]
        return metadata

    def load_metadata(self) -> Dict | None:
        """searches the output_path pointer for file named 'metadata.yaml',
        if found loads yaml into dict and instatiates path objects if there are keywords with the word path
        otherwise raises error and returns none: interally used in the metadata property
        """

        try:
            with open(self.path_to_metadata.as_posix()) as f:
                raw_metadata = yaml.load(f, Loader=yaml.FullLoader)
                metadata = self._instatntiate_path_objs(raw_metadata)
                return metadata

        except FileNotFoundError as e:
            logger.error(
                f"metadata.yaml not found in {self.path_to_metadata.name}, returned None"
            )
            return None

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.load_metadata()
        return self._metadata

    @property
    def timeseries_dict(self):
        if self._timeseries_dict is None:
            self._timeseries_dict = {}
        return self._timeseries_dict

    def _get_strategy(self, stem: str) -> str:
        """returns the read strategy for the file based on the stem of the file"""
        if "z_score" in stem or "dff" in stem:
            return "recording"
        else:
            return "event"

    def _load_data_from_path(self, file_path: Path) -> Dict[str, np.array]:
        """
        reads data from a single file in the guppy output directory. method for reading is defined by the read_strategy_type
        first checks that read_strategy_type is either 'recording' or 'event'
        raises AssertionError if not
        returns data from the file as np.array

        """

        name = file_path.stem
        read_strategy_key = self._get_strategy(name)

        data = self.read_strategy[read_strategy_key](file_path)
        self.timeseries_dict[name] = data

    # fetch files to read
    def load_timeseries_data(self):
        paths_to_timeseries = filter_metadata_keys_for_keywords(
            "paths", metadata=self.metadata
        )
        list(map(self._load_data_from_path, paths_to_timeseries))
        # load metadata dataframe

    def load_categorical_data_into_ts_dataframe(self):
        cat_dict = {k: v for k, v in self.metadata.items() if "path" not in k}
        cat_df = pl.DataFrame(cat_dict)
        updated_df = pl.concat(
            [self.timeseries_dataframe, cat_df], how="horizontal"
        ).select(pl.all().forward_fill())
        self.timeseries_dataframe = updated_df

    def load_timeseries_dataframe(self):
        if self._timeseries_dataframe is None:
            padded_dict = pad_dict_arrays(self._timeseries_dict)

            self._timeseries_dataframe = pl.DataFrame(padded_dict)
        return self._timeseries_dataframe

    def compute_time(self):
        self.timeseries_dataframe = self.timeseries_dataframe.with_columns(
            time=np.linspace(
                0,
                self.timeseries_dataframe.shape[0] / 1070,
                self.timeseries_dataframe.shape[0],
            )
        )

    def encode_timestamps(self):
        self.timeseries_dataframe = create_encoded_ts_df(self.timeseries_dataframe)

    @property
    def timeseries_dataframe(self):
        try:
            assert self._timeseries_dataframe is not None
        except AssertionError:
            raise AssertionError(
                "you have not loaded the dataframe yet, call load_timeseries_dataframe first"
            )
        return self._timeseries_dataframe

    @timeseries_dataframe.setter
    def timeseries_dataframe(self, val):
        self._timeseries_dataframe = val

    def write_to_parquet(self):
        dir_to_save = self.path_to_metadata.parents[1] / "compiled_timeseries_data"
        dir_to_save.mkdir(exist_ok=True)
        file_stem = re.sub(
            "metadata", "compiled_timeseries_data", self.path_to_metadata.stem
        )
        file_name = f"{file_stem}.parquet"
        self.timeseries_dataframe.write_parquet(dir_to_save / file_name)


def guppy_processing_strategy(fetcher: GuppyDataFetcher):
    fetcher.load_timeseries_data()
    fetcher.load_timeseries_dataframe()
    fetcher.load_categorical_data_into_ts_dataframe()
    fetcher.compute_time()
    fetcher.encode_timestamps()
    fetcher.write_to_parquet()
