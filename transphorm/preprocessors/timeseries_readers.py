from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def label_learned(x: int, threshold=80) -> Literal[0, 1]:
    """
    labels learned or not learned as binary 0 or if over threshold
    """
    if x >= threshold:
        return 1
    else:
        return 0


def combine_data(fp_df, percent_avoid_df) -> pl.DataFrame:
    """
    takes in two dataframes and combines them on day, cage, and mouse_id
    """

    combined_data = fp_df.join(
        percent_avoid_df, on=["day", "cage", "mouse_id"], how="left"
    ).with_columns(
        pl.col("perc_avoid")
        .map_elements(lambda x: 1 if x >= 80 else 0, return_dtype=pl.Int64)
        .alias("learned")
    )
    return combined_data


def get_arrays(df: pl.DataFrame) -> np.array:
    """takes in a dataframe and returns a numpy array of the learned column and the z_score_DA column only"""
    learned = df.select("learned").collect().to_numpy()[0]
    da_signal = df.select("z_score_DA").collect().to_numpy()
    combined_array = np.append(learned, da_signal)
    return combined_array.reshape(combined_array.shape[0], 1)


def aggregate_data_to_arrays(
    fp_df: pl.DataFrame, percent_avoid_df: pl.DataFrame
) -> np.array:
    """takes in two dataframes, joins them and returns a numpy array of the learned column and the z_score_DA column only"""
    combined_data = combine_data(fp_df, percent_avoid_df)
    arr = get_arrays(combined_data)
    return arr


def aggregate_all_data_to_arrays(
    df_list: List[pl.DataFrame], percent_avoid_df: pl.DataFrame
) -> List[np.array]:
    """takes in a list of dataframes and returns a list of numpy arrays of the learned column and the z_score_DA column only"""
    return [
        aggregate_data_to_arrays(fp_df=df, percent_avoid_df=percent_avoid_df)
        for df in df_list
    ]


# get min length for list
def get_cutoff_idx(array_list: List[np.array]) -> int:
    """takes in a list of numpy arrays and returns the minimum length of the arrays, intended to be used as an index to ensure all arrays are the same length"""
    lengths = np.array([a.shape[0] for a in array_list])
    return np.min(lengths)


# drop all idx after min length
def shape_arrays(
    array_list: List[np.array], cut_off_idx: int, tranpose_array: False
) -> np.array:
    """takes in a list of numpy arrays and an index and returns a numpy array of the arrays concatenated along the columns and cut off at the index provided"""
    reshaped_list = [a[:cut_off_idx, :] for a in array_list]
    reshaped_array = np.concatenate(reshaped_list, axis=1)
    # transponse arrays for each instance as a row
    if tranpose_array:
        return reshaped_array.T
    else:
        return reshaped_array


# concat arrays
def filter_data_for_subjects(
    df_list: List[pl.DataFrame], percent_avoid_df: pl.DataFrame
):
    valid_subjects = (
        percent_avoid_df.select("mouse_id").unique().collect().to_numpy()[:, 0]
    )
    updated_df_list = []
    for df in df_list:
        subject = df.select("mouse_id").unique().collect().to_numpy()[0, 0]
        if subject in valid_subjects:
            updated_df_list.append(df)
    return updated_df_list


def read_parquets(path: Path) -> List[pl.DataFrame]:
    # assumes all data are parquets and will be read intentionaly
    return [pl.scan_parquet(p) for p in list(path.iterdir())]


def save_array_as_npy(path: Path, file_name: str, arr: np.array):
    path_to_save = path / f"{file_name}.npy"
    np.save(path_to_save, arr, allow_pickle=False)


def save_array_as_df_parquet(path, file_name: str, arr):
    path_to_save = path / f"{file_name}.parquet"
    df = pl.DataFrame(arr)
    df.write_parquet(path_to_save)
