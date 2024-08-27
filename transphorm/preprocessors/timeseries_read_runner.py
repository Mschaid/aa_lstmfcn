import logging

import numpy as np
from pathlib import Path
import polars as pl

from transphorm.preprocessors import (
    aggregate_all_data_to_arrays,
    filter_data_for_subjects,
    get_cutoff_idx,
    read_parquets,
    save_array_as_npy,
    save_array_as_df_parquet,
    shape_arrays,
)

LOG_LEVEL = logging.INFO
logger = logging.getLogger("__name__")
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(main_path):
    recording_path = main_path / "compiled_timeseries_data"
    perc_avoid_path = main_path / "percent_avoid.parquet"

    percent_avoid_df = pl.scan_parquet(perc_avoid_path).rename(
        {"mouse_numb": "mouse_id"}
    )

    all_data_df = read_parquets(recording_path)
    logger.info("parquets read into mem")

    updated_subjects_df = filter_data_for_subjects(
        df_list=all_data_df, percent_avoid_df=percent_avoid_df
    )
    logger.info("data filtered for included subjects")

    array_list = aggregate_all_data_to_arrays(
        updated_subjects_df, percent_avoid_df=percent_avoid_df
    )
    logger.info(f"dataframes converted to np arrays")
    cut_off_idx = get_cutoff_idx(array_list)
    reshaped_and_combined_array = shape_arrays(
        array_list, cut_off_idx=cut_off_idx, tranpose_array=True
    )
    logger.info("data reshaped")
    save_array_as_npy(
        path=main_path,
        file_name="dopamine_full_timeseries_array",
        arr=reshaped_and_combined_array,
    )
    logger.info(f"data saved to {main_path}")


if __name__ == "__main__":
    main_path = Path(
        "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/core_data"
    )
    main(main_path=main_path)
