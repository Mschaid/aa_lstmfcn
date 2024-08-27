import logging
import multiprocessing as mp
import re
from functools import partial
from pathlib import Path
from typing import Callable, List

from transphorm.preprocessors import (
    DataFetcher,
    GuppyDataFetcher,
    data_fetcher_factory,
    guppy_processing_strategy,
)

LOG_LEVEL = logging.INFO
logger = logging.getLogger("__name__")
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    PATH = Path(
        "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/core_data/metadata_files"
    )
    if not PATH.exists():
        raise FileNotFoundError(f"Path {PATH} does not exist.")

    logger.info("batch processing")
    CPUS = mp.cpu_count() - 2

    logger.info(f"fetching metadata files from {PATH}")
    fetcher_paths = [p for p in PATH.iterdir() if p.name.endswith("metadata.yaml")]

    logger.info(f"creating data_fetchers")
    data_fetchers = [data_fetcher_factory(GuppyDataFetcher, p) for p in fetcher_paths]

    with mp.Pool(processes=CPUS) as pool:
        logger.info(f"processing data with {CPUS} cpus")
        pool.map(guppy_processing_strategy, data_fetchers)
        pool.close()
        pool.join()
        logger.info("processing complete")


if __name__ == "__main__":
    main()
