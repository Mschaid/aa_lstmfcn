import logging
import pretty_errors
from pathlib import Path
from typing import List
from src.data_processing.pipelines import AAMetaDataFetcher, meta_data_factory, directory_finder

LOG_LEVEL = logging.INFO
# set up logger for metadata runner
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(path_to_data):
    path_to_data = Path(path_to_data)
    path_to_save = path_to_data / "metadata_files"
    path_to_save.mkdir(exist_ok=True)
    logger.info(f"Looking for directories in {path_to_data}")

    files = directory_finder(
        main_path=path_to_data, directory_keyword="output", keywords_to_drop=["Bad_Photometry"])

    logger.info(f"Found {len(files)} directories")
    fetchers = [meta_data_factory(
        path=file, path_to_save=path_to_save, fetcher=AAMetaDataFetcher) for file in files]
    logger.info(f"Created {len(fetchers)} fetchers")

    for _ in fetchers:
        _.save_metadata_to_yaml()

    logger.info("Saved metadata to yaml files")


if __name__ == "__main__":
    path_to_data = "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/core_data"
    main(path_to_data)
