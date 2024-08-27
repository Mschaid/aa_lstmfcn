import logging
import re
from pathlib import Path
from typing import Generator, List, Protocol

import yaml

LOG_LEVEL = logging.DEBUG
# set up logger for metadata runner
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class MetaDataFetcher(Protocol):
    def __init__(self, path: Path, path_to_save: Path):
        """takes output guppy path as Path object"""
        ...

    def metadata(self) -> dict:
        """returns a dictionary containing the metadata extracted from the directory parameters, as well as any filepaths."""
        ...

    def save_metadata_to_yaml(self) -> None:
        """saves the metadata to a yaml file in the directory pointed to by the path attribute."""
        ...


# simple methods used by the MetaDataFetcher class implementation and factory function


def directory_finder(
    main_path: Path, directory_keyword: str, keywords_to_drop: List[str] = None
) -> List[Path]:

    paths_found = main_path.glob(f"**/*{directory_keyword}*")
    paths_found = [path for path in paths_found if path.is_dir()]
    if not keywords_to_drop:
        return paths_found
    else:

        def filter_paths(p):
            return all(kw not in p.as_posix() for kw in keywords_to_drop)

        return list(filter(filter_paths, paths_found))


def meta_data_factory(
    path: Path, path_to_save: Path, fetcher: MetaDataFetcher
) -> MetaDataFetcher:
    return fetcher(path=path, path_to_save=path_to_save)


class AAMetaDataFetcher:
    """
        class used to extract metadata from active avoidance experiments from a given path.

    Attributes
    ----------
    path : Path
        a Path object that points to the file from which metadata is extracted
    day : int
        the day extracted from the file name (default is None)
    cage : int
        the cage number extracted from the file name (default is None)
    mouse_id : int
        the mouse ID extracted from the file name (default is None)
    D1 : bool
        a flag indicating if "D1" is in the file name (default is None)
    D2 : bool
        a flag indicating if "A2A" is in the file name (default is None)
    dDA : bool
        a flag always set to True


    """

    def __init__(self, path: Path, path_to_save: Path):
        self.path = path
        self.path_to_save = path_to_save
        self._metadata: dict = None

    def fetch_day(self) -> int:
        """extracts the day from the file name and returns it as an int."""
        try:
            search_match = re.search("[dD]ay[0-9]", self.path.as_posix())
            day_string = search_match.group()
            day_number = re.sub("[dD]ay", "", day_string)

            day = int(day_number)
            return day
        except ValueError:
            print(f"{self.path} has an does not contain day string: {day_string}")

    def fetch_cage(self) -> int:
        """extracts the cage number from the file name and returns it as an int."""
        parent_name = self.path.name
        cage_string = parent_name.split("-")[0]
        cage = int(cage_string)
        return cage

    def fetch_mouse_id(self) -> int:
        """extracts the mouse ID from the file name and returns it as an int."""
        name_split = self.path.name.split("-")
        ids = name_split[1].split("_")
        has_copy = "copy" in self.path.name
        if not has_copy:
            mouse_id = int(ids[0])
        else:
            mouse_id = int(ids[1])
        return mouse_id

    def fetch_D1(self) -> bool:
        """returns True if "D1" is in the file name, False otherwise."""
        is_D1 = "D1" in self.path.as_posix()
        return is_D1

    def fetch_D2(self) -> bool:
        """returns True if "A2A" is in the file name, False otherwise."""
        is_D2 = "A2A" in self.path.as_posix()
        return is_D2

    def fetch_DA(self) -> bool:
        """always returns True, given that this dataset always records dopamine"""
        return True

    def _fetch_hdf5_paths(self, *keyword) -> Generator[str, None, None]:
        """returns a generator that yields the paths of hdf5 files that contain the keyword in their name."""
        for k in keyword:
            for file in self.path.glob(f"*{k}*.hdf5"):
                yield file.as_posix()

    def load_metadata(self):
        """returns a dictionary containing the metadata extracted from the file name."""
        metadata = {
            "day": self.fetch_day(),
            "cage": self.fetch_cage(),
            "mouse_id": self.fetch_mouse_id(),
            "D1": self.fetch_D1(),
            "D2": self.fetch_D2(),
            "DA": self.fetch_DA(),
            "full_z_scored_recording_paths": list(self._fetch_hdf5_paths("z_score_")),
            "full_dff_recording_paths": list(self._fetch_hdf5_paths("dff_")),
            "event_paths": list(
                self._fetch_hdf5_paths(
                    "CueA",
                    "CueB",
                    "CrsA",
                    "CrsB",
                    "ShkA",
                    "ShkB",
                    "AvdA",
                    "AvdB",
                    "EspA",
                    "EspB",
                )
            ),
        }
        return metadata

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.load_metadata()
        return self._metadata

    def save_metadata_to_yaml(self):
        logger.debug(f"Metadata: {self.metadata}")
        """ saves the metadata to a yaml file in the directory pointed to by the path attribute)."""
        cage = self.metadata["cage"]
        mouse = self.metadata["mouse_id"]
        day = self.metadata["day"]
        file_path_name = Path(f"cage_{cage}_mouse_{mouse}_day_{day}_metadata.yaml")

        file_path = self.path_to_save / file_path_name

        with open(file_path, "w") as f:
            yaml.dump(self.metadata, f)
            logger.info(f"Metadata saved to {file_path}")
