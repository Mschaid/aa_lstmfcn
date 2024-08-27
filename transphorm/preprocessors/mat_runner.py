from transphorm.preprocessors import MatFetcher
import logging
from pathlib import Path

LOG_LEVEL = logging.INFO
logger = logging.getLogger("__name__")
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(path):
    fetcher = MatFetcher(path, day_filter=5)
    fetcher.load_data()
    logger.info("Data loaded")

    fetcher.write_to_torch()
    logger.info(f"Data written to pt binary in {path}")


if __name__ == "__main__":
    PATH = Path(
        "/Users/mds8301/Library/CloudStorage/OneDrive-NorthwesternUniversity/gaby_data"
    )
    main(PATH)
