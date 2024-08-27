import os
import comet_ml as comet
from dotenv import load_dotenv

load_dotenv()


def setup_comet_experimet(project_name, workspace):
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    experiment = comet.Experiment(
        api_key=COMET_API_KEY,
        project_name=project_name,
        workspace="transphorm",
    )
    return experiment
