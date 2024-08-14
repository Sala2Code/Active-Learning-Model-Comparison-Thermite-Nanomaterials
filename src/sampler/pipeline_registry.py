"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import sampler.pipelines.prep.pipeline as prep_input
import sampler.pipelines.irbs.pipeline as irbs_sampler
import sampler.pipelines.rerun_exp.pipeline as rerun_exp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["rerun_exp"] = rerun_exp.create_pipeline()
    pipelines["__default__"] = prep_input.create_pipeline() + irbs_sampler.create_pipeline()
    return pipelines
