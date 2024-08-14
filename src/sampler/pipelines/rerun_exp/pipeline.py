"""
This is a boilerplate pipeline 'rerun_exp'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from sampler.pipelines.rerun_exp.nodes import getBackData, rerun
from sampler.common.storing import join_history


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=getBackData,
            inputs=dict(
                path='params:rerun_path_history',
                run_condition='params:run_condition',
                initial_size='params:initial_size'
            ),
            outputs=dict(
                data='rerun_data',
                rerun_condition='rerun_condition'
            )
        )
        ,
        node(
            func=rerun,
            inputs=dict(
                rerun_exp="params:rerun_exp",
                data='rerun_data',
                treatment='treatment',
                features='params:features',
                targets='params:targets',
                additional_values='params:additional_values',
                run_condition='rerun_condition',
                dic_params='params:rerun_dic_params',
                simulator_env='params:simulator_env'
            ),
            outputs='rerun_history'
        ),
        node(
            func=join_history,
            inputs=dict(
                history='rerun_history',
                run_condition='rerun_condition',
                initial_size='params:initial_size'
            ),
            outputs='rerun_increased_data'
        )
    ])
