"""
This is a boilerplate pipeline 'rerun_exp'
generated using Kedro 0.18.5
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import glob


from sampler.pipelines.metrics.postprocessing_functions import create_dict, prepare_new_data, prepare_benchmark, get_result

from sampler.common.data_treatment import DataTreatment
from sampler.pipelines.irbs.nodes import irbs_sampling

from sampler.common.storing import parse_results
from sampler.models.wrapper_for_0d import SimulationProcessor
from sampler.models.fom import FigureOfMerit
from datetime import datetime
from tqdm import tqdm


def getBackData(
            path: str, run_condition: dict,
            initial_size: int
        ) -> tuple:

    files = glob.glob(f"{path}/*.csv")
    files.sort()
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    rerun_condition = {
        "run_until_max_size": run_condition["run_until_max_size"],
        "batch_size": run_condition["batch_size"],
        
        "max_size": run_condition["max_size"] - len(df) + initial_size,
        "n_interest_max": run_condition["n_interest_max"] - sum(df['quality']=="interest")
    }
    print(f"Rerun condition: {rerun_condition}")
    return dict(data=df, rerun_condition=rerun_condition)


def rerun(
    rerun_exp: str, data: pd.DataFrame, treatment: DataTreatment,
    features: List[str], targets: List[str], additional_values: List[str],
    run_condition: dict, dic_params: dict, simulator_env: dict
):
    # if rerun_exp == "irbs":
    fom_terms=dic_params['irbs_fom_terms']
    simulator_env=simulator_env
    run_condition=run_condition
    opt_iters=dic_params['irbs_opt_iters']
    opt_points=dic_params['irbs_opt_sampling_points']

    max_size, n_interest_max, run_until_max_size, batch_size = run_condition['max_size'], run_condition['n_interest_max'], run_condition['run_until_max_size'], run_condition['batch_size']

    # Set figure of merite (acquisition function)
    model = FigureOfMerit(
        features=features, targets=targets, terms=fom_terms,
        interest_region=treatment.scaled_interest_region
    )

    # Set simulator environement
    simulator = SimulationProcessor(
        features=features, targets=targets, additional_values=additional_values,
        treatment=treatment, n_proc=batch_size, simulator_env=simulator_env
    )
    data = simulator.adapt_targets(data)

    res = data
    yield parse_results(res, current_history_size=0)

    n_total = 0  # counting all simulations
    n_inliers = 0  # counting only inliers
    n_interest = 0  # counting only interesting inliers
    iteration = 0
    end_condition = n_inliers < max_size if run_until_max_size else n_interest < n_interest_max 
    progress_bar = tqdm(total=max_size, dynamic_ncols=True) if run_until_max_size else tqdm(total=n_interest_max, dynamic_ncols=True)  # Initialize tqdm progress bar with estimated time remaining
    print(f"Iteration {iteration:03} - Total size {n_total} - Inliers size {n_inliers} - Interest count {n_interest}")
    while end_condition:
        model.update(res, optimizer_kwargs=dict(shgo_iters=opt_iters, shgo_n=opt_points))  # Set the new model that will be used in next iteration

        new_x, scores = model.optimize(batch_size=batch_size, shgo_iters=opt_iters, shgo_n=opt_points)  # Search new candidates to add to res dataset

        new_df = simulator.process_data(new_x, real_x=False, index=n_total, treat_output=True)  # Launch time expensive simulations
        model.excluder.update_outliers_set(new_df)

        print(f'Round {iteration:03} (continued): simulation results' + '-'*49)
        print(f'irbs_sampling -> New samples after simulation:\n {new_df}')

        new_df = pd.concat([new_df, scores], axis=1, join='inner', ignore_index=False)

        # Add maximum found value for surrogate GP combined std
        new_df['max_std'] = model.gp_surrogate.max_std

        # Add model prediction to selected (already simulated) points
        prediction = model.gp_surrogate.predict(new_df[features].values)
        prediction_cols = [f"pred_{t}" for t in targets]
        new_df[prediction_cols] = prediction if len(targets) > 1 else prediction.reshape(-1, 1)

        # Add column is_interest with True if targets are inside the interest region
        new_df = treatment.classify_quality_interest(new_df, data_is_scaled=True)
        
        # Add iteration number and datetime
        timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_df['datetime'] = timenow
        new_df['iteration'] = iteration
        
        # Store final batch results
        yield parse_results(new_df, current_history_size=res.shape[0])

        # Concatenate new values to original results DataFrame
        res = pd.concat([res, new_df], axis=0, ignore_index=True)
        
        # Update stopping conditions
        n_new_samples = new_df.shape[0]
        n_new_inliers = new_df.dropna(subset=targets).shape[0]
        n_new_interest = new_df[new_df['quality'] == 'interest'].shape[0]
    
        n_total += n_new_samples
        n_inliers += n_new_inliers
        n_interest += n_new_interest
        iteration += 1

        # Update progress bar based on the condition
        progress_bar.update(n_new_inliers if run_until_max_size else n_new_interest)

        end_condition = (n_inliers < max_size) if run_until_max_size else (n_interest < n_interest_max)

        print(f"Iteration {iteration:03} - Total size {n_total} - Inliers size {n_inliers} - Interest count {n_interest}")
    progress_bar.close()

