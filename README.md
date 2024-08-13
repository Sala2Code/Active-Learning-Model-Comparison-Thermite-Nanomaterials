# Active Learning Model Comparison for Thermite Nanomaterials

## Overview

This repository contains the code and documentation for my internship project, where I focused on comparing active learning models using custom-developed metrics in the context of Al/CuO thermite nanomaterials. The goal was to identify optimal material compositions and process parameters that meet specific user-defined performance criteria, such as combustion pressure and temperature.

## Internship Project

During my internship, I explored different methodologies for intelligent sequential experimental design. These methodologies were applied to the Al/CuO thermite system, a promising class of metal-based reactive materials. These materials have applications in various fields such as miniature autonomous systems, nanosatellites, and high-energy actuations, particularly in harsh environments.

The project aimed to overcome the challenges associated with the vast material design space and the lack of reliable design guidelines. To address these challenges, we implemented and compared various active learning and Bayesian optimization techniques, focusing on efficient exploration of the design space to identify optimal thermite compositions.

## Methodology

We initiated our approach by training a Gaussian Process Regression (GPR) model to accurately predict the combustion pressure and temperature of Al/CuO thermite materials. The GPR model serves as a predictive tool, providing estimates of these critical properties across the design space. To enhance the efficiency of our exploration within this space, we developed a custom acquisition function. This function specifically targets a *region of interest* (ROI), which is defined by the desired range of combustion properties, such as specific pressure and temperature thresholds that are relevant to practical applications.

### 1. Gaussian Process Regression with Custom Acquisition Function (*irbs*)

The first model we developed, named **Interest Region Bayesian Sampling (irbs)**, leverages this acquisition function to guide the sampling process. The goal of irbs is to selectively sample data points that are most likely to fall within the ROI, thereby optimizing the search for ideal thermite compositions. By focusing the sampling efforts on the most promising regions, irbs aims to minimize unnecessary evaluations in less relevant areas of the design space, making the experimental design more efficient and effective.

### 2. Efficient Global Optimization (EGO) Algorithm

We extended the traditional EGO algorithm to a multi-objective optimization problem using a scalarized objective function. The algorithm, known as ParEGO, converts multiple objectives into a single scalar objective using a parameterized weight vector. We further modified this approach to focus on exploring the ROI by introducing the sGA_tent algorithm.

#### 2.1 Replacing SHGO with Genetic Algorithm (*GA_GP*)

In another variant, we replaced the Simplicial Homology Global Optimization (SHGO) algorithm with a genetic algorithm (GA). This approach, named GA_GP, focuses on maximizing the probability of sampling points within the ROI, leveraging the genetic algorithm's ability to explore the design space effectively.

#### 2.2 Scalarization (*sGA_tent*)

The sGA_tent algorithm is a modified version of the ParEGO algorithm. It replaces the traditional scalarization approach with a function that emphasizes exploration within the ROI. The algorithm employs a genetic algorithm to iteratively optimize the scalar objective function, balancing exploration and exploitation.

### Metrics

To compare the performance of the different algorithms, we used three custom-developed metrics:

1. **No Interest**: The number of points sampled outside the region of interest (ROI). A lower value indicates better performance.
   
2. **Coverage**: The total volume of hyper-spheres centered at each sampled point, normalized by the number of points. This metric indicates how well the algorithm covers the design space.

3. **Voronoi Volume**: The volume of the region around each sampled point where the point is closer to any other sampled point. This metric provides insight into the spatial arrangement and influence of the sampled points.

### Results

Below, there are some plots that illustrate how the active learning algorithms performed. For a detailed exploration of the specific regions of interest (ROI) used in real applications and the interpretation of the data, please refer to the [project report](path-to-report) (see the section titled "Results and Discussion"). In the report, you can find an in-depth analysis of the results, including the specific regions of interest relevant to practical applications and a thorough interpretation of the experimental data.

## References

For more detailed information about the methodology and results, please refer to the accompanying [project report](path-to-report).

# Set up
## Rules and guidelines
In order to get the best out of the template:
* Don't remove any lines from the `.gitignore` file we provide
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## Install dependencies
* When using **pip** (recommended)
The dependencies are declared in `src/requirements.txt`.  
To install them, we suggest to create a virtual environment and then install them.
When running with Python version than 3.6 run (from this folder):
```
python -m venv venv
. venv/bin/activate
pip install -r src/requirements.txt
```

* When using **pip** (recommended only if your OS is Windows)
```
conda create -n venv pip
conda actiate venv
pip install -r src/requirements.txt
```

# Run project
In order to run this project is necessary that you have an initial file, stored in the `data` folder inside this project,
and to add it to the file in `conf/base/catalog.yml` under the name `initial_data`, which is the first entry on the file.
You can change the location of all files, but make sure that those are inside `data`.
In order to give to the user an idea of how this file should look like, there is an initial file included in this repository,
located on the direction written on the catalog.

The code is divided into 3 steps:
1. <span style="color:yellow"> Prepare input data </span>: This pipeline will prepare the input data so that can be run in the next pipeline, while also creating the necessary classes to known all the parameters that define the regions of interest as well as outliers/inliers. 
2. <span style="color:green"> Run IRBS (Interest Region Bayesian Sampling) experiment </span>: This pipeline will conduct the Bayesian Optimization in order to increase the initial dataset with interest points, while improving the surrogate model. It will output a DB with several samples.
3. <span style="color:red"> Perform analysis on different experiments </span>: This pipeline contains plots to analyze results, it's created to compare several experiments at the same time (described on `conf/base/parameters/metrics.json`).

## Common parameters
There are some parameters shared between all pipelines, these are stored in `conf/base/parameters.json`, and they are:
* paths: Some programs need paths and store files bypassing the kedro structure,
* features: Physical variables known, that limit the design space,
* targets: Physical variables of interest, which cannot be fixed but are to be known after simulating,
* additional_values: Other values of interest that the user wishes to keep track (but not use).
* interest_region: Boundaries of the targets that will define the region of interest, 

The parameters that only belong to one pipeline are located in `conf/base/parameters/` and are named after each pipeline. The description of each parameter is contained in the README.md file located inside each pipeline in `src/sampler/pipelines/`.


## Environments
In case some parameters need to be modified, they could be stored in a new folder on `conf`.
This new folder should copy the structure on base, and should only contain the parameters or catalog entries that are modified (since all other values will be inherited from base).
This new configuration can be run by adding `--env new_env_folder_name` to the command to run the pipeline.  

## <span style="color:yellow"> 1. Prepare Initial Data and create classification classes </span>
```
kedro run --pipeline prep --env base
``` 
## <span style="color:green"> 2. Run IRBS (Interest Region Bayesian Sampling) experiment </span>
``` 
kedro run --pipeline irbs --env base
``` 
## <span style="color:red"> 3. Perform metrics </span>
``` 
kedro run --pipeline metrics --env base
```
