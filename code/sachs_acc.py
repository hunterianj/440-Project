#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:30:49 2025

@author: jkai
"""

import dataUtils
import utils
import ccpg
import ci_tests

import numpy as np

import json
import os
import sys
sys.path.append("causal-learn")

from causallearn.graph.SHD import SHD
from causallearn.graph.SCM import SCM

sachs_ground_truth = dataUtils.loadSachsGroundTruth()

results_filename = "benchmark_results_int.json"

# Create file if it doesn't exist
if not os.path.exists(results_filename):
    with open(results_filename, "w") as f:
        json.dump([], f)

def save_result(data):
    with open(results_filename, "r+") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
        existing_data.append(data)
        f.seek(0)
        json.dump(existing_data, f, indent=2)
        f.truncate()

def run_benchmark_suite(data,
                        i_data,
                        i_nodes,
                        data_label,
                        node_names,
                        ci_test_name,
                        params,
                        param_name,
                        ci_test_kwargs = None):
    bestSHD = np.inf
    bestParam = None
    bestResult = None
    print(f"\n--- Benchmarking: {data_label} + {ci_test_name} ---")

    for param in params:
        result = utils.benchmark_i_ccpg_against_ground_truth(
            data=data,
            i_data=i_data,
            i_nodes=i_nodes,
            ground_truth_graph=sachs_ground_truth,
            node_names=node_names,
            ci_test_name=ci_test_name,
            alpha=param,
            verbose=True,
            ci_test_kwargs=ci_test_kwargs,
        )

        # record = result.copy()
        # record["graph"] = None
        # record["components"] = [list(comp) for comp in record["components"]]
        # record["edges"] = list(record["edges"])
        # result_record = {
        #     "data_label": data_label,
        #     "ci_test_name": ci_test_name,
        #     "param_name": param_name,
        #     "param_value": bestParam,
        #     "shd": record["shd"],
        #     "result_details": record
        # }
        # save_result(result_record)

        if result["shd"] < bestSHD:
            bestSHD = result["shd"]
            bestParam = param
            bestResult = result

    print(f"Best {param_name} for {data_label} + {ci_test_name}: {bestParam} with SHD = {bestSHD}")
    print(f"Result: {bestResult}")

    result = bestResult.copy()
    result["graph"] = None
    result["edges"] = list(result["edges"])
    result["components"] = [list(comp) for comp in result["components"]]
    result_record = {
        "data_label": data_label,
        "ci_test_name": ci_test_name,
        "param_name": param_name,
        "param_value": bestParam,
        "shd": bestResult["shd"],
        "result_details": result
    }
    save_result(result_record)
    return bestParam, bestResult


# Parameters for tuning
alphas = [0.5, 0.0001] # [0.1, 0.05, 0.01, 0.001, 0.0001]
bic_thresholds = [0.0, 1.0, 2.5, 5.0, 10.0]

# Perform discrete data analysis
disc_obs_data, disc_i_data, disc_i_nodes = dataUtils.loadSachsInterventionalDiscrete()
disc_names = list(disc_obs_data.columns)

# utils.plot_graph(d_i_cg, "sachs_discrete_i_ccpg")

ci_test_kwargs = {
    # "kernelX": "Linear", # default is Gaussian
    # "kernelY": "Linear", # default is Gaussian
    # "kernelZ": "Gaussian", # default is Gaussian
    # "est_width": "median",
    # "approx": True,
    # "null_ss": 1000
}
# chi^2 CI test on discrete data
run_benchmark_suite(
    data=disc_obs_data.to_numpy(),
    i_data=[i_dat.to_numpy() for i_dat in disc_i_data],
    i_nodes=disc_i_nodes,
    data_label="discrete",
    node_names=disc_names,
    ci_test_name="chisq",
    params=alphas,
    param_name="alpha",
    ci_test_kwargs=ci_test_kwargs
)

ci_tests.register_ci_tests()

# Perform continuous data analysis
cont_obs_data, cont_i_data, cont_i_nodes = dataUtils.loadSachsInterventionalContinuous()
cont_names = list(cont_obs_data.columns)

# Gaussian BIC custom CI test on log-transformed data
run_benchmark_suite(
    data=cont_obs_data.to_numpy(),
    i_data=[i_dat.to_numpy() for i_dat in cont_i_data],
    i_nodes=cont_i_nodes,
    data_label="log-continuous",
    node_names=cont_names,
    ci_test_name="gaussbic",
    params=bic_thresholds,
    param_name="threshold"
)


# ground_SHD_sanity_check = SHD(sachs_ground, sachs_ground)
# print(f"SHD: {ground_SHD_sanity_check.get_shd()}")
# ground_SCM_sanity_check = SCM(sachs_ground, sachs_ground)
