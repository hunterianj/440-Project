import dataUtils, utils
from ccpg import ccpg
from causallearn.graph.SHD import SHD
from causallearn.graph.SCM import SCM
import numpy as np
import ci_tests
import json
import os

ci_tests.register_ci_tests()

verbose = True
ground_truth = dataUtils.tegGroundTruth()
# utils.plot_graph(ground_truth, "teg_ground_truth")

dataset = dataUtils.loadTEGData()
# reduce to 500 samples for faster benches
# dataset = dataset[:100, :]

results_filename = "benchmark_results_obs_teg_pc.json"

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
                        data_label,
                        ci_test_name,
                        params,
                        param_name,
                        ci_test_kwargs = None):
    bestSHD = np.inf
    bestParam = None
    bestResult = None
    print(f"\n--- Benchmarking: {data_label} + {ci_test_name} ---")

    for param in params:
        result = utils.benchmark_pc_against_ground_truth(
            data=data,
            ground_truth_graph=ground_truth,
            ci_test_name=ci_test_name,
            alpha=param,
            verbose=verbose,
            ci_test_kwargs=ci_test_kwargs,
            compute_scm=False,
        )

        if result["shd"] < bestSHD:
            bestSHD = result["shd"]
            bestParam = param
            bestResult = result

    print(f"Best {param_name} for {data_label} + {ci_test_name}: {bestParam} with SHD = {bestSHD}")
    print(f"Result: {bestResult}")

    result = bestResult.copy()
    result["graph"] = None
    result["edges"] = [(edge.get_node1().get_name(), edge.get_node2().get_name()) for edge in result["edges"]]
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
alphas = [0.1, 0.05, 0.01, 0.001, 0.0001]
bic_thresholds = [0.0, 1.0, 2.5, 5.0, 10.0]


# this isn't too bad for speed, could go for some extensive hyper-parameter tuning
ci_test_kwargs = {
    # "approx": "chi2",
    # "approx": "gamma",
    # "approx": "hbe",
    # "approx": "lpd4", # default
    # "approx": "perm",
    "num_f": 50, # default = 100
    "num_f2": 5 # default = 5
}
alphas = [0.0001]
# RCIT CI test on continuous data
run_benchmark_suite(
    data=dataset,
    data_label="continuous",
    ci_test_name="rcit",
    params=alphas,
    param_name="alpha",
    ci_test_kwargs=ci_test_kwargs
)
#
# bic_thresholds = [0.1]
# # Gaussian BIC custom CI test on continuous data
# run_benchmark_suite(
#     data=dataset,
#     data_label="continuous",
#     ci_test_name="gaussbic",
#     params=bic_thresholds,
#     param_name="threshold"
# )
#
# alphas = [0.001]
# # fisherz CI test on continuous data
# run_benchmark_suite(
#     data=dataset,
#     data_label="continuous",
#     ci_test_name="fisherz",
#     params=alphas,
#     param_name="alpha"
# )

# ci_test_kwargs = {
#     "K": 5,
#     "J": 4,
#     # "alpha": 500,
#     # "use_gp": True,
# }
# # FastKCI CI test on continuous data
# run_benchmark_suite(
#     data=dataset,
#     data_label="continuous",
#
#     ci_test_name="fastkci",
#     params=alphas,
#     param_name="alpha",
#     ci_test_kwargs=ci_test_kwargs
# )
#
# # FastKCI CI test on log-transformed data
# run_benchmark_suite(
#     data=log_obs_data,
#     data_label="log-continuous",
#
#     ci_test_name="fastkci",
#     params=alphas,
#     param_name="alpha",
#     ci_test_kwargs=ci_test_kwargs
# )

# loading smaller dataset to keep runtimes relatively manageable
dataset = dataset[:100, :]

alphas = [0.001]

ci_test_kwargs = {
    # "kernelX": "Linear", # default is Gaussian
    # "kernelY": "Linear", # default is Gaussian
    # "kernelZ": "Gaussian", # default is Gaussian
    "est_width": "median",
    "approx": True,
    "null_ss": 1000
}
# KCI CI test on continuous data
run_benchmark_suite(
    data=dataset,
    data_label="continuous",

    ci_test_name="kci",
    params=alphas,
    param_name="alpha",
    ci_test_kwargs=ci_test_kwargs
)
#
# # KCI CI test on log-transformed data
# run_benchmark_suite(
#     data=log_obs_data,
#     data_label="log-continuous",
#
#     ci_test_name="kci",
#     params=alphas,
#     param_name="alpha",
#     ci_test_kwargs=ci_test_kwargs
# )