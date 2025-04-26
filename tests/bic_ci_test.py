from causallearn.search.ConstraintBased import PC
from pytorch_lightning import seed_everything
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

import ccpg as CCPG
import utils
import ci_tests

# use the same seed as the CCPG authors
seed_everything(42)

# register gaussian BIC CI test
ci_tests.register_ci_tests()

# Load airfoil dataset
fileName = "data/airfoil-self-noise.continuous.txt"
data = np.loadtxt(fileName, skiprows=1)
with open(fileName, 'r') as f:
    node_names = f.readline().strip().split('\t')
n, d = data.shape


# Our implementation
graph = CCPG.ccpg(data, alpha=1e-3, node_names=node_names, ci_test_name="gaussbic")

# PC
# graph_PC = PC.pc(data, alpha=1e-3, node_names=node_names)

utils.plot_graph(graph, "airfoil_our_ccpg_bic")
# utils.plot_graph(graph_PC, "airfoil_pc_bic")
