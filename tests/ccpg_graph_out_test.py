from causallearn.search.ConstraintBased import PC, CCPG
from pytorch_lightning import seed_everything
import numpy as np
import utils
from CCPG import ccpg as ccpg_original
from causaldag import (partial_correlation_suffstat,
                       partial_correlation_test,
                       MemoizedCI_Tester)
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

# import ccpg as CCPG

# use the same seed as the CCPG authors
seed_everything(42)

# Load airfoil dataset
fileName = "data/airfoil-self-noise.continuous.txt"
data = np.loadtxt(fileName, skiprows=1)
with open(fileName, 'r') as f:
    node_names = f.readline().strip().split('\t')
n, d = data.shape

# Author's implementation
suffstat = partial_correlation_suffstat(data)
ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
c, e = ccpg_original.ccpg(set(range(d)), ci_tester, verbose=False)
print("Authors' CCPG:")
print('Components: ', [{node_names[j] for j in i} for i in c])
print('Edges: ', e)

# Our implementation
graph = CCPG.ccpg(data, alpha=1e-3, node_names=node_names)

# PC
graph_PC = PC.pc(data, alpha=1e-3, node_names=node_names)

utils.plot_graph(graph, "airfoil_our_ccpg")
utils.plot_graph(graph_PC, "airfoil_pc")
