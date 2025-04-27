#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:30:49 2025

@author: jkai
"""

import dataUtils
import utils
import ccpg

import sys
sys.path.append("causal-learn")

from causallearn.graph.SHD import SHD
from causallearn.graph.SCM import SCM

sachs_ground = dataUtils.loadSachsGroundTruth()

alpha=1e-4

# Perform discrete data analysis
disc_obs_data, disc_i_data, disc_i_nodes = dataUtils.loadSachsInterventionalDiscrete()
d_cg, d_components, d_edges = ccpg.i_ccpg(
                                disc_obs_data.to_numpy(),
                                [i_dat.to_numpy() for i_dat in disc_i_data],
                                disc_i_nodes,
                                alpha=alpha,
                                verbose=True,
                                ci_test_name="gsq",
                                node_names=list(disc_obs_data.columns))

utils.plot_graph(d_cg, "sachs_discrete_i_ccpg")

# Perform continuous data analysis
cont_obs_data, cont_i_data, cont_i_nodes = dataUtils.loadSachsInterventionalContinuous()
c_cg, c_components, c_edges = ccpg.i_ccpg(
                                cont_obs_data.to_numpy(),
                                [i_dat.to_numpy() for i_dat in cont_i_data],
                                cont_i_nodes,
                                alpha=alpha,
                                verbose=True,
                                ci_test_name="fisherz",
                                node_names=list(cont_obs_data.columns))

x = 1
# ground_SHD_sanity_check = SHD(sachs_ground, sachs_ground)
# print(f"SHD: {ground_SHD_sanity_check.get_shd()}")
# ground_SCM_sanity_check = SCM(sachs_ground, sachs_ground)
