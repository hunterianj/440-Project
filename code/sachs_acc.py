#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:30:49 2025

@author: jkai
"""

import dataUtils

import sys
sys.path.append("causal-learn")

from causallearn.graph.SHD import SHD
from causallearn.graph.SCM import SCM


sachs_ground = dataUtils.loadSachsGroundTruth()
ground_SHD_sanity_check = SHD(sachs_ground, sachs_ground)
print(f"SHD: {ground_SHD_sanity_check.get_shd()}")
ground_SCM_sanity_check = SCM(sachs_ground, sachs_ground)

sachsExpDataset = dataUtils.loadSachsDataset()
sachsExpDataset = sachsExpDataset.astype({col: "float64" for col in sachsExpDataset.columns})