#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:30:49 2025

@author: jkai
"""

import dataUtils
import importlib  
causal_learn = importlib.import_module("causal-learn")

sachs_ground = dataUtils.loadSachsGroundTruth()

sachsExpDataset = dataUtils.loadSachsDataset()
sachsExpDataset = sachsExpDataset.astype({col: "float64" for col in sachsExpDataset.columns})