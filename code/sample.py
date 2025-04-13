#!/usr/bin/env python
import dataUtils
import pytetrad.tools.TetradSearch as ts

# ihdpDataset = dataUtils.loadIHDPDataset()
# print(ihdpDataset.head())
# data = ihdpDataset.astype({col: "float64" for col in ihdpDataset.columns})
# search = ts.TetradSearch(data)
# search.set_verbose(False)
# search.use_sem_bic()
# search.use_fisher_z(alpha=0.05)
#
# print("\nPC\n")
# search.run_pc()
# print(search.get_string())

# sachsDataset = dataUtils.loadSachsDataset()
# print(sachsDataset.head())
# data = sachsDataset.astype({col: "float64" for col in sachsDataset.columns})
# # print(data)
# search = ts.TetradSearch(data)
# search.set_verbose(False)
# search.use_sem_bic()
# search.use_fisher_z(alpha=0.05)
#
# print("\nPC\n")
# search.run_pc()
# print(search.get_string())
#
# print('\nFCI\n')
# search.run_fci()
# print(search.get_string())

# import pandas as pd
# import pytetrad.tools.translate as ptt
# import edu.cmu.tetrad.search as ts
# import edu.cmu.tetrad.data as td
#
# sachsExpDataset = dataUtils.loadSachsDatasetWithExperiments()
# tetradData = ptt.pandas_data_to_tetrad(sachsExpDataset)
#
# knowledge = td.Knowledge()
# tiers = [
#     ['cd3_cd28', 'icam2', 'aktinhib', 'g0076', 'psitect', 'u0126', 'ly', 'pma', 'b2camp'],
#     ['raf', 'mek', 'plc', 'pip2', 'pip3', 'erk', 'akt', 'pka', 'pkc', 'p38', 'jnk']
# ]
#
# for tierNum, values in enumerate(tiers):
#     for val in values:
#         knowledge.addToTier(tierNum, val)
#     knowledge.setTierForbiddenWithin(tierNum, True)
#
# # print("\nConstructed Knowledge:")
# # print(knowledge)
#
# score = ts.score.ConditionalGaussianScore(tetradData, 0.0, True)
#
# fges = ts.Fges(score)
# fges.setKnowledge(knowledge)
#
# dag = fges.search()
#
# print(dag)

import pytetrad.tools.translate as ptt
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.data as td
import pytetrad

sachsExpDataset = dataUtils.loadSachsDataset()
sachsExpDataset = sachsExpDataset.astype({col: "float64" for col in sachsExpDataset.columns})
data = ptt.pandas_data_to_tetrad(sachsExpDataset)

# knowledge = td.Knowledge()
# tiers = [
#     ['cd3_cd28', 'icam2', 'aktinhib', 'g0076', 'psitect', 'u0126', 'ly', 'pma', 'b2camp'],
#     ['raf', 'mek', 'plc', 'pip2', 'pip3', 'erk', 'akt', 'pka', 'pkc', 'p38', 'jnk']
# ]

# for tierNum, values in enumerate(tiers):
#     for val in values:
#         knowledge.addToTier(tierNum, val)
#     knowledge.setTierForbiddenWithin(tierNum, True)

# print(knowledge)

# score = ts.score.ConditionalGaussianScore(data, 0.0, True)
#
# fges = ts.Fges(score)
# # fges.setKnowledge(knowledge)
#
# dag = fges.search()
#
# print(dag)


indTest = ts.test.IndTestFisherZ(data, 0.05)
fci = ts.Fci(indTest)

pag = fci.search()
print(pag)