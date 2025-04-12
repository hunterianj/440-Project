#!/usr/bin/env python
import dataUtils
import pytetrad.tools.TetradSearch as ts

ihdpDataset = dataUtils.loadIHDPDataset()
print(ihdpDataset)

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
