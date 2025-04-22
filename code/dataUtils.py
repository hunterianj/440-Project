import pandas as pd
from enum import Enum
from pathlib import Path
import numpy as np
# from pytetrad import resources

# class Format(Enum):
#     RAW = 1
#     TETRAD = 2

def loadSachsDataset():
    # dataset = pd.read_csv("../data/sachs/data/sachs.2005.continuous.txt", sep=r'\s+')
    dataset = pd.read_csv("../data/sachs/data/sachs.2005.logxplus10.continuous.txt", sep=r'\s+')
    # if format == Format.TETRAD:
    # dataset =

    return dataset

def loadSachsDatasetWithExperiments():
    dataset = pd.read_csv("../data/sachs/data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt", sep=r'\s+')
    return dataset

def loadSachsGroundTruth():
    # dataset = pd.read_csv("../data/sachs/ground_truth/sachs.2005.ground.truth.graph.txt", sep=r'\s+')
    # if format == Format.TETRAD:
    dataset = ""

    return dataset

class IHDPFormat(Enum):
    TRAIN = 1
    TEST = 2

# possible keys are:
# ate - average treatment effect
# mu1 - ?
# mu0 - ?
# yadd - ?
# yf - factual outcomes
# ycf - counter-factual outcomes
# t - treatment assignments
# x - data
# ymul - ?
def loadIHDPDataset(replication: int = 0, format: IHDPFormat = IHDPFormat.TRAIN):
    if format == IHDPFormat.TRAIN:
        filename = "ihdp_npci_1-100.train.npz"
    else:
        filename = "ihdp_npci_1-100.test.npz"

    fn = Path("..", "data", filename)
    data = np.load(fn, allow_pickle=True)
    # breakpoint()
    # for k in data.keys():
    #     print(k)
    # print(f"IHDP shape: {np.ndarray([data[k] for k in data.keys()]).shape}")
    # data = [data[k] for k in data.keys()]
    # https://github.com/clinicalml/cfrnet/blob/master/cfr/loader.py
    # dataDict = dict([(k, data[k]) for k in data.keys()])
    # data = pd.DataFrame.from_dict({item: data[item] for item in data.files}, orient='index')

    x = data["x"][:, :, replication]
    t = data["t"][:, replication]
    yf = data["yf"][:, replication]

    dataframe = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    dataframe["treatment"] = t
    dataframe["outcome"] = yf
    return dataframe
