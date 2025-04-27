import pandas as pd
from enum import Enum
from pathlib import Path
import numpy as np
from io import StringIO
import re

import sys
sys.path.append("causal-learn")

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode

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


# def loadSachsInterventionalContinuous():
#     df = pd.read_csv("../data/sachs/data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt", sep=r'\s+')
    
#     interventions = {
#         (1, 0, 0, 0, 0, 0, 0, 0, 0): "obs",  # (853 samples) cd3_cd28 only (general pertubation)
#         (1, 1, 0, 0, 0, 0, 0, 0, 0): "obs",  # (901 samples) cd3_cd28 + icam2 (general pertubation)
#         (1, 0, 1, 0, 0, 0, 0, 0, 0): "akt",  # (911 samples) cd3_cd28 + aktinhib
#         (1, 0, 0, 1, 0, 0, 0, 0, 0): "pkc",  # (723 samples) cd3_cd28 + g0076 (inhibitor)
#         (1, 0, 0, 0, 1, 0, 0, 0, 0): "pip2", # (810 samples) cd3_cd28 + psitect
#         (1, 0, 0, 0, 0, 1, 0, 0, 0): "mek",  # (799 samples) cd3_cd28 + u-126
#         (1, 0, 0, 0, 0, 0, 1, 0, 0): "obs",  # (848 samples) cd3_cd28 + ly (P13k inhibitor - “subsequent” relation to Akt)
#         (0, 0, 0, 0, 0, 0, 0, 1, 0): "pkc",  # (913 samples) pma (activator)
#         (0, 0, 0, 0, 0, 0, 0, 0, 1): "pka"   # (707 samples) b2camp
#     }
    
#     data_cols = ["raf", "mek", "plc", "pip2", "pip3", "erk", "akt", "pka", "pkc", "p38", "jnk"]
#     intv_cols = ["cd3_cd28", "icam2", "aktinhib", "g0076", "psitect", "u0126", "ly", "pma", "b2camp"]
    
#     data = pd.DataFrame()
#     i_data = {
#         "akt":  pd.DataFrame(),
#         "pkc":  pd.DataFrame(),
#         "pip2": pd.DataFrame(),
#         "mek":  pd.DataFrame(),
#         "pka":  pd.DataFrame()
#     }
    
#     for interv_row in interventions:
#         i_row = list(interv_row)
#         # TODO: This doesn't quite work... the idea is just to match interventional columns to tuple keys
#         # Alternately, we can just do this manually and make a new version of
#         # sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt with intervention numbers like
#         match_row = df[intv_cols].apply(lambda row: row == i_row, axis=1, broadcast=True)
#         _data = df.where(match_row)
#         if interventions[interv_row] == "obs":
#             data.append(_data)
#         else:
#             i_data[interventions[interv_row]].append(_data)
        
#     return data, i_data.values(), [(key, data_cols.index(key)) for key in i_data.keys()]


def loadSachsInterventionalContinuous():
    df = pd.read_csv("../data/sachs/data/sachs.2005.continuous.interventional.txt", sep=r'\s+')
    
    # interventions = {
    #     (1, 0, 0, 0, 0, 0, 0, 0, 0): "obs",  # (853 samples) cd3_cd28 only (general pertubation)
    #     (1, 1, 0, 0, 0, 0, 0, 0, 0): "obs",  # (901 samples) cd3_cd28 + icam2 (general pertubation)
    #     (1, 0, 1, 0, 0, 0, 0, 0, 0): "akt",  # (911 samples) cd3_cd28 + aktinhib
    #     (1, 0, 0, 1, 0, 0, 0, 0, 0): "pkc",  # (723 samples) cd3_cd28 + g0076 (inhibitor)
    #     (1, 0, 0, 0, 1, 0, 0, 0, 0): "pip2", # (810 samples) cd3_cd28 + psitect
    #     (1, 0, 0, 0, 0, 1, 0, 0, 0): "mek",  # (799 samples) cd3_cd28 + u-126
    #     (1, 0, 0, 0, 0, 0, 1, 0, 0): "obs",  # (848 samples) cd3_cd28 + ly (P13k inhibitor - “subsequent” relation to Akt)
    #     (0, 0, 0, 0, 0, 0, 0, 1, 0): "pkc",  # (913 samples) pma (activator)
    #     (0, 0, 0, 0, 0, 0, 0, 0, 1): "pka"   # (707 samples) b2camp
    # }
    
    unique_ints = df["INT"].unique()
    # get the list of intervention targets and list of dataframe associated with each intervention
    intervention_targets = [(df.columns[idx], idx) for idx in unique_ints]
    data_cols = [col for col in df.columns if col != "INT"]
    data = pd.DataFrame()
    i_data = []
    for interv_idx in unique_ints:
        _data = df[df["INT"] == interv_idx][data_cols]
        if interv_idx != 0:
            i_data.append(_data)
        else:
            data = _data
        
    unique_idxs = [i-1 for i in unique_ints]
    return data, i_data, [(df.columns[i], set([i])) for i in unique_idxs if i >= 0]


def loadSachsInterventionalDiscrete():
    df = pd.read_csv("../data/sachs/data/sachs.interventional.txt", sep=r'\s+')
    unique_ints = df["INT"].unique()
    # get the list of intervention targets and list of dataframe associated with each intervention
    intervention_targets = [(df.columns[idx], idx) for idx in unique_ints]
    data_cols = [col for col in df.columns if col != "INT"]
    data = pd.DataFrame()
    i_data = []
    for interv_idx in unique_ints:
        _data = df[df["INT"] == interv_idx][data_cols]
        if interv_idx != 0:
            i_data.append(_data)
        else:
            data = _data
        
    unique_idxs = [i-1 for i in unique_ints]
    return data, i_data, [(df.columns[i], set([i])) for i in unique_idxs if i >= 0]


def loadSachsGroundTruth():
    # Read content and decode to string
    f = open('../data/sachs/ground_truth/sachs.2005.ground.truth.graph.txt')
    content = f.read().splitlines()
    f.close()
    
    # see https://github.com/cmu-phil/example-causal-datasets/blob/main/formatting.txt
    edge_re = re.compile(r"(?<![^\s>])[0-9]+\. ([^\s]*) --> ([^\s]*)")
    
    if content[0] == "Graph Nodes:":
        node_list = [GraphNode(name) for name in content[1].split(";")]
        ground_truth_graph = Dag(node_list)
        if content[3] == "Graph Edges:":
            for i in range(4, len(content)):
                line_str = content[i]
                if line_str:
                    line_edge = edge_re.match(content[i]).group(1, 2)
                    node_from = ground_truth_graph.get_node(line_edge[0])
                    node_to = ground_truth_graph.get_node(line_edge[1])
                    ground_truth_graph.add_directed_edge(node_from, node_to)
        else:
            print("loadSachsGroundTruth: Failed to find any graph edges")
            return ""
    else:
        print("loadSachsGroundTruth: Failed to find graph nodes")
        return ""

    return ground_truth_graph


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

sachs_dat, sachs_i_dat, sachs_i_nodes = loadSachsInterventionalContinuous()

x=1