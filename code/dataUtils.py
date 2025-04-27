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

def loadSachsDataset():
    dataset = pd.read_csv("../data/sachs/data/sachs.2005.logxplus10.continuous.txt", sep=r'\s+')
    return dataset

def loadSachsDatasetWithExperiments():
    dataset = pd.read_csv("../data/sachs/data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt", sep=r'\s+')
    return dataset

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

def loadSachsObservationalDiscrete():
    df = pd.read_csv("../data/sachs/data/sachs.interventional.txt", sep=r'\s+')
    data_cols = [col for col in df.columns if col != "INT"]
    data = df[df["INT"] == 0][data_cols]
    return data

def loadSachsObservationalContinuous():
    dataset = pd.read_csv("../data/sachs/data/sachs.2005.logxplus10.continuous.txt", sep=r'\s+')
    return dataset

def loadSachsObservational():
    intervention_cols = ['cd3_cd28', 'icam2', 'aktinhib', 'g0076', 'psitect', 'u0126', 'ly', 'pma', 'b2camp']
    dataset_log = pd.read_csv("../data/sachs/data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt", sep=r'\s+')
    dataset_continuous = pd.read_csv("../data/sachs/data/sachs.2005.continuous.txt", sep=r'\s+')
    # retrieve activated Th cells, which include cd3_cd28 and icam2 groups
    filtered_data_log = dataset_log[((dataset_log['cd3_cd28'] == 1) | (dataset_log['icam2'] == 1)) &
                                    (dataset_log[[col for col in intervention_cols if col not in ['cd3_cd28', 'icam2']]] == 0).all(axis=1)]
    filtered_data_continuous = dataset_continuous.loc[filtered_data_log.index]
    filtered_data_log = filtered_data_log.drop(columns=intervention_cols)
    return filtered_data_log, filtered_data_continuous

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


sachs_ground = loadSachsGroundTruth()