import gzip
import numpy as np
import pickle
import torch
from scipy.sparse import csc_array
import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import glob


def pklz_to_incmat(pkl):
    """
    Convert obtained dataset (.pklz) to hypergraph incidence matrix (scipy.sparse.csc)
    return: H (scipy.sparse), net2node (list)
    """


    # with gzip.open(f'../DREAMPlace/install/dataset/{top}/{top}_0.7_1.0.icc2.pklz') as f:
    with gzip.open(pkl) as f:
        xx = pickle.load(f)

    pin2net = np.array(xx['pin_info']['pin2net_map'])
    pin2node = np.array(xx['pin_info']['pin2node_map'])

    H = csc_array((np.ones(len(pin2net)), (pin2node, pin2net)))

    # flat_net2pin_map = np.array(xx['net_info']['flat_net2pin_map'])s
    # flat_node2pin_map =  np.array(xx['node_info']['flat_node2pin_map'])
    # flat_node2pin_start_map =  np.array(xx['node_info']['flat_node2pin_start_map'])

    # my_net2pin_map = np.array([flat_net2pin_map[flat_net2pin_start_map[i]:flat_net2pin_start_map[i+1]] \
    #                             for i in range(len(flat_net2pin_start_map) - 1)], dtype=object)
    
    net2node = [[] for _ in range(H.shape[1])]
    for pin, net in enumerate(pin2net):
        net2node[net].append(pin2node[pin])

    for i, net in enumerate(net2node):
        net2node[i] = list(set(net))

    return H, net2node