import torch
# import dgl
import numpy as np
import gzip
import re
from glob import glob
import random
import os
import ray
import torch.nn.functional as F

# from utils.HyperEF import HyperEF
from utils.hypergraph_conversion import *
from utils.functions import *

def HPWL(pklz_, new_net_mapper, dir="dataset/y_HPWL", num_labels = 10, labeled = True):
    dset_name = pklz_[0]
    pklz = pklz_[1]
    dset_parse = re.match(
        "^(.*)_(\d\.\d+)_(\d\.\d+)\.icc2.pklz$", os.path.basename(dset_name)
    )
    top = dset_parse.group(1)
    core_util = dset_parse.group(2)
    core_aspect = dset_parse.group(3)

    pin2node = np.array(pklz["pin_info"]["pin2node_map"])
    flat_net2pin_map = np.array(pklz["net_info"]["flat_net2pin_map"])
    flat_net2pin_start_map = np.array(pklz["net_info"]["flat_net2pin_start_map"])

    po_x = np.array(pklz["pin_info"]["pin_offset"])[0]
    po_y = np.array(pklz["pin_info"]["pin_offset"])[1]
    pos_x = np.array(pklz["node_info"]["node_position"])[0]
    pos_y = np.array(pklz["node_info"]["node_position"])[1]

    hpwl_collection = []
    my_net2pin_map = np.array(
        [
            flat_net2pin_map[flat_net2pin_start_map[i] : flat_net2pin_start_map[i + 1]]
            for i in range(len(flat_net2pin_start_map) - 1)
        ],
        dtype=object,
    )

    for n in new_net_mapper:
        pins = my_net2pin_map[n]
        pin_offset = np.vstack([po_x[pins], po_y[pins]]).T
        pos = np.vstack([pos_x[pin2node[pins]], pos_y[pin2node[pins]]]).T
        pin_pos = pos + pin_offset
        amax = np.max(pin_pos, axis=0)
        amin = np.min(pin_pos, axis=0)

        hpwl_collection.append((amax - amin).sum())
    hpwl_collection = torch.tensor(hpwl_collection)

    #total_info = torch.vstack([hpwl_collection, label]).T

    if labeled:
        logged = torch.log10(hpwl_collection)
        bins = torch.histogram(logged, bins=num_labels)[1]
        ground_truth = (torch.bucketize(logged, bins, right=True) - 1)
        ground_truth[ground_truth == num_labels] = num_labels - 1
        
        torch.save(
            ground_truth, os.path.join(dir, f"{top}_{core_util}_{core_aspect}_label.pt")
        )

    else:
        torch.save(
            hpwl_collection, os.path.join(dir, f"{top}_{core_util}_{core_aspect}_raw.pt")
        )

    
    return len(hpwl_collection)

def node_feature(H, pklz_, dtl_A, src_dst, node_mapper, dir="dataset/x_node_feature"):
    ## node feature
    # cell area
    # num_fan_ins, num_fan_outs
    # node degree
    # TODO: cell type; macro / cell / io

    dset_name = pklz_[0]
    pklz = pklz_[1]
    dset_parse = re.match(
        "^(.*)_(\d\.\d+)_(\d\.\d+)\.icc2.pklz$", os.path.basename(dset_name)
    )
    top = dset_parse.group(1)
    core_util = dset_parse.group(2)
    core_aspect = dset_parse.group(3)

    num_movable_nodes = pklz["db_place_info"]["num_movable_nodes"]
    num_terminals = pklz["db_place_info"]["num_terminals"]
    num_terminal_NIs = pklz["db_place_info"]["num_terminal_NIs"]
    nd_size_x = pklz["node_info"]["node_size"][0][
        : num_movable_nodes + num_terminals + num_terminal_NIs
    ][node_mapper]
    nd_size_y = pklz["node_info"]["node_size"][1][
        : num_movable_nodes + num_terminals + num_terminal_NIs
    ][node_mapper]
    node_degree = H.sum(axis=1)

    num_fanout = np.zeros(len(nd_size_x), dtype=int)
    for _net in src_dst.keys():
        src = src_dst[_net][0]
        dsts = src_dst[_net][1]
        num_fanout[src] = len(dsts)

    num_fanin = dtl_A.sum(axis=0) - num_fanout

    collection = torch.vstack(
        [
            torch.tensor(nd_size_x),
            torch.tensor(nd_size_y),
            torch.tensor(node_degree),
            torch.tensor(num_fanin),
            torch.tensor(num_fanout),
        ]
    )

    nfeat_collection = collection.T
    torch.save(
        nfeat_collection, os.path.join(dir, f"{top}_{core_util}_{core_aspect}.pt")
    )

    return nfeat_collection.shape

@ray.remote
def generate_dataset(dsets):
    for dname in dsets:
        with gzip.open(dname, "rb") as f:
            obj = pickle.load(f)

        H, net2node, new_net_mapper, new_node_mapper = pklz_to_incmat(obj)
        dtl_A, src_dst = driver2load(H, obj, new_node_mapper)
        yy = HPWL((dname, obj), new_net_mapper, labeled=True)
        # xx = node_feature(H, (dname, obj), dtl_A, src_dst, new_node_mapper)

    return dname

if __name__ == "__main__":
    dset_dirs = glob('../DREAMPlace/install/dataset/*')
    dset_files = []
    for dset_dir in dset_dirs:
        dset_name = os.path.basename(dset_dir)
        dset_files += glob(f'../DREAMPlace/install/dataset/{dset_name}/*.icc2.pklz')
    random.shuffle(dset_files)

    dset_file_split = []
    files_per_chunk = 10
    th_num = round(len(dset_files) / files_per_chunk)
    for t in range(th_num - 1):
        dset_file_split.append(dset_files[t:files_per_chunk*(t+1)])
    dset_file_split[-1] += dset_files[files_per_chunk*t:]

    ray.init()
    job_list = [generate_dataset.remote(x) for x in dset_file_split]
    print(f"Jobs : {len(job_list)}")
    result = ray.get(job_list)

    # with gzip.open('../DREAMPlace/install/dataset/bsg_chip/bsg_chip_0.7_1.0.icc2.pklz', 'rb') as f:
    #     obj = pickle.load(f)
    # H, net2node, new_net_mapper = pklz_to_incmat(obj)
    # xx = star_hetero(H)
    # print(xx)
