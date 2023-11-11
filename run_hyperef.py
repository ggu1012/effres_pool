import gzip
import pickle
from glob import glob
import os
import re
import ray


from utils.HyperEF.HyperEF import HyperEF
from utils.hypergraph_conversion import *
from utils.functions import *

####
main_level = 2
tail = 2
device = 'cuda:0'
top = 'bsg_chip'
####
sub_level = 2

@ray.remote
def run_HyperEF(dataset, main_level, sub_level):
    basen = os.path.basename(dataset)
    top_var = re.match(r'^(.*)\.icc2\.pklz$', basen).group(1)
    chunkname = f'graph/pre_process/{top_var}.m{main_level}.s{sub_level}.pkl'
    with gzip.open(dataset, 'rb') as f:
        dataset = pickle.load(f)
    H, net2node, net_map, node_map = pklz_to_incmat(dataset)
    idx_mat, new_net2nodes = HyperEF(net2node, main_level, sub_level, 42312)
    net2nodes = [net2node, *new_net2nodes]
    with gzip.open(chunkname, 'wb') as f:
        pickle.dump((H, idx_mat, net2nodes, net_map, node_map), f)

    return 1

def main():
    top = ''
    ray.init(num_cpus = 16)
    if top:
        dsets = glob(f'../DREAMPlace/install/dataset/{top}/*.icc2.pklz')
    else:
        dsets = glob(f'../DREAMPlace/install/dataset/*/*.icc2.pklz')

    jobs = []
    for ml in [2, 3]:
        for sl in [2, 3]:
            jobs += [run_HyperEF.remote(dset, ml, sl) for dset in dsets]
    result = ray.get(jobs)

    print(result)

if __name__ == '__main__':
    main()
