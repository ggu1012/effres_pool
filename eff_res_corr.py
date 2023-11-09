#!/usr/bin/env python
# coding: utf-8

import gzip
import numpy as np
import pickle
from scipy.sparse import csr_array
import os
import ray
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import re
import random
import matplotlib
from matplotlib import colors
import copy



@ray.remote
def th_job(dset_files):

    result_list = []
    for dset in dset_files:
        
        dset_parse = re.match('^(.*)_(\d\.\d+)_(\d\.\d+)\.icc2.pklz$', os.path.basename(dset))
        top = dset_parse.group(1)
        core_util = dset_parse.group(2)
        core_aspect = dset_parse.group(3)

        with gzip.open(dset) as f:
            dataset = pickle.load(f)

        pin2net = np.array(dataset['pin_info']['pin2net_map'])
        pin2node = np.array(dataset['pin_info']['pin2node_map'])

        # hypergraph incidence matrix
        H = csr_array((np.ones(len(pin2net)), (pin2node, pin2net)))

        flat_net2pin_map = np.array(dataset['net_info']['flat_net2pin_map'])
        flat_net2pin_start_map = np.array(dataset['net_info']['flat_net2pin_start_map'])
        flat_node2pin_map =  np.array(dataset['node_info']['flat_node2pin_map'])
        flat_node2pin_start_map =  np.array(dataset['node_info']['flat_node2pin_start_map'])

        my_net2pin_map = np.array([flat_net2pin_map[flat_net2pin_start_map[i]:flat_net2pin_start_map[i+1]] \
                                        for i in range(len(flat_net2pin_start_map) - 1)], dtype=object)

        po_x = np.array(dataset['pin_info']['pin_offset'])[0]
        po_y = np.array(dataset['pin_info']['pin_offset'])[1]
        pos_x = np.array(dataset['node_info']['node_position'])[0]
        pos_y = np.array(dataset['node_info']['node_position'])[1]

        my_net2pin_map = np.array([flat_net2pin_map[flat_net2pin_start_map[i]:flat_net2pin_start_map[i+1]] \
                                    for i in range(len(flat_net2pin_start_map) - 1)], dtype=object)
        my_node2pin_map = np.array([flat_node2pin_map[flat_node2pin_start_map[i]:flat_node2pin_start_map[i+1]] \
                                    for i in range(len(flat_node2pin_start_map) - 1)], dtype=object)

        net2node = [[] for _ in range(H.shape[1])]
        for pin, net in enumerate(pin2net):
            net2node[net].append(str(pin2node[pin] + 1))

        fname = f'{top}_{core_util}_{core_aspect}'        
        hpwl_collection = []
        for n in range(len(my_net2pin_map)):
            pins = my_net2pin_map[n]
            pin_offset = np.vstack([po_x[pins], po_y[pins]]).T
            pos = np.vstack([pos_x[pin2node[pins]], pos_y[pin2node[pins]]]).T

            pin_pos = pos + pin_offset

            amax = np.max(pin_pos, axis=0)
            amin = np.min(pin_pos, axis=0)

            hpwl_collection.append((amax - amin).sum())
        hpwl_collection = np.array(hpwl_collection)

        # load eratio
        eratio = np.load(f'HyperEF/src/tmp/{top}.npz')

        # Approx. eff. res.
        evec = np.sort(eratio, axis=1)[:,::-1]
        P = evec[:,0]
        
        # ignore singleton net
        zero_idx = np.argwhere(hpwl_collection == 0).flatten()
        hpwl_ = np.delete(hpwl_collection, zero_idx)
        evec_ = np.delete(P, zero_idx)
        degree_ = np.delete(np.array([len(x) for x in net2node]), zero_idx)
        
        evec_[evec_ < 10e-10] = 10e-10
        
        fig = plt.figure(figsize=(12,8))
        fig.set_facecolor('white')
        my_cmap = copy.copy(matplotlib.colormaps['afmhot']) # copy the default cmap
        my_cmap.set_bad((0,0,0))
        
        ax1 = fig.add_subplot(221)
        h = ax1.hist2d(np.log10(hpwl_), np.log10(evec_), bins=40, norm=colors.LogNorm(), cmap=my_cmap)
        ax1.set_xlabel('HPWL (log)')
        ax1.set_ylabel('Eff_res (log)')
        cur_ax = plt.gca() ## 현재 Axes
        fig.colorbar(h[3],ax=cur_ax) ## 컬러바 추가
        
        ax2 = fig.add_subplot(222)
        h = ax2.hist2d(np.log10(hpwl_), np.log10(evec_), bins=40, norm=colors.LogNorm(), cmap=matplotlib.colormaps['afmhot'])
        ax2.set_xlabel('HPWL (log)')
        ax2.set_ylabel('Eff_res (log)')
        cur_ax = plt.gca() ## 현재 Axes
        fig.colorbar(h[3],ax=cur_ax) ## 컬러바 추가
        
        ax3 = fig.add_subplot(223)
        h = ax3.hist2d(np.log10(hpwl_), np.log10(degree_), bins=40, norm=colors.LogNorm(), cmap=my_cmap)
        ax3.set_xlabel('HPWL (log)')
        ax3.set_ylabel('degree (log)')
        cur_ax = plt.gca() ## 현재 Axes
        fig.colorbar(h[3],ax=cur_ax) ## 컬러바 추가
        
        ax4 = fig.add_subplot(224)
        h = ax4.hist2d(np.log10(hpwl_), np.log10(degree_), bins=40, norm=colors.LogNorm(), cmap=matplotlib.colormaps['afmhot'])
        ax4.set_xlabel('HPWL (log)')
        ax4.set_ylabel('degree (log)')
        cur_ax = plt.gca() ## 현재 Axes
        fig.colorbar(h[3],ax=cur_ax) ## 컬러바 추가
        
        fig.savefig(f'eff_res_plot/{top}_{core_util}_{core_aspect}.png')
        
        
        df = pd.DataFrame()
        df['hpwl'] = np.log10(hpwl_)
        df['eff_res'] = np.log10(evec_)
        df['degree'] = np.log10(degree_)

        # pearson, kendall, spearman, pearson_deg, kendall_deg, spearman_deg
        result = [
            df.corr(method='pearson')['hpwl']['eff_res'],
            df.corr(method='kendall')['hpwl']['eff_res'],
            df.corr(method='spearman')['hpwl']['eff_res'],
            df.corr(method='pearson')['hpwl']['degree'],
            df.corr(method='kendall')['hpwl']['degree'],
            df.corr(method='spearman')['hpwl']['degree']
        ]

        plt.close(fig)

        result_list.append([(top, core_util, core_aspect), result])
    return result_list
    
def main():
    dset_dirs = glob('../DREAMPlace/install/dataset/*')
    dset_files = []
    for dset_dir in dset_dirs:
        dset_name = os.path.basename(dset_dir)
        dset_files += glob(f'../DREAMPlace/install/dataset/{dset_name}/*.icc2.pklz')
        print(dset_name)
        print(glob(f'../DREAMPlace/install/dataset/{dset_name}/*.icc2.pklz'))
        print(len(dset_files))
    random.shuffle(dset_files)
    print(len(dset_files))

    dset_file_split = []
    files_per_chunk = 10
    th_num = round(len(dset_files) / files_per_chunk)
    for t in range(th_num - 1):
        dset_file_split.append(dset_files[t:files_per_chunk*(t+1)])
    dset_file_split[-1] += dset_files[files_per_chunk*t:]

    ray.init()
    col = ['top', 'util', 'aspect', 'pearson', 'kendall', 'spearman', 'pearson_deg', 'kendall_deg', 'spearman_deg']
    job_list = [th_job.remote(x) for x in dset_file_split]
    print(f"Jobs : {len(job_list)}")
    result = ray.get(job_list)

    with open('eff_res_rpt.csv', 'w') as f:
        f.write(','.join(col) + '\n')
        for chunk in result:
            for ln in chunk:
                xres = [str(x) for x in ln[1]]
                f.write(f"{ln[0][0]},{ln[0][1]},{ln[0][2]}," + ','.join(xres) + '\n')

if __name__ == '__main__':
    main()
