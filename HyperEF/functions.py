import numpy as np
from scipy.sparse import diags, lil_array

# return approx. eigvecs of L
def Filter(rand_vec, gram_schmidt_iter, adj_mat, num_nodes, interval, num_vecs):
    sm_vec = np.zeros((num_nodes, gram_schmidt_iter))

    adj_mat = adj_mat.astype(float)
    adj_mat.row = np.append(adj_mat.row, np.arange(adj_mat.shape[0]))
    adj_mat.col = np.append(adj_mat.col, np.arange(adj_mat.shape[0]))
    adj_mat.data = np.append(adj_mat.data, 0.1 * np.ones(adj_mat.shape[0]))
    adj_mat = lil_array(adj_mat)

    deg = adj_mat.sum(axis=0)
    # D_sqrt = diags(1/np.sqrt(deg), 0)
    one_vec = np.ones(len(rand_vec), dtype=int)
    sm_ot = rand_vec - ((rand_vec.dot(one_vec) / one_vec.dot(one_vec)) * one_vec) # orth. to one_vec
    ## span(sm_ot, one_vec) = span(rand_vec, one_vec)
    sm = sm_ot / np.linalg.norm(sm_ot)

    D_inv = diags(1/deg, 0)
    Ax = D_inv @ adj_mat
    for loop in range(gram_schmidt_iter):
        # sm = D_sqrt @ sm
        # sm = adj_mat @ sm
        # sm = D_sqrt @ sm
        sm = Ax @ sm
        sm_ot = sm - ((sm.dot(one_vec) / one_vec.dot(one_vec)) * one_vec) # orth. to one_vec and sm
        ## span(sm, one_vec) = span(sm_ot, one_vec)
        sm_norm = sm_ot / np.linalg.norm(sm_ot)
        # print(sm_ot[0])
        # print(sm_norm[0])
        # print("K: ", loop, " ", sm.dot(one_vec) / one_vec.dot(one_vec))
        sm_vec[:, loop] = sm_norm[:num_nodes] # get edge-nodes for star
        
    return sm_vec[:,::interval]

def HyperEdgeScore(hg, SV):
    score = np.zeros((len(hg), SV.shape[1]))
    for eid, nodes in enumerate(hg):
        x = SV[nodes]
        mx = np.max(x, axis=0)
        mn = np.min(x, axis=0)
        score[eid] = np.power((mx - mn), 2)
    return score

def get_num_nodes(hinc):
    num_nodes = 0
    for x in hinc:
        for y in x:
            if y > num_nodes:
                num_nodes = y
    num_nodes += 1
    return num_nodes