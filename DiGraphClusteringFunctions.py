#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
from scipy.linalg import eigh as eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from itertools import permutations
import random
import matplotlib.pyplot as plt


# In[ ]:


def directed_cycle(k):
    """
    adjacency matrix of a directed k cycle
    """
    A = np.zeros((k,k))
    for i in range(k-1):
        A[i, i+1] = 1
    A[k-1, 0] = 1
    return A


# In[ ]:


def DSBM(k, n, p, q, F):
    """
    Directed Stochastic Block Model 
    (see *Hermitian matrices for clustering directed graphs: insights and applications*).
    Note: This assumes evenly sized clusters
    
    Input
     k: Number of clusters {C_1, C_2, ..., C_k}
     n: Number of vertices in each cluster
     p: probability of an edge between two vertices in the same cluster
     q: probability of an edge between two vertices in different clusters
     F: k x k matrix controlling the orientation of the edges. F_{uv} + F_{vu} = 1
     
    Return
     G: nx.DiGraph()
    """
    N = n*k
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(N))
    
    # add the edges
    for cluster in range(k):
        # do intra cluster first
        for i in range(n-1):
            u = i + n*cluster
            for j in range(i+1, n):
                v = j + n*cluster
                if np.random.rand() < p: #edge exists
                    # pick orientation uniformly at random
                    #  corresponds to diagonal of F
                    if np.random.rand() < 0.5:
                        G.add_edge(u,v)
                    else:
                        G.add_edge(v,u)
    
    # Do inter cluster now
    for clust in range(k-1):
        for other in range(clust + 1, k):
            for i in range(n):
                # get the elements in 
                u = i + n*clust
                for j in range(n):
                    #iterating through other's nodes
                    v = j + n*other
                    if np.random.rand() < q: # add edge
                        if np.random.rand() < F[clust, other]:
                            G.add_edge(u,v)
                        else:
                            G.add_edge(v,u)
                    
                    
    return G
    


# In[ ]:


def Hermitian_A(G):
    """
    Given directed graph G, return the matrix A_H.
    A_H(u,v) = i if edge (u,v) exists, -i if (v, u) exists, and 1 if both exist.
    Otherwise 0
    
    NOTE: Probably better to make a function that deals with adjacency list
    rather than a 2d np.array. To do later
    
    Parameters
     G: nx.DiGraph()
     
    Returns
     Ah: np.array with complex entries
    """
    A = nx.to_numpy_array(G)
    N = len(A)
    Ah = np.zeros_like(A, dtype=complex)
    for u in range(N):
        for v in range(u+1, N):
            uv = A[u,v]
            vu = A[v,u]
            if uv == 1 and vu == 0:
                Ah[u,v] = 1j
                Ah[v,u] = -1j
            if uv == 1 and vu == 1:
                Ah[u,v] = 1
                Ah[v,u] = 1
            if uv == 0 and vu == 1:
                Ah[u,v] = -1j
                Ah[v,u] = 1j
    return Ah


# In[ ]:


def Algorithm1(G, ell, n_init = 'auto', rw = False, tau=0):
    """
    Algorithm 1 as found in
    "Hermitian matrices for clustering directed graphs: insights and applications"
    
    Parameters
     G: nx.DiGraph
     ell: number of eigenpairs to use
     rw: (bool) if do random walk Hermitian A or not
     
    Returns
     partition: partition of nodes into clusters
     
    NOTES: Important imports implicit here
     eigh:    scipy.linalg.eigh
     KMeans:  sklearn.clusters.KMeans
     np:      numpy
    """
    N = len(G)
    
    if not rw:
        A = Hermitian_A(G)
        
    if rw:
        A = Hermitian_A(G)
        #d = np.abs(A)@np.ones(N) # more efficient way surely but later
        d = np.sum(np.abs(A), axis=1) # row sums of |A|
        d = d + tau
        d = d**(-1/2)
        A = np.diag(d)@A
        
        
    w, v = eigh(A, subset_by_index = [N-ell, N-1]) # w is the ell largest eigenvalues
    P = v@v.conj().T # v is N x ell matrix of ell largest eigenvectors
    #NOTE!!: possible difference here from paper
    #      as we don't use an \epsilon, also w has the largest, not
    #      largest magnitude. Results haven't seemed to change too much
    #      but probably good to correct at some point
        
    
    # Do kmeans on the rows of P
    # kmeans doesn't work on complex
    #  so make the dimension twice as large but real valued
    P2N = np.hstack([P.real, P.imag])
    
    kmeans = KMeans(n_clusters = ell, n_init=n_init) #consider putting a random_state in for testing
    kmeans.fit(P2N)
    
    return kmeans.labels_


# In[ ]:


def create_F(k,eta):
    """
    Creates F for DSBM, k x k matrix with noise level eta
    NOTE: Currently only works for directed cycle
    """
    F = np.zeros((k,k))

    # k x k adj of directed cycle (with noise)
    for j in range(k):
        for ell in range(k):
            if (j % k) == ((ell-1) % k):
                F[j, ell] = 1 - eta
            elif (j % k) == ((ell + 1)%k):
                F[j, ell] = eta
            else:
                F[j, ell] = 0.5
    return F


# In[ ]:


def DSBM_true_labels(k, n):
    """
    list (np.array) of size n*k where the first k elements are 0, then k 1s, ..., k (n-1)s
    """
    true_labels = []
    for cluster in range(k):
        for node in range(n):
            true_labels.append(cluster)
    return np.array(true_labels)


# In[ ]:


def get_cut(G, cl, i, j, score='ratio', signed = False, use_matrix = False, matrix_to_use = []):
    """
    Cut scores between cluster i and cluster j
    as in *Hermitian matrices for clustering directed graphs: insights and applications*
    
    Input
     G: nx.DiGraph
     cl: (array-like) cl[k] = cluster membership of node k
     i, j: (ints) clusters of interest (i --> j)
     score: (str) cut_score variant. 
         'ratio' -- unnormalized
         'size' -- multiplied by min of sizes of clusters i and j
         'volume' -- multiplied by min of volume of cluser i and j
     signed: (bool) If signed is False take the absolute value of the score
     use_matrix: (bool) if false scores calculated using G. If True, scores
                         calculated using an np.array (e.g. Adjacency matrix)
     matrix_to_use: (np.array) If not empty, use that matrix to calculate cut score
                                 (e.g. if want a weighted graph. can accept weighted adjacency or 
                                 weighted hermitian adjacency)/
    
    Returns:
     cut score cluster i to cluster j
     
    NOTES: probably could/should have not included use_matrix and matrix_to_use
            as currently not trying to deal with weighted graphs, just checking for
            possible bugs in code
    """
    #if (i not in np.arange(0, k)) or (j not in np.arange(0, k)):
    #    print("Clusters out of bounds: i,j must be in \{0, 1, ..., k-1\}")
    #    return 0
    X = np.where(cl == i)[0]
    Y = np.where(cl == j)[0]
    
    wXY = 0
    wYX = 0
                
    if use_matrix:
        # use an np.array to get edge data
        if len(matrix_to_use) > 0:
            A = matrix_to_use
            for vi in X:
                for vj in Y:
                    wXY += max(A[vi, vj], 0)
                    wYX += max(A[vj, vi], 0)
                    
        if len(matrix_to_use) == 0:
            # Didn't supply a matrix
            use_matrix = False
            
            
    if not use_matrix:
        # we just use graph data structure. i.e. not weighted edges
        for vi in X:
            for vj in Y:
                #print(f"x = {vi}, y = {vj}, has_edge(x,y) = {G.has_edge(vi, vj)}")
                wXY += G.has_edge(vi, vj) # adds 1 if edge exists, else adds 0
                wYX += G.has_edge(vj, vi) # similar
                
                
    # get the ratio
    # recall SCI is Signed Cut Imbalance
    try:
        SCI = wXY/(wXY + wYX) - 0.5
    except ZeroDivisionError:
        print(f"ZeroDivisionError: Cluster {i} and Cluster {j} have no edges between them?")
        return 0
    
    if score == 'ratio':
        if signed:
            return SCI
        else:
            return np.abs(SCI)
    
    
        
    if score == 'size':
        if signed:
            return SCI * min(len(X), len(Y))
        else:
            return np.abs(SCI * min(len(X), len(Y)))
        
    
    
    if score == 'vol':
        volX = 0
        volY = 0
        if use_matrix:
            for z in range(A.shape[0]):
                for x in X:
                    volX += max(A[x, z], 0)
                for y in Y:
                    volY += max(A[y, z], 0)
        else:
            for x in X:
                volX += G.degree(x)
            for y in Y:
                volY += G.degree(y)
                
        if signed:
            return SCI * min(volX, volY)
        else:
            return np.abs(SCI * min(volX, volY))


# In[ ]:


def cutscore_meta_matrix(G, cl, score='ratio', signed = False, use_matrix = False, matrix_to_use = []):
    """
    create a meta adjacency matrix with the cut score defining the weights
    
    Input
     G: nx.DiGraph
     cl: (array-like) cl[k] = cluster membership of node k
     i, j: (ints) clusters of interest (i --> j)
     score: (str) cut_score variant. 
         'ratio' -- unnormalized
         'size' -- multiplied by min of sizes of clusters i and j
         'volume' -- multiplied by min of volume of cluser i and j
     signed: (bool) If signed is False take the absolute value of the score
     use_matrix: (bool) if false scores calculated using G. If True, scores
                         calculated using an np.array (e.g. Adjacency matrix)
     matrix_to_use: (np.array) If not empty, use that matrix to calculate cut score
                                 (e.g. if want a weighted graph. can accept weighted adjacency or 
                                 weighted hermitian adjacency)
    Returns
     M: 2d np.array
     
    
    NOTE: update to use **kwargs maybe for cleaner definition?
    """
    meta_nodes = np.unique(cl)
    k = len(meta_nodes)
    M = np.zeros((k,k))
    
    # Loop through the clusters
    #  get the cut scores
    #  use those as the entries of the meta adjacency matrix
    for i in meta_nodes:
        for j in meta_nodes:
            M[i,j] = get_cut(G, cl, i, j, score=score, signed = signed, use_matrix = use_matrix, matrix_to_use = matrix_to_use)
            
    return M
            


# In[ ]:


def make_unweighted(M, eps = 0.2):
    """
    Convert a real skew-symmetric matrix to a 0/1 directed adjacency matrix by zeroing out entries
    small in magnitude. Eliminates negative values too.
    (M being skew-symmetric doesn't come into play in the code,
     its just the types of matrices we are using)
    
    Input:
     M: (2d np.array) real matrix (corresponding to the meta_matrix above)
     eps: (numeric) tolerance. If M[i,j] < eps set M[i,j]=0
    """
    k = M.shape[0]
    N = np.zeros((k,k))
    
    for i in range(k):
        for j in range(k):
            if M[i, j] > eps:
                N[i,j] = 1
    return N


# In[ ]:


def find_min_perm(M, target, order = 'fro'):
    """
    Find permutation that minimizes |pi^TMpi - target|
    
    Input:
     M: (2d np.array) - given matrix (meta matrix in our case)
     target: (2d np.array) - 0/1 adjacency matrix we are hoping to recover
     order: (float or str) - ord used in np.linalg.norm (what type of matrix or vector norm)
     
    Returns:
     minperm: (tuple) - the permutation pi that minimized the distance
     minMat: (2d np.array) - the matrix that achieved the minimum
     
    Imports assumed:
     import numpy as np
     from itertools import permutations
    """
    k = target.shape[0]
    perms = list(permutations(np.arange(k)))
    
    minperm = []
    minMat = M
    mindist = np.inf # initialize to infinity
    
    for perm in perms:
        Mhat = M[list(perm)]
        Mhat = Mhat[:,list(perm)] # Mhat is M with permuted rows and columns
        currdist = np.linalg.norm(Mhat - target, ord = order)
        if currdist < mindist:
            mindist = currdist
            minperm = perm
            minMat = Mhat
            
    return minperm, minMat
        


# In[ ]:


def flip_edges(G, M, cl, i, j, want_meta_edge):
    """
    flip edges from cluster i to cluster j to better match what we expect in
    the target matrix (side information)
    
    Input:
     G: nx.DiGraph
     M: (2d np.array) meta_cut_matrix with score='ratio'
     cl: (array-like) clustering of G
     i, j: (int) clusters of interest. If want an edge it will be from i to j
     want_meta_edge: (bool) True--flip to create meta edge i to j (make cut score large)
                            False--flip to destroy meta edge (make cut score near 0)
    Return:
     H: (nx.DiGraph) G with some flipped edges
     
     NOTE: Assuming M[i,j] >= 0
    """
    cl1 = np.where(cl == i)[0]
    cl2 = np.where(cl == j)[0]
    H = G.copy()
    
    if not want_meta_edge:
        # e.g. i goes to j but should be random
        # flipping probability
        alpha = M[i,j]
        prob = alpha/(alpha + 0.5) # some minor algebra on explanation in 285J report page 7
        for u in cl1:
            for v in cl2:
                if H.has_edge(u, v):
                    if random.random() < prob: #random.random() \in [0, 1)
                        # flip the edge with probability prob
                        H.remove_edge(u, v)
                        H.add_edge(v, u)
                        
        return H
    
    # if we create the edge instead
    for u in cl1:
        for v in cl2:
            if H.has_edge(v, u): # has wrong orientation
                H.remove_edge(v, u)
                H.add_edge(u, v)
    return H


# In[ ]:


def meta_graph_match(G, k, target, true_labels, order='fro', maxiters = 10, rw=True, eps=0.2, **kwargs):
    """
    iteratively cluster, find best permutation to match the target meta matrix,
    flip the edges accordingly in the clusters matching the largest discrepancy
    between the current meta matrix and the target.
    (When flipping edges, we are working in a copy of G, not altering G directly)
    As described in Algorithm MGM in 285J project report
    
    NOTE: This is the function that all the previous ones were building to
    
    NOTE: for now at least, working with DSBM so we have access to the true clustering
            so calculate the ari at each stage
    
    Input: 
     G: (nx.DiGraph)
     k: (int) number of clusters
     ##M: (2d np.array) Meta Graph of cluster
     target: (2d np.array) side information (desired meta graph)
     true_labels: (array-like) the true labeling of 
     order: (str) order command for find_min_perm
     maxiters: (int) max number of iterations of 'edge fixing' we'll do
     **kwargs: same as cutscore_meta_matrix 
     
    Return:
    cl_new: the final clustering
    ari_list: list of the ari scores at each step
    """
    # maybe don't input cl, just do it here
    ari_list = []
    
    H = G.copy()
    i = 0
    while i < maxiters:
        # calculate a clustering and ari score
        C = Algorithm1(H, k, rw = rw)
        ari = adjusted_rand_score(true_labels, C)
        ari_list.append(ari)
        
        # get the meta flow matrix (i.e. adjacency from cluster cut scores)
        A = cutscore_meta_matrix(H, C, **kwargs)
        A[A < 0] = 0 # makes negative values 0, to better look like adj matrix
        B = make_unweighted(A, eps=eps) # this is possibly redundant now? I guess A still has small pos values
        
        # do the graph matching
        minperm, permA = find_min_perm(A, target, order = order)
        permB = B[list(minperm)]
        permB = permB[:,list(minperm)]
        
        # if we already have the right meta graph
        if np.max(permB - target) == 0:
            return C, ari_list, permB, minperm
        
        # i,j giving largest |permA - target|_{i,j}
        imax, jmax = np.unravel_index(np.argmax(np.abs(permA - target)), (k, k))
        
        if (permA - target)[imax, jmax] > 0:
            # i.e. we have a meta edge that we don't want 
            #  since permA_{imax, jmax} > target_{imax, jmax} so 
            #  must have permA_{imax, jmax} > 0 and target_{imax, jmax} = 0
            H = flip_edges(H, permA, C, imax, jmax, want_meta_edge=False)
            
        else:
            #(permA - target)[imax, jmax] < 0:
            # we don't have a meta edge but want i --> j
            H = flip_edges(H, permA, C, imax, jmax, want_meta_edge=True)
            
        # recluster on the updated graph
        i += 1
        
    return C, ari_list, permA, minperm
            
        
            


# In[ ]:


"""
~~~NOTE~~~ 
This code is not in a 'finished' state and so may be a little ugly or weird at times.
However, it should be sufficiently commented to make it clear what is being done.

meta_graph_match does NOT seem to improve ARI scores, but rather drop them. This being said,
I have not done a perfect and exhaustive study showing this for varying parameters of
noise, edge probability, etc.
"""


# In[ ]:


"""
# New iterative idea: 
In the best cluster, 'fix' the worst guys
"""


# In[ ]:


def flip_unhappy_edges(G, cl, i, j):
    """
    Given a graph G, clustering cl, meta edge i-->j, 
    flip all the edges that go from j to i so they better match 
    the correct meta edge
    
    Inputs:
     G: (nx.DiGraph) -- graph of interest
     cl: (1d np.array) -- cluster membership list. cl[u] = k if node u is in cluster k
     i: (int) -- cluster i where the meta edge begins
     j: (int) -- cluster j where the meta edge ends
     
    Returns:
     H: (nx.DiGraph) -- graph with 'fixed' edges
    """
    cl1 = np.where(cl == i)[0]
    cl2 = np.where(cl == j)[0]
    H = G.copy()
    
    for u in cl1:
        for v in cl2:
            if H.has_edge(v, u):
                H.remove_edge(v, u)
                H.add_edge(u, v)
                
    return H

def iterative_small_fixes(G, target, true_labels, maxiters = 20):
    """
    cluster -> find 'most happy' meta edge -> flip_unhappy_edges
    repeat until meta graph matches target (up to permutation).
    That is, use Algorithm1 to find a cluster, C. Create a meta adjacency matrix, A,
    with entries given by cut scores. Find argmax_{ij} A_{ij}. Flip the remaining 
    edges so as to drive this particular cut score as large as possible. 
    Repeat until the meta graph matches the target meta graph or we hit our maxiters. 
    
    
    Inputs:
     G: (nx.DiGraph) -- graph of interest
     target: (2d np.array) -- (side information) 'true' adjacency matrix of meta graph
     
    Returns:
     cl_list: (list of lists) -- the clusterings at each iteration 
                                 (cl_list[k] is clustering at kth time step)
     ari_list: (list) -- the ARI score at each time step
    """
    k = target.shape[0]
    curr_iter = 0
    ari_list = []
    cl_list = []
    
    H = G.copy()
    
    while curr_iter < maxiters:
        #H = G.copy()
        
        # get a cluster and ari score 
        C = Algorithm1(H, k)
        cl_list.append(C)
        ari = adjusted_rand_score(true_labels, C)
        ari_list.append(ari)
        
        # get meta matrix of the current clustering
        A_S = cutscore_meta_matrix(H, C, score='ratio', signed = True)

        #A_S is skew-symmetric
        #  make negative values 0 to turn into directed adjacency
        A = A_S.copy()
        A[A < 0] = 0
        
        # turn A into a 0,1 matrix by setting all positive values = 1
        B = A.copy()
        B[B > 0] = 1
        
        # stop if B is the target
        minperm, permB = find_min_perm(B, target)
        if np.max(permB - target) == 0:
            return cl_list, ari_list

        # i,j index of largest entry of A
        imax, jmax = np.unravel_index(np.argmax(A), (k, k))
        
        # flip the outlier edges of this 'best' edge
        H = flip_unhappy_edges(H, C, imax, jmax)
        
        curr_iter += 1
        
    return cl_list, ari_list

