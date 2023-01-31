import numpy as np
import pandas as pd
import networkx as nx
import time
import math
# Import to dataframe
#%%
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=1000,
    
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    import scipy as sp
    import scipy.sparse  # call as sp.sparse
    start= time.time()

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight, dtype=float)
    S = np.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    M = Q * M

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x = x / x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p = p / p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for t in range(max_iter):
        xlast = x
        x = alpha * (x * M ) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        print(f"Iteration {t+1}:",x)
        if err < N * tol:
            end = time.time()
            NM_time = end-start
            print('PageRank running time:' ,NM_time)
            return dict(zip(nodelist, map(float, x)))
    print()
    raise nx.PowerIterationFailedConvergence(max_iter)
#%%
def AdaptovePR(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    import scipy as sp
    import scipy.sparse  # call as sp.sparse
    start= time.time()

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight, dtype=float)
    S = np.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    M = Q * M

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x = x / x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p = p / p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]
    list_N = list(range(N))
    # power iteration: make up to max_iter iterations

    for t in range(max_iter):
        xlast = x.copy()
        x_change = alpha * (x * M[:,list_N]) + (1 - alpha) * p[list_N]
        x[list_N] = x_change
        print(f"Iteration {t+1}:",x)
        
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        list_N = np.where(((abs(x-xlast)/abs(x))) > 1e-3)[0]
        if err < N * tol or not any(list_N):
            end = time.time()
            AP_time = end-start
            print('AdaptovePR runnign time: ',AP_time)
            return dict(zip(nodelist, map(float, x)))
    print()
    raise nx.PowerIterationFailedConvergence(max_iter)
    
#%%
def ModifiedPR(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    import scipy as sp
    import scipy.sparse  # call as sp.sparse
    start= time.time()
    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight, dtype=float)
    S = np.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    M = Q * M

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x = x / x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p = p / p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]
    list_N = list(range(N))
    y = 0
    list_C = []
    # power iteration: make up to max_iter iterations
    for t in range(max_iter):
        xlast = x.copy()
        x_change = alpha * (x[list_N]* M[:,list_N][list_N] + y ) + (1 - alpha) * p[list_N]

        x[list_N] = x_change
        print(f"Iteration {t+1}:",x)
        
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        
        list_N = np.where((abs(x-xlast)/abs(x)) > 1e-3)[0]
        list_C = np.where((abs(x-xlast)/abs(x)) <= 1e-3)[0]
        if err < N * tol :
            end = time.time()
            MP_time = end-start
            print('ModifiedPR running time: ', MP_time)
            return dict(zip(nodelist, map(float, x)))
        y = x[list_C]* M[:,list_N][list_C]
    print()
    raise nx.PowerIterationFailedConvergence(max_iter)

#%%
print("DBLP collaboration network: ")

data = pd.read_csv('./dataset/com-DBLP.txt', sep="	"or "	"or "	", header=None)
data.columns = ["FromNodeId", "ToNodeId"]
egdes = [tuple(x) for x in data.to_numpy()]
Graph = nx.Graph()
Graph.add_edges_from(egdes)
print("Nodes: ",str(len(Graph)))
print("Edges: ",str(len(egdes)))
#%%
print()
pagerank_list_NM = pagerank(Graph,alpha=0.15,tol=1e-9)
print()
pagerank_list_PR = AdaptovePR(Graph,alpha=0.15,tol=1e-9)
print()
pagerank_list_MP = ModifiedPR(Graph,alpha=0.15,tol=1e-9)
print()
#%%
PR_differen = 0
MP_differen = 0
for i in range(len(Graph)):
    PR_differen += (pagerank_list_NM[i]-pagerank_list_PR[i])**2
    MP_differen += (pagerank_list_NM[i]-pagerank_list_MP[i])**2
print(PR_differen)
print(MP_differen)
    
#%%
print("Web graph from Google network: ")

data = pd.read_csv('./dataset/web-google.txt', sep="	"or "	"or "	" or" ", header=None)
data.columns = ["FromNodeId", "ToNodeId"]
egdes = [tuple(x) for x in data.to_numpy()]
Graph = nx.Graph()
Graph.add_edges_from(egdes)
print("Nodes: ",str(len(Graph)))
print("Edges: ",str(len(egdes)))

print()
pagerank_list_NM = pagerank(Graph,alpha=0.15,tol=1e-9)
print()
pagerank_list_PR = AdaptovePR(Graph,alpha=0.15,tol=1e-9)
print()
pagerank_list_MP = ModifiedPR(Graph,alpha=0.15,tol=1e-9)
print()
#%%
PR_differen = 0
MP_differen = 0
for i in range(len(Graph)):
    PR_differen += (pagerank_list_NM[i]-pagerank_list_PR[i])**2
    MP_differen += (pagerank_list_NM[i]-pagerank_list_MP[i])**2
print(math.sqrt(PR_differen))
print(math.sqrt(MP_differen))
   
#%%
print("Web graph of Berkeley and Stanford : ")

data = pd.read_csv('./dataset/web-BerkStan.txt', sep="	"or "	"or "	"or" ", header=None)
data.columns = ["FromNodeId", "ToNodeId"]
egdes = [tuple(x) for x in data.to_numpy()]
Graph = nx.Graph()
Graph.add_edges_from(egdes)
print("Nodes: ",str(len(Graph)))
print("Edges: ",str(len(egdes)))

print()
pagerank_list_NM = pagerank(Graph,alpha=0.15,tol=1e-9)
print()
pagerank_list_PR = AdaptovePR(Graph,alpha=0.15,tol=1e-9)
print()
pagerank_list_MP = ModifiedPR(Graph,alpha=0.15,tol=1e-9)
print()
#%%
PR_differen = 0
MP_differen = 0
for i in range(1,len(Graph)):
    PR_differen += (pagerank_list_NM[i]-pagerank_list_PR[i])**2
    MP_differen += (pagerank_list_NM[i]-pagerank_list_MP[i])**2
print(PR_differen)
print(MP_differen)
   