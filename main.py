# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
from collections import defaultdict
import time
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

df = pd.read_csv(
    filepath_or_buffer='data/train.dat', 
    header=None, 
    sep='\n',
    error_bad_lines=False)

X = df.iloc[:,:].values.tolist()

clas = []
features = []

for row in X:
    clas.append(row[0].split(',')[0])
    features.append(row[0].split(',')[1:-2])
    
df = pd.read_csv(
    filepath_or_buffer='data/test.dat', 
    header=None, 
    sep='\n',
    error_bad_lines=False)

test_X = df.iloc[:,:].values.tolist()

for row in test_X:
    features.append(row[0].split(' ')[:-2])
    
# extract the vectors from the Pandas data file
mat = build_matrix(features)
mat2 = csr_l2normalize(mat, copy=True)

x_train = mat2.todense().tolist()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
X_train_svd = svd.fit_transform(x_train)

#knn.fit(x_train[:800], clas)
#y = knn.predict(x_train[800:])
#
#
#file = open("results.dat","w+")
#for clas in y:
#    file.write(clas + "\n")
#file.close()