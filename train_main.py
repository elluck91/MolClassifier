# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter

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
    
# extract the vectors from the Pandas data file
mat = build_matrix(features)
mat2 = csr_l2normalize(mat, copy=True)

x_train = mat2.todense().tolist()


# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score


logreg = linear_model.LogisticRegression()
clf = DecisionTreeClassifier()
gaus = GaussianNB()
ada = AdaBoostClassifier()
sgd = linear_model.SGDClassifier()
svmc = svm.SVC()
best_k = 11
knn = KNeighborsClassifier(n_neighbors=best_k)
clf_rf = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini')

#best_score = 0
#for k in range(1, 25, 2):
#    knn = KNeighborsClassifier(n_neighbors=k)
#
#    scores = cross_val_score(knn, mat2, clas, cv=10, scoring='f1_micro')
#    print("F1 score for k=" + str(k) + ": " + str(scores.mean()))
#    
#    if (scores.mean() > best_score):
#        best_k = k
#        best_score = scores.mean()
#
#print("Best k: " + str(best_k) + " | F1: " + str(best_score) + "\n")

# DIMENSIONALITY REDUCTION
#print "================================= Sparse PCA reduction =============================="
#from sklearn.decomposition import SparsePCA
#best_n_com = 0
#best_n_score = 0
#
#for n_comp in range(0,100,10):
#    sparse_pca = SparsePCA(n_components=n_comp)
#    X_train_pca = sparse_pca.fit_transform(x_train)
#    
#    scores = cross_val_score(knn, X_train_pca, clas, cv=10, scoring='f1_micro')
#    if (scores.mean() > best_n_score):
#        best_n_com = n_comp
#        best_n_score = scores.mean()
#    print(str(n_comp) + " component PCA: " + str(scores.mean()))
#
#    print("Best n_comp=" + str(best_n_com) + " f1_score: " + str(best_n_score))




#print "================================= SVD reduction =============================="
from sklearn.decomposition import TruncatedSVD
#best_n_com = 0
#best_n_score = 0
#
#for n_comp in range(10,110,10):
svd = TruncatedSVD(n_components=30, n_iter=7, random_state=42)
X_train_svd = svd.fit_transform(x_train)
#    
#    scores = cross_val_score(knn, X_train_svd, clas, cv=10, scoring='f1_micro')
#    if (scores.mean() > best_n_score):
#        best_n_com = n_comp
#        best_n_score = scores.mean()
#    print(str(n_comp) + " component SVD: " + str(scores.mean()))
#
#print("Best n_comp=" + str(best_n_com) + " f1_score: " + str(best_n_score))


#print "====================== RANDOM PROJECTION ==============================="
from sklearn import random_projection
#best_n_com = 0
#best_n_score = 0
#
#for n_comp in range(10,110,10):
rp = random_projection.SparseRandomProjection()
X_train_rp = rp.fit_transform(x_train)

#print("Best n_comp=" + str(len(X_train_rp[0])) + " f1_score: " + str(best_n_score))

    
#print "============================ Chi^2 feature selection ======================="
from sklearn.feature_selection import SelectKBest, chi2
#best_n_com = 0
#best_n_score = 0
#
#for n_comp in range(10,110,10):
test = SelectKBest(score_func=chi2, k=20)
fit = test.fit_transform(x_train, clas)
#    scores = cross_val_score(knn, fit, clas, cv=10, scoring='f1_micro')
#    if (scores.mean() > best_n_score):
#        best_n_com = n_comp
#        best_n_score = scores.mean()
#    print(str(n_comp) + " component Chi^2: " + str(scores.mean()))
#    
#print("Best n_comp=" + str(best_n_com) + " f1_score: " + str(best_n_score))

print "============================ VarianceTreshold feature selection ======================="
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold()
x_train_v = sel.fit_transform(x_train)
scores = cross_val_score(knn, x_train_v, clas, cv=10, scoring='f1_micro')

    
print("VarianceThreshold f1_score: " + str(scores.mean()))

print "==================Best results========================"
print "***KNN***\n"
scores = cross_val_score(knn, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(knn, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(knn, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(knn, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())

print "***Decision Tree***\n"
scores = cross_val_score(clf, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(clf, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(clf, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(clf, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())

print "***Naive Bayes***\n"
scores = cross_val_score(gaus, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(gaus, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(gaus, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(gaus, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())

print "***adaBoost***\n"
scores = cross_val_score(ada, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(ada, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(ada, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(ada, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())

print "***logistic regression***\n"
scores = cross_val_score(logreg, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(logreg, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(logreg, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(logreg, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())

print "***SGD***\n"
scores = cross_val_score(sgd, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(sgd, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(sgd, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(sgd, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())

print "***SVM***\n"
scores = cross_val_score(svmc, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(svmc, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(svmc, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(svmc, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())

print "***Random Forest***\n"
scores = cross_val_score(clf_rf, X_train_svd, clas, cv=10, scoring='f1_micro')
print "SVD f1_score: " + str(scores.mean())
scores = cross_val_score(clf_rf, X_train_rp, clas, cv=10, scoring='f1_micro')
print "Random Projection f1_score: " + str(scores.mean())
scores = cross_val_score(clf_rf, fit, clas, cv=10, scoring='f1_micro')
print "Chi^2 f1_score: " + str(scores.mean())
scores = cross_val_score(clf_rf, x_train_v, clas, cv=10, scoring='f1_micro')
print "VarianceThreshold f1_score: " + str(scores.mean())