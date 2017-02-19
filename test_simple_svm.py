import sys
import numpy as np
from svm_simple import *


#return (TRAIN_X,TEST_X,TRAIN_Y, TEST_Y)
def load_data(file_name):
    fp = open(file_name)
    X = []
    Y = []
    while True:
        line = fp.readline()
        if not line:
            break
        toks = line.strip().split('\t')
        X.append(toks[:-1])
        Y.append(toks[-1])
    fp.close()
    n = len(Y)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=int)
    TRAIN_X_N = n * 4 / 5
    print TRAIN_X_N
    TEST_Y_N = n - TRAIN_X_N
    return (X[:TRAIN_X_N], X[TRAIN_X_N:], Y[:TRAIN_X_N], Y[TRAIN_X_N:])

if __name__ == '__main__':
    print 'start to load test data ...'
    (TRAIN_X, TEST_X, TRAIN_Y,TEST_Y) = load_data('testSet.txt')
    svm = SIMPLE_SVM(C=0.2, error_theta=0.001, kernel = 'linear')
    print 'start to train_data ...'
    svm.train(TRAIN_X, TRAIN_Y, iter_num = 500)
    print 'start to show ' 
    svm.show_svm()
    
