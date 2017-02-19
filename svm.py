#coding: utf-8
'''
@author: huangjin (Jeff)
@contact: hj51yc@gmail.com

SVM implemented using classic method with  kdd condition and dual method!

'''

import numpy as np
import sys
import random
import matplotlib.pyplot as plt 
import math

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2):
    sigma = 1.0
    return np.exp(np.dot(x1, x2)/ (-2.0 * sigma**2))


class SVM:

    def check_data(X,Y):
        if len(train_data.shape) != 2:
            print 'error train_data';
            exit(0)
        (row_num, feat_dim) = train_data.shape


    def __init__(self, C = 0.6, kernel = 'linear', error_theta = 0.000001):
        self._w = None
        self._b = random.random()
        self._C = C
        ## X must be normalized between [0, 1]
        self._X = None
        ## Y must be -1 or 1
        self._Y = None
        self._alfa = None
        if kernel == 'rbf':
            self._kernel = rbf_kernel
        else:
            self._kernel = linear_kernel
        self._N = 0
        self._feat_dim = 0
        ##_error_theta must be greater than 0
        self._error_theta = error_theta
        self._support_vecs_alfa = None
        self._support_vecs = None
        self._support_vecs_label = None
    

    def is_point_violate_kkt(self, i):
        ai = self._alfa[i]
        ui = self.u_func(self._X[i])
        yi = self._Y[i]
        tmp = ui * yi
        tmp_diff = yi - ui

        #if ai > 0 or ai < self._C:
        #    if abs(tmp_diff) <= self._error_theta:
        #        return False
        ##satisify KKT Condition
        ##1: ui * yi > 1 , ai = 0
        ##2: ui * yi < 1 , ai = C
        ##3: ui * yi = 1, 0 < ai < C
        ## when tmp > 1+theta, then ai should be 0
        if tmp > 1 + self._error_theta and ai > 0:
            return True
        if tmp < 1 - self._error_theta and ai < self._C:
            return True
        if abs(tmp_diff) <= self._error_theta and (ai == 0 or ai ==self._C):
            return True
        return False


    def u_func(self, x):
        u = 0.0
        for i in xrange(self._N):
            v = self._kernel(self._X[i], x)
            u += v * self._Y[i] * self._alfa[i]
        ## this is for : f(x) = w * x + b
        #return u + self._b

        ## this is for : f(x) = w * x - b
        return u - self._b


    def E_func(self, i):
        return self.u_func(self._X[i]) - self._Y[i]
    
    
    def find_point_j(self, i):
        Ei = self.E_func(i)
        max_E_diff = 0
        s_j = -1
        ##find the second point, using MAX|E1-E2|
        for j in xrange(self._N):
            if j == i:
                continue
            Ej = self.E_func(j)
            if Ej == 0:
                continue
            abs_diff = abs(Ei - Ej)
            if max_E_diff < abs_diff:
                max_E_diff = abs_diff
                s_j = j
        if s_j == -1:
            while s_j != i:
                s_j = int(random.uniform(0, self._N))            
        #print 'select:i',i,'j',s_j,'Ei',Ei,'Ej',self.E_func(s_j)
        return s_j


    def select_two_points(self):
        s_i = -1;
        s_j = -1;
        s_i_tmp = -1;
        ##find the first point
        for i in xrange(self._N):
            is_between_0_C = self.is_alfa_between_0_C(self._alfa[i])
            is_violate_kkt = False
            if is_between_0_C and self.is_point_violate_kkt(i):
                s_i = i
                break
            elif s_i_tmp == -1 and self.is_point_violate_kkt(i):
                s_i_tmp = i
        if s_i == -1 and s_i_tmp == -1:
            ## all point fit kkt condition
            return None
        elif s_i == -1 and s_i_tmp >= 0:
            s_i = s_i_tmp
        
        s_j = self.find_point_j(s_i)
        return (s_i, s_j)
    

    def init_w(self):
        self._w = np.zeros(self._feat_dim)
        for i in xrange(self._N):
            xi = self._X[i]
            yi = self._Y[i]
            alfa_i = self._alfa[i]
            self._w += yi * alfa_i * xi
        return


    def update_w(self, i, j, alfa_i_new, alfa_j_new):
        alfa_i_old = self._alfa[i]
        alfa_j_old = self._alfa[j]
        xi = self._X[i]
        xj = self._X[j]
        yi = self._Y[i]
        yj = self._Y[j]
        self._w += (alfa_i_new - alfa_i_old) * yi * xi + (alfa_j_new - alfa_j_old) * yj * xj
    

    ## update_b must after update_w
    def update_b_v1(self):
        positive_v = -sys.float_info.max
        negative_v = sys.float_info.max
        for i in xrange(self._N):
            tmp = self._kernel(self._w, self._X[i])
            if self._Y[i] == 1 and positive_v < tmp:
                positive_v = tmp
            elif self._Y[i] == -1 and negative_v > tmp:
                negative_v = tmp
        self._b = - (positive_v + negative_v)/2

    
    def is_alfa_between_0_C(self, a):
        if a >= self._C or a <= 0:
            return False
        return True

    ## update b version2
    def update_b_v2(self, i, j, alfa_i_new, alfa_j_new, Ei, Ej, Kii, Kjj, Kij):
        alfa_i_diff = alfa_i_new - self._alfa[i]
        alfa_j_diff = alfa_j_new - self._alfa[j]
        
        ## this is for : f(x) = w * x - b
        b1_new = self._b + Ei + self._Y[i] * alfa_i_diff * Kii + self._Y[j] * alfa_j_diff * Kij
        b2_new = self._b + Ej + self._Y[j] * alfa_j_diff * Kjj + self._Y[i] * alfa_i_diff * Kij
        
        ## this is for : f(x) = w * x + b
        #b1_new = self._b - Ei - self._Y[i] * alfa_i_diff * Kii - self._Y[j] * alfa_j_diff * Kij
        #b2_new = self._b - Ej - self._Y[j] * alfa_j_diff * Kjj - self._Y[i] * alfa_i_diff * Kij
        

        alfa_i_in = self.is_alfa_between_0_C(alfa_i_new)
        alfa_j_in = self.is_alfa_between_0_C(alfa_j_new)
        if alfa_i_in:
            self._b = b1_new
        elif alfa_j_in:
            self._b = b2_new
        else:
            self._b = (b1_new + b2_new) / 2
        return

    
    def update_once(self, i, j):
        Ej = self.E_func(j)
        Ei = self.E_func(i)
        Kij = self._kernel(self._X[i], self._X[j])
        Kii = self._kernel(self._X[i], self._X[i])
        Kjj = self._kernel(self._X[j], self._X[j])
        #print 'before'
        #print 'ai',self._alfa[i],'aj',self._alfa[j]
        #print 'Ei',Ei,'Ej',Ej
        eta = Kii + Kjj - 2*Kij
        s = self._Y[i] * self._Y[j]
        L = 0.0
        H = self._C
        if s == -1:
            ##different sign
            L = max(0, self._alfa[j] - self._alfa[i])
            H = min(self._C, self._C + self._alfa[j] - self._alfa[i])
        else:
            ##same sign
            L = max(0, self._alfa[j] + self._alfa[i] - self._C)
            H = min(self._C, self._alfa[i] + self._alfa[j])
        #print 's',s,'L',L,'H',H
        alfa_i_new = None
        alfa_j_new = None
        #print 'eta:',eta
        if eta > 0:
            ## when eta > 0, has the normal solution
            alfa_j_new = self._alfa[j] + self._Y[j] * (Ei - Ej) / eta
            #print 'alfa_j_new:',alfa_j_new
            if alfa_j_new > H:
                alfa_j_new = H
            elif alfa_j_new < L:
                alfa_j_new = L
        elif eta <= 0 or H == L:
            return 0
        else:
            ## find solution on the two side point
            f1 = self._Y[i] * (Ei + self._b) - self._alfa[i] * Kii - s * self._alfa[j] * Kij
            f2 = self._Y[j] * (Ej + self._b) - self._alfa[j] * Kjj - s * self._alfa[i] * Kij
            L1 = self._alfa[i] + s * (self._alfa[j] - L)
            H1 = self._alfa[i] + s *(self._alfa[j] - H)
            FI_L = L1 * f1 + L * f2 + L1*L1*Kii/2 + L*L*Kjj/2 + s*L*L1*Kij
            FI_H = H1 * f1 + H * f2 + H1*H1*Kii/2 + H*H*Kjj/2 + s*H*H1*Kij
            if FI_L < FI_H:
                alfa_j_new = L
            else:
                alfa_j_new = H
        #print 'aj_new_old_diff',abs(alfa_j_new - self._alfa[j])
        #print 'aj', self._alfa[j]
        if abs(alfa_j_new - self._alfa[j]) == 0:
            return 0

        alfa_i_new = self._alfa[i] + s * (self._alfa[j] - alfa_j_new)
        
        #self.update_w(i, j, alfa_i_new, alfa_j_new)
        ##self.update_b_v1()
        self.update_b_v2(i, j, alfa_i_new, alfa_j_new, Ei, Ej, Kii, Kjj, Kij)
        self._alfa[i] = alfa_i_new;
        self._alfa[j] = alfa_j_new;

        Ej = self.E_func(j)
        Ei = self.E_func(i)
        #print 'after'
        #print 'ai',self._alfa[i],'aj',self._alfa[j]
        #print 'Ei',Ei,'Ej',Ej
        return 1


    def find_support_vector(self):
        indexes = np.nonzero(self._alfa > 0)[0]
        self._support_vecs_alfa = self._alfa[indexes]
        self._support_vecs = self._X[indexes]
        self._support_vecs_label = self._Y[indexes]

    
    def train_batch(self,use_all):
        changes = 0
        index_range = []
        for i in xrange(self._N):
            if use_all:
                index_range.append(i)
            else:
                is_between_0_C = self.is_alfa_between_0_C(self._alfa[i])
                if is_between_0_C:
                    index_range.append(i)

        for i in index_range:
            if self.is_point_violate_kkt(i):
                #print 'use_all',use_all,'i',i
                j = self.find_point_j(i)
                #changes += self.update_once(i, j)
                tmp_change = self.update_once(j, i)
                k = 0
                while tmp_change == 0 and k < self._N:
                    if k != i:
                        tmp_change = self.update_once(k, i)
                    k += 1
                changes += tmp_change
        print ''
        return changes
                    
    
    def train(self, X, Y, iter_num = 10000):
        (row_num, feat_dim) = X.shape
        #alfa = self._C * np.random.rand(row_num)
        alfa = np.zeros((row_num,))
        self._alfa = alfa
        self._X = X
        self._Y = Y
        self._b = random.random()
        self._N = row_num
        self._feat_dim = feat_dim
        ## may not need w when using kernel
        ##init_w()
        use_all = True
        for it in xrange(iter_num):

            if it % 1 == 0:
                print 'start iter ', it
            #points = self.select_two_points()
            changes = self.train_batch(use_all)
            print 'changes:', changes
            if use_all == True and changes == 0:
                print 'finished'
                break
            if changes > 0:
                use_all = False
            elif changes == 0:
                use_all = True
            print self._alfa
            
        print 'start to find support vector'
        self.find_support_vector()
        print "finish train!"
    

    def predict_using_w(self, x):
        return np.dot(self._w, x) + b
    

    def predict_one(self, x):
        u = 0.0
        for i in xrange(len(self._support_vecs_alfa)):
            support_vec = self._support_vecs[i]
            support_vec_y = self._support_vecs_label[i]
            support_vec_alfa = self._support_vecs_alfa[i]
            kernel_v = self._kernel(support_vec, x)
            u += support_vec_y * support_vec_alfa * kernel_v
        u += self._b
        if u >= 0:
            return 1 
        else:
            return -1
    
    
    # show your trained svm model only available with 2-D data
    def show_svm(self):
        print self._alfa
        if self._X.shape[1] != 2:
            print "Sorry! I can not draw because the dimension of your data is not 2!"
            return 1

        # draw all samples
        for i in xrange(self._N):
            if self._Y[i] == -1:
                plt.plot(self._X[i, 0], self._X[i, 1], 'or')
            elif  self._Y[i] == 1:
                plt.plot(self._X[i, 0], self._X[i, 1], 'ob')
        
        # mark support vectors
        supportVectorsIndex = np.nonzero(self._alfa > 0)[0]
        for i in supportVectorsIndex:
            plt.plot(self._X[i, 0], self._X[i, 1], 'oy')
        
        #plt.show()
        #return
        # draw the classify line
        w = np.zeros((2,))
        for i in supportVectorsIndex:
            w +=  self._alfa[i] * self._Y[i] * self._X[i]
        print 'w',w,'b',self._b
        min_x = min(self._X[:, 0])
        max_x = max(self._X[:, 0])
        ## this one is for: f(x) = w*x + b
        #y_min_x = float(-self._b - w[0] * min_x) / w[1]
        #y_max_x = float(-self._b - w[0] * max_x) / w[1]

        ## this one is for: f(x) = w*x - b
        y_min_x = float(+self._b - w[0] * min_x) / w[1]
        y_max_x = float(+self._b - w[0] * max_x) / w[1]
        print min_x, y_min_x
        print max_x, y_max_x
        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
        plt.show()
        
        
        
