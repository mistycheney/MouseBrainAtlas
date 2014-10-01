# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.datasets import make_gaussian_quantiles

n_samples = 300
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
X_train = np.r_[10*np.random.random((n_samples, 2)),
                np.dot(np.random.randn(n_samples, 2), C),]
X_train = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                -4+np.dot(np.random.randn(n_samples, 2), C),]
Y_train = np.r_[np.ones((n_samples,)), -np.ones((n_samples,))]


# X1, y1 = make_gaussian_quantiles(cov=2.,
#                                  n_samples=200, n_features=2,
#                                  n_classes=2, random_state=1)
# X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
#                                  n_samples=300, n_features=2,
#                                  n_classes=2, random_state=1)
# X_train = np.concatenate((X1, X2))
# Y_train = np.concatenate((y1, - y2 + 1))
# Y_train[Y_train==0]=-1

# <codecell>

import itertools

def ADT(x,y,n_iter=3):
    n_samples, n_feat = x.shape
    
    eps = 1./n_samples
    satPs = np.zeros((n_samples, n_iter), dtype=np.bool)
    satBs = np.zeros((n_samples, n_iter), dtype=np.bool)

    r0 = y.mean()
    h0 = .5*np.log((1+r0)/(1-r0))
    Ps = [np.ones_like(y, dtype=np.bool)]
    rules = [h0]
    descriptions = ['all']
    
    w = np.zeros((n_samples, n_iter+1))
    w[:,0] = 1./(1+r0*y)*1./n_samples
    
    orders = x.argsort(axis=0)
    
    for t in range(n_iter):
        print 'iter %d' % t
        
        Z_t = np.inf
        
        for satPid, satP in enumerate(Ps):
#             print 'descriptions', descriptions[satPid]
            for jb in range(n_feat):
#         for satP, jb in itertools.product(Ps, range(n_feat)):
#         for jp, thrp, smaller_than, jb in itertools.product(range(n_feat), np.linspace(0,1,10), [0,1], range(n_feat)):
        
#             satP = x[:,jp] < thrp if smaller_than else x[:,jp] >= thrp
                Wt_notP = w[~satP, t].sum()
    
                x_jb_order = x[orders[:,jb], :]
                thresh_points = np.r_[x_jb_order[0,jb]-0.1, .5*(x_jb_order[:-1,jb]+x_jb_order[1:,jb]), x_jb_order[-1,jb]+0.1]

                Z_diff_thrb = np.zeros_like(thresh_points)
                c0_diff_thrb = np.zeros_like(thresh_points)
                c1_diff_thrb = np.zeros_like(thresh_points)
                for thrb_id, thrb in enumerate(thresh_points):
                    satB =  x[:,jb] < thrb

                    Wt_PB_pos = w[satP&satB&(y==1), t].sum()
                    Wt_PB_neg = w[satP&satB&(y==-1), t].sum()
                    Wt_PnotB_pos = w[satP&~satB&(y==1), t].sum() 
                    Wt_PnotB_neg = w[satP&~satB&(y==-1), t].sum()
                    Z_diff_thrb[thrb_id] = Wt_notP + 2*np.sqrt(Wt_PB_pos * Wt_PB_neg) + 2*np.sqrt(Wt_PnotB_pos * Wt_PnotB_neg)
                    c0_diff_thrb[thrb_id] = .5*np.log((Wt_PB_pos+eps)/(Wt_PB_neg+eps))
                    c1_diff_thrb[thrb_id] = .5*np.log((Wt_PnotB_pos+eps)/(Wt_PnotB_neg+eps))
#                     if satPid == 2 and thrb - 1.41 < 0.01 \
#                     and c1_diff_thrb[thrb_id] - 0.46 < 0.01\
#                     and c0_diff_thrb[thrb_id] - (-2.72) < 0.01:
#                         print Wt_PnotB_pos, Wt_PnotB_neg, w[satP&~satB&(y==1), t].max(), w[satP&~satB&(y==-1), t].max()

                        
                Z = Z_diff_thrb.min()
                thrb_id_best = Z_diff_thrb.argmin()
                thrb = thresh_points[thrb_id_best]
                c0 = c0_diff_thrb[thrb_id_best]
                c1 = c1_diff_thrb[thrb_id_best]
                
#                 c0,c1,thrb,err = find_stump_for_one_feature2(x[:,jb], y, w[:,t], orders[satP,jb])
#                 Z = 2*np.sqrt(err*(1-err)) + Wt_notP
                if Z < Z_t:
                    Z_t = Z
#                     err_t = err
                    satP_t = satP
                    satPid_t = satPid
                    satB_t = x[:,jb] < thrb
                    jb_t, c0_t, c1_t, thrb_t = jb, c0, c1, thrb

#         alpha_t = .5*np.log((1-err_t+eps)/(err_t+eps))
        
        rules += [(satPid_t, jb_t, thrb_t, c0_t, c1_t)]
#         print '%d: x[%d] ^ %.2f #[%.2f,%.2f] ' % (satPid_t, jb_t, thrb_t, c0_t, c1_t)
#         print 'Zt = %f, err = %f, alpha = %f' % (Z_t, err_t, alpha_t)
        print 'Zt = %.2f' % (Z_t)
        
        satPs[:,t] = satP_t
        satBs[:,t] = satB_t
        
        Ps += [satP_t & satB_t, satP_t & ~satB_t]
        
        precond = descriptions[satPid_t]
        descriptions += [precond + ' => %d, x[%d] < %.2f value %.2f ' % (satPid_t, jb_t, thrb_t, c0_t),
                         precond + ' => %d, x[%d] >= %.2f value %.2f ' % (satPid_t, jb_t, thrb_t, c1_t)]
        print 'added', precond + ' => %d, x[%d] < %.2f value %.2f ' % (satPid_t, jb_t, thrb_t, c0_t)
        print 'added', precond + ' => %d, x[%d] >= %.2f value %.2f ' % (satPid_t, jb_t, thrb_t, c1_t)
        
        ht = np.zeros_like(y)
        ht[~satP_t] = 0
        ht[satP_t & satB_t] = c0_t
        ht[satP_t & ~satB_t] = c1_t        
#         ht = np.where(x[:,jb_t] < thrb_t, c0_t, c1_t)
#         w[:,t+1] = w[:,t]*np.exp(-alpha_t*y*ht)
        w[:,t+1] = w[:,t]*np.exp(-y*ht)
        print 'W_sum =%f' %w[:,t+1].sum()

        w[:,t+1] /= w[:,t+1].sum()
    
    
        curr_prediction = apply_ADT(x, rules)
        training_error = np.count_nonzero(np.sign(curr_prediction) != y)/float(len(y))
        print 'training_error = %f' % training_error
        if training_error == 0: break
    
    return satPs, satBs, rules, descriptions, w

# satPs, satBs, rules, descriptions, weights = ADT(X_train, Y_train, n_iter=10)

# x, y = X_train, Y_train
# for t, (p, b, rule) in enumerate(zip(satPs.T, satBs.T, nodes[1:])):
#     plt.scatter(x[~p, 0], x[~p, 1], s=weights[:,t]*10000, c='g',
#                 marker='.',
#                 cmap=plt.cm.Paired, label='abstain')
 
#     plt.scatter(x[p&b&(y==1), 0], x[p&b&(y==1), 1], s= weights[p&b&(y==1),t]*10000, c='b',
#             marker='+',
#             cmap=plt.cm.Paired, label='+, c0=%.2f'%rule[3])


#     plt.scatter(x[p&b&(y==-1), 0], x[p&b&(y==-1), 1], s= weights[p&b&(y==-1),t]*10000, c='b',
#                 marker='_',
#                 cmap=plt.cm.Paired, label='-, c0=%.2f'%rule[3])


#     plt.scatter(x[p&~b&(y==1), 0], x[p&~b&(y==1), 1], s=weights[p&~b&(y==1),t]*10000, c='r',
#                 marker='+',
#                 cmap=plt.cm.Paired, label='+, c1=%.2f'%rule[4])

#     plt.scatter(x[p&~b&(y==-1), 0], x[p&~b&(y==-1), 1], s=weights[p&~b&(y==-1),t]*10000, c='r',
#                 marker='_',
#                 cmap=plt.cm.Paired, label='-, c1=%.2f'%rule[4])

#     print 'c0=%.2f, c1=%.2f'%(rule[3], rule[4])
# #     plt.legend(loc='lower right',)
#     plt.show()


# <codecell>

def apply_ADT(X, rules):
    
    children = dict([])
    for i, rule in enumerate(rules):
        if i == 0: continue
        par, j, thr, c0, c1 = rule
        if par not in children:
            children[par] = [i]
        else:
            children[par] += [i]
    
#     print children
        
    def collect_rule(x, rule_id):
        
        par, j, thr, c0, c1 = rules[rule_id]
        s = c0 if x[j] < thr else c1
#         print 'rule ', rule_id, s

        if x[j] < thr:
            left_child = rule_id*2-1
            sl = 0
            if left_child in children:
                for rule in children[left_child]:
                    sl += collect_rule(x, rule)
            return s + sl
        else:
            right_child = rule_id*2
            sr = 0
            if right_child in children:
                for rule in children[right_child]:
                    sr += collect_rule(x, rule)
            return s + sr
        
    n_test, n_feat = X.shape
    F = np.zeros((n_test, ))
    for i, x in enumerate(X):
        s = rules[0]
        for rule_id, rule in enumerate(rules):
            if isinstance(rule, tuple):
                if rule[0] == 0:
#                     print 'initial rules', rule_id
                    s += collect_rule(x, rule_id)
        F[i] = s
    
    return F

# F = apply_classifiers(X_train, nodes)

# <codecell>

# apply_classifiers(np.atleast_2d([0,0]), rules)

# <codecell>

# plot_step = 0.02
# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                      np.arange(y_min, y_max, plot_step))

# F = apply_classifiers(np.c_[xx.ravel(), yy.ravel()], rules)

# Ff = F.reshape(xx.shape)
# cs = plt.contourf(xx, yy, Ff, cmap=plt.cm.coolwarm)
# plt.axis("tight")

# for i, n, c in zip([+1,-1], ['+1','-1'], 'rb'):
#     idx = np.where(Y_train == i)
#     plt.scatter(X_train[idx, 0], X_train[idx, 1],
#                c=c, cmap=plt.cm.Paired,
#                label="Class %s" % n)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.legend(loc='upper right')
# plt.xlabel("Decision Boundary")
# plt.show()

# <codecell>

# c0 and c1 are real values, and does not have to have equal magnitude
def find_stump_for_one_feature_binary_different_values(x, y, w, order):
    x = x[order]
    y = y[order]
    w = w[order]
    
    n_samples = len(y)
    W0 = np.zeros((2,))
    W1 = np.zeros((2,))
    W0[1] = 0
    W0[0] = 0
    W1[1] = w[y==1].sum()
    W1[0] = w[y==-1].sum()
    
    err_best = min(W1[1], W1[0])
    c0 = 1 if W0[1] > W0[0] else -1
    c1 = 1 if W1[1] > W1[0] else -1
    v = fj[0] - 0.1
    for i in range(n_samples):
        W0[1 if y[i]==1 else 0] = min(W0[1 if y[i]==1 else 0] + w[i], 1)
        W1[1 if y[i]==1 else 0] = max(W1[1 if y[i]==1 else 0] - w[i], 0)
        
        if i != n_samples-1:
            if fj[i] == fj[i+1]: continue
        
        err = min(W0[0],W0[1]) + min(W1[0],W1[1])
        if err < err_best:
            err_best = err
            if i < n_samples-1:
                v = .5*(fj[i] + fj[i+1])
            else:
                v = fj[n_samples-1] + 0.1
            c0 = 1 if W0[1] > W0[0] else -1
            c1 = 1 if W1[1] > W1[0] else -1
    
    return c0,c1,v,err_best
    

# <codecell>

def find_stump_for_one_feature2(fj, y, w, order):
    
    fj = fj[order]
    y = y[order]
    w = w[order]
    
    n_samples = len(y)
    W0 = np.zeros((2,))
    W1 = np.zeros((2,))
    W0[1] = 0
    W0[0] = 0
    W1[1] = w[y==1].sum()
    W1[0] = w[y==-1].sum()
    
    err_best = min(W1[1], W1[0])
    c0 = 1 if W0[1] > W0[0] else -1
    c1 = 1 if W1[1] > W1[0] else -1
    v = fj[0] - 0.1
    for i in range(n_samples):
        W0[1 if y[i]==1 else 0] = min(W0[1 if y[i]==1 else 0] + w[i], 1)
        W1[1 if y[i]==1 else 0] = max(W1[1 if y[i]==1 else 0] - w[i], 0)
        
        if i != n_samples-1:
            if fj[i] == fj[i+1]: continue
        
        err = min(W0[0],W0[1]) + min(W1[0],W1[1])
        if err < err_best:
            err_best = err
            if i < n_samples-1:
                v = .5*(fj[i] + fj[i+1])
            else:
                v = fj[n_samples-1] + 0.1
            c0 = 1 if W0[1] > W0[0] else -1
            c1 = 1 if W1[1] > W1[0] else -1
    
    return c0,c1,v,err_best

# <codecell>

epsilon = 1./500000

def predict(jt,c, curr_frame, scale):
    score = np.zeros(frame_size_multiscale[scale][::-1], dtype=np.float)
    for j,vals in zip(jt,c):
        if j is None: break
        partitioned = function_pool[j](curr_frame, scale)
        for k, v in enumerate(vals):
            score[partitioned == k] += v
    return score

def adaboost_dompart(curr_frame, function_pool, K_pool, y):
    
    n_samples = np.prod(frame_size_multiscale[0])
    n_funcs = len(function_pool)
    
    w = np.ones((4, n_samples), dtype=np.float)/n_samples
#     r0 = np.sum(y).astype(np.float)/y.size
#     w[y==1] = 1./(1+r0)
#     w[y==-1] = 1./(1-r0)
    
    wu = np.ones((4, n_samples), dtype=np.float)/n_samples
    jt = [None for t in range(3)]
    s = [None for t in range(3)]
    c = [None for t in range(3)]

    phi = np.zeros((n_funcs, n_samples), dtype=np.int)
    for func_id, phi_func in enumerate(function_pool):
            j = func_id
            phi[j] =  
            plot_with_colorbar_noticks(phi[j].reshape(frame_size_multiscale[0][::-1]),
                               'frame %d, label map, j=%d, func=%d(%s), scale=%d'%(curr_frame, j, func_id, feature_name_pool[func_id], scale))    
    
    for t in range(3):
        Z = np.zeros((n_funcs,))
        Wpos = [None for j in range(n_funcs)]
        Wneg = [None for j in range(n_funcs)]
        for func_id, phi_func in enumerate(function_pool):
            j = func_id
            Wpos[j] = np.asarray([sum(w[t, (phi[j]==k) & (y==1)]) for k in range(K_pool[func_id])])
            Wneg[j] = np.asarray([sum(w[t, (phi[j]==k) & (y==-1)]) for k in range(K_pool[func_id])])
            Z[j] = 2*np.sum([np.sqrt(Wpos[j][k]*Wneg[j][k]) for k in range(K_pool[func_id])])

#             draw_double_histogram(Wpos[j], Wneg[j], labels=['Wpos', 'Wneg'], title='class weights in each bin, j=%d(%s)'%(j, feature_name_pool[j]))
            
        print 'Z:', Z
        
        jt[t] = np.argmin(Z)
        func_id_t = jt[t]
        Zt = Z[jt[t]]
        Kjt = K_pool[jt[t]]
        c[t] = [.5*np.log((Wpos[jt[t]][k]+epsilon)/(Wneg[jt[t]][k]+epsilon)) for k in range(Kjt)]
        
#         plot_with_colorbar_noticks(phi[jt[t]].reshape((original_h, original_w)),
#                            't=%d, label map, j=%d(%s)'%(t, jt[t], feature_name_pool[jt[t]]))
        
        partition_map = np.zeros((n_samples, ), dtype=np.float)
        for k in range(Kjt):
            partition_map[phi[jt[t]] == k] = c[t][k]
        
        plot_with_colorbar_noticks(partition_map.reshape(frame_size_multiscale[scale][::-1]),
                                   't=%d, weak hypothesis partition output, j=%d(%s)'%(t, jt[t], feature_name_pool[jt[t]]))
        
        mult_factor_pos = np.zeros((Kjt, ), dtype=np.float)
        mult_factor_neg = np.zeros((Kjt, ), dtype=np.float)
        for k in range(Kjt):
            mult_factor_pos[k] = np.exp(-c[t][k])
            mult_factor_neg[k] = np.exp(c[t][k])
            w[t+1, (phi[jt[t]] == k) & (y==1)] = w[t, (phi[jt[t]] == k) & (y==1)]*mult_factor_pos[k]
            w[t+1, (phi[jt[t]] == k) & (y==-1)] = w[t, (phi[jt[t]] == k) & (y==-1)]*mult_factor_neg[k] 
            wu[t+1, (phi[jt[t]] == k) & (y==1)] = wu[t, (phi[jt[t]] == k) & (y==1)]*mult_factor_pos[k]
            wu[t+1, (phi[jt[t]] == k) & (y==-1)] = wu[t, (phi[jt[t]] == k) & (y==-1)]*mult_factor_neg[k]
            
#         for i in range(n_samples):
#             w[t+1,i] = w[t,i]*np.exp(-y[i]*c[t][phi[jt[t],i]])
        w[t+1] = w[t+1]/w[t+1].sum()
        
        print 'Zu=%f, KL(w[t+1], w[t])=%f, -2ln(Zt)=%f'% (wu[t+1].sum(), KL_divergence(w[t+1], w[t]), -2*np.log(Zt))
        
#         plt.hist(w[t+1])
                
#         plot_with_colorbar_noticks(np.reshape(w[t+1], (original_h, original_w)),
#                                    "sample weights, t=%d, Z=%f"%(t,Z[jt[t]]))
        
        curr_score = predict(jt,c, curr_frame, scale)
        plot_with_colorbar_noticks(curr_score, "t=%d, current prediction"%t)
        
    return jt, c

# <codecell>

def AdaBoostMH(x, Y):
    T = 10
    m = len(x)
    K = len(Y)
    D = np.zeros((T, m, K))
    h = np.zeros((T, m, K))
    D[0] = 1./(m*K)
    for t in range(T):
        h[t,i,:]
        for i in range(m):
            D[t+1, i, :] = D[t, i]*np.exp(-alpha[t]*Y[i,:]*h[t,i,:])
        D[t+1] /= D[t+1].sum()

# <codecell>

def find_stump_for_one_feature(fj, y, w):
    n_samples = len(y)
    W0 = np.zeros((2,))
    W1 = np.zeros((2,))
    W0[1] = 0
    W0[0] = 0
    W1[1] = w[y==1].sum()
    W1[0] = w[y==-1].sum()
    
    err_best = min(W1[1], W1[0])
    c0 = 1 if W0[1] > W0[0] else -1
    c1 = 1 if W1[1] > W1[0] else -1
    v = fj[0] - 0.1
    for i in range(n_samples):
        W0[1 if y[i]==1 else 0] += w[i]
        W1[1 if y[i]==1 else 0] -= w[i]
        
        err = min(W0[0],W0[1]) + min(W1[0],W1[1])
        if err < err_best:
            err_best = err
            if i < n_samples-1:
                v = fj[i]
            else:
                v = fj[n_samples-1]
            c0 = 1 if W0[1] > W0[0] else -1
            c1 = 1 if W1[1] > W1[0] else -1
    
    return c0,c1,v,err_best
                            
def adaboost_decision_stump(haar_vals, y):
    n_iter = 1
    n_samples = len(y)
    epsilon = 1./n_samples

    jt = np.zeros((n_iter, ), dtype=np.int)
    params_t = np.zeros((n_iter, 3))
    alpha_t = np.zeros((n_iter, ))
    Z_t = np.zeros((n_iter, ))
    
    orders = [haar_vals[:,j].argsort() for j in range(n_haars)]
    f_sorted_j = [haar_vals[orders[j],j] for j in range(n_haars)]
    y_sorted_j = [y[orders[j]] for j in range(n_haars)]
    w = 1./n_samples * np.ones((n_samples,))
    for t in range(n_iter):
        print 'iter %d'% t
        w_sorted_j = [w[orders[j]] for j in range(n_haars)]
        err_best = np.inf
        for j in range(n_haars):
            c0,c1,v,err = find_stump_for_one_feature(f_sorted_j[j], y_sorted_j[j], w_sorted_j[j])
            if err < err_best:
                err_best = err
                j_best = j
                params_best = [c0,c1,v]
        
        Z_t[t] = 2*np.sqrt(err_best*(1-err_best))
        print 'err_best=%.2f, Zt = %.2f'%(err_best,Z_t[t])
        
        alpha_t[t] = .5*np.log((1-err_best + epsilon)/(err_best+epsilon))
        
        jt[t] = j_best
        params_t[t] = params_best
        c0, c1, v = params_best
        
        hjt = c1*np.ones_like(y)
        hjt[haar_vals[:,jt[t]]<v] = c0
        w[(hjt == c0) & (y == 1)] *= np.exp(-alpha_t[t]*c0)
        w[(hjt == c1) & (y == 1)] *= np.exp(-alpha_t[t]*c1)
        w[(hjt == c0) & (y == -1)] *= np.exp(alpha_t[t]*c0)
        w[(hjt == c1) & (y == -1)] *= np.exp(alpha_t[t]*c1)
        
    return jt, params_t, alpha_t

def predict(img, prev_pos, jt, params_t, alpha_t):
    II = cv2.integral(img)
#     n_predict = 10
    n_predict = np.prod(predict_window_size)
    predict_sample_vals = np.zeros((n_predict, len(jt)))
    predict_patches = np.zeros((n_predict, patch_size[1], patch_size[0]), dtype=np.uint8)
    predict_sampler = patch_coordinates_in_window(prev_pos, patch_size, predict_window_size, frame_size)
    
    predict_positions = []
    for i, sampl_position in enumerate(predict_sampler):
        predict_positions.append(sampl_position)
        predict_patches[i] = subimage(img, sampl_position, patch_size)
        predict_sample_vals[i] = compute_all_haar_for_patch(II, sampl_position, patch_size, haars[jt])
        
        F = np.zeros((n_predict, ))
        for t, ((c0,c1,v),a) in enumerate(zip(params_t, alpha_t)):
            hjt = c1*np.ones((n_predict, ))
            hjt[predict_sample_vals[:,t]<v] = c0
            F += a*hjt
            
    image_grid(predict_patches, titles=['%.2f'%(s) for i,s in enumerate(F)])
    
    return predict_positions[F.argmax()]

def apply_classifiers(haar_vals, params_t, alpha_t):
    n_samples = len(haar_vals)
    hjt = np.zeros_like(haar_vals)
    for t, ((c0,c1,v),a) in enumerate(zip(params_t, alpha_t)):
        hjt[t] = c1*np.ones((n_samples, ))
        hjt[t, haar_vals[:,t]<v] = c0
    return hjt

