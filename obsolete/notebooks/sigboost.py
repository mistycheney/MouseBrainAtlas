# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from joblib import Parallel, delayed
import pathos
from pathos.multiprocessing import ProcessingPool as Pool

class ModelDetector(object):
    def __init__(self, param, segmentation, p, q, output_dir, bg_superpixels, neighbors):
        self.param = param
        self.p = p
        self.q = q
        self.output_dir = output_dir
        self.bg_superpixels = bg_superpixels
        self.neighbors = neighbors
        
    def _grow_cluster_relative_entropy(self, seed, frontier_contrast_diff_thresh = 0.1,
                                      max_cluster_size = 100):
        '''
        find the connected cluster of superpixels that have similar texture, starting from a superpixel as seed
        '''
        re_thresh_min = 0.01
        re_thresh_max = 0.8
        
        bg_set = set(self.bg_superpixels.tolist())

        if seed in bg_set:
            return [], -1

        prev_frontier_contrast = np.inf
        for re_thresh in np.arange(re_thresh_min, re_thresh_max, .01):

            curr_cluster = set([seed])
            frontier = [seed]

            while len(frontier) > 0:
                u = frontier.pop(-1)
                for v in neighbors[u]:
                    if v in bg_superpixels or v in curr_cluster: 
                        continue

                    if chi2(p[v], p[seed]) < re_thresh:
                        curr_cluster.add(v)
                        frontier.append(v)

            surround = set.union(*[neighbors[i] for i in curr_cluster]) - set.union(curr_cluster, bg_set)
            assert len(surround) != 0, seed

            frontier_in_cluster = set.intersection(set.union(*[neighbors[i] for i in surround]), curr_cluster)
            frontier_contrasts = [np.nanmax([chi2(p[i], p[j]) for j in neighbors[i] if j not in bg_set]) for i in frontier_in_cluster]
            frontier_contrast = np.max(frontier_contrasts)

            if len(curr_cluster) > max_cluster_size or \
            frontier_contrast - prev_frontier_contrast > frontier_contrast_diff_thresh:
                return curr_cluster, re_thresh

            prev_frontier_contrast = frontier_contrast
            prev_cluster = curr_cluster
            prev_re_thresh = re_thresh

        return curr_cluster, re_thresh
    

    def _grow_cluster_likelihood_ratio(self, seed, texton_model, dir_model, lr_grow_thresh = 0.1, precompute=False):
        '''
        find the connected cluster of superpixels that are more likely to be explained by given model than by null, starting from a superpixel as seed
        '''

        if seed in bg_superpixels:
            return [], -1

        curr_cluster = set([seed])
        frontier = [seed]

        while len(frontier) > 0:
            u = frontier.pop(-1)
            for v in neighbors[u]:
                if v in bg_superpixels or v in curr_cluster: 
                    continue

                if precompute:
                    ratio_v = D_texton_null[v] - D_texton_model[v] +\
                        D_dir_null[v] - D_dir_model[v]
                else:
                    ratio_v = D_texton_null[v] - chi2(p[v], texton_model) +\
                            D_dir_null[v] - chi2(q[v], dir_model)

                if ratio_v > lr_grow_thresh:
                    curr_cluster.add(v)
                    frontier.append(v)

        return curr_cluster, lr_grow_thresh

    def _visualize_cluster(self, scores, cluster='all', title='', filename=None):
        vis = scores[segmentation]
        if cluster != 'all':
            cluster_selection = np.equal.outer(segmentation, cluster).any(axis=2)
            vis[~cluster_selection] = 0

        plt.matshow(vis, cmap=plt.cm.Greys_r);
        plt.axis('off');
        plt.title(title)
        if filename is not None:
            plt.savefig(os.path.join(args.output_dir, 'stages', filename + '.png'), bbox_inches='tight')
    #     plt.show()
        plt.close();
    
    def _paint_cluster_on_img(self, cluster, title, filename=None):
        cluster_map = -1*np.ones_like(segmentation)
        for s in cluster:
            cluster_map[segmentation==s] = 1
        vis = label2rgb(cluster_map, image=img)
        plt.imshow(vis, cmap=plt.cm.Greys_r);
        plt.axis('off');
        plt.title(title)
        if filename is not None:
            plt.savefig(os.path.join(args.output_dir, 'stages', filename + '.png'), bbox_inches='tight')
    #     plt.show()
        plt.close();

    def _paint_clusters_on_img(self, clusters, title, filename=None):
        cluster_map = -1*np.ones_like(segmentation)
        for i, cluster in enumerate(clusters):
            for j in cluster:
                cluster_map[segmentation==j] = i
        vis = label2rgb(cluster_map, image=img)
        plt.imshow(vis, cmap=plt.cm.Greys_r);
        plt.axis('off');
        plt.title(title)
        if filename is not None:
            plt.savefig(os.path.join(args.output_dir, 'stages', filename + '.png'), bbox_inches='tight')
    #     plt.show()
        plt.close();
    
    
    def compute_all_clusters(self):
        '''
        compute clusters for each superpixel, using each superpixel as seed
        '''
        n_superpixels = int(self.param['n_superpixels'])
        frontier_contrast_diff_thresh = self.param['frontier_contrast_diff_thresh']
        
        pool = Pool()
        r = pool.map(self._grow_cluster_relative_entropy, range(n_superpixels))
        
#         r = Parallel(n_jobs=16)(delayed(self._grow_cluster_relative_entropy)(i, frontier_contrast_diff_thresh=frontier_contrast_diff_thresh) 
#                                 for i in range(n_superpixels))
#         r = Parallel(n_jobs=16)(delayed(self._grow_cluster_relative_entropy)(i, frontier_contrast_diff_thresh=frontier_contrast_diff_thresh) 
#                                 for i in range(n_superpixels))
        self.clusters = [list(c) for c, t in r]
        print 'clusters computed'
    
    def sigboost(self):
        '''
        perform sigboost
        '''

        n_models = self.param['n_models']
        frontier_contrast_diff_thresh = self.param['frontier_contrast_diff_thresh']
        lr_grow_thresh = self.param['lr_grow_thresh']
        beta = self.param['beta']
            
    
        f = os.path.join(self.output_dir, 'stages')
        if not os.path.exists(f):
            os.makedirs(f)

        texton_models = np.zeros((n_models, n_texton))
        dir_models = np.zeros((n_models, n_angle))

        seed_indices = np.zeros((n_models,))

        weights = np.ones((n_superpixels, ))/n_superpixels
        weights[bg_superpixels] = 0

        for t in range(n_models):

            print 'model %d' % (t)

            sig_score = np.zeros((n_superpixels, ))
            for i in fg_superpixels:
                cluster = clusters[i]
                sig_score[i] = np.mean(weights[cluster] * \
                                       (D_texton_null[cluster] - np.array([chi2(p[j], p[i]) for j in cluster]) +\
                                       D_dir_null[cluster] - np.array([chi2(q[j], q[i]) for j in cluster])))

            seed_sp = sig_score.argsort()[-1]
            print "most significant superpixel", seed_sp

            self.visualize_cluster(sig_score, 'all', title='significance score for each superpixel', filename='sigscore%d'%t)

            curr_cluster = clusters[seed_sp]
            self.visualize_cluster(sig_score, curr_cluster, title='distance cluster', filename='curr_cluster%d'%t)

            model_texton = sp_texton_hist_normalized[curr_cluster].mean(axis=0)
            model_dir = sp_dir_hist_normalized[curr_cluster].mean(axis=0)

            # RE(pj|pm)
            D_texton_model = np.empty((n_superpixels,))
            D_texton_model[fg_superpixels] = np.array([chi2(sp_texton_hist_normalized[i], model_texton) for i in fg_superpixels])
            D_texton_model[bg_superpixels] = np.nan

            # RE(qj|qm)
            D_dir_model = np.empty((n_superpixels,)) 
            D_dir_model[fg_superpixels] = np.array([chi2(sp_dir_hist_normalized[i], model_dir) for i in fg_superpixels])
            D_dir_model[bg_superpixels] = np.nan

            # RE(pj|p0)-RE(pj|pm) + RE(qj|q0)-RE(qj|qm)
            match_scores = np.empty((n_superpixels,))
            match_scores[fg_superpixels] = D_texton_null[fg_superpixels] - D_texton_model[fg_superpixels] +\
                                            D_dir_model[fg_superpixels] - D_dir_model[fg_superpixels]
            match_scores[bg_superpixels] = 0

            self.visualize_cluster(match_scores, 'all', title='match score', filename='grow%d'%t)


            matched, _ = self._grow_cluster_likelihood_ratio(seed_sp, model_texton, model_dir)
            matched = list(matched)

            self.visualize_cluster(match_scores, matched, title='growed cluster', filename='grow%d'%t)

            weights[matched] = weights[matched] * np.exp(-5*(D_texton_null[matched] - D_texton_model[matched] +\
                                                           D_dir_null[matched] - D_dir_model[matched])**beta)
            weights[bg_superpixels] = 0
            weights = weights/weights.sum()
            self.visualize_cluster((weights - weights.min())/(weights.max()-weights.min()), 'all', 
                              title='updated superpixel weights', filename='weight%d'%t)

            labels = -1*np.ones_like(segmentation)
            for i in matched:
                labels[segmentation == i] = 1
            real_image = label2rgb(labels, img)
            save_img(real_image, 'real_image_model%d'%t)

            seed_indices[t] = seed_sp
            texton_models[t] = model_texton
            dir_models[t] = model_dir

            
    def _compute_model_score_per_proc(i):
        model_score = np.empty((n_models, ))

        if i in bg_superpixels:
            return -1
        else:
            for m in range(n_models):
                matched, _ = grow_cluster_likelihood_ratio_precomputed(i, D_texton_model[m], D_dir_model[m], 
                                                                       lr_grow_thresh=lr_grow_thresh)
                matched = list(matched)
                model_score[m] = np.mean(D_texton_null[matched] - D_texton_model[m, matched] +\
                                         D_dir_null[matched] - D_dir_model[m, matched])

            best_sig = model_score.max()
            if best_sig > lr_decision_thresh: # sp whose sig is smaller than this is assigned null
              return model_score.argmax()
        return -1

            
    def apply_image(self):
        
        D_texton_model = -1*np.ones((n_models, n_superpixels))
        D_dir_model = -1*np.ones((n_models, n_superpixels))
        D_texton_model[:, fg_superpixels] = cdist(sp_texton_hist_normalized[fg_superpixels], texton_models, chi2).T
        D_dir_model[:, fg_superpixels] = cdist(sp_dir_hist_normalized[fg_superpixels], dir_models, chi2).T

        lr_decision_thresh = self.param['lr_decision_thresh']

        r = Parallel(n_jobs=16)(delayed(self._compute_model_score_per_proc)(i) for i in range(n_superpixels))
        labels = np.array(r, dtype=np.int)
        save_array(labels, 'labels')

        labelmap = labels[segmentation]
        save_array(labelmap, 'labelmap')

        labelmap_rgb = label2rgb(labelmap.astype(np.int), image=img)
        save_img(labelmap_rgb, 'labelmap')

