import random
import numpy as np
from shape_matching import rigid_transform_from_pairs, rigid_transform_to

def stable(rankings, A, B):
    partners = dict((a, (rankings[(a, 1)], 1)) for a in A)
    is_stable = False # whether the current pairing (given by `partners`) is stable
    while is_stable == False:
        is_stable = True
        for b in B:
            is_paired = False # whether b has a pair which b ranks <= to n
            for n in range(1, len(B) + 1):
                a = rankings[(b, n)]
                a_partner, a_n = partners[a]
                if a_partner == b:
                    if is_paired:
                        is_stable = False
                        partners[a] = (rankings[(a, a_n + 1)], a_n + 1)
                    else:
                        is_paired = True
    return sorted((a, b) for (a, (b, n)) in partners.items())


def stable_marriage_matchings(D_boundaries):

    A = ['A'+str(i) for i in range(n_boundaries1)]
    B = ['B'+str(i) for i in range(n_boundaries2)]

    ao = np.zeros(D_boundaries)
    for q in range(n_boundaries1):
        ao[q, np.argsort(D_boundaries[q,:])] = np.arange(n_boundaries2)

    bo = np.zeros_like(D_boundaries.T)
    for q in range(n_boundaries2):
        bo[q, np.argsort(D_boundaries[:,q])] = np.arange(n_boundaries1)

    rankings1 = zip(A, ao+1)
    rankings2 = zip(B, bo+1)
    rank = dict(rankings1+rankings2)

    Arankings = dict(((a, rank[a][b_]), B[b_]) for (a, b_) in product(A, range(n_boundaries2)))
    Brankings = dict(((b, rank[b][a_]), A[a_]) for (b, a_) in product(B, range(n_boundaries1)))
    rankings = Arankings
    rankings.update(Brankings)

    m = stable(rankings, A, B)
    m = [(int(i[1:]), int(j[1:])) for i, j in m]
    m = sorted([(D_boundaries[i,j], i,j) for i,j in m if D_boundaries[i,j]<np.inf])

#     for s, i,j in m:
#         print s, i, j
    
    return m

def greedy_matching(D_boundaries, thresh_percentage=.2, verified_matchings=[], excluded_matchings=[]):
    
    Dnz = D_boundaries[D_boundaries < np.inf]
    if len(Dnz) == 0:
        return []
    
    th = np.sort(Dnz.flat)[int(len(Dnz.flat)*thresh_percentage)]
    print 'thresh', th
    
    matchings = [(0,i,j) for i,j in verified_matchings]
    rs, cs = np.unravel_index(np.argsort(D_boundaries.flat), D_boundaries.shape)
    for r, c in zip(rs, cs):
        if (r,c) in excluded_matchings:
            continue
        if D_boundaries[r,c] > th:
            break
        if r not in [i for d,i,j in matchings] and c not in [j for d,i,j in matchings]:
            matchings.append((D_boundaries[r,c],r,c))
            
    return matchings

from collections import defaultdict

def knn_matching(D_boundaries, boundaries1, boundaries2, k=2, centroid_dist_limit=500):
    
    import networkx as nx
    
    n_boundaries1, n_boundaries2 = D_boundaries.shape

    nn1 = D_boundaries.argsort(axis=1)
    dd1 = np.sort(D_boundaries,axis=1)
    nn1 = [nn[:np.searchsorted(d, d[0]+0.2)] for d, nn in zip(dd1, nn1)]
    
    nn2 = D_boundaries.argsort(axis=0).T
    dd2 = np.sort(D_boundaries,axis=0).T
    nn2 = [nn[:np.searchsorted(d, d[0]+0.2)] for d, nn in zip(dd2, nn2)]

    DD = np.zeros((n_boundaries1+n_boundaries2, n_boundaries1+n_boundaries2))
    G = nx.Graph(DD)
    G = nx.relabel_nodes(G, dict([(i,(0,i)) for i in range(n_boundaries1)]+[(n_boundaries1+j,(1,j)) for j in range(n_boundaries2)]))
    matches = []
    for i in range(n_boundaries1):
        for j in range(n_boundaries2):
#             if j in nn1[i,:k] and i in nn2[j,:k]:
            if j in nn1[i] and i in nn2[j]:
                matches.append((i,j))
                G.add_edge((0,i), (1,j))

    ms = [sorted(g) for g in sorted(list(nx.connected_components(G)), key=len, reverse=True) if len(g) >= 2]
#     print len(ms), 'matchings'
    
    groups = []
    for mi, m in enumerate(ms):
        d = defaultdict(list)
        for sec_i, bnd_i in m:
            d[sec_i].append(bnd_i)
        A = D_boundaries[d[0]][:,d[1]]
        rs, cs = np.unravel_index(np.argsort(D_boundaries[d[0]][:,d[1]].flat), (len(d[0]), len(d[1])))
    #     print rs, cs
    #     print [((sec1, d[sec1][r]), (sec2, d[sec2][c]), D_boundaries[d[sec1][r], d[sec2][c]]) for r, c in zip(rs, cs)
    #           if D_boundaries[d[sec1][r], d[sec2][c]] < np.inf]

        g = []
        for r, c in zip(rs, cs):
            if D_boundaries[d[0][r], d[1][c]] < np.inf:
    #             print ((sec1, d[sec1][r]), (sec2, d[sec2][c]), D_boundaries[d[sec1][r], d[sec2][c]])
                g.append([d[0][r], d[1][c]])

        groups.append(g)

    #     print '\n'
    
    boundary1_centers = np.array([b[4][::-1] for b in boundaries1])
    boundary2_centers = np.array([b[4][::-1] for b in boundaries2])

    matches = []
    scores = []

    for ransac_iter in range(5000):

        boundary_samples = [random.sample(g, 1)[0] for g in random.sample(groups, 3)]
        X = []
        Y = []
        for b1, b2 in boundary_samples:
            X.append(boundary1_centers[b1])
            Y.append(boundary2_centers[b2])
        X = np.array(X)
        Y = np.array(Y)

        T, angle = rigid_transform_from_pairs(X,Y)
    #     print T, angle
        if angle > np.pi/2:
    #         print 'angle too wide'
            matches.append([])
            scores.append(0)
            continue

        boundary1_centers_trans = rigid_transform_to(boundary1_centers, T)

        match = [(bi,bj) for g in groups for bi,bj in g 
                 if np.linalg.norm(boundary1_centers_trans[bi] - boundary2_centers[bj]) < centroid_dist_limit]

        score = len(match)

        matches.append(match)
        scores.append(score)

    best = np.argmax(scores)
    s_best = scores[best]
    m_best = matches[best]

    g = nx.Graph()
    g.add_edges_from([((0, i),(1, j), {'weight': D_boundaries[D_boundaries!=np.inf].max()-D_boundaries[i,j]}) 
                      for i,j in m_best])
    m = nx.matching.max_weight_matching(g, maxcardinality=True)
    
    best_match = set(((0, dict([n1,n2])[0]), (1, dict([n1,n2])[1])) for n1,n2 in m.iteritems())
    best_match = [(D_boundaries[t1[1],  t2[1]], t1[1], t2[1]) for t1,t2 in best_match]

    return best_match