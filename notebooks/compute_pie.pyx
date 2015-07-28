import numpy as np
cimport numpy as np

cimport cython

from libcpp.vector cimport vector

@cython.boundscheck(False) # turn of bounds-checking for entire function
def compute_pie_histogram(np.ndarray[np.int_t, ndim=2] textonmap,
                          np.ndarray[np.int16_t, ndim=2] pie_indices_s, 
                          np.ndarray[np.int16_t, ndim=1] mys, 
                          np.ndarray[np.int16_t, ndim=1] mxs,
                          np.int_t radius, 
                          np.int_t height, 
                          np.int_t width,
                          np.int_t n_texton,
                          np.int_t ri,
                          np.int_t s):
    
    cdef np.ndarray[np.int_t, ndim=3] Hs = np.zeros((height, width, n_texton), np.int)

    cdef unsigned int n = pie_indices_s.shape[0]

    cdef unsigned int i, j
    cdef unsigned int L = len(mys)
    cdef unsigned int yc, xc, shifted_y, shifted_x
   
    cdef int t
    cdef np.ndarray[np.int_t, ndim=1] h = np.zeros((n_texton,), np.int)

    for i in range(L):
        
        yc = mys[i]
        xc = mxs[i]
        check = yc <= radius or yc >= height - radius or xc <= radius or xc >= width - radius

        for k in range(n_texton):
            h[k] = 0

        for j in range(n):

            shifted_y = pie_indices_s[j,0] + yc
            shifted_x = pie_indices_s[j,1] + xc

            if check:
                if not (shifted_y >= 0 and shifted_y < height and shifted_x >= 0 and shifted_x < width):
                    continue

            t = textonmap[shifted_y, shifted_x]
            if t > 0:
                h[t] += 1

        Hs[yc, xc] = h

    return Hs

@cython.wraparound(False)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def compute_connection_weight(np.ndarray[np.float_t, ndim=2] G_nonmaxsup,
                              np.ndarray[np.int16_t, ndim=2] circle_j,
                          list conns_ij_y, 
                          list conns_ij_x, 
                          np.ndarray[np.int16_t, ndim=1] mys, 
                          np.ndarray[np.int16_t, ndim=1] mxs,
                          np.int_t height, 
                          np.int_t width,
                          np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    
    cdef unsigned int i, j, p_i
    cdef unsigned int L = len(mys)
    cdef unsigned int n = len(conns_ij_y)
    cdef unsigned int conn_len
    cdef int xx, yy, x, y, yj, xj
    cdef float vmax = 0, v
#     cdef np.ndarray[np.float_t, ndim=1] vmaxs
    cdef np.ndarray[np.int_t, ndim=1] ys
    cdef np.ndarray[np.int_t, ndim=1] xs
    cdef np.ndarray[np.float_t, ndim=2] ret = np.zeros((L, n), np.float)
    
    for i in range(L):
        y = mys[i]
        x = mxs[i]
                            
        for p_i in range(n):
            
            ys = conns_ij_y[p_i]
            xs = conns_ij_x[p_i]
            conn_len = len(ys)
            
            yj = y + circle_j[p_i,0]
            xj = x + circle_j[p_i,1]

            if yj < 0 or yj >= height or xj < 0 or xj >= width or mask[yj,xj] == 0:
                continue
            
            vmax = 0
            for j in range(conn_len):
                yy = y + ys[j]
                xx = x + xs[j]
                
                if yy < 0 or yy >= height or xx < 0 or xx >= width or mask[yy,xx] == 0:
                    continue
                v = G_nonmaxsup[yy, xx]
                if v > vmax:
                    vmax = v
            
#             if yy == 36 and xx == 2529:
#                 print vmax
            
            if vmax > 0:
                ret[i, p_i] = vmax
                
            
#             vmaxs[p_i] = vmax
#             if vmax > 0:
#                 ret.append((i, p_i, vmax))
#         print 'vmaxs', vmaxs
#         ret.append(vmaxs)

    return ret
        
        

# @cython.boundscheck(False) # turn of bounds-checking for entire function
# def compute_halfdisc_histogram_diff(np.ndarray[np.int_t, ndim=4] H, 
#                                     np.int_t start_bin, 
#                                     np.ndarray[np.int16_t, ndim=1] mys, 
#                                     np.ndarray[np.int16_t, ndim=1] mxs,
#                                     np.int_t height, 
#                                     np.int_t width,
#                                     np.int_t n_theta):
    
#     cdef np.ndarray[np.float_t, ndim=2] Gs = np.zeros((height, width), np.float)

#     first_half_bins = np.arange(start_bin, start_bin+n_theta/2)%n_theta
#     second_half_bins = np.arange(start_bin+n_theta/2, start_bin+n_theta)%n_theta

#     H_halfdisk1 = np.sum(H[first_half_bins], axis=0).astype(np.float)
#     H_halfdisk2 = np.sum(H[second_half_bins], axis=0).astype(np.float)
        
#     H_halfdisk1 /= H_halfdisk1.sum(axis=-1)[:,:,None]
#     H_halfdisk2 /= H_halfdisk2.sum(axis=-1)[:,:,None]
    
#     for y,x in zip(mys, mxs):
# #         q = time.time()
#         Gs[y,x] = chi2(H_halfdisk1[y,x], H_halfdisk2[y,x])    
# #         print time.time() - q
        
#     return Gs

# @cython.boundscheck(False)
# def chi2s(np.ndarray[np.float_t, ndim=2] h1, np.ndarray[np.float_t, ndim=2] h2):
#     cdef unsigned int n  = h1.shape[1]
#     cdef unsigned int N  = h1.shape[0]
#     cdef float d, s
#     cdef unsigned int i,j
#     cdef np.ndarray[np.float_t, ndim=1] r = np.zeros((N,), dtype=np.float)
    
#     for j in range(N):
#         for i in range(n):
#             if h1[j,i] == 0:
#                 if h2[j,i] != 0:
#                     r[j] += h2[j,i]
#             else:
#                 if h2[j,i] == 0:
#                     r[j] += h1[j,i]
#                 else:
#                     d = h1[j,i]-h2[j,i]
#                     s = h1[j,i]+h2[j,i]
#                     r[j] += d*d/<float>s
                
#     return r
    

# @cython.boundscheck(False)
# def chi2(np.ndarray[np.float_t, ndim=1] h1, np.ndarray[np.float_t, ndim=1] h2):
#     cdef unsigned int n = len(h1)
#     cdef float t = 0
#     cdef float d, s
#     cdef unsigned int i
    
#     for i in range(n):
#         if h1[i] == 0:
#             if h2[i] != 0:
#                 t += h2[i]
#         else:
#             if h2[i] == 0:
#                 t += h1[i]
#             else:
#                 d = h1[i]-h2[i]
#                 s = h1[i]+h2[i]
#                 t += d*d/<float>s
                
#     return t
    

    
    
