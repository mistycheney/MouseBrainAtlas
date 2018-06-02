import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#from scipy.optimize import leastsq
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

def res(p,src,dst):
    T = np.matrix([[np.cos(p[2]),-np.sin(p[2]),p[0]],
                   [np.sin(p[2]), np.cos(p[2]),p[1]],
                   [0           ,0            ,1   ]])
    n  = np.size(src,0)
    xt = np.ones([n,3])        
    xt[:,:-1] = src
    xt = (xt*T.T).A
    d  = np.zeros(np.shape(src))
    d[:,0] = xt[:,0]-dst[:,0]
    d[:,1] = xt[:,1]-dst[:,1]

    r = np.sum(np.square(d[:,0])+np.square(d[:,1]))
    return r

def jac(p,src,dst):
    T = np.matrix([[np.cos(p[2]),-np.sin(p[2]),p[0]],
                   [np.sin(p[2]), np.cos(p[2]),p[1]],
                   [0           ,0            ,1   ]])
    n  = np.size(src,0)
    xt = np.ones([n,3])        
    xt[:,:-1] = src
    xt = (xt*T.T).A
    d  = np.zeros(np.shape(src))
    d[:,0] = xt[:,0]-dst[:,0]
    d[:,1] = xt[:,1]-dst[:,1]

    #look at square as g(U)=sum U_i^TU_i, U_i=f_i([t_x,t_y,theta]^T)
    dUdth_R = np.matrix([[-np.sin(p[2]),-np.cos(p[2])],
                        [ np.cos(p[2]),-np.sin(p[2])]])
    dUdth = (src*dUdth_R.T).A
    g = np.array([  np.sum(2*d[:,0]),
                    np.sum(2*d[:,1]),
                    np.sum(2*(d[:,0]*dUdth[:,0]+d[:,1]*dUdth[:,1])) ])
    return g

def hess(p,src,dst):
    n  = np.size(src,0)
    T = np.matrix([[np.cos(p[2]),-np.sin(p[2]),p[0]],
                   [np.sin(p[2]), np.cos(p[2]),p[1]],
                   [0           ,0            ,1   ]])
    n  = np.size(src,0)
    xt = np.ones([n,3])        
    xt[:,:-1] = src
    xt = (xt*T.T).A
    d  = np.zeros(np.shape(src))
    d[:,0] = xt[:,0]-dst[:,0]
    d[:,1] = xt[:,1]-dst[:,1]

    H = np.zeros([3,3])
    
    dUdth_R = np.matrix([[-np.sin(p[2]),-np.cos(p[2])],
                        [ np.cos(p[2]),-np.sin(p[2])]])
    dUdth = (src*dUdth_R.T).A

    H[0,0] = n*2
    H[0,1] = 0
    H[0,2] = np.sum(2*dUdth[:,0])
    
    H[1,0] = 0
    H[1,1] = n*2
    H[1,2] = np.sum(2*dUdth[:,1])
    
    H[2,0] = H[0,2]
    H[2,1] = H[1,2]

    d2Ud2th_R = np.matrix([[-np.cos(p[2]), np.sin(p[2])],
                           [-np.sin(p[2]),-np.cos(p[2])]])
    d2Ud2th = (src*d2Ud2th_R.T).A
    
    H[2,2] = np.sum(2*(np.square(dUdth[:,0])+np.square(dUdth[:,1]) + d[:,0]*d2Ud2th[:,0]+d[:,0]*d2Ud2th[:,0]))
    return H

def debug_gradient(p,src,dst):
    '''
    Compare gradient with numerical approxmimation 
    '''    
    r_t_x = r_t_y = 1
    
    g_a = jac(p,src,dst)
    g_n = approx_fprime(p,res,[1.0e-10,1.0e-10,1.0e-10],src,dst)
    
    print "g_a:",g_a
    print "g_n:",g_n

    H_a = hess(p,src,dst)

    #element of gradient
    def g_p_i(p,src,dst,i):
        g = jac(p,src,dst)
        return g[i]

    #assuming analytical gradient is correct!
    H_x_n     = approx_fprime(p,g_p_i,1.0e-10,src,dst,0)
    H_y_n     = approx_fprime(p,g_p_i,1.0e-10,src,dst,1)
    H_theta_n = approx_fprime(p,g_p_i,1.0e-10,src,dst,2)
    H_n = np.zeros([3,3])
    H_n[0,:] = H_x_n
    H_n[1,:] = H_y_n
    H_n[2,:] = H_theta_n

    print "H_a:\n",H_a
    print "H_n:\n",H_n

def least_squared_2d_transform(src,dst,p0):
    '''
    Find the translation and roation (matrix) that
    gives a local optima to
    
    sum (T(src[i])-dst[i])^T*(T(src[i])-dst[i])

    src: (nx2) [x,y]
    dst: (nx2) [x,y]
    p0:  (3x,) [x,y,theta]
   
    '''
   

    #least squares want's 1d functions
    #result = leastsq(res,p0,Dfun=jac,col_deriv=1,full_output=1)
        
    #p_opt  = fmin_bfgs(res,p0,fprime=jac,args=(src,dst),disp=1)        
    result  = minimize(res,p0,args=(src,dst),method='Newton-CG',jac=jac,hess=hess)
    #print result
    p_opt = result.x
    T_opt  = np.array([[np.cos(p_opt[2]),-np.sin(p_opt[2]),p_opt[0]],
                       [np.sin(p_opt[2]), np.cos(p_opt[2]),p_opt[1]]])
    return p_opt,T_opt
                       

def icp(a, b, init_pose=(0,0,0), no_iterations = 13):    
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.



    1. For each point in the source point cloud, find the closest point in the reference point cloud.
    2. Estimate the combination of rotation and translation using a mean squared error cost function that will best align each source point to its match found in the previous step.
    3. Transform the source points using the obtained transformation.
    4. Iterate (re-associate the points, and so on).
    '''

    print "init_pose:",init_pose
    
    #print "a: ",np.shape(a)
    #print "b: ",np.shape(b)
    
    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #print "src1: ",np.shape(src)
    #print "dst1: ",np.shape(dst)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])
    #print "src2: ",np.shape(src)

    p_opt = np.array(init_pose)
        
    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        #T = cv2.estimateRigidTransform(src, dst[0, indices.T], False) #this thing can return None for unknown reasons!!!

        if i==0:
            print "squared error at p0 = " + str(res([0,0,0],src[0],dst[0, indices.T][0]))

            
        #debug_gradient([0,0,0],src[0],dst[0, indices.T][0])
            
        p,T = least_squared_2d_transform(src[0],dst[0, indices.T][0],[0,0,0])
                               
        #Transform the previous source and update the
        #current source cloudpoint
        p_opt[:2]  = (p_opt[:2]*np.matrix(T[:2,:2]).T).A       
        p_opt[0] += p[0]
        p_opt[1] += p[1]
        p_opt[2] += p[2]
        
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint        
        Tr = (np.matrix(np.vstack((T,[0,0,1])))*np.matrix(Tr)).A
        
    p_opt[2] = p_opt[2] % (2*np.pi)
    print "squared error at p_opt = " + str(res([0,0,0],src[0],dst[0, indices.T][0]))
    print "p_opt:",p_opt

    return p_opt,np.matrix(Tr)

if __name__ == "__main__":
    import pylab
    import numpy.random
    
    fig = pylab.figure(figsize=(10,10))
    ax  = fig.add_subplot(111,aspect='equal')
    
    #Create the datasets
    ang = np.linspace(-np.pi/2, np.pi/2, 520)
    a = np.array([ang, np.sin(ang)])

    #reference is a rotated by pi/2 and translated [0.2,0.3]
    th = np.pi/2
    rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    b = np.dot(rot, a) + np.array([[0.2], [0.3]]) #reference

    idx = numpy.random.choice(520,size=60,replace=False)
    a= a[:,idx]
    
    #plot them
    ref_h,   = ax.plot(b[0],b[1],'b')
    input_h  = ax.scatter(a[0],a[1],marker='x',color='r')

    #homogeneous coords
    a_h = np.ones([3,np.size(a,1)])
    a_h[:-1,:] = a
    
    #guess for correct pose 
    #init_pose=[-1.0,0,0.1]
    init_pose=[0.2+5,0.3-7,th+0.7]
    #init_pose=[0,0,0]
    T_g = np.matrix([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    a_g = T_g*a_h
    guess_h  = ax.scatter(a_g[0],a_g[1],marker='o',color='g')
    

    #Run the icp
    p_opt,T_opt = icp(a, b,init_pose,no_iterations=35)
    a_opt = T_opt*a_h
    
    result_h = ax.scatter(a_opt[0],a_opt[1],marker='o',color='k')
    
    ax.legend((ref_h,input_h,guess_h,result_h),('reference','input','guess','result'),scatterpoints=1)
    
    pylab.show()