import collections

stem='PMD1305_reduce2_region0_0244_param5'
seg=np.load(open(stem+'_segmentation.npy','r'))
texton=np.load(open(stem+'_sp_texton_hist_normalized.npy','r'))
directions=np.load(open(stem+'_sp_dir_hist_normalized.npy','r'))

seg_shape=shape(seg)
seg_no=shape(texton)[0]

# count the number of pixels in each super-pixel
S=list(seg.flatten())
C=collections.Counter(S)
total=sum(C.values())

assert seg_no==len(C)

def calc_Null(C,texton):
    """
    calculate Null distribution over textons
    """
    Null=np.zeros([20])
    total=0;
    for i in range(len(C)):
        if isnan(texton[i,:]).any():
            print 'skipping',i
            print texton[i,:]
            continue
        Null = Null + texton[i,:]*C[i]
        total+= C[i]

    return Null/total


def RE(p1,p2):
    """
    Calculate the relative entropy between p1 and p2 (The KL distance)
    """
    c=0
    for i in range(len(p1)):
        if p1[i]>0:
            c+= p1[i] * np.log(p1[i]/p2[i])
    if isnan(c):
        c=1
    return c

def JS(p1,n1,p2,n2):
    """
    Calculate the Jensen Shannon distance between p1 and p2
    n1,n2 are the number of elements drawn from p1,p2 respectively 
    """
    f=float(n1)/(n1+n2)
    p0=p1*f+p2*(1-f)
    return f*RE(p1,p0)+(1-f)*RE(p2,p0)

def Distance_from_Null(Null,texton):
    D_Null={}  # Distance from the null
    for j in range(seg_no):
        D_Null[j]=RE(texton[j,:],Null)
    return D_Null

def most_sig_segments(D_Null):
    most_sig=[]
    for e in sorted(D_Null,key=D_Null.get,reverse=True):
        most_sig.append(e)
    return most_sig

def calc_pairwise_JS(seg_no,C,texton)
    Distances=-0.1*np.ones([seg_no,seg_no])
    for i in range(seg_no):
        print '\r',i,
        for j in range(i):
            if C[i]==0 | C[j]==0:
                continue
            D=JS(texton[i,:],C[i],texton[j,:],C[j])
            if isnan(D): continue
            if D<0: 
                D,i,j,texton[i,:],C[i],texton[j,:],C[j]
                raise Exception
            if D>1: D=1
            Distances[i,j]=D
            Distances[j,i]=D
    return Distances

Distances=calc_pairwise_JS(seg_no,texton)
H=Distances.flatten()
print shape(H)
hist(H,bins=300);
xlim([0,0.2])

def calc_neighbor_graph(seg,seg_shape):
    neighbors={}
    def add_edge(s,so):
        if not s in neighbors.keys():
            neighbors[s]=set([so])
        else:
            neighbors[s].add(so)

    for i in range(seg_shape[0]-1):
        print '\r',i,
        for j in range(seg_shape[1]-1):
            s=seg[i,j]
            s1=seg[i+1,j]
            s2=seg[i,j+1]
            if s!=s1 | s!=s2:
                if s!=s1:
                    so=s1
                else:
                    so=s2
                add_edge(s,so)
                add_edge(so,s)
    return neighbors


Sets=[]; Dists=[]; Sizes=[]
All=set([])
Pixel_no=size(seg)
for i in range(30):

def grow_region(initial,texton,C,neighbors):
    """
    grow a connected region starting from an initial set.
    initial=set of initial elements (do not have to be connected)
    """
    Count=0.0
    S=initial.copy()
    P=np.zeros([np.shape(texton)[1]])
    for e in initial:
        P=P+texton[e]*C[e]
        Count+= C[e]
    P=P/Count

    
    
    while True:
        Stmp=S.copy()
        for e in S:
            for f in neighbors[e].difference(S):
                JSD=JS(P,Count,texton[f,:],C[f])
                Dist_from_Null=RE(P,Null)
                print e,f,JSD,D_Null[f],Dist_from_Null,
                if Dist_from_Null < 0.01:
                        print 'Too close to Null'
                        break
                if JSD<D_Null[f]/4.0:
                    print ' Adding',
                    Stmp.add(f)
                    q=Count/(Count+C[f])
                    P=P*q+texton[f,:]*(1-q)
                    Count += C[f]
                print
        if S==Stmp:
            if (Dist_from_Null > 0.1) & (size(list(S))>1):
                Sets.append(S)
                Dists.append(P)
                Sizes.append(Count)
                All=All.union(S)
            break
        S=Stmp
        print S

        


# In[211]:

shape(Dists)


# In[212]:

plot(Null,label='Null')
for i in range(shape(Dists)[0]):
    if Sizes[i]<1000000:
        plot(Dists[i],label=str(i))
legend(loc=2)


# In[213]:

Sizes


# In[214]:

print size(list(All))
print seg_no


# In[215]:

Sets


# In[216]:

len(Dists)


# In[217]:

for i in range(len(Dists)):
    for j in range(i):
        print i,j,JS(Dists[i],Sizes[i],Dists[j],Sizes[j])


# In[ ]:



