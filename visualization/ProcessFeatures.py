"""
Information-Geometry based analysis of the super-pixel descriptors. Ultimately resulting in detection models.
"""

import collections
import numpy as np

def RE(p1,p2):
    """
    Calculate the relative entropy between p1 and p2 (The KL distance)
    """
    c=0
    for i in range(len(p1)):
        if p1[i]>0:
            c+= p1[i] * np.log(p1[i]/p2[i])
    if np.isnan(c):
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

class Analyzer:
    def __init__(self,segmentation,texton,directions,labeling):

        self.seg=segmentation
        self.texton=texton
        self.directions=directions
        self.labeling=labeling

        self.seg_shape=np.shape(self.seg)
        self.seg_no=np.shape(self.texton)[0]

        # C = the number of pixels in each super-pixel
        S=list(self.seg.flatten())
        self.C=collections.Counter(S)
        self.total=sum(self.C.values())
        assert self.seg_no==len(self.C)

        self.Sets=[]; self.Dists=[]; self.Sizes=[]
        self.All=set([])
        self.Pixel_no=np.size(self.seg)

        self.calc_Null()
        self.Distance_from_Null()
        self.calc_neighbor_graph()
        
    def calc_Null(self):
        """
        calculate Null distribution over textons
        """
        Null=np.zeros([20])
        total=0;
        for i in range(len(self.C)):
            if np.isnan(self.texton[i,:]).any():
                print 'skipping',i
                print self.texton[i,:]
                continue
            Null = Null + self.texton[i,:]*self.C[i]
            total+= self.C[i]
        self.Null=Null/total
        return self.Null

    def Distance_from_Null(self):
        self.D_Null={}  # Distance from the null
        for j in range(self.seg_no):
            self.D_Null[j]=RE(self.texton[j,:],self.Null)
        return self.D_Null

    def most_sig_segments(self):
        self.most_sig=[]
        for e in sorted(self.D_Null,key=self.D_Null.get,reverse=True):
            self.most_sig.append(e)
        return self.most_sig

    def calc_pairwise_JS(self):
        self.Distances=-0.1*np.ones([self.seg_no,self.seg_no])
        for i in range(self.seg_no):
            print '\r',i,
            for j in range(i):
                if self.C[i]==0 | self.C[j]==0:
                    continue
                D=JS(self.texton[i,:],self.C[i],self.texton[j,:],self.C[j])
                if np.isnan(D): continue
                if D<0: 
                    print D,i,j,self.texton[i,:],self.C[i],self.texton[j,:],self.C[j]
                    raise Exception
                if D>1: D=1
                self.Distances[i,j]=D
                self.Distances[j,i]=D
        return self.Distances

    def calc_neighbor_graph(self):
        self.neighbors={}
        def add_edge(s,so):
            if not s in self.neighbors.keys():
                self.neighbors[s]=set([so])
            else:
                self.neighbors[s].add(so)

        for i in range(self.seg_shape[0]-1):
            print '\r',i,
            for j in range(self.seg_shape[1]-1):
                s  = self.seg[i,j]
                s1 = self.seg[i+1,j]
                s2 = self.seg[i,j+1]
                if s!=s1 | s!=s2:
                    if s!=s1:
                        so=s1
                    else:
                        so=s2
                    add_edge(s,so)
                    add_edge(so,s)
        return self.neighbors

        def recalc(self,prev_labels,new_labels):
            """
            Calculate new models and new labeling from given labeling
            """
            Labels=set(new_labels.flatten())
            
            
