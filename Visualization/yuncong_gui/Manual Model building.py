
# coding: utf-8

# In[1]:

pylab.rcParams['figure.figsize'] = (10.0, 6.0)


# In[2]:

get_ipython().system(u'ls data')


# In[3]:

seg=np.load(open('data/PMD1305_reduce2_region0_0244_param5_segmentation.npy','r'))
texton=np.load(open('data/PMD1305_reduce2_region0_0244_param5_texton_hist_normalized.npy','r'))
directions=np.load(open('data/PMD1305_reduce2_region0_0244_param5_dir_hist_normalized.npy','r'))


# In[24]:

import pickle
labeling=pickle.load(open('data/PMD1305_reduce2_region0_0244_param5_labeling.pkl','r'))
labeling.pop('labeling')


# In[25]:

labeling.keys()


# In[26]:

labellist=labeling['labellist']


# In[27]:

print set(labellist)
print labeling['names']


# In[28]:

seeds={}
for i in range(7):
    name=labeling['names'][i+1]
    seeds[name]=[j for j in range(len(labellist)) if labellist[j]==i]
    print i,name,len(seeds[name])


# In[29]:

for name in seeds.keys():
    print name
    print ','.join([str(e) for e in seeds[name]])


# In[30]:

get_ipython().system(u'ls *.py')


# In[31]:

from ProcessFeatures import *


# In[32]:

Anal=Analyzer(seg,texton,directions,labeling)


# In[33]:

Models={}
for name in seeds.keys():
    element=seeds[name]
    plot(Anal.Null,'b',label='Null',linewidth=3.0)
    mean=np.zeros_like(texton[0])
    Count=0
    for e in element:
        mean=mean+Anal.C[e]*texton[e]
        Count = Count + Anal.C[e]
        plot(texton[e],'.',label=str(e))
    model=mean/Count
    plot(model,'r',label='Mean',linewidth=3.0)
    legend()
    title(name)
    figure()
    Models[name]= {'model':model, 'Count':Count}


# In[34]:

for name in seeds.keys():
    element=seeds[name]
    plot(Anal.Null,'b',label='Null',linewidth=3.0)
    mean=np.zeros_like(texton[0])
    Count=0
    for e in element:
        mean=mean+Anal.C[e]*texton[e]
        Count = Count + Anal.C[e]
        plot(texton[e],'.',label=str(e))
    model=mean/Count
    plot(model,'r',label='Mean',linewidth=3.0)
    legend()
    title(name)
    figure()


# In[35]:

Stats={}
Null=Anal.Null

for name in seeds.keys():
    print name
    print ' dist to Null \t dist to Model \t difference'
    element=seeds[name]
    Alternative=Models[name]['model']
    Stats[name]={'to_null':[],
                 'to_alt':[]
                 }
    for e in element:
        to_null=RE(texton[e],Null)
        Stats[name]['to_null'].append(to_null)
        to_alt=RE(texton[e],Alternative)
        Stats[name]['to_alt'].append(to_alt)
        print '%f \t %f \t %f'%(to_null,to_alt,to_null-to_alt)
    


# ### Finding the super-pixels that have a good model ###
# Regardless of neighborhood relations.

# In[45]:

import operator
TextonModel=[]
print 'A list of superpixels that can be classified with some confidence'
print 'Current label      1st score Label      2nd-score label \t null\t1st\t2nd'
for i in range(shape(texton)[0]):
    T=texton[i,:]
    NullFit=RE(T,Null) 
    ModelFit={}
    for name in Models.keys():
        model=Models[name]['model']
        ModelFit[name]=RE(T,model)
    ModelFit=sorted(ModelFit.items(),key=operator.itemgetter(1))
    if (NullFit>0.2) & (ModelFit[1][1]-ModelFit[0][1]>0.05):
        j=labellist[i]
        name=labeling['names'][int(j)+1]
        print '%3d:%s \t%15s \t%15s \t%5.3f \t%5.3f,\t%5.3f'%                (i,name,ModelFit[0][0],ModelFit[1][0],NullFit,ModelFit[0][1],ModelFit[1][1])
    


# ### Studying the directionality distributions ###

# In[109]:

dirSigs=[]
Strong_dirs={}
for i in range(shape(directions)[0]):
    dirs=directions[i,:]
    Mx=np.max(dirs)
    Mn=np.mean(dirs)
    dirSigs.append(Mx/Mn)
    if Mx/Mn > 1.3:
        Strong_dirs[i]={'loc':argmax(dirs),
                   'strength':Mx/Mn}


# In[110]:

hist(dirSigs,bins=40);


# In[116]:

import collections
CC=collections.Counter([Strong_dirs[k]['loc'] for k in Strong_dirs.keys()])


# In[123]:

[(i,CC[i]) for i in sorted(CC)]


# In[107]:




# In[ ]:



