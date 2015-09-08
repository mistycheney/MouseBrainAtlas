import os
import argparse
import sys
sys.path.append(os.environ['GORDON_REPO_DIR'] + '/pipeline_scripts')
from utilities2014 import *

sys.path.append('/home/yuncong/project/opencv-2.4.9/release/lib/python2.7/site-packages')
import cv2

from skimage.filters import gaussian_filter
from joblib import Parallel, delayed
from skimage.util import img_as_ubyte

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='Compute structure tensor map')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
args = parser.parse_args()

dm = DataManager(generate_hierarchy=False, stack='RS141', resol='x5', section=args.slice_ind)
dm._load_image()

t = 10
smoothed_im = gaussian_filter(dm.image, t)
dy, dx = np.gradient(smoothed_im)

display(smoothed_im)

sigma = 50
dx2 = gaussian_filter(dx*dx, sigma)
dxdy = gaussian_filter(dx*dy, sigma)
dy2 = gaussian_filter(dy*dy, sigma)
M = np.array([[dx2, dxdy],[dxdy, dy2]])
M = np.rollaxis(np.rollaxis(M, 2), 3, 1)
M = M.reshape((-1,2,2))

def tensors_eigh(tensors):
    n = len(tensors)
    vec = np.empty((n, 2))
    coh = np.empty((n, ))
    for i, m in enumerate(tensors):
        w, v = np.linalg.eigh(m)
        coh[i] = ((w[0]-w[1])/(w[0]+w[1]))**2
        vec[i] = v[:, np.argmax(np.abs(w))]
    return coh, vec
        
res = Parallel(n_jobs=16)(delayed(tensors_eigh)(tensors) for tensors in np.array_split(M, 16))

cohs, vecs = zip(*res)
coherence_map = np.reshape(np.concatenate(cohs), dm.image.shape)
eigenvec_map = np.reshape(np.vstack(vecs), (dm.image.shape[0], dm.image.shape[1], -1))
    
dm.save_pipeline_result(coherence_map, 'coherenceMap', 'npy')
dm.save_pipeline_result(eigenvec_map, 'eigenvecMap', 'npy')

# segmentation_vis = dm.load_pipeline_result('segmentationWithoutText', 'jpg')
# viz = img_as_ubyte(segmentation_vis)

viz = img_as_ubyte(dm.image_rgb.copy())

# viz = img_as_ubyte(np.ones_like(dm.image_rgb))

xs, ys = np.mgrid[:dm.image_width:30, :dm.image_height:30]
for x,y in np.c_[xs.flat, ys.flat]:
    
    if coherence_map[y,x] < .1:
        continue
    
    if not dm.mask[y,x]: continue
    vec = eigenvec_map[y,x]
    angle = np.arctan2(vec[1], vec[0])
    length = coherence_map[y,x]*100 if not np.isnan(coherence_map[y,x]) else 0
    end = [x,y] + np.array([length*np.sin(angle), -length*np.cos(angle)], dtype=np.int)
    cv2.line(viz, (x,y), tuple(end), (255,0,0), thickness=5, lineType=8, shift=0)
    
display(viz)
dm.save_pipeline_result(viz, 'structureTensorMap', 'jpg')