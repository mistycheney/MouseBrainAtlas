# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

import sigboost
import numpy as np
import argparse, os, json, pprint

if __name__ == '__main__':

    class args:
        param_id = 'nissl324'
    #     img_file = '../ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif'
        img_file = '../DavidData/RS155_x5/RS155_x5_0004.tif'
        output_dir = '/oasis/scratch/csd181/yuncong/output'
        params_dir = '/oasis/projects/nsf/csd181/yuncong/Brain/params'
    
#     parser = argparse.ArgumentParser(
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     description='Test SigBoost module',
#     epilog="""Example:
#     python %s ../ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif nissl324
#     """%(os.path.basename(sys.argv[0]), ))

#     parser.add_argument("img_file", type=str, help="path to image file")
#     parser.add_argument("param_id", type=str, help="parameter identification name")
#     parser.add_argument("-o", "--output_dir", type=str, help="output directory (default: %(default)s)", default='/oasis/scratch/csd181/yuncong/output')
#     parser.add_argument("-p", "--params_dir", type=str, help="directory containing csv parameter files %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/Brain/params')
#     args = parser.parse_args()

    params_dir = os.path.realpath(args.params_dir)
    param_file = os.path.join(params_dir, 'param_%s.json'%args.param_id)
    param_default_file = os.path.join(params_dir, 'param_default.json')
    param = json.load(open(param_file, 'r'))
    param_default = json.load(open(param_default_file, 'r'))

    for k, v in param_default.iteritems():
        if not isinstance(param[k], basestring):
            if np.isnan(param[k]):
                param[k] = v

#     pprint.pprint(param)

    img_file = os.path.realpath(args.img_file)
    img_path, ext = os.path.splitext(img_file)
    img_dir, img_name = os.path.split(img_path)
    
    output_dir = os.path.realpath(args.output_dir)

    result_name = img_name + '_param_' + str(param['param_id'])
    result_dir = os.path.join(output_dir, result_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    segmentation = np.load(os.path.join(result_dir, result_name + '_segmentation.npy'))
    p = np.load(os.path.join(result_dir, result_name + '_sp_texton_hist_normalized.npy'))
    q = np.load(os.path.join(result_dir, result_name + '_sp_dir_hist_normalized.npy'))
        
    detector = sigboost.ModelDetector(param, segmentation, p, q, result_dir, bg_superpixels)
    detector.compute_all_clusters()
#     detector.sigboost()
        

