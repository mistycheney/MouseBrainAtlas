import os
import sys
import time
print(os.environ['REPO_DIR'])
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from preprocess_utility import *
from data_manager import *
from metadata import *
#for stack in all_stacks:
    
for stack in ['MD603']:
        
    first_sec, last_sec = metadata_cache['section_limits'][stack]

    #################################

    t = time.time()
    sys.stderr.write('running svm classifier ...')

    exclude_nodes = [33]

    run_distributed_aws(command='%(script_path)s %(stack)s %%(first_sec)d %%(last_sec)d' % \
                    {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/svm_predict.py',
                    'stack': stack},
                    kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                    exclude_nodes=exclude_nodes,
                    argument_type='partition')

    sys.stderr.write('done in %f seconds\n' % (time.time() - t)) 

#     #################################
    print("CHECKING");

    t = time.time()
    sys.stderr.write('interpolating scoremaps ...')


    run_distributed_aws(command='%(script_path)s %(stack)s %%(first_sec)d %%(last_sec)d' % \
                    {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/interpolate_scoremaps_v2.py',
                    'stack': stack},
                    kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                    exclude_nodes=exclude_nodes,
                    argument_type='partition')

    sys.stderr.write('done in %f seconds\n' % (time.time() - t)) # ~240 seconds 

    #################################

    t = time.time()
    sys.stderr.write('visualize scoremaps ...')

    add_annotation = False

    exclude_nodes = [33]
    # first_sec, last_sec = DataManager.load_cropbox(stack)[4:]

    run_distributed_aws(command='%(script_path)s %(stack)s -b %%(first_sec)d -e %%(last_sec)d %(add_annotation)s' % \
                    {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/visualize_scoremaps_v2.py',
                    'stack': stack,
                    'add_annotation': '-a' if add_annotation else ''},
                    kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                    exclude_nodes=exclude_nodes,
                    argument_type='partition')

    sys.stderr.write('done in %f seconds\n' % (time.time() - t)) # 464 seconds / stack

    #################################

    paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                        'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
    singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
    structures = paired_structures + singular_structures

    #################################

    t = time.time()
    sys.stderr.write('constructing score volumes ...')

    exclude_nodes = [33]

    run_distributed_aws(command='%(script_path)s %(stack)s %%(label)s' % \
                    {'script_path': os.path.join(os.environ['REPO_DIR'], 'reconstruct') + '/construct_score_volume_v2.py',
                    'stack': stack},
                    kwargs_list=dict(label=structures),
                    exclude_nodes=exclude_nodes,
                    argument_type='single')

    sys.stderr.write('done in %f seconds\n' % (time.time() - t))

    #################################

    downscale_factor = 32

    #################################

    print stack

    volume_allLabels = {}

    for name_u in structures:
        volume = DataManager.load_score_volume(stack, label=name_u, downscale=downscale_factor, train_sample_scheme=1)
        volume_allLabels[name_u] = volume
#         del volume

    t1 = time.time()

    gradient_dir = create_if_not_exists(os.path.join(VOLUME_ROOTDIR, stack, 'score_volume_gradients'))

    for name_u in structures:

        t = time.time()

        gy_gx_gz = np.gradient(volume_allLabels[name_u].astype(np.float32), 3, 3, 3) 
        # 3.3 second - re-computing is much faster than loading
        # .astype(np.float32) is important; 
        # Otherwise the score volume is type np.float16, np.gradient requires np.float32 and will have to convert which is very slow
        # 20s (float32) vs. 2s (float16)

        sys.stderr.write('Gradient %s: %f seconds\n' % (name_u, time.time() - t))

        t = time.time()

        bp.pack_ndarray_file(gy_gx_gz[0], os.path.join(gradient_dir, '%(stack)s_down%(ds)d_scoreVolume_%(label)s_trainSampleScheme_1_gy.bp' % {'stack':stack, 'label':name_u, 'ds': downscale_factor}))
        bp.pack_ndarray_file(gy_gx_gz[1], os.path.join(gradient_dir, '%(stack)s_down%(ds)d_scoreVolume_%(label)s_trainSampleScheme_1_gx.bp' % {'stack':stack, 'label':name_u, 'ds': downscale_factor}))
        bp.pack_ndarray_file(gy_gx_gz[2], os.path.join(gradient_dir, '%(stack)s_down%(ds)d_scoreVolume_%(label)s_trainSampleScheme_1_gz.bp' % {'stack':stack, 'label':name_u, 'ds': downscale_factor}))

        del gy_gx_gz

        sys.stderr.write('save %s: %f seconds\n' % (name_u, time.time() - t))


    sys.stderr.write('overall: %f seconds\n' % (time.time() - t1))
