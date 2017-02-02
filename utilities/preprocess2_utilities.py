import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *




def delete_file_or_directory(fp):
    execute_command("rm -rf %s" % fp)

def download_from_remote(fp_remote, fp_local):
    execute_command("scp -r oasis-dm.sdsc.edu:%(fp_remote)s %(fp_local)s" % \
                    dict(fp_remote=fp_remote, fp_local=fp_local))

def download_from_remote_synced(fp_relative, remote_root='/home/yuncong/csd395/CSHL_data_processed', local_root='/home/yuncong/CSHL_data_processed'):
    remote_fp = os.path.join(remote_root, fp_relative)
    local_fp = os.path.join(local_root, fp_relative)
    create_if_not_exists(os.path.dirname(local_fp))
    execute_command("scp -r oasis-dm.sdsc.edu:%(fp_remote)s %(fp_local)s" % \
                    dict(fp_remote=remote_fp, fp_local=local_fp))

def parse_submask_review_results_one_section_from_file(review_fp):

    if not os.path.exists(review_fp):
        return

    decisions = {}
    with open(review_fp, 'r') as f:
        for line in f:
            mask_ind, decision = map(int, line.split())
            decisions[mask_ind] = decision == 1

    if len(decisions) == 0:
        return
    else:
        return decisions

def cleanup_mask_review(d):
    """
    Return {filename: {submask_ind: bool}}
    """

    labels_reviewed = {}

    for fn, alg_labels in d.iteritems():
        if len(alg_labels) == 0:
            labels_reviewed[fn] = {}
        else:
            alg_positives = [index for index, l in alg_labels.iteritems() if l == 1]
            if len(alg_positives) > 0:
                assert len(alg_positives) == 1
                correct_index = alg_positives[0]
            else:
                correct_index = 1

            alg_labels[correct_index] = 1
            for idx in alg_labels:
                if idx != correct_index:
                    alg_labels[idx] = -1

            labels_reviewed[fn] = {i: r == 1 for i, r in alg_labels.iteritems()}

    return labels_reviewed
