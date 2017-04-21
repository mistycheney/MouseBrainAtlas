import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *

DEFAULT_BORDER_DISSIMILARITY_PERCENTILE = 30
# DEFAULT_FOREGROUND_DISSIMILARITY_THRESHOLD = .2
# DEFAULT_FOREGROUND_DISSIMILARITY_THRESHOLD = None
DEFAULT_MINSIZE = 100

def generate_submask_review_results(stack, filenames):
    sys.stderr.write('Generate submask review...\n')

    mask_alg_review_results = {}
    for img_fn in filenames:
        decisions = generate_submask_review_results_one_section(stack=stack, fn=img_fn)
        if decisions is None:
            sys.stderr.write('No review results found: %s.\n' % img_fn)
            mask_alg_review_results[img_fn] = {}
        else:
            mask_alg_review_results[img_fn] = decisions

    return cleanup_mask_review(mask_alg_review_results)


def generate_submask_review_results_one_section(stack, fn):
    review_fp = os.path.join(THUMBNAIL_DATA_DIR, "%(stack)s/%(stack)s_submasks/%(img_fn)s/%(img_fn)s_submasksAlgReview.txt") % \
                dict(stack=stack, img_fn=fn)
    return parse_submask_review_results_one_section_from_file(review_fp)

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
