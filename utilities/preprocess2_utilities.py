def cleanup_mask_review(d):
    """
    Return {filename: {submask_ind: bool}}
    """

    labels_reviewed = {}

    for fn, alg_labels in d.iteritems():
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
