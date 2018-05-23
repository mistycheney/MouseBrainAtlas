# Train classifiers

`$ ./train_classifiers <structure_name> <path_to_annotation_file>`

- Infer patch labels from aligned atlas structures or hand-drawn structures using `learning/identify_patch_class_based_on_labeling_v2.ipynb`
- Optionally, one can compute patch features for all images using `learning/compute_features_for_entire_stacks.ipynb`.
- Train classifiers using `learning/train_and_test_classifier_performance_v4_UCSD_brains.ipynb`
