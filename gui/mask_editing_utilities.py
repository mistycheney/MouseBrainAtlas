import sys, os
from collections import deque

from skimage.segmentation import slic
from skimage.morphology import remove_small_objects, disk, remove_small_holes, binary_dilation
from skimage.future.graph import rag_mean_color, cut_normalized
from multiprocess import Pool
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'preprocess'))
import morphsnakes

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *
from registration_utilities import find_contour_points
from annotation_utilities import contours_to_mask
from gui_utilities import *
from distributed_utilities import transfer_data_synced
from preprocess_utilities import DEFAULT_MINSIZE, DEFAULT_BORDER_DISSIMILARITY_PERCENTILE

VMAX_PERCENTILE = 99
VMIN_PERCENTILE = 1

SLIC_SIGMA = 2
SLIC_COMPACTNESS = 5
SLIC_N_SEGMENTS = 400 # 200 causes many superpixels to cross obvious boundaries. 400 is good, 20 per dimension.
SLIC_MAXITER = 100

SUPERPIXEL_SIMILARITY_SIGMA = 50. # higher value means greater affinity between superpixels.
SUPERPIXEL_MERGE_SIMILARITY_THRESH = .2
# threshold on affinity; edge whose affinity is above this value is not further split.
# if edge affinity is below this value, do further split.
# Higher value means more sensitive to slight intensity difference.
GRAPHCUT_NUM_CUTS = 20

BORDER_DISSIMILARITY_PERCENTILE = DEFAULT_BORDER_DISSIMILARITY_PERCENTILE
MIN_SIZE = DEFAULT_MINSIZE

FOREGROUND_DISSIMILARITY_THRESHOLD_MAX = 1.5
INIT_CONTOUR_COVERAGE_MAX = .5

INIT_CONTOUR_MINLEN = 50

MORPHSNAKE_SMOOTHING = 1
MORPHSNAKE_LAMBDA1 = 1 # imprtance of inside pixels
MORPHSNAKE_LAMBDA2 = 20 # imprtance of outside pixels
# Only relative lambda1/lambda2 matters, large = shrink, small = expand
MORPHSNAKE_MAXITER = 600
MORPHSNAKE_MINITER = 10
PIXEL_CHANGE_TERMINATE_CRITERIA = 3
AREA_CHANGE_RATIO_MAX = 1.2
AREA_CHANGE_RATIO_MIN = .1

def generate_submask_review_results(submasks_rootdir, filenames=None, which='user'):

    mask_alg_review_results = {}

    if filenames is None:
        filenames = os.listdir(submasks_rootdir)

    for img_fn in filenames:
        try:
            mask_alg_review_results[img_fn] = generate_submask_review_results_one_section(submasks_rootdir=submasks_rootdir, fn=img_fn, which=which)
        except Exception as e:
            # sys.stderr.write('%s\n' % e)
            mask_alg_review_results[img_fn] = []

    return mask_alg_review_results

def generate_submask_review_results_one_section(submasks_rootdir, fn, which):

    if which == 'auto':
        review_fp = os.path.join(submasks_rootdir, fn, fn + "_submasksAlgReview.txt")
    elif which == 'user':
        review_fp = os.path.join(submasks_rootdir, fn, fn + "_submasksUserReview.txt")
    else:
        raise Exception("Argument which must be either auto or user.")
    decisions = map(bool, np.atleast_1d(np.loadtxt(review_fp, dtype=np.float)))
    return decisions

def load_masking_parameters(submasks_rootdir):
    snake_lambda1_allFiles = {}
    dissim_threshold_allFiles = {}
    channel_allFiles = {}
    for fn in os.listdir(submasks_rootdir):
        params_fp = os.path.join(submasks_rootdir, fn, fn + '_maskingParameters.txt')
        with open(params_fp, 'r') as f:
            for line in f.readlines():
                param_name, val_str = line.split()
                if param_name == 'snake_lambda1':
                    snake_lambda1_allFiles[fn] = int(val_str)
                elif param_name == 'dissim_threshold':
                    dissim_threshold_allFiles[fn] = float(val_str)
                elif param_name == 'channel':
                    channel_allFiles[fn] = int(val_str)

    return snake_lambda1_allFiles, dissim_threshold_allFiles, channel_allFiles

def load_final_decisions(stack):

    submask_decisions_allFiles = {}
    accept_which_allFiles = {}

    fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_submasks_finalDecisions.txt')
    with open(fp, 'r') as f:
        for line in f.readlines():
            elements = line.split()
            sec = int(elements[0])
            fn = elements[1]
            accept_which = int(elements[2])
            submask_decisions = map(lambda x: bool(int(x)), elements[3:])

            accept_which_allFiles[fn] = accept_which
            submask_decisions_allFiles[fn] = submask_decisions

    return accept_which_allFiles, submask_decisions_allFiles


# def load_submask_decisions(fp):
#     assert os.path.exists(fp), 'No review file'
#     review_df = pandas.read_csv(fp, header=0, index_col=0)
#     submask_decisions = {fn: [bool(dec) for submask_i, dec in decisions.iteritems()] for fn, decisions in review_df.iterrows()}
#     return  submask_decisions

def load_submasks(submasks_rootdir):

    submasks = defaultdict(dict)
    for fn in os.listdir(submasks_rootdir):
        submasks_dir = os.path.join(submasks_rootdir, fn)
        for fp in os.listdir(submasks_dir):
            match_obj = re.match(fn + '_submask_([0-9]{1,2}).png', fp)
            if match_obj is not None:
                submask_ind = int(match_obj.groups()[0])
                submasks[fn][submask_ind] = imread(os.path.join(submasks_dir, fp)).astype(np.bool)
        submasks[fn] = [m for i, m in submasks[fn].iteritems()]
    submasks.default_factory = None
    return submasks

def auto_judge_submasks(submasks):
    n = len(submasks)

    submask_areas = [np.count_nonzero(m) for m in submasks]
    rank1 = np.argsort(submask_areas)[::-1] # sort by area from large to small

    image_center = np.r_[submasks[0].shape[1]/2, submasks[0].shape[0]/2]

    bbox_to_image_center_distance = []
    for m in submasks:
        xmin, xmax, ymin, ymax = bbox_2d(m)
        dist = np.sqrt(np.sum((image_center - ((xmin + xmax)/2, (ymin+ymax)/2))**2))
        bbox_to_image_center_distance.append(dist)

    rank2 = np.argsort(bbox_to_image_center_distance) # sort by center distance from small to large

    r1 = np.asarray([r for r, i in sorted(enumerate(rank1), key=lambda (r,i): i)])
    r2 = np.asarray([r for r, i in sorted(enumerate(rank2), key=lambda (r,i): i)])
    print 'area', submask_areas
    print 'area penalty', r1
    print 'box_to_center_distance', bbox_to_image_center_distance
    print 'bbox center penalty', r2
    rank = np.argsort(r1 + 0.2 * r2)
    best_mask_ind = rank[0]

    decisions = [False for _ in range(n)]
    decisions[best_mask_ind] = True

    return decisions

# def do_snake_worker(image, init_snake_contour_vertices, lambda1):
#     submasks = snake(img=image, init_contours=[init_snake_contour_vertices], lambda1=lambda1)
#     return submasks
#
# def snake_parallel(images_all_sections, init_snake_contours_all_sections, lambda1=1.):
#
#     pool = Pool(NUM_CORES)
#     submasks_all_sections = pool.map(lambda (img, init_cnt): do_snake_worker(img, init_cnt, lambda1),
#                                     zip(images_all_sections, init_snake_contours_all_sections))
#     pool.close()
#     pool.join()
#
#     return submasks_all_sections

def snake(img, init_submasks=None, init_contours=None, lambda1=MORPHSNAKE_LAMBDA1, return_masks=True, min_size=MIN_SIZE):

    # Find contours from mask.
    if init_contours is None:
        init_contours = []
        for submask in init_submasks:
            cnts = find_contour_points(submask.astype(np.int), sample_every=1)
            if 1 not in cnts or len(cnts[1]) == 0:
                continue
            for cnt in cnts[1]:
                if len(cnt) > INIT_CONTOUR_MINLEN:
                    init_contours.append(cnt)
    else:
        init_contours = [c.astype(np.int) for c in init_contours]

    xmin, ymin = np.min([np.min(c, axis=0) for c in init_contours], axis=0)
    xmax, ymax = np.max([np.max(c, axis=0) for c in init_contours], axis=0)
    margin = 50
    crop_xmin = max(0, xmin - margin)
    crop_ymin = max(0, ymin - margin)
    crop_xmax = min(img.shape[1], xmax + margin)
    crop_ymax = min(img.shape[0], ymax + margin)
    cropped_img = img[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]
    # cropped_img_height, cropped_img_width = cropped_img.shape[:2]
    init_contours_on_cropped_img = [c-(crop_xmin, crop_ymin) for c in init_contours]

    # init_contours = [xys for submask in submasks
    #                  for xys in find_contour_points(submask.astype(np.int), sample_every=1)[1]
    #                  if len(xys) > INIT_CONTOUR_MINLEN]
    sys.stderr.write('Extracted %d contours from mask.\n' % len(init_contours))

    # Create initial levelset
    init_levelsets = []
    for cnt in init_contours_on_cropped_img:
        init_levelset = np.zeros_like(cropped_img, np.float)

        t = time.time()
        init_levelset[contours_to_mask([cnt], cropped_img.shape[:2])] = 1.
        sys.stderr.write('Contour to levelset: %.2f seconds\n' % (time.time() - t)) # 10s

        init_levelset[:10, :] = 0
        init_levelset[-10:, :] = 0
        init_levelset[:, :10] = 0
        init_levelset[:, -10:] = 0
        init_levelsets.append(init_levelset)

    # img_enhanced = img.copy()

    #####################
    # Evolve morphsnake #
    #####################

    final_masks = []

    for levelset_ind, init_levelset in enumerate(init_levelsets):

        sys.stderr.write('\nContour %d\n' % levelset_ind)

        discard = False
        init_area = np.count_nonzero(init_levelset)

        t = time.time()

        msnake = morphsnakes.MorphACWE(cropped_img.astype(np.float), smoothing=int(MORPHSNAKE_SMOOTHING),
                                       lambda1=lambda1, lambda2=MORPHSNAKE_LAMBDA2)

        msnake.levelset = init_levelset.copy()

        dq = deque([None, None])
        for i in range(MORPHSNAKE_MAXITER):
            # At stable stage, the levelset (thus contour) will oscilate,
            # so instead of comparing to previous levelset, must compare to the one before the previous
            oneBefore_levelset = dq.popleft()

            # If less than 3 pixels are changed, stop.
            if i > MORPHSNAKE_MINITER:
                if np.count_nonzero(msnake.levelset - oneBefore_levelset) < PIXEL_CHANGE_TERMINATE_CRITERIA:
                    break

            area = np.count_nonzero(msnake.levelset)

            if area < min_size:
                discard = True
                sys.stderr.write('Too small, stop iteration.\n')
                break

            # If area changes more than 2, stop.
            labeled_mask = label(msnake.levelset.astype(np.bool))
            for l in np.unique(labeled_mask):
                if l != 0:
                    m = labeled_mask == l
                    if np.count_nonzero(m)/float(init_area) > AREA_CHANGE_RATIO_MAX:
                        msnake.levelset[m] = 0
                        sys.stderr.write('Area expands too much - nullified.\n')

            if  np.count_nonzero(msnake.levelset)/float(init_area) < AREA_CHANGE_RATIO_MIN:
                discard = True
                sys.stderr.write('Area shrinks too much, stop iteration.\n')
                break

            dq.append(msnake.levelset)

    #         t = time.time()
            msnake.step()
    #         sys.stderr.write('Step: %f seconds\n' % (time.time()-t)) # 0.6 second/step, roughly 200 steps takes 120s

        sys.stderr.write('Snake finished at iteration %d.\n' % i)
        sys.stderr.write('Snake: %.2f seconds\n' % (time.time()-t)) # 72s

        if discard:
            sys.stderr.write('Discarded.\n')
            continue
        else:
            # Handles the case that a single initial contour morphs into multiple contours
            labeled_mask = label(msnake.levelset.astype(np.bool))
            for l in np.unique(labeled_mask):
                if l != 0:
                    m = labeled_mask == l
                    if np.count_nonzero(m) > min_size:
                        final_masks.append(m)
                        sys.stderr.write('Final masks added.\n')

    if len(final_masks) == 0:
        sys.stderr.write('Snake return no valid submasks.\n')

    if return_masks:
        final_masks_uncropped = []
        for m in final_masks:
            uncropped_mask = np.zeros(img.shape[:2], np.bool)
            uncropped_mask[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1] = m
            final_masks_uncropped.append(uncropped_mask)
        return final_masks_uncropped
    else:
        final_contours = []
        for m in final_masks:
            cnts = [cnt_on_cropped + (crop_xmin, crop_ymin) for cnt_on_cropped in find_contour_points(m)[1]]
            final_contours += cnts
        return final_contours

def get_submasks(ncut_labels, sp_dissims, dissim_thresh):
    """Generate mask for snake's initial contours."""

    # t = time.time()
    superpixel_mask = np.zeros_like(ncut_labels, np.bool)
    for l, d in sp_dissims.iteritems():
        if d > dissim_thresh:
            superpixel_mask[ncut_labels == l] = 1
    superpixel_mask = remove_small_objects(superpixel_mask, min_size=200)
    # sys.stderr.write('Get mask from foreground superpixels: %.2f seconds.\n' %  (time.time() - t)) # 50 seconds.

    labelmap, n_submasks = label(superpixel_mask, return_num=True)

    dilated_superpixel_submasks = []

    for i in range(1, n_submasks+1):
        m = labelmap == i
        dilated_m = binary_dilation(m, disk(10))
        dilated_m = remove_small_objects(dilated_m, min_size=MIN_SIZE)
        dilated_superpixel_submasks.append(dilated_m)

    return dilated_superpixel_submasks


def merge_overlapping_masks(submasks):
    """
    Args:
        submasks (list)
    """
    n = len(submasks)
    overlap = np.zeros((n,n), np.int)
    for i in range(n):
        for j in range(i, n):
            overlap[i,j] = np.count_nonzero(np.logical_and(submasks[i], submasks[j]))

    import networkx as nx
    g = nx.from_numpy_matrix(overlap)
    components = nx.connected_components(g)
    output_masks = [np.any([submasks[i] for i in nodes], axis=0) for nodes in components]
    return output_masks

def generate_submasks_viz(img, submasks, color=(255,0,0), linewidth=3):
    """Generate visualization of submasks."""

    viz = gray2rgb(img)
    for i, submask in enumerate(submasks):
        cnts = find_contour_points(submask)
        if 1 not in cnts or len(cnts[1]) == 0:
            sys.stderr.write('Submask %d has no contour.\n' % i)
            continue
        for cnt in cnts[1]:
            cv2.polylines(viz, [cnt.astype(np.int)], True, color, linewidth) # blue
    return viz

    # plt.figure(figsize=(15,15));
    # plt.imshow(viz);
    # plt.show();

# def normalized_cut_superpixels(img, slic_labels):
#     # Build affinity graph.
#
#     t = time.time()
#     sim_graph = rag_mean_color(img, slic_labels, mode='similarity', sigma=SUPERPIXEL_SIMILARITY_SIGMA)
#     sys.stderr.write('Build affinity graph: %.2f seconds.\n' % (time.time() - t)) # 20 seconds
#
#     edge_weights = np.array([a['weight'] for n, d in sim_graph.adjacency_iter() for a in d.itervalues()])
#
#     # Recursively perform binary normalized cut.
#     for _ in range(3):
#         try:
#
#             t = time.time()
#             ncut_labels = cut_normalized(slic_labels, sim_graph, in_place=False,
#                                          thresh=SUPERPIXEL_MERGE_SIMILARITY_THRESH,
#                                          num_cuts=GRAPHCUT_NUM_CUTS)
#
#             sys.stderr.write('Normalized Cut: %.2f seconds.\n' % (time.time() - t)) # 1.5s for SLIC_N_SEGMENTS=200 ~ O(SLIC_N_SEGMENTS**3)
#             break
#
#         except ArpackError as e:
#             sys.stderr.write('ArpackError encountered.\n')
#             continue
#
#     # ncut_boundaries_viz = mark_boundaries(img, label_img=ncut_labels, background_label=-1, color=(1,0,0))
#     return ncut_labels

def compute_sp_dissims_to_border(img, ncut_labels):

    # Find background superpixels.
    background_labels = np.unique(np.concatenate([ncut_labels[:,0], ncut_labels[:,-1], ncut_labels[0,:], ncut_labels[-1,:]]))

    # Collect border superpixels.
    border_histos = [np.histogram(img[ncut_labels == b], bins=np.arange(0,256,5), density=True)[0].astype(np.float)
                    for b in background_labels]

    # Compute dissimilarity of superpixels to border superpixels.
    allsp_histos = {l: np.histogram(img[ncut_labels == l], bins=np.arange(0,256,5), density=True)[0].astype(np.float)
                    for l in np.unique(ncut_labels)}

    hist_distances = {l: np.percentile([chi2(h, th) for th in border_histos], BORDER_DISSIMILARITY_PERCENTILE)
                    for l, h in allsp_histos.iteritems()}
        # min is too sensitive if there is a blob at the border

    return hist_distances

def generate_dissim_viz(sp_dissims, ncut_labels):

    superpixel_border_distancemap = np.zeros_like(ncut_labels, np.float)
    for l, s in sp_dissims.iteritems():
        superpixel_border_distancemap[ncut_labels == l] = s

    # superpixel_border_distances_normed = superpixel_border_distances.copy()
    # superpixel_border_distances_normed[superpixel_border_distances > 2.] = 2.
    viz = img_as_ubyte(plt.cm.jet(np.minimum(superpixel_border_distancemap, 2.)))

    # plt.figure(figsize=(20,20));
    # im = plt.imshow(superpixel_border_distances, vmin=0, vmax=2);
    # plt.title('Superpixels distance to border');
    # plt.colorbar(im, fraction=0.025, pad=0.02);
    #
    # import io
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # plt.close();
    return viz[..., :3] # discard alpha channel

def determine_dissim_threshold(sp_dissims, ncut_labels):

    dissim_vals = np.asarray(sp_dissims.values())
    ticks = np.linspace(0, dissim_vals.max(), 100)
    dissim_cumdist = [np.count_nonzero(dissim_vals < th) / float(len(dissim_vals)) for th in ticks]

    # Strategy: Select the lowest threshold (most inclusive) while covers less than 50% of the image (avoid over-inclusiveness).

    # def moving_average(interval, window_size):
    #     window = np.ones(int(window_size))/float(window_size)
    #     return np.convolve(interval, window, 'same')

    grad = np.gradient(dissim_cumdist, 3)
    # smoothed_grad = moving_average(grad, 1)

    # Identify the leveling point
    hessian = np.gradient(grad, 3)

    # plt.plot(ticks, hessian);
    # plt.title('Hessian - minima is the plateau point of cum. distr.');
    # plt.xlabel('Dissimlarity threshold');
    # plt.savefig(os.path.join(submask_dir, '%(fn)s_spDissimCumDistHessian.png' % dict(fn=fn)));
    # plt.show()

    # print ticks[h.argsort()]
    # print ticks[np.argsort(smoothed_grad, kind='mergesort')]

    # fig, axes = plt.subplots(3, 1, sharex=True)
    # axes[0].plot(ticks, dissim_cumdist);
    # # axes[0].set_title('Cumulative Distribution of Superpixel dissimilarity to border superpixels');
    # axes[1].plot(ticks, grad);
    # axes[2].plot(ticks, hessian);

    # import io
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # plt.close();

    ticks_sorted = ticks[10:][hessian[10:].argsort()]
    # ticks_sorted = ticks[find_score_peaks(-h, min_distance=5)[0]]
    # ticks_sorted = ticks[np.argsort(smoothed_grad, kind='mergesort')] # Only mergesort is "stable".
    ticks_sorted_reduced = ticks_sorted[ticks_sorted < FOREGROUND_DISSIMILARITY_THRESHOLD_MAX]

    init_contour_coverages = np.asarray([np.sum([np.count_nonzero(ncut_labels == l)
                                                   for l, d in sp_dissims.iteritems()
                                                   if d > th]) / float(ncut_labels.size)
                                           for th in ticks_sorted_reduced])

    threshold_candidates = ticks_sorted_reduced[(init_contour_coverages < INIT_CONTOUR_COVERAGE_MAX) & \
                                                (init_contour_coverages > 0)]
    # np.savetxt(os.path.join(submask_dir, '%(fn)s_spThreshCandidates.txt' % dict(fn=fn)), threshold_candidates, fmt='%.3f')

    print threshold_candidates[:10]
    FOREGROUND_DISSIMILARITY_THRESHOLD = threshold_candidates[0]

    print 'FOREGROUND_DISSIMILARITY_THRESHOLD =', FOREGROUND_DISSIMILARITY_THRESHOLD

    return FOREGROUND_DISSIMILARITY_THRESHOLD


def contrast_stretch_image(img):
    """
    Args:
        img (2D np.ndarray): single-channel image.
    """

    # Stretch contrast
    # img_flattened = img.flatten()
    img_flattened = img[(img > 0) & (img < 255)] # ignore 0 and 255 which are likely artificial background

    vmax_perc = VMAX_PERCENTILE
    while vmax_perc > 80:
        vmax = np.percentile(img_flattened, vmax_perc)
        if vmax < 255:
            break
        else:
            vmax_perc -= 1

    vmin_perc = VMIN_PERCENTILE
    while vmin_perc < 20:
        vmin = np.percentile(img_flattened, vmin_perc)
        if vmin > 0:
            break
        else:
            vmin_perc += 1

    sys.stderr.write('%d(%d percentile), %d(%d percentile)\n' % (vmin, vmin_perc, vmax, vmax_perc) )

    from skimage.exposure import rescale_intensity
    img = img_as_ubyte(rescale_intensity(img, in_range=(vmin, vmax)))

    # img[(img <= vmax) & (img >= vmin)] = 255./(vmax-vmin)*(img[(img <= vmax) & (img >= vmin)]-vmin)
    # img[img > vmax] = 255
    # img[img < vmin] = 0
    # img = img.astype(np.uint8)

    return img

# def contrast_stretch_and_slic_image(stack, sec):
#
#     contrast_stretched_images_allChannels = contrast_stretch_image(stack, sec)
#
#     # def do_slic(c):
#     #     t = time.time()
#     #     slic_labels_ = slic(images[c].astype(np.float), sigma=SLIC_SIGMA, compactness=SLIC_COMPACTNESS,
#     #                        n_segments=SLIC_N_SEGMENTS, multichannel=False, max_iter=SLIC_MAXITER)
#     #     sys.stderr.write('SLIC: %.2f seconds.\n' % (time.time() - t)) # 10 seconds, iter=100, nseg=1000;
#     #     sp_max_std = np.percentile([images[c][slic_labels_ == l].std() for l in np.unique(slic_labels_)], 90)
#     #     return slic_labels_, sp_max_std
#
#     # pool = Pool(3)
#     # result_list = map(do_slic, range(3))
#     # slic_labels_allChannel, sp_max_stds = zip(*result_list)
#
#     slic_labels_allChannel = []
#     sp_max_stds = []
#     for c in range(3):
#         t = time.time()
#         slic_labels_ = slic(contrast_stretched_images_allChannels[c].astype(np.float), sigma=SLIC_SIGMA, compactness=SLIC_COMPACTNESS,
#                            n_segments=SLIC_N_SEGMENTS, multichannel=False, max_iter=SLIC_MAXITER)
#         sys.stderr.write('SLIC: %.2f seconds.\n' % (time.time() - t)) # 10 seconds, iter=100, nseg=1000;
#         slic_labels_allChannel.append(slic_labels_)
#
#         sp_max_std = np.percentile([contrast_stretched_images_allChannels[c][slic_labels_ == l].std() for l in np.unique(slic_labels_)], 90)
#         sys.stderr.write('sp_max_std = %.2f.\n' % sp_max_std)
#         sp_max_stds.append(sp_max_std)
#
#     best_channel_id = np.argmin(sp_max_stds)
#     sys.stderr.write('Use channel %s.\n' % ['RED', 'GREEN', 'BLUE'][best_channel_id])
#     slic_labelmap = slic_labels_allChannel[best_channel_id]
#     contrast_stretched_image = images[best_channel_id]
#
#     return contrast_stretched_image, slic_labelmap
