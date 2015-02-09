import matplotlib.pyplot as plt
import os
from collections import defaultdict
import argparse
import numpy as np
from PIL import Image
from skimage.io import imread

from preprocess_utility import create_if_not_exists

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")
args = parser.parse_args()

stack = args.stack

slide_dir = os.path.join(os.environ['LOCAL_SLIDEDATA_DIR'], stack, 'x0.3125')
all_slide_files = sorted([f for f in os.listdir(slide_dir) if f.endswith('tif')])
# all_slides = [slide_fn[:-4].split('_') for slide_fn in all_slide_files]

masked_sections_dir = os.path.join(os.environ['LOCAL_SECTIONDATA_DIR'], stack, 'autogen_maskedimg_x0.3125')
all_section_files = [f for f in os.listdir(masked_sections_dir) if f.endswith('tif')]
# all_sections = [fn[:-4].split('_') for fn in os.listdir(masked_sections_dir)]

sections = defaultdict(list)
for fn in all_section_files:
	stack, _, slide_str, section_ind = fn[:-4].split('_')
	sections[slide_str].append(fn)

for k, v in sections.iteritems():
	sections[k] = sorted(v)

for fn in all_slide_files:

	plt.figure()

	stack, slide_str, _, _ = fn[:-4].split('_')

	slide_img = np.array(Image.open(os.path.join(slide_dir, fn)).convert('RGB'))

	n_sections = len(sections[slide_str])

	ax1 = plt.subplot2grid((2,n_sections), (0,0), colspan=n_sections)

	ax1.imshow(slide_img, aspect='equal')
	ax1.set_title("slide %s" % slide_str)
	ax1.axis('off')

	for i, fn in enumerate(sections[slide_str]):
		section_img = imread(os.path.join(masked_sections_dir, fn))

		section_ind = int(fn[:-4].split('_')[3])

		ax = plt.subplot2grid((2,n_sections), (1,i), colspan=1)
		ax.imshow(section_img, aspect='equal')
		ax.set_title('section %d' % section_ind)
		ax.axis('off')

	out_dir = create_if_not_exists('evaluation_session')
	plt.savefig(out_dir + '/' + slide_str + '.jpg')

	# plt.show()
