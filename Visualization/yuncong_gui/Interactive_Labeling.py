"""
The master script that manages the GUI and the analytical code
"""
import argparse
import pickle
import datetime
import sys
import os
from PIL import Image
import numpy as np

from brain_labelling_gui import PickByColorsGUI
from ProcessFeatures import Analyzer

class Interactive_Labeling:
    def recalc_handler(self,event):
        print 'recalc_handler',event
        print self.main_window.get_labels()

    def save_handler(self,event):
        print 'save_handler'
        state = self.main_window.get_labels()
        print state
        pickle.dump(state,open(self.labellist_filename,'w'))

    def full_name(self,extension):
        return os.path.join(self.data_dir, self.result_name + extension)

    def main(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='GUI for coloring superpixels',
            epilog="""Example:
            python %s PMD1305_region0_reduce2_0244_param_nissl324
            """%(os.path.basename(sys.argv[0]), ))

        parser.add_argument("result_name", type=str, help="name of the result")
        parser.add_argument("-d", "--data_dir", type=str, help="result data directory (default: %(default)s)",\
                            default='.')
        args = parser.parse_args()
        self.result_name=args.result_name
        self.data_dir=args.data_dir
        data_dir = os.path.realpath(args.data_dir)

        # The brain image with superpixel boundaries drawn on it
        self.img_filename = self.full_name('_segmentation.tif')
        img = np.array(Image.open(self.img_filename)).mean(axis=-1)

        # a matrix of labels indicating which superpixel a pixel belongs to 
        # each label is an integer from 0 to n_superpixels.
        # -1 means background, 0 to n_superpixel-1 corresponds to each superpixel
        self.seg_filename = self.full_name('_segmentation.npy')
        segmentation = np.load(self.seg_filename)

        # Texton histogram for each superpixel
        self.texton_filename = self.full_name('_texton_hist_normalized.npy')
        texton=np.load(self.texton_filename)
        
        # Direction histogram for each superpixel
        self.directions_filename=self.full_name('_dir_hist_normalized.npy')
        directions=np.load(self.directions_filename)

        # a list of labels indicating which model a suerpixel is associated with. 
        # Each label is an integer from -1 to n_models-1.
        # -1 means background, 0 to n_models-1 corresponds to each model
        self.labellist_filename = self.full_name('_labeling.pkl') 

        try:
            self.labeling = pickle.load(open(self.labellist_filename,'r'))
        except:
            self.labeling= None

        # Initiate modules
        # self.analyzer=Analyzer(segmentation,texton,directions,self.labeling)

        self.main_window = PickByColorsGUI(img=img, segmentation=segmentation,\
                                      recalc_callback=self.recalc_handler,\
                                      save_callback=self.save_handler,\
                                      labeling=self.labeling)

        self.main_window.show()
        self.main_window.app.exec_()

if __name__ == "__main__":
    IL=Interactive_Labeling()
    IL.main()

