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
import re
from pprint import pprint

from brain_labelling_gui3 import PickByColorsGUI3
# from ProcessFeatures import Analyzer

class Interactive_Labeling:
    def recalc_handler(self,event):
        print self.gui.get_labels()

    def save_handler(self,event):
        state = self.gui.get_labels()
        pprint(state)
        print 'labeling saved into %s' % self.labeling_filename
        pickle.dump(state, open(self.labeling_filename,'w'))

    def full_name(self,extension):
        return os.path.join(self.data_dir, self.result_name + extension)

    def main(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='GUI for coloring superpixels',
            epilog="""Example:
            python %s -d /home/yuncong/BrainLocal/output/RS141_x5_0000_param_redNissl RS141_x5_0000_param_redNissl
            """%(os.path.basename(sys.argv[0]), ))

        parser.add_argument("result_name", type=str, help="name of the result")
        parser.add_argument("-d", "--data_dir", type=str, help="result data directory (default: %(default)s)",\
                            default='.')

        args = parser.parse_args()

        self.result_name = args.result_name
        self.data_dir = args.data_dir
        # data_dir = os.path.realpath(args.data_dir)

        # The brain image with superpixel boundaries drawn on it
        self.img_filename = self.full_name('_segmentation.tif')
        img = np.array(Image.open(self.img_filename)).mean(axis=-1)

        # a matrix of labels indicating which superpixel a pixel belongs to.
        # each label is an integer from 0 to n_superpixels.
        # -1 means background, 0 to n_superpixel-1 corresponds to each superpixel
        self.seg_filename = self.full_name('_segmentation.npy')
        segmentation = np.load(self.seg_filename)

        # Texton histogram for each superpixel
        # self.texton_filename = self.full_name('_sp_texton_hist_normalized.npy')
        # texton=np.load(self.texton_filename)
        
        # Direction histogram for each superpixel
        # try:
        #     self.directions_filename=self.full_name('_sp_dir_hist_normalized.npy')
        #     directions=np.load(self.directions_filename)
        # except:
        #     print 'failed to load',self.directions_filename

        # a list of labels indicating which model a suerpixel is associated with. 
        # Each label is an integer from -1 to n_models-1.
        # -1 means background, 0 to n_models-1 corresponds to each model
        self.labeling_filename = self.full_name('_labeling.pkl') 
        try:
            self.labeling = pickle.load(open(self.labeling_filename,'r'))				
        except:
            self.labeling= None

        # pprint(self.labeling)

        #This allows the ability to load previous pickle file to get previous label names used
        oldindex = re.findall('_00.._',self.labeling_filename)[0]

        newindex = '_'+str(int(re.findall('_00.._',self.labeling_filename)[0].replace('_',''))-1).zfill(4)+'_'
        
        prev_labellist_filename = re.sub(oldindex,newindex,self.labeling_filename)
        print prev_labellist_filename
        try:
            self.prev_labeling = pickle.load(open(prev_labellist_filename,'r'))
        except:
            self.prev_labeling = None
             
        # Initiate modules
        # self.analyzer = Analyzer(segmentation, texton, directions, self.labeling)

        self.gui = PickByColorsGUI3(img=img, segmentation=segmentation,\
                                    recalc_callback=self.recalc_handler,\
                                    save_callback=self.save_handler,\
                                    labeling=self.labeling,\
                                    prev_labeling=self.prev_labeling,\
                                    title=self.result_name)

        self.gui.show()
        self.gui.app.exec_()

if __name__ == "__main__":
    IL=Interactive_Labeling()
    IL.main()
