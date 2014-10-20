import sys
from pprint import pprint
import cPickle as pickle
import numpy as np
import datetime

labeling = pickle.load(open(sys.argv[1], 'r'))

history = labeling['label_history']
history = [h['data'] for h in history if h['Full']==False]

dt = datetime.datetime.now().strftime("%y%m%d%H%M%S")

new_labeling = {
'username': None,
'parent_labeling_name': None,
'login_time': dt,
'init_labellist': None,
'final_labellist': labeling['labellist'].astype(np.int),
'labelnames': labeling['names'],
'history': history
}

new_labeling_fn = sys.argv[1][:-12]+'anon_'+dt+'.pkl'
pickle.dump(new_labeling, open(new_labeling_fn, 'w'))
