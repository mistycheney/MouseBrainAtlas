# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from utilities import *

# <codecell>

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Generate textons from a set of filtered images',
# epilog="""%s
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("begin_slice", type=str, help="slice number to begin, zero-padding to 4 digits")
# parser.add_argument("end_slice", type=str, help="slice number to end, zero-padding to 4 digits")
# parser.add_argument("param_id", type=str, help="parameter identification name")
# args = parser.parse_args()


class args:
    stack_name = 'RS141'
    resolution = 'x5'
    gabor_param_id = '1'
    

paramset = ParameterSet(gabor_param_id=args.gabor_param_id)
    
instance = Instance(args.stack_name, args.resolution, paramset=args.param_id)

# <codecell>

theta_interval = paramset.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = paramset.gabor_params['freq_step']
freq_max = 1./paramset.gabor_params['min_wavelen']
freq_min = 1./paramset.gabor_params['max_wavelen']
bandwidth = paramset.gabor_params['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies 
          for t in np.arange(0, n_angle)*np.deg2rad(theta_interval)]
kernels = map(np.real, kernels)

n_kernel = len(kernels)

print '=== filter using Gabor filters ==='
print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

