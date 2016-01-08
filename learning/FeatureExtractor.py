import mxnet as mx
import numpy as np
import os


class FeatureExtractor:
    def __init__(self, model_dir, batch_size=1, ctx='cpu'):
        model = mx.model.FeedForward.load(model_dir, 1)
        internals = model.symbol.get_internals()
        fea_symbol = internals["fc7_output"]
        if ctx == 'cpu':
            ctx = mx.cpu()
        else:
            ctx = mx.gpu()
        self.net = mx.model.FeedForward(ctx=ctx, symbol=fea_symbol, numpy_batch_size=batch_size,
                                        arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    def extract(self, images):
        return self.net.predict(images)


if __name__ == '__main__':
    data_dir = '/home/jiaxuzhu/data/landmark_patches'
    model_dir = '/home/jiaxuzhu/developer/CSD395/model/vgg16'
    names = ['5N', '7n', '7N', '12N', 'Gr', 'LVe', 'Pn', 'SuVe', 'VLL']

    fe = FeatureExtractor(model_dir, batch_size=16, ctx='cpu')

    for name in names:
        images = np.load(os.path.join(data_dir, '%s_patches.npy' % name)).transpose(0, 3, 1, 2)
        print images.shape
        
        features = fe.extract(images)
        print features.shape
        np.save(os.path.join(data_dir, '%s_features.npy' % name), features)