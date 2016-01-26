import sys
import os
sys.path.append(os.environ['MXNET_DIR'] + '/python')
import numpy as np
import scipy
import mxnet as mx
import cv2
import logging

def data_separation(data):
    n_data = len(data)
    train_data = data[:n_data/2, :]
    val_data = data[n_data/2:n_data/4*3, :]
    test_data = data[n_data/4*3:, :]
    return {'train': train_data, 'val': val_data, 'test': test_data}


class TextureClassifier:
    def __init__(self,  model_dir, prefix, n_iter, n_class):
        self.n_class = n_class
        self.iter = {}
        self.model_dir = model_dir
        #self.init_model = mx.model.FeedForward.load(os.path.join(model_dir, prefix), n_iter, ctx=mx.gpu())
        self.init_model = mx.model.FeedForward.load(os.path.join(model_dir, prefix), n_iter, ctx=[mx.cpu(i) for i in range(16)])
        internals = self.init_model.symbol.get_internals()
        symbol = internals['flatten_output']
        symbol = mx.symbol.FullyConnected(data=symbol, name='fullc', num_hidden=n_class)
        self.symbol = mx.symbol.SoftmaxOutput(data=symbol, name='softmax')
        self.net = None

    def mx_init(self, data_dir, b_size, sets = ['train', 'eval', 'test']):
        for dataset in sets:
            rec_file = os.path.join(data_dir, '%s.rec' % dataset)
            if not os.path.exists(rec_file):
                continue
            self.iter[dataset] = mx.io.ImageRecordIter(
                path_imgrec=rec_file,
                batch_size=b_size,
                data_shape=(3, 224, 224),
                mean_img=os.path.join(self.model_dir, 'mean_224.nd'),
            )

    def mx_training(self, n_epoch, l_rate, b_size, dst_prefix):
        opt = mx.optimizer.SGD(learning_rate=l_rate)
        
        #self.net = mx.model.FeedForward(ctx=mx.gpu(), symbol=self.symbol, num_epoch=n_epoch, optimizer=opt,
        self.net = mx.model.FeedForward(ctx=[mx.cpu(i) for i in range(16)], symbol=self.symbol, num_epoch=n_epoch, optimizer=opt,
                                        arg_params=self.init_model.arg_params, aux_params=self.init_model.aux_params,
                                        allow_extra_params=True)
        self.net.fit(self.iter['train'], eval_data=self.iter['eval'],
                     batch_end_callback=mx.callback.Speedometer(b_size, 30),
                     epoch_end_callback=mx.callback.do_checkpoint(dst_prefix))

        self.net.save(dst_prefix)

    def mx_confusion(self):
        prob = self.init_model.predict(self.iter['test'])
        logging.info('Finish predict...')
        self.iter['test'].reset()
        y_batch = []
        for dbatch in self.iter['test']:
            label = dbatch.label[0].asnumpy()
            pad = self.iter['test'].getpad()
            real_size = label.shape[0] - pad
            y_batch.append(label[0:real_size])
        y = np.concatenate(y_batch)
        # get prediction label from
        py = np.argmax(prob, axis=1)
        acc1 = float(np.sum(py == y)) / len(y)
        logging.info('testing accuracy = %f', acc1)
        confusion = np.zeros((self.n_class, self.n_class))
        for i in range(len(py)):
            confusion[y[i], py[i]] += 1
        return confusion

    def mx_predict(self, data_path, b_size):
        test_iter = mx.io.ImageRecordIter(
            path_imgrec=data_path,
            batch_size=b_size,
            data_shape=(3, 224, 224),
            mean_img=os.path.join(self.model_dir, 'mean_224.nd'),
        )
        prob = self.init_model.predict(test_iter)
        return prob
