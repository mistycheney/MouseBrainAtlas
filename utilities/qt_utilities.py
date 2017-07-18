from PyQt4.QtCore import *
from PyQt4.QtGui import *

import copy
import numpy as np

gray_color_table = [qRgb(i, i, i) for i in range(256)]

def numpy_to_qimage(img):
    """
    Convert numpy array to QImage.
    """
    h, w = img.shape[:2]
    if img.ndim == 3:
        qimage = QImage(img.flatten(), w, h, 3*w, QImage.Format_RGB888)
    else:
        qimage = QImage(img.flatten(), w, h, w, QImage.Format_Indexed8)
        qimage.setColorTable(gray_color_table)

    return qimage

# def qimage_to_numpy(qimage):
#     '''
#     Converts  QImage to a numpy array.
#
#     Reference: https://stackoverflow.com/a/36646415
#     '''
#
#     qimage = qimage.convertToFormat(QImage.Format_RGB888)
#
#     width = qimage.width()
#     height = qimage.height()
#
#     ptr = qimage.constBits()
#     print len(np.array(ptr)), width, height
#     arr = np.array(ptr).reshape(height, width, 3)  #  Copies the data
#     return arr

def qimage_to_numpy(img, share_memory=False):
    """ Creates a numpy array from a QImage.

        If share_memory is True, the numpy array and the QImage is shared.
        Be careful: make sure the numpy array is destroyed before the image,
        otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QImage), "img must be a QtGui.QImage object"
    assert img.format() == QImage.Format_RGB32, \
        "img format must be QImage.Format.Format_RGB32, got: {}".format(img.format())

    img_size = img.size()
    buffer = img.constBits()

    # Sanity check
    n_bits_buffer = len(buffer) * 8
    n_bits_image  = img_size.width() * img_size.height() * img.depth()
    assert n_bits_buffer == n_bits_image, \
        "size mismatch: {} != {}".format(n_bits_buffer, n_bits_image)

    assert img.depth() == 32, "unexpected image depth: {}".format(img.depth())

    # Note the different width height parameter order!
    arr = np.ndarray(shape  = (img_size.height(), img_size.width(), img.depth()//8),
                     buffer = buffer,
                     dtype  = np.uint8)

    if share_memory:
        return arr
    else:
        return copy.deepcopy(arr)
