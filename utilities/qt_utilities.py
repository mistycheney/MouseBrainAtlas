from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
if use_pyside:
    #print 'Using PySide'
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    #print 'Using PyQt4'
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

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
