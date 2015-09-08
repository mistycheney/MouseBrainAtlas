import numpy as np
from matplotlib import pyplot as plt

def visualize_correspondances(img1, img2, points1, points2, alignment):
    src = []
    dst = []
    for i, j in alignment:
        src.append(points1[i])
        dst.append(points2[j])
    src = np.array(src)
    dst = np.array(dst)

    img_combined = np.concatenate((img1, img2), axis=1)

    plt.gray()

    plt.imshow(img_combined, interpolation='nearest')
    plt.axis('off')

    color = 'red'
    plt.plot((src[:, 1], dst[:, 1] + img1.shape[1]), (src[:, 0], dst[:, 0]), '-')
    plt.plot(src[:, 1], src[:, 0], '.', markersize=10, color=color)
    plt.plot(dst[:, 1] + img1.shape[1], dst[:, 0], '.', markersize=10, color=color)

    plt.show()
