import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.use('Agg')

def save_figure(img, tar, pred, path, note):
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    fig = plt.figure(figsize=(30, 30))
    plt.subplot(2, 3, 1)
    ax = plt.gca()
    plt.title('TP - 0')
    im = ax.imshow(img[0, ..., 0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2, 3, 2)
    ax = plt.gca()
    plt.title('TP - 1')
    im = ax.imshow(img[0, ..., 1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2, 3, 3)
    ax = plt.gca()
    plt.title('TP - 2')
    im = ax.imshow(img[0, ..., 2])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2, 3, 4)
    ax = plt.gca()
    plt.title('Target')
    im = ax.imshow(tar[0, ..., 0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2, 3, 5)
    ax = plt.gca()
    plt.title('Pred')
    im = plt.imshow(pred[0, ..., 0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2, 3, 6)
    ax = plt.gca()
    plt.title('Diff')
    im = ax.imshow(np.abs(tar[0, ..., 0]-pred[0, ..., 0]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fig.savefig(f'{path}/visual-{note}.png', dpi=300)
    plt.close()
