import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import cPickle as pickle
from skimage.feature import greycomatrix, greycoprops
from skimage.exposure import equalize_hist, equalize_adapthist
from segment import make_folder


def glide(image, w, d, theta, levels=16, step=2):

    image = np.pad(image, int(w/2), mode='reflect')
    M, N = image.shape
    # map_homo = np.zeros((M, N))
    # map_iner = np.zeros((M, N))
    # map_clsh = np.zeros((M, N))
    map_Q1 = np.zeros((M, N))
    map_Q2 = np.zeros((M, N))
    map_Q4 = np.zeros((M, N))

    for m in xrange(0, M, step):
        print m
        for n in xrange(0, N, step):
            window = image[m:m+w, n:n+w]
            P = greycomatrix(
                window, d, theta*np.pi/180, levels,
                symmetric=True, normed=True,
            ).mean(axis=(2,3)) / float(len(d) * len(theta))
            mu = np.mean(window)
            # map_homo[m:m+step, n:n+step] = homogeneity(P)
            # map_iner[m:m+step, n:n+step] = inertia(P)
            # map_clsh[m:m+step, n:n+step] = clustershade(P)
            map_Q1[m:m+step, n:n+step] = Q1(P)
            map_Q2[m:m+step, n:n+step] = Q2(P)
            map_Q4[m:m+step, n:n+step] = Q4(P)

    # return map_homo, map_iner, map_clsh
    return map_Q1, map_Q2, map_Q4


levels = 16  # hardcoded number of levels
i = np.arange(levels)
j = np.arange(levels)
j, i = np.meshgrid(j, i, sparse=True)

weight_homo = 1. / (1. + (i - j)**2)
def homogeneity(P):
    return np.sum(weight_homo * P)

weight_iner = (i - j)**2
def inertia(P):
    return np.sum(weight_iner * P)

ipj = i + j
def clustershade(P):
    return np.sum((ipj - (np.sum(i*P) + np.sum(j*P)))**3 * P)

def Q1(P):
    return P[0:levels/2 , 0:levels/2].sum() / P.sum()
def Q2(P):
    return P[0:levels/2 , levels/2:levels].sum() / P.sum()
def Q21(P):
    return P[0:levels/4 , 2*levels/4:3*levels/4].sum() / P.sum()
def Q22(P):
    return P[0:levels/4 , 3*levels/4:levels].sum() / P.sum()
def Q23(P):
    return P[levels/4 : 2*levels/4 , 2*levels/4 : 3*levels/4].sum() / P.sum()
def Q24(P):
    return P[3*levels/4:levels , levels/4:2*levels/4].sum() / P.sum()
def Q3(P):
    return P[levels/2:levels , 0:levels/2].sum() / P.sum()
def Q4(P):
    return P[levels/2:levels , levels/2:levels].sum() / P.sum()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default=None, nargs='?')
    parser.add_argument('--n', '--subimg',   type=int, default=None)
    parser.add_argument('--g', '--grid',     type=int, default=2)
    parser.add_argument('--d', '--distance', type=int, default=1)
    parser.add_argument('--a', '--angle',  type=float, default=[0], nargs='+')
    parser.add_argument('--w', '--window',   type=int, default=31)
    parser.add_argument('--s', '--step',     type=int, default=2)
    parser.add_argument('--show', action='store_true')  # show figures
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--img', action='store_true')  # make input image figure
    parser.add_argument('--hist', action='store_true')  # make histogram figure
    parser.add_argument('--glcm', action='store_true')
    parser.add_argument('--glide', action='store_true')
    parser.add_argument('--weight', action='store_true')  # make image of weights
    args = parser.parse_args()
    d = args.d
    angle = np.array(args.a)
    w = args.w
    step = args.s

    if not args.filename is None:
        try:
            img = np.array(Image.open(args.filename), dtype='uint8')
        except IOError:
            # Assume MATLAB matrix.
            mat = sio.loadmat(args.filename)
            for key in mat.keys():
                if not key.endswith("__"):
                    img = np.array(mat[key], dtype='uint8')
                    break
        if not args.n is None:
            N = np.array(img.shape, dtype=int) / args.g
            nx, ny = (args.n - 1) % args.g, (args.n - 1) / args.g
            img = img[N[0]*ny : N[0]*(ny+1), N[1]*nx : N[1]*(nx+1)]
        img /= (256/levels)  # Fewer graylevels.
        title = "%s" % args.filename.split('/')[~0].split('.')[0]
        if not args.n is None:
            title += "-%d" % args.n
    else:
        if not args.weight:
            print "You must provide an image to do something with."
            sys.exit(1)

    if args.save:
        make_folder("figures")
        make_folder("features")

    if args.img:
        plt.figure()
        imgplot = plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()

    if args.hist:
        plt.figure()
        plt.hist(img.flatten(), levels, range=(0,levels))
        plt.title(title)
        plt.tight_layout()
        if args.save:
            plt.savefig("figures/hist_%s.png" % title)


    if args.glcm:
        P = greycomatrix(
            img, [d], angle*np.pi/180, levels,
            symmetric=True, normed=True,
        ).mean(axis=(2,3)) / float(len(angle))
        plt.figure()
        plt.imshow(P, cmap='hot', interpolation='nearest')
        plt.title(
            r"%s, d=%d, $\theta$=%s, w=%d"
            % (title, d, str(angle.astype(int)), w),
            fontsize=20,
        )
        plt.colorbar()
        plt.tight_layout()
        if args.save:
            plt.savefig("figures/glcm_%s_d%02d_a%03d-%1d_w%02d.png" \
                % (title, d, int(angle[0]), len(angle), w))

    if args.glide:
        # map_homo, map_iner, map_clsh = glide(img, w, [d], angle, levels, step)
        map_Q1, map_Q2, map_Q4 = glide(img, w, [d], angle, levels, step)
        fig = plt.figure()
        ax = []
        ax.append(fig.add_subplot(2,2,1))
        ax.append(fig.add_subplot(2,2,2))
        ax.append(fig.add_subplot(2,2,3))
        # ax[0].set_title("homogeneity")
        # ax[1].set_title("inertia")
        # ax[2].set_title("cluster shade")
        ax[0].set_title("Q1")
        ax[1].set_title("Q2")
        ax[2].set_title("Q4")
        plt.colorbar(
            ax[0].imshow(map_Q1, cmap='hot', interpolation='nearest'),
            ax=ax[0],
        )
        plt.colorbar(
            ax[1].imshow(map_Q2, cmap='hot', interpolation='nearest'),
            ax=ax[1],
        )
        plt.colorbar(
            ax[2].imshow(map_Q4, cmap='hot', interpolation='nearest'),
            ax=ax[2],
        )
        degtext = str(angle.astype(int)) + " deg"
        degtext1 = degtext[:21]
        degtext2 = degtext[21:]
        fig.text(
            0.55, 0.20,
            "window  = %2d px\n"
            "distance = %2d px\n"
            "angle      = %s\n"
            "                  %s"
            % (w, d, degtext1, degtext2)
        )
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        if args.save:
            plt.savefig("figures/glide_%s_d%02d_a%03d-%1d_w%02d.png" \
                % (title, d, int(angle[0]), len(angle), w))
            for fname, fmap in \
                zip(("Q1", "Q2", "Q4"), (map_Q1, map_Q2, map_Q4)):
                # zip(("homo", "iner", "clsh"), (map_homo, map_iner, map_clsh)):
                outfile = open(
                    "features/%s_%s_d%02d_a%03d-%1d_w%02d.p"
                    % (fname, title, d, int(angle[0]), len(angle), w),
                    "wb",
                )
                pickle.dump(fmap, outfile)

    if args.weight:
        fig = plt.figure()
        ax = []
        ax.append(fig.add_subplot(2,2,1))
        ax.append(fig.add_subplot(2,2,2))
        ax.append(fig.add_subplot(2,2,3))
        ax.append(fig.add_subplot(2,2,4))
        plt.colorbar(
            ax[0].imshow(weight_homo, cmap='hot', interpolation='nearest'),
            ax=ax[0],
        )
        plt.colorbar(
            ax[1].imshow(weight_iner, cmap='hot', interpolation='nearest'),
            ax=ax[1],
        )
        if not args.n is None:
            img = img
        else:
            img = np.random.randint(levels, size=(256,256))
        a = greycomatrix(
            img, [5], [0.00*np.pi], levels,
            symmetric=True, normed=True,
        )[:,:,0,0]
        P = greycomatrix(
            img, [5], [0.00*np.pi], levels,
            symmetric=True, normed=True,
        )[:,:,0,0]
        plt.colorbar(
            ax[2].imshow(
                (ipj - (np.sum(i*P) + np.sum(j*P)))**3,
                cmap='hot', interpolation='nearest',
            ), ax=ax[2],
        )
        imgr = np.random.randint(levels, size=(256,256))
        P = greycomatrix(
            imgr, [5], [0.00*np.pi], levels,
            symmetric=True, normed=True,
        )[:,:,0,0]
        plt.colorbar(
            ax[3].imshow(
                (ipj - (np.sum(i*P) + np.sum(j*P)))**3,
                cmap='hot', interpolation='nearest',
            ), ax=ax[3],
        )
        fig.tight_layout()
        if args.save:
            plt.savefig("figures/weights.png")

    if args.show:
        plt.show()
