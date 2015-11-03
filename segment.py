import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from skimage.filters import gaussian_filter


def make_folder(directory):
    """Make the directory, but do nothing if it already exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def segment(name, tresh=None, mode="geq"):

    infile = open("features/%s.p" % name)
    data = gaussian_filter(pickle.load(infile), sigma=args.sigma)
    infile.close()

    if args.img:
        fig1 = plt.figure()
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.tight_layout()
        plt.colorbar()
        plt.title(name)
        fig1.show()

    if mode == "geq":
        data_tresh = data >= tresh
        sign = "\geq"
    elif mode == "g":
        data_tresh = data >  tresh
        sign = ">"
    elif mode == "leq":
        data_tresh = data <= tresh
        sign = "\leq"
    elif mode == "l":
        data_tresh = data <  tresh
        sign = "<"

    fig2 = plt.figure()
    plt.imshow(data_tresh, cmap='hot', interpolation='nearest')
    plt.tight_layout()
    plt.title(r"%s $%s$ %g" % (name, sign, tresh))
    if args.save:
        make_folder("figures")
        plt.savefig("figures/segment_%s_%s.png" % (mode[0], name))
    if args.show:
        fig2.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default=None, nargs='?')
    parser.add_argument('--t', '--treshold', type=float, default=None)
    parser.add_argument('--m', '--mode',     type=str,   default='geq')
    parser.add_argument('--sigma', '--std',  type=float, default=16)
    parser.add_argument('--img', action='store_true')  # show input image
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.name:
        segment(args.name, args.t, args.m)

    elif False:
        # If no input image was provided, do this instead:
        ## mosaic1-1,4
        segment("homo_mosaic1_d07_a070-1_w41", 0.27, "l")
        ## mosaic1-2
        segment("iner_mosaic1_d13_a000-4_w41", 2.4, "leq")
        ## mosaic1-3
        segment("homo_mosaic1_d07_a070-1_w41", 0.37, "geq")
        ## mosaic2-2
        segment("homo_mosaic2_d21_a000-1_w41", 0.40, "g")
        # ## mosaic2-1
        segment("iner_mosaic2_d06_a090-1_w41", 25, "geq")
        ## mosaic2-3
        segment("homo_mosaic2_d06_a090-1_w41", 0.40, "geq")

    if args.show or args.img:
        raw_input("<enter> to exit")
