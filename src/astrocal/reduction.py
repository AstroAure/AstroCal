import glob
import os
import argparse
import numpy as np
import numpy.ma as ma
from astropy.io import fits

def reduce_one(image_path,
               master_dark,
               master_flat=1,
               mask=False,
               verbose=False):
    # Science image reduction
    if verbose: print("Reducing science image")
    hdu = fits.open(image_path, memmap=False)[0]
    hdu.data = (ma.array(hdu.data, mask=mask)-master_dark) / master_flat
    hdu.data = hdu.data.astype(np.single)
    return hdu

def reduce(generic_filename, 
           savedir,
           master_dark_name,
           master_flat_name=None,
           mask_name=[],
           every_X=1, 
           first=None, 
           last=None,
           clean=False,
           verbose=False):
    os.makedirs(savedir, exist_ok=True)
    if verbose: print("Loading master dark")
    master_dark = fits.open(master_dark_name, memmap=False)[0].data
    if verbose: print("Loading master flat")
    if master_flat_name is None:
        master_flat = 1
    else:
        master_flat = fits.open(master_flat_name, memmap=False)[0].data
    if verbose: print("Loading mask(s)")
    mask = False
    for name in mask_name:
        mask = mask | fits.open(name, memmap=False)[0].data.astype(bool)
    if verbose: print("Finding frames")
    last = None if last is None else last+1
    list_frames = glob.glob(generic_filename)[first:last:every_X]
    nb_frames = len(list_frames)
    for i, frame in enumerate(list_frames):
        # Load frame
        if verbose: print("---------------------------")
        if verbose: print(f"Frame : {i+1:>{int(np.log10(nb_frames))+1}}/{nb_frames}")
        # Reduction
        if verbose: print("Reducing frame")
        hdu = reduce_one(frame, master_dark, master_flat, mask=mask, verbose=False)
        # Cleaning
        if clean:
            if verbose: print("Cleaning masked")
            print("\033[91mCleaning if not implemented for now\033[0m") #TODO
        # Save frame
        if verbose: print("Saving frame")
        frame_name = frame.replace("\\","/").split("/")[-1]
        hdu.writeto(f"{savedir}/{frame_name.replace('.fits', '_df.fits')}", overwrite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sci', '-s', help='generic filename for science frames', type=str)
    parser.add_argument('--first', help='index of first frame to reduce', type=int, default=0)
    parser.add_argument('--last', help='index of last frame to reduce', type=int, default=None)
    parser.add_argument('--step', help='step of frames in folder', type=int, default=1)
    parser.add_argument('--master-dark', '-d', help='filename of master dark', type=str)
    parser.add_argument('--master-flat', '-f', help='filename of master flat', type=str)
    parser.add_argument('--mask', '-m', help='filename of mask, can be passed multiple times', action='append', type=str, default=[])
    parser.add_argument('--out', '-o', help='folder to save reduced frames', type=str)
    parser.add_argument('--clean', help='clean bad pixels in reduced frames', type=bool, default=False)
    parser.add_argument('--verbose', '-v', type=bool, default=False)
    args = vars(parser.parse_args())

    for arg in args:
        if type(args[arg])==str:
            args[arg] = args[arg].replace('\\','/')

    reduce(generic_filename=args['sci'], 
           savedir=args['out'], 
           master_dark_name=args['master-dark'], 
           master_flat_name=args['master-flat'], 
           mask_name=args['mask'],
           every_X=args['step'], first=args['first'], last=args['last'],
           clean=args['clean'],
           verbose=args['verbose'])

if __name__=='__main__':
    main()