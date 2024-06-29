import os
import argparse
import glob
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time

def make_master_dark(generic_filename, 
                     gen_hot_px=False,
                     threshold=3,
                     clean_bad_px=False,
                     mask=False,
                     verbose=False):
    # Store frames in nparray
    if verbose : print("Loading dark frames")
    list_frames = [img.replace('\\','/') for img in glob.glob(generic_filename)]
    hdu = fits.open(list_frames[0], memmap=False)[0]
    size_x  = hdu.header['NAXIS1']
    size_y  = hdu.header['NAXIS2']
    nb_frames = len(list_frames)
    frames = ma.zeros((size_y, size_x, nb_frames), dtype=np.single)
    for i in range(nb_frames):
        hdu = fits.open(list_frames[i], memmap=False)[0]
        frames[:,:,i] = ma.array(hdu.data, mask=mask)
        if verbose: print(f"Frame : {i+1}/{nb_frames}")
    # Master frame generation
    if verbose: print("Generating master dark")
    master = ma.median(frames, axis=2, overwrite_input=True)
    # Hot pixel detection
    if gen_hot_px:
        if verbose: print("Detecting hot pixels")
        bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(master, sigma=3.0)
        hot_px = ma.where(master > bkg_median+threshold*bkg_sigma)
        mask_hot = np.full((size_y, size_x), False)
        mask_hot[hot_px] = True
        if verbose: print( f'Number of pixels in the frame  : {len(master.flatten()):8d}')
        if verbose: print( f'Number of hot pixels           : {len(master[hot_px]):8d}')
        if verbose: print( f'Fraction of hot pixels (%)     : {100*len(master[hot_px])/len(master.flatten()):.2f}')
    # Bad pixel removal
    if clean_bad_px:
        if verbose: print("Removing bad pixels")
        master[mask | mask_hot] = ma.median(master)
    if gen_hot_px:
        return master, mask_hot
    return master

def make_master_flat(generic_filename,
                     master_dark=0, 
                     gen_dead_px=False,
                     threshold=0.5,
                     clean_bad_px=False,
                     mask=False, 
                     verbose=False):
    # Store frames in nparray
    if verbose : print("Loading flat frames")
    list_frames = [img.replace('\\','/') for img in glob.glob(generic_filename)]
    hdu = fits.open(list_frames[0], memmap=False)[0]
    size_x  = hdu.header['NAXIS1']
    size_y  = hdu.header['NAXIS2']
    nb_frames = len(list_frames)
    frames = ma.zeros((size_y, size_x, nb_frames), dtype=np.single)
    if verbose: print("Flat reduction and normalization")
    for i in range(nb_frames):
        hdu = fits.open(list_frames[i], memmap=False)[0]
        frames[:,:,i] = ma.array(hdu.data, mask=mask)
        if verbose: print(f"Frame : {i+1}/{nb_frames}")
        flat_minus_dark = frames[:,:,i] - master_dark
        frames[:,:,i] = flat_minus_dark / ma.median(flat_minus_dark)
    # Master frame generation
    if verbose: print("Generating master flat")
    master = ma.median(frames, axis=2, overwrite_input=True)
    # Dead pixel detection
    if gen_dead_px:
        if verbose: print("Detecting dead pixels")
        dead_px = ma.where(master<threshold)
        mask_dead = np.full((size_y, size_x), False)
        mask_dead[dead_px] = True
        if verbose: print( f'Number of pixels in the frame  : {len(master.flatten()):8d}')
        if verbose: print( f'Number of dead pixels          : {len(master[dead_px]):8d}')
        if verbose: print( f'Fraction of dead pixels (%)    : {100*len(master[dead_px])/len(master.flatten()):.2f}' )
    # Bad pixel removal
    if clean_bad_px:
        if verbose: print("Removing bad pixels")
        master[mask | mask_dead] = ma.median(master)
    # Normalization
    if verbose: print("Normalizing master flat")
    master = master/ma.median(master)
    if gen_dead_px:
        return master, mask_dead
    return master

def save_master(savedir,
                master, 
                sample_frame,
                out_name=None):
    os.makedirs(savedir, exist_ok=True)
    hdu_sample = fits.open(sample_frame)[0]
    hdu = fits.PrimaryHDU(data=master, header=hdu_sample.header)
    # hdu.data = master.astype(np.single)
    # hdu.update_header()
    if out_name is None:
        sample_name = sample_frame.split("/")[-1]
        out_name = f"MASTER_{sample_name}"
    hdu.writeto(f"{savedir}/{out_name}", overwrite=True)

def save_mask(savedir,
              mask,
              sample_frame,
              out_name=None):
    os.makedirs(savedir, exist_ok=True)
    hdu_sample = fits.open(sample_frame)[0]
    hdu = fits.PrimaryHDU(data=mask.astype(np.byte), header=hdu_sample.header)
    # hdu.data = mask.astype(np.byte)
    # hdu.update_header()
    if out_name is None:
        sample_name = sample_frame.split("/")[-1]
        out_name = f"MASK_{sample_name}"
    hdu.writeto(f'{savedir}/{out_name}', overwrite=True)

def process_master(dark_filename, flat_filename, dark_flat_filename=None,
                   savedir=None,
                   gen_hot=True, gen_dead=True,
                   savedir_dark=None, savedir_flat=None, savedir_mask=None,
                   master_dark_name=None, master_flat_name=None,
                   mask_hot_name=None, mask_dead_name=None,
                   clean=False, combine_mask=False, mask_name=None,
                   save_dark_flat=True, master_dark_flat_name=None,
                   verbose=False):
    if savedir is None: savedir = "/".join(dark_filename.split("/")[:-1])
    if savedir_dark is None: savedir_dark = f"{savedir}/DARK"
    if savedir_flat is None: savedir_flat = f"{savedir}/FLAT"
    if savedir_mask is None: savedir_mask = f"{savedir}/MASK"
    dark_sample = glob.glob(dark_filename)[0].replace('\\','/')
    if mask_hot_name is None: mask_hot_name = 'MASK-HOT_' + dark_sample.split("/")[-1]
    if mask_dead_name is None: mask_dead_name = 'MASK-DEAD_' + dark_sample.split("/")[-1]
    if mask_name is None: mask_name = 'MASK_' + dark_sample.split("/")[-1]

    if verbose: print("-Creating master dark")
    if gen_hot:
        master_dark, mask_hot = make_master_dark(dark_filename, gen_hot_px=True, clean_bad_px=clean, verbose=verbose)
    else:
        master_dark = make_master_dark(dark_filename, gen_hot_px=False, clean_bad_px=clean, verbose=verbose)
        mask_hot = False
    
    if verbose: print("-Saving master dark")
    save_master(savedir_dark, master_dark, dark_sample, out_name=master_dark_name)

    if dark_flat_filename is not None:
        if verbose: print("-Creating master dark for flats")
        master_dark_flat = make_master_dark(dark_flat_filename, gen_hot_px=False, mask=mask_hot, clean_bad_px=clean, verbose=verbose)
        if save_dark_flat:
            if verbose: print("-Saving master dark for flats")
            save_master(savedir_dark, master_dark_flat, glob.glob(dark_flat_filename)[0].replace('\\','/'), out_name=master_dark_flat_name)   
    else:
        master_dark_flat = 0

    if verbose: print("-Creating master flat")
    if gen_dead:
        master_flat, mask_dead = make_master_flat(flat_filename, master_dark_flat, gen_dead_px=True, clean_bad_px=clean, mask=mask_hot, verbose=verbose)
    else:
        master_flat = make_master_flat(flat_filename, master_dark_flat, gen_dead_px=False, clean_bad_px=clean, mask=mask_hot, verbose=verbose)
        mask_dead = False

    mask = mask_hot|mask_dead
    if verbose: print("-Saving master flat")
    save_master(savedir_flat, master_flat, glob.glob(flat_filename)[0].replace('\\','/'), out_name=master_flat_name)

    if verbose: print("-Saving masks")
    if combine_mask & (gen_dead|gen_hot):
        save_mask(savedir_mask, mask, dark_sample, out_name=mask_name)
        return
    if gen_hot:
        save_mask(savedir_mask, mask_hot, dark_sample, out_name=mask_hot_name)
    if gen_dead:
        save_mask(savedir_mask, mask_dead, dark_sample, out_name=mask_dead_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dark', '-d', help='generic filename for dark frames', type=str)
    parser.add_argument('--flat', '-f', help='generic filename for flat frames', type=str)
    parser.add_argument('--dark-flat', '-df', help='generic filename for dark frames for the flats', type=str, default=None)
    parser.add_argument('--out', '-o', help='folder to save master frames', type=str)
    parser.add_argument('--out-dark', '-od', help='folder to save master dark(s)', type=str, default=None)
    parser.add_argument('--master-dark-name', '-md', help='filename for the master dark', type=str, default=None)
    parser.add_argument('--out-flat', '-of', help='folder to save master flat', type=str, default=None)
    parser.add_argument('--master-flat-name', '-mf', help='filename for the master flat', type=str, default=None)
    parser.add_argument('--gen-hot', help='generate mask for hot pixels', type=bool, default=True)
    parser.add_argument('--mask-hot-name', '-mh', help='filename for the hot pixels mask', type=str, default=None)
    parser.add_argument('--gen-dead', help='generate mask for dead pixels', type=bool, default=True)
    parser.add_argument('--mask-dead-name', '-mk', help='filename for the dead pixels mask', type=str, default=None)
    parser.add_argument('--out-mask', '-om', help='folder to save masks', type=str, default=None)
    parser.add_argument('--clean', help='clean bad pixels in master frames', type=bool, default=False)
    parser.add_argument('--save-dark-flat', help='save master dark for flat frames', type=bool, default=True)
    parser.add_argument('--master-dark-flat-name', '-mdf', help='filename for the master dark for flats', type=str, default=None)
    parser.add_argument('--combine-mask', help='save only one dead pixels mask', type=bool, default=False)
    parser.add_argument('--mask-name', '-m', help='filename for the bad pixels mask', type=str, default=None)
    parser.add_argument('--verbose', '-v', type=bool, default=False)
    args = vars(parser.parse_args())

    for arg in args:
        if type(args[arg])==str:
            args[arg] = args[arg].replace('\\','/')
    print(args)

    process_master(dark_filename=args['dark'], flat_filename=args['flat'], dark_flat_filename=args['dark_flat'],
                   savedir=args['out'],
                   gen_hot=args['gen_hot'], gen_dead=args['gen_dead'],
                   savedir_dark=args['out_dark'], savedir_flat=args['out_flat'], savedir_mask=args['out_mask'],
                   master_dark_name=args['master_dark_name'], master_flat_name=args['master_flat_name'],
                   mask_hot_name=args['mask_hot_name'], mask_dead_name=args['mask_dead_name'],
                   clean=args['clean'], combine_mask=args['combine_mask'], mask_name=args['mask_name'],
                   save_dark_flat=args['save_dark_flat'], master_dark_flat_name=args['master_dark_flat_name'],
                   verbose=args['verbose'])


if __name__=='__main__':
    main()