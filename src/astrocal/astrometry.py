import os
import glob
import numpy as np
from astropy.io import fits
from astroquery.astrometry_net import AstrometryNet
from astropy.wcs import WCS
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from . import photometry

def plate_solve(hdu, sources=None, ast=None, parity=2):
    if ast is None: ast = AstrometryNet()
    if ('PXSCALE' in list(hdu.header)) or (('XPIXSZ' in list(hdu.header)) and('FOCALLEN' in list(hdu.header))):
        pxscale_est = photometry.pixel_scale(hdu).to(u.arcsec).value
    else:
        pxscale_est = None
    wcs_header = ast.solve_from_source_list(sources["xcentroid"], sources["ycentroid"], 
                                            hdu.data.shape[1], hdu.data.shape[0], 
                                            solve_timeout=60, crpix_center=True, parity=parity,
                                            scale_units='arcsecperpix', scale_type='ev', 
                                            scale_est=pxscale_est, scale_err=0.1)
    if wcs_header:
        wcs = WCS(wcs_header)
        return wcs
    else:
        print("\033[91mWARNING\033[00m: Plate-solving failed")

def save_astrometry(hdu, savename, wcs):
    hdu.header.update(wcs.to_header())
    hdu.writeto(savename, overwrite=True)

def astrometry_one(filename, savename, ast=None, mask=False, fwhm=10, threshold=7, parity=2, verbose=False):
    with fits.open(filename) as hdul:
        hdu = hdul[0].copy()
    if verbose: print("Finding sources")
    sources = photometry.find_sources(hdu, mask, fwhm, threshold)
    if verbose: print(f"Sources found : {len(sources)}")
    if verbose: print("Plate-solving")
    if ast is None: ast = AstrometryNet()
    wcs = plate_solve(hdu, sources, ast=ast, parity=parity)
    fwhm = photometry.fwhm_estimate(hdu, sources, angle=True)
    if verbose: print(f"FWHM : {fwhm.value:.2f}\"")
    pxscale = photometry.pixel_scale(hdu)
    if verbose: print(f"Pixel scale : {pxscale.value:.2f}\"/px")
    if verbose: print("Saving image with updated WCS")
    save_astrometry(hdu, savename, wcs)

def astrometry_all(generic_filename,
                   savedir,
                   overwrite=False,
                   mask_name=[],
                   mask=False,
                   every_X=1, 
                   first=None, 
                   last=None,
                   fwhm=8,
                   threshold=5,
                   parity=2,
                   verbose=False):
    os.makedirs(savedir, exist_ok=True)
    if verbose: print("Loading mask(s)")
    for name in mask_name:
        mask = mask | fits.open(name, memmap=False)[0].data.astype(bool)
    if verbose: print("Finding frames")
    last = None if last is None else last+1
    list_frames = glob.glob(generic_filename)[first:last:every_X]
    nb_frames = len(list_frames)
    ast = AstrometryNet()
    for i, frame in enumerate(list_frames):
        # Load frame
        if verbose: print("---------------------------")
        if verbose: print(f"Frame : {i+1:>{int(np.log10(nb_frames))+1}}/{nb_frames}")
        # Plate-solving
        frame_name = frame.replace("\\","/").split("/")[-1]
        save_name = frame_name if overwrite else frame_name.replace('.fits', '_wcs.fits')
        astrometry_one(frame, f"{savedir}/{save_name}", ast=ast, mask=mask, fwhm=fwhm, threshold=threshold, parity=parity, verbose=verbose)


def main():
    pass #TODO

if __name__=='__main__':
    main()
    