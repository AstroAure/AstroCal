import os
import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from astropy.nddata import Cutout2D
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales, pixel_to_skycoord
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.visualization import ImageNormalize, MinMaxInterval, ZScaleInterval, LogStretch
from astroquery.vizier import Vizier
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder
from sklearn import linear_model

def pixel_scale(hdu):
    if proj_plane_pixel_scales(WCS(hdu.header))[0] != 1:
        pxscale = ((proj_plane_pixel_scales(WCS(hdu.header))[0]*u.deg).to(u.arcsec)) # Correct WCS
    else:
        pxscale = ((hdu.header['XPIXSZ']*u.um)/(hdu.header['FOCALLEN']*u.mm)).to(u.arcsec, equivalencies=u.dimensionless_angles()) # NINA
    hdu.header['PXSCALE'] = pxscale.value
    return pxscale

def find_sources(hdu, mask=False, fwhm=None, threshold=7, plot=False):
    if fwhm is None: fwhm = hdu.header['FWHM'] if 'FWHM' in hdu.header else 10
    # Masking bad edges
    mask_edges = np.ones(hdu.data.shape, dtype=bool)
    ny, nx = mask_edges.shape
    mask_edges[int(0.01*ny):int(0.99*ny),int(0.01*nx):int(0.99*nx)] = False
    mask = mask | mask_edges
    # Initializing the DAO Star Finder
    bkg_mean, bkg_median, bkg_sigma = sigma_clipped_stats(hdu.data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*bkg_sigma, min_separation=fwhm)
    # Search sources in the frame
    frame = hdu.data
    sources = daofind(frame-bkg_median, mask=mask)
    sources.sort("flux")
    sources.reverse()
    # Add RA/DEC (if WCS present in header)
    try:
        is_wcs = True
        wcs = WCS(hdu.header)
        sources_radec = pixel_to_skycoord(sources['xcentroid'], sources['ycentroid'], wcs)
        sources.add_columns([sources_radec.ra, sources_radec.dec], indexes=[1,1], names=['ra', 'dec'])
    except:
        is_wcs = False
        pass
    # Plot
    if plot:
        if is_wcs:
            fig, ax = plt.subplots(1,1,figsize=(12,12), subplot_kw={'projection':WCS(hdu.header)})
        else:
            fig, ax = plt.subplots(1,1,figsize=(12,12))
        norm = ImageNormalize(hdu.data, interval=ZScaleInterval())
        ax.imshow(hdu.data, cmap='gray', origin='lower', norm=norm)
        if is_wcs:
            ax.scatter(sources['ra'], sources['dec'], transform=ax.get_transform('world'), s=50, edgecolor='yellow', facecolor='none', alpha=0.3)
        else:
            ax.scatter(sources['xcentroid'], sources['ycentroid'], s=50, edgecolor='yellow', facecolor='none', alpha=0.3)
        plt.show()
    return sources

def fwhm_estimate(hdu, sources, n_src=50, src_siz=51, angle=False, plot=False, verbose=False):
    if n_src == -1: n_src = np.inf
    n_src = min(len(sources), n_src)
    X_SUM = []
    src_selection = np.random.choice(sources, n_src, replace=False)
    for src_sel in src_selection:
        src_pos = (src_sel['xcentroid'], src_sel['ycentroid'])
        # Cutout around source
        src_cut = Cutout2D(hdu.data, position=src_pos, size=src_siz)
        # Source profile
        x_arr = np.array([x for x in range(src_siz)])
        x_sum = np.sum(src_cut.data,axis=1) 
        x_sum = x_sum - sigma_clipped_stats(x_sum)[1]
        X_SUM.append(x_sum/np.max(x_sum))
    # Define 1D Gaussian function
    def gaus(x,a,x0,sigma):
        return (a/np.sqrt(2*np.pi*sigma*sigma))*np.exp(-(x-x0)**2/(2*sigma**2))
    _, x_med, _ = sigma_clipped_stats(X_SUM, axis=0)
    param, _ = curve_fit(gaus, x_arr, x_med, p0=[1,src_siz/2.,5])
    fwhm = np.abs(param[2])*gaussian_sigma_to_fwhm
    # Plot profiles and model
    if plot:
        fig, ax = plt.subplots(figsize=(6,4))
        for x_sum in X_SUM:
            ax.plot(x_arr, x_sum, c='gray', alpha=0.2)
        ax.plot(x_arr, x_med, c='k')
        ax.plot(x_arr, gaus(x_arr, param[0], param[1], param[2]), c='red', ls=':', label=f"FWHM : {fwhm:.2f}px")
        ax.set_ylim(-0.5, 1.2)
        ax.legend()
        plt.show()
    if angle:
        pxscale = pixel_scale(hdu)
        fwhm = (fwhm*pxscale).to(u.arcsec)
        if verbose: print(f"FWHM : {fwhm:.2f}\"")
        hdu.header['FWHM'] = fwhm.value
    else:
        if verbose: print(f"FWHM : {fwhm:.2f}px")
        hdu.header['FWHM'] = fwhm
    return fwhm

def relative_aperture_photometry(hdu, sources, fwhm=None, mask=False):
    if fwhm is None:
        fwhm = hdu.header['FWHM'] if 'FWHM' in hdu.header else fwhm_estimate(hdu, sources)
    pxscale = hdu.header['PXSCALE']*u.arcsec if 'PXSCALE' in hdu.header else pixel_scale(hdu)
    if type(fwhm)==u.Quantity: fwhm = (fwhm.to(u.deg)/((pxscale).to(u.deg))).value
    # Defining aperture radiuses
    aperture_radius = 1.5 * fwhm
    annulus_radius= [aperture_radius*2,aperture_radius*3]
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=aperture_radius)
    # Calculating aperture photometry
    phot_table = aperture_photometry(hdu.data, apertures, mask=mask)
    # Define annuli
    annulus_aperture = CircularAnnulus(positions, 
                                    r_in=annulus_radius[0],
                                    r_out=annulus_radius[1])
    annulus_masks = annulus_aperture.to_mask(method='center')
    # For each source, compute the median (through sigma/clipping)
    bkg_median_arr = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(hdu.data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median_arr.append(median_sigclip)
    # Store background stat in phot_table
    bkg_median_arr = np.array(bkg_median_arr)
    phot_table['annulus_median'] = bkg_median_arr
    phot_table['aper_bkg'] = bkg_median_arr * apertures.area
    phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['aper_bkg']
    # Estimating noise and SNR
    phot_table['noise'] = np.sqrt(phot_table['aper_sum_bkgsub'] +  # photon noise: source
                                phot_table['aper_bkg'])         # photon noise: sky
    phot_table['SNR'] = phot_table['aper_sum_bkgsub'] / phot_table['noise']
    # Compute instrumental magnitude
    exptime = hdu.header['EXPTIME']
    ins_mag = -2.5*np.log10(phot_table['aper_sum_bkgsub']/exptime)
    ins_err = ins_mag - -2.5*np.log10((phot_table['aper_sum_bkgsub']+phot_table['noise'])/exptime)
    phot_table['ins_mag'] = ins_mag
    phot_table['ins_err'] = ins_err
    try: 
        phot_table.add_columns([sources['ra'], sources['dec']], indexes=[1,1], names=['ra', 'dec'])
    except:
        pass
    return phot_table, apertures, annulus_aperture

def query_catalog(hdu, catalog='II/349/ps1'):
    # 'II/349/ps1' : SDSS
    wcs = WCS(hdu.header)
    center = SkyCoord(wcs.wcs.crval[0], wcs.wcs.crval[1], unit=u.deg).to_string()
    pxscale = hdu.header['PXSCALE']*u.arcsec if 'PXSCALE' in hdu.header else pixel_scale(hdu)
    pxscale = pxscale
    width = hdu.header['NAXIS1']*pxscale
    height = hdu.header['NAXIS2']*pxscale
    result = Vizier(row_limit=-1).query_region(center, width=width, height=height, catalog=catalog)
    sdss_sources = result[0].copy()
    return sdss_sources

def cross_match(hdu, apertures, 
                phot_table, cat_table,
                max_sep=3*u.arcsec,
                radec_phot=['ra','dec'], radec_cat=['RAJ2000','DEJ2000'],
                mag_col='gmag', emag_col='e_gmag'):
    # Match sources
    wcs = WCS(hdu.header)
    coord_apertures = apertures.to_sky(wcs).positions
    coord_catalog = SkyCoord(ra=cat_table[radec_cat[0]],
                            dec=cat_table[radec_cat[1]])
    xm_id, xm_ang_distance, _ = coord_apertures.match_to_catalog_sky(coord_catalog, nthneighbor=1)
    # Seeing restriction for matches
    sep_constraint = xm_ang_distance < max_sep
    # fig, ax = plt.subplots(figsize=(6,4))
    # ax.hist(xm_ang_distance, bins=20)
    # plt.show()
    # Record the RA/Dec of apertures
    phot_table[radec_phot[0]] = coord_apertures.ra.value
    phot_table[radec_phot[1]] = coord_apertures.dec.value
    # Record catalog magnitude
    cat_mag = cat_table[mag_col][xm_id[sep_constraint]]
    cat_err = cat_table[emag_col][xm_id[sep_constraint]]
    matched_phot_table = phot_table[sep_constraint]
    matched_phot_table["cat_mag"] = cat_mag.value
    matched_phot_table["cat_err"] = cat_err.value
    return matched_phot_table

def ZP_ransac(table, mag_bounds=(10,18), plot=False, verbose=False):
    # Selection from magnitude range
    mag_min, mag_max = mag_bounds
    cat_mag = table['cat_mag'].data
    ins_mag = table['ins_mag'].data
    cond = (cat_mag>mag_min) & (cat_mag<mag_max) & (~np.isnan(cat_mag)) & (~np.isnan(ins_mag))
    # Create two mock arrays for linear regression
    X = ins_mag[cond].reshape(-1, 1)
    Y = cat_mag[cond].reshape(-1, 1)
    # Sigma clipping pour choisir le threshold
    MAD = median_abs_deviation(X-Y, axis=None)
    # RANSAC linear regressions
    ransac = linear_model.RANSACRegressor(residual_threshold=3*MAD)
    ransac.fit(X, Y)
    slope = ransac.estimator_.coef_[0][0]
    ZP = ransac.estimator_.intercept_[0]
    if verbose: print(f"ZP : {ZP:.2f}")
    if abs(1-ransac.estimator_.coef_[0][0]) > 0.1:
        print("\033[91mWARNING\033[00m: Zero Point estimate may be wrong !")
        print(f"\033[91mWARNING\033[00m: Slope = {slope:.3f}")
        print(f"\033[91mWARNING\033[00m: ZP = {ZP:.2f}")
    if plot:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(table['ins_mag'], table['cat_mag'], marker='+', c='k')
        min_max = np.array([np.min(table['ins_mag']), np.max(table['ins_mag'])])
        ax.plot(min_max, ransac.predict(min_max.reshape(-1,1)).flatten(), color='r', ls='--')
        ax.set_xlabel('$mag_{ins}$')
        ax.set_ylabel('$mag_{cat}$')
        ax.set_aspect('equal')
        plt.show()
    return ransac

def calibrate_sources(table, ransac):
    table['app_mag'] = np.nan
    table['app_mag'][~np.isnan(table['ins_mag'])] = ransac.predict(table['ins_mag'][~np.isnan(table['ins_mag'])].reshape(-1,1)).flatten()
    return table

def save_sources(table, savename):
    ascii.write(table, savename, format='csv', overwrite=True)

def photometry_one(filename, 
                   savename,
                   mask=False, 
                   detect_fwhm=None, detect_threshold=7,
                   fwhm_n_src=-1,
                   calib_sources=None, calib_mag_col='gmag', calib_emag_col='e_gmag',
                   ransac_mag_bounds=(10,18),
                   plot=False, verbose=False):
    hdu = fits.open(filename)[0]
    hdu.data = ma.array(hdu.data, mask=mask)
    wcs = WCS(hdu.header)
    if verbose: print("Finding sources")
    sources = find_sources(hdu, mask=mask, fwhm=detect_fwhm, threshold=detect_threshold, plot=plot)
    fwhm = fwhm_estimate(hdu, sources, n_src=fwhm_n_src, angle=False, plot=plot, verbose=verbose)
    if verbose: print("Aperture photometry")
    phot_table, apertures, annulus_apertures = relative_aperture_photometry(hdu, sources, fwhm, mask=mask)
    if calib_sources is None: 
        if verbose: print("Querying catalog")
        calib_sources = query_catalog(hdu)
    if verbose: print("Cross-matching catalogs")
    matched_phot_table = cross_match(hdu, apertures, phot_table, calib_sources, 
                                     max_sep=hdu.header['FWHM']*hdu.header['PXSCALE']*u.arcsec, 
                                     mag_col=calib_mag_col, emag_col=calib_emag_col)
    ransac = ZP_ransac(matched_phot_table, mag_bounds=ransac_mag_bounds, plot=plot, verbose=verbose)
    hdu.header['ZP'] = ransac.estimator_.intercept_[0]
    if verbose: print("Calibrating sources")
    phot_table = calibrate_sources(phot_table, ransac)
    if verbose: print("Saving table")
    save_sources(phot_table, savename)

def photometry_all(generic_filename,
                   savedir,
                   mask_name=[],
                   every_X=1, 
                   first=None, 
                   last=None,
                   detect_fwhm=None, detect_threshold=7,
                   fwhm_n_src=-1,
                   calib_sources=None, calib_mag_col='gmag', calib_emag_col='e_gmag',
                   ransac_mag_bounds=(10,18),
                   plot=False, verbose=False):
    os.makedirs(savedir, exist_ok=True)
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
        # Photometry
        frame_name = frame.replace("\\","/").split("/")[-1]
        save_name = frame_name.replace('.fits', '_phot.csv')
        if i%100 == 0:
            if verbose: print("Querying catalog")
            calib_sources = query_catalog(fits.open(frame)[0])
        photometry_one(frame, f"{savedir}/{save_name}", mask=mask, 
                       detect_fwhm=detect_fwhm, detect_threshold=detect_threshold, fwhm_n_src=fwhm_n_src,
                       calib_sources=calib_sources, calib_mag_col=calib_mag_col, calib_emag_col=calib_emag_col,
                       ransac_mag_bounds=ransac_mag_bounds,
                       plot=plot, verbose=verbose)