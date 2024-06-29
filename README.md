# Pipeline for light-curve

## Calibration

### Level 1
1. Create master files
    1. $m_d$ : Dark (darkflat)
    2. $m_d^f$ : Dark for flats
    3. $m_f$ : Flat
2. Create dead pixels maps:
    1. Hot pixels map from $m_d$ (mask, or set them to median)
    2. Dead pixels map from $m_f$ (mask, or set them to local gaussian)
3. Calibrate images : $s=\frac{s-m_d}{m_f-m_d^f}$

### Level 2
1. Clean $s$ with dead pixels maps (local gaussian *or mask*)
2. Find sources in $s$ (coarse) ([photutils](https://photutils.readthedocs.io/en/stable/detection.html) or [SExtractor](https://astromatic-wrapper.readthedocs.io/en/latest/))
3. Plate-solve and update WCS ([astroquery](https://astroquery.readthedocs.io/en/latest/astrometry_net/astrometry_net.html) or [SCAMP](https://astromatic-wrapper.readthedocs.io/en/latest/))
4. Calibrate photometry
    1. [Search stars in SDSS](https://astroquery.readthedocs.io/en/latest/api/astroquery.sdss.SDSSClass.html#astroquery.sdss.SDSSClass.query_region) (once for all frames)
    3. RANSAC fit to find ZP and slope
    4. Calibrate pixel values with RANSAC (to Jy)

### *Level 3*
1. [Clean cosmic rays](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/08-03-Cosmic-ray-removal.html) (local gaussian *or mask*)
2. [Remove sky](https://photutils.readthedocs.io/en/stable/background.html)

### *PSF*
1. Co-add images ([reproject](https://reproject.readthedocs.io/en/stable/mosaicking.html) or [SWarp](https://astromatic-wrapper.readthedocs.io/en/latest/))
2. Build PSF ([photutils](https://photutils.readthedocs.io/en/stable/epsf.html) or [PSFEx](https://astromatic-wrapper.readthedocs.io/en/latest/))

## Light-curve

1. For all $s$:
    1. Find sources in $s$ (fine) ([photutils](https://photutils.readthedocs.io/en/stable/detection.html) or [SExtractor](https://astromatic-wrapper.readthedocs.io/en/latest/))
    2. Aperture *and PSF photometry* ([photutils](https://photutils.readthedocs.io/en/stable/psf.html) or [SExtractor](https://astromatic-wrapper.readthedocs.io/en/latest/))
2. Match sources in different catalogs by DBSCAN clustering
3. Find moving sources
    1. Calculate dispersion of RA/DEC (MAD + threshold)
    2. Linear regression for RA/DEC
    3. Plot RA/DEC for moving sources
4. Find variable and stable sources
    1. Calculate dispersion of flux/mag (MAD + threshold)
    2. Plot flux/mag
5. Identify target: Match target RA/DEC to global catalog