#!/home/polaris/miniconda3/envs/labastro/bin/python3.10
# -*- coding: utf-8 -*-
from __future__ import division, print_function

__doc__ = """Create a fake, random-position catalogue for testing the false association rate.

For each source, tries to find a new position, by choosing randomly one of its K
nearest neighbors and picking a random location between them. If the new
location is within --radius (arcsec) of an old or new source, the random process
is repeated. With 2/3 probability K=10, with 1/3 probability K=100, giving a
good balance between reproducing local structures and filling the field.

Example: nway-create-fake-catalogue.py --radius 20 COSMOS-XMM.fits fake-COSMOS-XMM.fits
"""

import argparse
import sys

import astropy.io.fits as pyfits
import healpy
import matplotlib.pyplot as plt
import numpy
from numpy import (arccos, arcsin, arctan, arctan2, cos, exp, log10,
                   logical_and, pi, sin, sqrt, tan)

import nwaylib.fastskymatch as match
import nwaylib.progress as progress


class HelpfulParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


parser = HelpfulParser(description=__doc__,
    epilog="""Johannes Buchner (C) 2013-2025 <johannes.buchner.acad@gmx.com>""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--radius', type=float, required=True,
    help='Remove sources which are near original sources, within this radius (arcsec).')

parser.add_argument('inputfile', type=str, help="input catalogue fits file")
parser.add_argument('outputfile', help='output catalogue fits file')

# parsing arguments
args = parser.parse_args()

outfile = args.outputfile
filename = args.inputfile
radius = args.radius

print('opening', filename)
inputfitsfile = pyfits.open(filename)
header_hdu = inputfitsfile[0]
table = inputfitsfile[1].data

ra_key = match.get_tablekeys(table, 'RA')
print('    using RA  column: %s' % ra_key)
dec_key = match.get_tablekeys(table, 'DEC')
print('    using DEC column: %s' % dec_key)

ra_orig = table[ra_key]
dec_orig = table[dec_key]
ra = ra_orig + 0
dec = dec_orig + 0
n = len(ra_orig)

i_select = numpy.random.randint(0, n, size=400)
ra_test = ra[i_select]
dec_test = dec[i_select]
phi_test = ra_test / 180 * pi
theta_test = dec_test / 180 * pi + pi / 2.
phi = ra / 180 * pi
theta = dec / 180 * pi + pi / 2.

# find a pixelation that has ~20 in each pixel
print('finding good pixelation...')
ntarget = 20

nside = 1
for nside_next in range(30):
    nside = 2 ** nside_next
    npix = healpy.pixelfunc.nside2npix(nside)
    if n > ntarget * npix:
        continue
    i = healpy.pixelfunc.ang2pix(nside, phi=phi_test, theta=theta_test, nest=True)
    k = healpy.pixelfunc.ang2pix(nside, phi=phi, theta=theta, nest=True)

    nneighbors = []
    for rai, deci, ii in zip(ra_test, dec_test, i):
        nneighbors.append((k == ii).sum())

    nneighbors_total = sum(nneighbors)
    print('  nside=%d: the %d test objects have a total of %d neighbors' % (nside, len(ra_test), nneighbors_total))
    if nneighbors_total < len(ra_test) * ntarget:
        print('    accepting.')
        break


def greatarc_interpolate(posa, posb, f):
    (a_ra, a_dec), (b_ra, b_dec) = posa, posb
    lon1 = a_ra / 180 * pi
    lat1 = a_dec / 180 * pi
    lon2 = b_ra / 180 * pi
    lat2 = b_dec / 180 * pi

    d = arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    A = sin((1 - f) * d) / sin(d)
    B = sin(f * d) / sin(d)
    x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
    y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
    z = A * sin(lat1) + B * sin(lat2)

    lat_f = arctan2(z, sqrt(x ** 2 + y ** 2))
    lon_f = arctan2(y, x)

    c_ra = lon_f * 180 / pi
    c_dec = lat_f * 180 / pi
    return c_ra, c_dec


# for each of them, create a new one without collision
pbar = progress.bar(ndigits=6)
for index in pbar(range(n)):
    a = index
    i = healpy.pixelfunc.ang2pix(nside, phi=phi[a], theta=theta[a], nest=True)
    j = healpy.pixelfunc.get_all_neighbours(nside, phi=phi[a], theta=theta[a], nest=True)
    neighbors = numpy.hstack((i, j))
    is_neighbor = (k.reshape((-1, 1)) == neighbors.reshape((1, -1))).any(axis=1)

    ra_nearby = ra_orig[is_neighbor]
    dec_nearby = dec_orig[is_neighbor]

    d = match.dist((ra_orig[a], dec_orig[a]), (ra_nearby, dec_nearby))
    b_nearest = numpy.argsort(d)
    dmask = d[b_nearest] * 60 * 60 > radius
    b_nearest = b_nearest[dmask]

    if len(b_nearest) == 0:
        print(f"[{index}] No neighbors found in HEALPix pixel — generating random offset point.")

        healpix_res_deg = healpy.nside2resol(nside, arcmin=False) * 180 / pi  # in degrees
        r_min = radius / 2.0 / 3600.0  # convert arcsec to degrees
        r_max = healpix_res_deg

        while True:
            r = numpy.random.uniform(r_min, r_max)
            theta_offset = numpy.random.uniform(0, 2 * pi)

            ra_i = ra_orig[a] + r * cos(theta_offset) / cos(dec_orig[a] * pi / 180.0)
            dec_i = dec_orig[a] + r * sin(theta_offset)

            ra_i = numpy.fmod(ra_i + 360, 360)
            dec_i = numpy.clip(dec_i, -90, 90)

            d = match.dist((ra_i, dec_i), (ra_nearby, dec_nearby))
            if (d * 60 * 60 < radius).any():
                continue

            d = match.dist((ra_i, dec_i), (ra[:index], dec[:index]))
            if (d * 60 * 60 < radius).any():
                continue

            ra[index] = ra_i
            dec[index] = dec_i
            break
    else:
        b_nearest = b_nearest[:100]
        while True:
            if numpy.random.randint(0, 3) == 0:
                b = b_nearest[numpy.random.randint(0, len(b_nearest))]
            else:
                b = b_nearest[numpy.random.randint(0, min(10, len(b_nearest)))]

            di = d[b]
            uexclude = radius / 60 / 60 / di
            u = numpy.random.uniform(uexclude, 1 - uexclude)
            ra_i, dec_i = greatarc_interpolate((ra_orig[a], dec_orig[a]), (ra_nearby[b], dec_nearby[b]), u)

            d = match.dist((ra_i, dec_i), (ra_nearby, dec_nearby))
            if (d * 60 * 60 < radius).any():
                continue

            d = match.dist((ra_i, dec_i), (ra[:index], dec[:index]))
            if (d * 60 * 60 < radius).any():
                continue

            ra[index] = numpy.fmod(ra_i + 360, 360)
            dec[index] = dec_i
            break

table[ra_key] = ra
table[dec_key] = dec

tbhdu = inputfitsfile[1]
print('writing "%s" (%d rows)' % (outfile, len(tbhdu.data)))

hdulist = pyfits.HDUList([header_hdu, tbhdu])
hdulist.writeto(outfile, **progress.kwargs_overwrite_true)
