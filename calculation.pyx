#cython: language_level=3
"""
Created on Tue August 27 09:42:57 2019
@coded by: yudhiprabowo
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned long[:,::1] calclut(double[:,::1] cdfinp, double[:,::1] cdfref, unsigned long[:,::1] lut, int band) nogil:
    cdef:
        int i, j, k, n
        double d, d0
    
    for i in range(band):
        n = 0
        for j in range(256):
            d = 1
            for k in range(n, 256):
                d0 = fabs(cdfinp[j, i] - cdfref[k, i])
                if(d0 > d):
                    break
                if(d0 < d):
                    d = d0; n = k
            lut[j, i] = n
    
    return lut

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned char[:,:,::1] histspec(unsigned char[:,:,::1] img, unsigned long[:,::1] lut, int row, int col, int band):
    cdef: 
        int i, j, k
    
    for i in range(row):
        for j in range(col):
            for k in range(band):
                img[i, j, k] = lut[img[i, j, k], k]
    
    return img

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef unsigned char[:,:,::1] histmatch(unsigned char[:,:,::1] iarr, unsigned char[:,:,::1] rarr,
                int irow, int icol, int rrow, int rcol, int band):
    cdef:
        int i ,j, k, num
        double pdfinp, pdfref
        double[:,::1] cdfinp, cdfref
        unsigned char[:,:,::1] oarr
        unsigned long[:,::1] ihist, rhist, lut
    
    ihist = np.zeros((256, band), dtype=np.uint32)
    rhist = np.zeros((256, band), dtype=np.uint32)
    lut = np.zeros((256, band), dtype=np.uint32)
    cdfinp = np.zeros((256, band), dtype=np.double)
    cdfref = np.zeros((256, band), dtype=np.double)
     
    num = 0
    for i in range(irow):
        for j in range(icol):
            num += 1
            for k in range(band):
                ihist[iarr[i, j, k], k] += 1
                rhist[rarr[i, j, k], k] += 1
    
    for i in range(band):
        for j in range(256):
            pdfinp = float(ihist[j, i]) / num
            pdfref = float(rhist[j, i]) / num
            if(j == 0):
                cdfinp[j, i] = pdfinp
                cdfref[j, i] = pdfref
            if(j > 0):
                cdfinp[j, i] = cdfinp[j-1, i] + pdfinp
                cdfref[j, i] = cdfref[j-1, i] + pdfref
    
    lut = calclut(cdfinp, cdfref, lut, band)
    oarr = histspec(iarr, lut, irow, icol, band)
    
    return oarr