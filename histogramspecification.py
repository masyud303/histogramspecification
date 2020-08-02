"""
Created on Tue August 27 09:42:57 2019
@coded by: yudhiprabowo
"""

import cv2
import numpy as np
import calculation as calc

inp = "C:\\WORK\\PYTHON\\input2.jpg"
ref = "C:\\WORK\\PYTHON\\reference2.jpg"
out = "C:\\WORK\\PYTHON\\output2.jpg"

iarr = cv2.imread(inp)
rarr = cv2.imread(ref)

irow, icol, iband = iarr.shape
rrow, rcol, rband = rarr.shape

oarr = np.asarray(calc.histmatch(iarr, rarr, irow, icol, rrow, rcol, 3))
cv2.imwrite(out, oarr)