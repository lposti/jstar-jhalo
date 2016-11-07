__author__ = 'lposti'

import numpy as np

def vrot_vc_P12(vc):
  d = np.array([[40, 16],
                [45, 23],
                [50, 32],
                [55, 42],
                [60, 53],
                [70, 77],
                [80, 102],
                [90, 125],
                [100, 147],
                [120, 185],
                [140, 218],
                [160, 244],
                [180, 267],
                [200, 286],
                [220, 303],
                [240, 318],
                [260, 333],
                [300, 360],
                [340, 387],
                [380, 416],
                [420, 449]], dtype=np.float)

  vhalo, vrot = d[:, 0], d[:, 1]
  return np.interp(vc, vhalo, vrot / vhalo, left= vrot[0]/vhalo[0], right=vrot[-1]/vhalo[-1])

def vrot_vc_D10(vc):
    x = np.asarray(vc)
    y0, x0, alpha, beta, gamma = 316.4, 1.078, 0.009, -1.16, 6.993
    return y0 * (x/x0)**alpha * (0.5*0.5*(x/x0)**gamma)**((beta-alpha)/gamma)

if __name__ == '__main__':
  print (vrot_vc_P12(30.))
