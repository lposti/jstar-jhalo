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

  vhalo, vrot = d[:,0], d[:, 1]
  return np.interp(vc, vhalo, vrot / vhalo, left= vrot[0]/vhalo[0], right=vrot[-1]/vhalo[-1])

if __name__ == '__main__':
  print (vrot_vc_P12(30.))
