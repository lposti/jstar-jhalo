from math import pi
from numpy import genfromtxt, power, log10, array, searchsorted, arange, interp, linspace,\
    asarray, log, exp, histogram2d, float32, percentile, sqrt, logspace, median, std, mean, append,\
    cumsum, sort, argmin
from numpy import abs as npabs
from numpy.random import normal, uniform, choice
from warnings import simplefilter, catch_warnings
import matplotlib.pylab as plt
from matplotlib.cm import get_cmap

class Mstar(object):

    def __init__(self, z, mode='high', model="L12", no_scatter=False):
        if model == "B10":
            self.d = genfromtxt("rsf_msmh2.txt")
        elif model == "L12":
            self.d = genfromtxt("rsf_msmh3.txt")
        else:
            raise ValueError("model must be 'B10' or 'L12'.")
        self.z = z
        self.mode = mode

        self._index_search = arange(399) * 1000
        self._z_file = self.d[self._index_search, 3]
        self.no_scatter = no_scatter

    def get_index(self):
        n = searchsorted(self._z_file, self.z)
        return self._index_search[n] + arange(1000)

    def __call__(self, Mh):
        w = self.get_index()
        if self.mode == 'low':
            mstar, mhalo = power(10., self.d[w[0:len(w):20], 0]), power(10., self.d[w[0:len(w):20], 1])
        elif self.mode == 'high':
            mstar, mhalo = power(10., self.d[w, 0]), power(10., self.d[w, 1])
        else:
            raise ValueError()
        Ms_med = interp(Mh, mhalo, mstar)
        # adding systematic scatter 0.2dex
        if self.no_scatter:
            return Ms_med
        else:
            return 10. ** normal(log10(Ms_med), 0.25, 1)[0]

def jstarRF(mstar, disc=True):
    mstar = asarray(mstar)
    if disc:
        j0, alpha = 3.21, 0.68
    else:
        j0, alpha = 2.73, 0.6

    return 10.**(j0+alpha*(mstar-11.))

size=100

Ms = Mstar(0., mode='high', model='L12', no_scatter=True)

mhalo = logspace(10.75, 14.5, num=size)
mstar = []

for m in mhalo:
    mstar.append(Ms(m))
mstar = array(mstar)

# j_halo da Romanowsky & Fall (2012) eq. (14)
# assuming lambda_halo=0.035 with no scatter
j_halo = 4.23e4 * 0.035 * power(mhalo / 1e12, 2./3.)
j_halo_mstar = 4.23e4 * 0.035 * power(mstar / 1e12, 2./3.)

# mstar[0]/mhalo[0]~0.003

lmhalo, lmstar, ljhalo, ljhalo_mstar = log10(mhalo), log10(mstar), log10(j_halo), log10(j_halo_mstar)
ljstar_d, ljstar_e = log10(jstarRF(lmstar, disc=True)), log10(jstarRF(lmstar, disc=False))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(211)
fig.subplots_adjust(hspace=0.)
ax.plot(lmstar, lmhalo, 'k-', lw=3)
ax.set_ylabel(r"$\rm \log M_h \,\,\, [M_\odot]$", fontsize=16)
ax.set_xticklabels([])
ax.set_ylim([9.75,14.5])

ax2 = fig.add_subplot(212)
ax2.plot(lmstar, ljhalo_mstar, 'k--', label=r"$\rm DM\,\,haloes$")
ax2.plot(lmstar, ljhalo, 'm-', lw=3, label=r"$\rm model\,\,galaxies$")
ax2.plot(lmstar, ljstar_d, 'b-', lw=3, label=r"$\rm obs.\,\,discs$")
ax2.plot(lmstar, ljstar_e, 'r-', lw=3, label=r"$\rm obs.\,\,bulges$")
ax2.set_xlabel(r"$\rm \log \,M_\ast\quad or\quad \log \,M_h \,\,\,[M_\odot]$", fontsize=16)
ax2.set_ylabel(r"$\rm \log \,j_\ast\quad or\quad \log \,j_h \,\,\,[kpc\,km\,s^{-1}]$", fontsize=16)
ax2.set_ylim([0,5.25])
ax2.legend(loc='best', frameon=False)

# arrows
dict_arrowstyle = dict(arrowstyle='<->', lw=2)
ax2.annotate("", xy=(lmstar[0], ljstar_d[0]), xytext=(lmstar[0],ljhalo[0]),
                     arrowprops=dict_arrowstyle)
ax2.annotate("", xy=(lmstar[-1], ljstar_e[-1]), xytext=(lmstar[-1],ljhalo[-1]),
             arrowprops=dict_arrowstyle)
ax2.text(8.25,1.8,r"$f_j$", fontsize=16)
ax2.text(11.6,3.9,r"$f_j$", fontsize=16)

plt.show()
