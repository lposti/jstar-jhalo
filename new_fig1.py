from numpy import genfromtxt, power, log10, array, searchsorted, arange, interp, asarray, logspace, diff,linspace
from numpy.random import normal
import matplotlib.pylab as plt

dot_dash_seq=[2,4,7,4]
long_dash_seq=[20,4]
dotted_seq=[2,2]

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

def Mstar_D10_ltg(mstar):
    a, b = -0.5, 0
    x0, y0, g = 10.**10.4, 10.**1.61, 1.

    mh = []
    for m in mstar:
        y = 10.**normal(log10(y0 * (m/x0)**a * (0.5 + 0.5*(m/x0)**g)**((b-a)/g)), 0.00001, 1)[0]
        mh.append(m*y*0.7)
    return array(mh)

def Mstar_D10_etg(mstar):
    a, b = -0.15, 0.85
    x0, y0, g = 10.**10.8, 10.**1.97, 2.

    mh = []
    for m in mstar:
        y = 10.**normal(log10(y0 * (m/x0)**a * (0.5 + 0.5*(m/x0)**g)**((b-a)/g)), 0.00001, 1)[0]
        mh.append(m*y*0.7)
    return array(mh)

def jstarRF(mstar, disc=True):
    mstar = asarray(mstar)
    if disc:
        j0, alpha = 3.18, 0.52
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

# logarithmic slope
# plt.plot(log10(mstar)[:-1], diff(log10(mhalo))/diff(log10(mstar)), 'ko-')
# plt.show()

mstar_ltg=logspace(9.6,11.3,num=13)
mstar_etg=logspace(10.1,11.7,num=10)
mhalo_ltg = Mstar_D10_ltg(mstar_ltg)
mhalo_etg = Mstar_D10_etg(mstar_etg)

# j_halo da Romanowsky & Fall (2012) eq. (14)
# assuming lambda_halo=0.035 with no scatter
j_halo = 4.23e4 * 0.035 * power(mhalo / 1e12, 2./3.)
j_halo_mstar = 4.23e4 * 0.035 * power(mstar / 1e12, 2./3.)
j_halo_ltg = 4.23e4 * 0.035 * power(mhalo_ltg / 1e12, 2./3.)
j_halo_etg = 4.23e4 * 0.035 * power(mhalo_etg / 1e12, 2./3.)


# mstar[0]/mhalo[0]~0.003

lmhalo, lmstar, ljhalo, ljhalo_mstar = log10(mhalo), log10(mstar), log10(j_halo), log10(j_halo_mstar)
lmstar_obs = linspace(8.75, 11.5)
ljstar_d, ljstar_e = log10(jstarRF(lmstar_obs, disc=True)), log10(jstarRF(lmstar_obs, disc=False))

# read RF12 data
rf12_discs = genfromtxt('rf12_discs.dat')
rf12_discs_lmstar = rf12_discs[:, 14]
rf12_discs_ljstar = log10(rf12_discs[:, 13])

rf12_sph = genfromtxt('rf12_sph.dat', dtype=None)
rf12_sph_lmstar = rf12_sph['f10']
rf12_sph_ljstar = log10(rf12_sph['f9'] * 1.65)
rf12_sph_ljstar[rf12_sph['f2']==b'S0'] += log10(1.21)-log10(1.65)
''' do not plot S0 '''
rf12_sph_lmstar = rf12_sph_lmstar[rf12_sph['f2']!=b'S0']
rf12_sph_ljstar = rf12_sph_ljstar[rf12_sph['f2']!=b'S0']

two_plots=True
three_plots=not two_plots

if two_plots:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(211)
else:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(311)
    pl=ax.plot(lmstar, lmhalo, 'k', lw=1.5, label=r"$\rm obs.\,\,from\,\,RF12$")
    pl[0].set_dashes(dotted_seq)

fig.subplots_adjust(hspace=0.)
pl=ax.plot(log10(mstar_ltg), log10(mhalo_ltg), 'k', lw=2, label=r"$\rm Dutton+10$")
pl[0].set_dashes(long_dash_seq)
pl=ax.plot(log10(mstar_ltg), log10(mhalo_ltg), 'b', lw=2)
pl[0].set_dashes(long_dash_seq)
pl=ax.plot(log10(mstar_etg), log10(mhalo_etg), 'r', lw=2)
pl[0].set_dashes(long_dash_seq)
ax.plot(lmstar, lmhalo, 'k-', lw=3, label=r"$\rm Leauthaud+12$")
ax.set_ylabel(r"$\rm \log \,\,M_h/M_\odot$", fontsize=16)
ax.set_xticklabels([])
ax.set_ylim([10.25, 14.5])

if two_plots:
    ax.legend(loc='upper left', frameon=False, numpoints=1, fontsize=14)
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel(r"$\rm \log \,\,j_\ast/kpc\,km\,s^{-1}$", fontsize=16)
    ax2.set_xlabel(r"$\rm \log \,M_\ast/M_\odot$", fontsize=16)
    ax2.plot(rf12_discs_lmstar, rf12_discs_ljstar, 's', label=r"$\rm RF12\,\,spirals$",
             markeredgecolor='c', markerfacecolor='None', markeredgewidth=2)
    ax2.plot(rf12_sph_lmstar, rf12_sph_ljstar, 'o', alpha=1, label=r"$\rm RF12\,\,ellipticals$",
             markeredgecolor='m', markerfacecolor='None', markeredgewidth=2)
    pl=ax2.plot(log10(mstar_ltg), log10(0.5*j_halo_ltg), 'b', lw=2, alpha=1)
    pl[0].set_dashes(long_dash_seq)
    pl=ax2.plot(log10(mstar_etg), log10(0.1*j_halo_etg), 'r', lw=2, alpha=1)
    pl[0].set_dashes(long_dash_seq)
    ax2.plot(lmstar, log10(0.5*j_halo), 'b-', alpha=1, lw=2)
    ax2.plot(lmstar, log10(0.1*j_halo), 'r-', alpha=1, lw=2)
    # ax2.set_ylim([0.5, 5.75])
    ax2.legend(loc='upper left', frameon=False, numpoints=1, fontsize=14)
    ax2.text(8.15,3.5,r"${\rm model\,\,galaxies\,\,w.} \,\,j_\ast=f_j\, j_h\propto f_j\,[M_h(M_\ast)]^{2/3}$", fontsize=16)
else:
    ax.legend(loc='upper left', frameon=False, numpoints=1, fontsize=14)
    ax2 = fig.add_subplot(312)
    # ax2.plot(lmstar, ljhalo_mstar, 'k--', label=r"$\rm DM:\,\,j_h\propto M_h^{2/3}$")
    ax2.plot(lmstar, ljhalo, 'k-', lw=3)
    pl=ax2.plot(log10(mstar_ltg), log10(j_halo_ltg), 'b', lw=2)
    pl[0].set_dashes(long_dash_seq)
    pl=ax2.plot(log10(mstar_etg), log10(j_halo_etg), 'r', lw=2)
    pl[0].set_dashes(long_dash_seq)
    pl=ax2.plot(lmstar_obs, ljstar_d, 'b', lw=1.5, label=r"$\rm obs. \,\,LTG\,\,(RF12)$")
    pl[0].set_dashes(dotted_seq)
    pl=ax2.plot(lmstar_obs, ljstar_e, 'r', lw=1.5, label=r"$\rm obs. \,\,ETG\,\,(RF12)$")
    pl[0].set_dashes(dotted_seq)
    ax2.set_xlabel(r"$\rm \log \,M_\ast/M_\odot$", fontsize=16)
    ax2.set_ylabel(r"$\rm \log \,\,j_\ast/kpc\,km\,s^{-1}$", fontsize=16)
    ax2.set_yticks(arange(1.5,5,0.5))
    ax2.set_xticklabels([])
    ax2.text(8.15,4.25,r"${\rm\bf Angular\,\,momentum\,\,conservation}:\,\,f_j=1$", fontsize=16)
    ax2.text(8.15,3.5,r"${\rm model\,\,galaxies\,\,w.} \,\,j_\ast=j_h\propto [M_h(M_\ast)]^{2/3}$", fontsize=16)

# ax2.plot(rf12_discs_lmstar[w_d], rf12_discs_ljstar[w_d], 'bo')
# ax2.plot(rf12_sph_lmstar[w_s], rf12_sph_ljstar[w_s], 'sr')

if three_plots:
    ax3 = fig.add_subplot(313)
    pl=ax3.plot(lmstar_obs, ljstar_d, 'b', lw=1.5)
    pl[0].set_dashes(dotted_seq)
    pl=ax3.plot(lmstar_obs, ljstar_e, 'r', lw=1.5)
    pl[0].set_dashes(dotted_seq)
    pl=ax3.plot(log10(mstar_ltg), log10(0.5*j_halo_ltg), 'b', lw=2, alpha=0.5)
    pl[0].set_dashes(long_dash_seq)
    pl=ax3.plot(log10(mstar_etg), log10(0.1*j_halo_etg), 'r', lw=2, alpha=0.5)
    pl[0].set_dashes(long_dash_seq)
    ax3.plot(lmstar, log10(0.5*j_halo), 'b-', alpha=0.5, lw=2)
    ax3.plot(lmstar, log10(0.1*j_halo), 'r-', alpha=0.5, lw=2)
    ax3.set_xlabel(r"$\rm \log \,M_\ast/M_\odot$", fontsize=16)
    ax3.set_ylabel(r"$\rm \log \,\,j_\ast/kpc\,km\,s^{-1}$", fontsize=16)

    ax3.text(8.15,4.25,r"${\rm \bf Constant\,\,angular\,\,momentum\,\,losses}:\,\,f_j=0.5\,{\rm(LTG),}\,\,f_j=0.1\,{\rm(ETG)}$",
        fontsize=16)
    ax3.text(8.15,3.5,r"${\rm model\,\,galaxies\,\,w.} \,\,j_\ast=f_j\, j_h\propto f_j\,[M_h(M_\ast)]^{2/3}$", fontsize=16)

plt.savefig("SHMR_RF12.pdf", bbox_inches='tight')
plt.show()
