__author__ = 'lposti'

from math import pi
from numpy import genfromtxt, power, log10, array, searchsorted, arange, interp, linspace,\
    asarray, log, exp, histogram2d, float32, percentile, sqrt, logspace, median, std, mean, append,\
    cumsum, sort, argmin, full_like, vstack
from numpy import abs as npabs
from numpy.random import normal, uniform, choice
from warnings import simplefilter, catch_warnings
import matplotlib.pylab as plt
from matplotlib.cm import get_cmap
import h5py
from vrot_vc import vrot_vc_P12
from hmf.sample import sample_mf
from hmf.fitting_functions import Tinker10
from mpl_toolkits.axes_grid1 import host_subplot
import corner


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


def median_bins(a, b, w, bin_size=11):
    bins = logspace(log10(a[w].min()), log10(a[w].max()), num=bin_size)
    a_bin = [0.5*(bins[i-1]+bins[i]) for i in range(1, len(bins))]
    b_bin = [median((b[w])[(a[w] <= bins[i]) & (a[w] > bins[i-1])]) for i in range(1, len(bins))]
    e_b_bin = [percentile((b[w])[(a[w] <= bins[i]) & (a[w] > bins[i-1])], 68.2) for i in range(1, len(bins))]

    return array(a_bin), array(b_bin), array(e_b_bin)


def bins_and_hist(a, b, w, size_hist, sigs, bin_size=11):
    bins = logspace(log10(a[w].min()), log10(a[w].max()), num=bin_size)
    a_bin = [0.5*(bins[i-1]+bins[i]) for i in range(1, len(bins))]
    b_bin = [median((b[w])[(a[w] <= bins[i]) & (a[w] > bins[i-1])]) for i in range(1, len(bins))]
    e_b_bin = [percentile((b[w])[(a[w] <= bins[i]) & (a[w] > bins[i-1])], 68.) for i in range(1, len(bins))]

    x_bin, y_bin = linspace(log10(a[w]).min(), log10(a[w]).max(), num=size_hist),\
                   linspace(log10(b[w]).min(), log10(b[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(a[w]), log10(b[w]), bins=(x_bin, y_bin))

    H_sorted, lev = sort(H.flatten())[::-1], []

    for i in range(len(sigs)):
        lev.append(H_sorted[argmin(npabs(cumsum(H_sorted)-H.sum()/100.*sigs[i]))])

    lev = array(lev)
    lev[lev < 1.5] = 1.

    x, y = linspace(log10(a[w]).min(), log10(a[w]).max(), num=H.shape[0]), \
           linspace(log10(b[w]).min(), log10(b[w]).max(), num=H.shape[1])

    return x, y, H, log10(lev[::-1]), a_bin, b_bin, e_b_bin


def Mstar_vU(mhalo, mode='def', no_scatter=False):

    if mode == 'def':
        # default
        # mh1, ms0, b1, b2 = 10.**12.06, 10.**11.16, 5.4, 0.15
        mh1, ms0, b1, b2 = 10.**12.5, 10.**11.15, 1.75, 0.25
    elif mode == 'TF':
        # stellar Tully-Fisher
        # mh1, ms0, b1, b2 = 10.**12.3, 10.**11.5, 2., 0.6
        mh1, ms0, b1, b2 = 10.**12.5, 10.**11.5, 1.75, 0.46
        # mh1, ms0, b1, b2 = 10.**12.15, 10.**11., 1.75, 0.6
    elif mode == 'FR':
        # Fall & Kravtsov relations
        mh1, ms0, b1, b2 = 10.**12.06, 10.**10.9, 1.25, 0.4
    else:
        raise ValueError("mode not recognized!")

    if no_scatter:
        s = 0.001
    else:
        # s = 0.14
        s = 0.2
    mhalo = asarray(mhalo)
    if mhalo[mhalo < 1e4] is []:
        mhalo = 10. ** mhalo

    ms = []
    for m in mhalo:
        ms.append(10. ** normal(log10(ms0 * power(m / mh1, b1) / power(1. + m / mh1, b1-b2)), s, 1)[0])

    return array(ms)


def Mstar_D10_ltg(mstar):
    a, b = -0.5, 0
    x0, y0, g = 10.**10.4, 10.**1.61, 1.

    mh = []
    for m in mstar:
        y = 10.**normal(log10(y0 * (m/x0)**a * (0.5 + 0.5*(m/x0)**g)**((b-a)/g)), 0.2, 1)[0]
        mh.append(m*y)
    return array(mh)


def Mstar_D10_etg(mstar):
    a, b = -0.15, 0.85
    x0, y0, g = 10.**10.8, 10.**1.97, 2.

    mh = []
    for m in mstar:
        y = 10.**normal(log10(y0 * (m/x0)**a * (0.5 + 0.5*(m/x0)**g)**((b-a)/g)), 0.2, 1)[0]
        mh.append(m*y)
    return array(mh)


def get_bt_distribution_none(mstar):
    """
        Returns -1 for all galaxies
    """
    return full_like(mstar, -1.)


def get_bt_distribution_from_SDSS(mstar):
    # Bulge-fraction distribution
    btf = genfromtxt('bt_distr_mstarOLD.txt')

    # for i in range(2, 9):
    #     btf[40:, i] /= 10.
    for i in range(2, 6):
        for k in range(40, 50):
            bth = btf[k, 0]
            btf[k, i] /= ((bth-0.8)/0.2 * 8.) + 1.
    for i in range(2, 10):
        for k in range(0, 10):
            bth = btf[k, 0]
            btf[k, i] *= (-(bth-0.2)/0.2 * 2.) + 1.

    bt = []
    for m in mstar:
        bt_bins = 0.5*(btf[:, 0]+btf[:, 1])
        if log10(m) <= 9.25:
            bt.append(choice(bt_bins, p=btf[:, 2]/sum(btf[:, 2]), size=1)[0])
        if (log10(m) > 9.25) & (log10(m) <= 9.5):
            bt.append(choice(bt_bins, p=btf[:, 3]/sum(btf[:, 3]), size=1)[0])
        if (log10(m) > 9.5) & (log10(m) <= 9.75):
            bt.append(choice(bt_bins, p=btf[:, 4]/sum(btf[:, 4]), size=1)[0])
        if (log10(m) > 9.75) & (log10(m) <= 10.):
            bt.append(choice(bt_bins, p=btf[:, 5]/sum(btf[:, 5]), size=1)[0])
        if (log10(m) > 10.) & (log10(m) <= 10.25):
            bt.append(choice(bt_bins, p=btf[:, 6]/sum(btf[:, 6]), size=1)[0])
        if (log10(m) > 10.25) & (log10(m) <= 10.5):
            bt.append(choice(bt_bins, p=btf[:, 7]/sum(btf[:, 7]), size=1)[0])
        if (log10(m) > 10.5) & (log10(m) <= 10.75):
            bt.append(choice(bt_bins, p=btf[:, 8]/sum(btf[:, 8]), size=1)[0])
        if (log10(m) > 10.75) & (log10(m) <= 11.):
            bt.append(choice(bt_bins, p=btf[:, 9]/sum(btf[:, 9]), size=1)[0])
        if log10(m) > 11.:
            bt.append(choice(bt_bins, p=btf[:, 10]/sum(btf[:, 10]), size=1)[0])

    return array(bt)


def get_bt_distribution_from_meert14(mstar):
    # Bulge-fraction distribution
    btf = genfromtxt('bt_distr_mstar_meert+14.dat')

    bt = []
    for m in mstar:
        bt_bins = 0.5*(btf[:, 0]+btf[:, 1])
        if log10(m) <= 9.5:
            bt.append(choice(bt_bins, p=btf[:, 2]/sum(btf[:, 2]), size=1)[0])
        if (log10(m) > 9.5) & (log10(m) <= 10.):
            bt.append(choice(bt_bins, p=btf[:, 3]/sum(btf[:, 3]), size=1)[0])
        if (log10(m) > 10.) & (log10(m) <= 10.5):
            bt.append(choice(bt_bins, p=btf[:, 4]/sum(btf[:, 4]), size=1)[0])
        if (log10(m) > 10.5) & (log10(m) <= 11.):
            bt.append(choice(bt_bins, p=btf[:, 5]/sum(btf[:, 5]), size=1)[0])
        if log10(m) > 11.:
            bt.append(choice(bt_bins, p=btf[:, 6]/sum(btf[:, 6]), size=1)[0])

    return array(bt)


def get_bt_distribution_from_lambda(mstar, mhalo, lambda_halo):
    # Bulge-fraction distribution
    from numpy import argsort
    btf = genfromtxt('bt_distr_mstar.txt')

    lowM  = [0., 9.25, 9.5, 9.75, 10., 10.25, 10.5, 10.75, 11.]
    highM = [9.25, 9.5, 9.75, 10., 10.25, 10.5, 10.75, 11., 20.]

    # sort the arrays as increasing mstar
    id_mstar = argsort(mstar)
    mstar, mhalo, lambda_halo = mstar[id_mstar], mhalo[id_mstar], lambda_halo[id_mstar]

    bt = []
    for i in range(len(lowM)):
        bt_bins = 0.5*(btf[:, 0]+btf[:, 1])

        idx = (log10(mstar) >= lowM[i]) & (log10(mstar) < highM[i])
        bt_distr_in_bin = choice(bt_bins, p=btf[:, 2+i]/sum(btf[:, 2+i]), size=len(mstar[idx]))

        # plt.hist(bt_distr_in_bin, 30)
        # plt.figure()
        # plt.hist(log10(lambda_halo[idx]), 20)
        # plt.show()

        # find ids that would sort lambda_halo distrib. in the mass bin
        id_lambda_in_bin = argsort(lambda_halo[idx])

        # sort the bt distribution and order it according to the lambda_halo ids
        bt_distr_in_bin[id_lambda_in_bin] = sort(bt_distr_in_bin)[::-1]

        for x in bt_distr_in_bin:
            bt.append(x)
        '''
        sort the three arrays and match the distributions of lambda and bt per mass bin
        or use a structured array and sort by one property
        '''

    return array(bt), mstar, mhalo, lambda_halo


def plot_angular_momentum_size_velocity():

    size = 50000
    # Planck parameters
    h, Om, OL = 0.677, 0.31, 0.69

    '''
    # uniform halo mass function
    mhalo = array(10. ** uniform(10.5, 13.5, size), dtype=float32)

    # read halo mass-function
    # mhf = genfromtxt('mVector_PLANCK-SMT_11-13.txt')
    # mhf = genfromtxt('mVector_PLANCK-SMT_10.5-13.txt')
    # mhf = genfromtxt('mVector_PLANCK-SMT_11-14.5.txt')
    # mhalo = array(choice(mhf[:, 0], p=mhf[:, 5]/sum(mhf[:, 5]), size=size), dtype=float32)

    # mhalo, _ = sample_mf(size, 11., Mmax=15.5, hmf_model=Tinker10)


    Ms = Mstar(0., mode='high', model='L12')
    mstar = []

    print("generating mstar...")
    for m in mhalo:
        mstar.append(Ms(m))
    mstar = array(mstar)

    # mstar = Mstar_vU(mhalo, mode='def')
    '''

    mstar = array(10. ** uniform(9.35, 11., size), dtype=float32)
    # mstar = array(10. ** uniform(9., 11., size), dtype=float32)
    mhalo = Mstar_D10_ltg(mstar)

    # ms_lowlim, ms_highlim = 7., 12.# 11.75
    # mhalo = mhalo[(log10(array(mstar)) > ms_lowlim) & (log10(array(mstar)) < ms_highlim)]
    # mstar = mstar[(log10(array(mstar)) > ms_lowlim) & (log10(array(mstar)) < ms_highlim)]

    # DM halo spin parameter distribution
    # from Maccio', Dutton & van den Bosch (2008)
    lambda_halo = 10. ** normal(-1.466, 0.253, len(mhalo))

    bt = get_bt_distribution_none(mstar)
    # bt = get_bt_distribution_from_SDSS(mstar)
    # bt = get_bt_distribution_from_meert14(mstar)
    # bt, mstar, mhalo, lambda_halo = get_bt_distribution_from_lambda(mstar, mhalo, lambda_halo)

    # Dutton & Maccio' (2014)
    # b = -0.101 + 0.026 * array([0., 1.])
    # a = 0.52 + (0.905 - 0.52) * exp(-0.617 * array([0., 1.]) ** 1.21)
    b = -0.097 + 0.024 * array([0., 1.])
    a = 0.537 + (1.025 - 0.537) * exp(-0.718 * array([0., 1.]) ** 1.08)
    # scatter 0.11dex
    cvir = []
    print ("generating concentrations...")
    for m in mhalo:
        cvir.append(10. ** normal(a[0] + b[0] * log10(m / (1e12 / h)), 0.11, 1)[0])
    cvir = array(cvir)

    # Circular velocity for NFW haloes
    G = 4.302e-3 * 1e-6  # Mpc Mo^-1 (km/s)^2
    f = lambda x: log(1.+asarray(x)) - asarray(x)/(1.+asarray(x))
    rho_c = 3. * (h * 100.)**2 / (8. * pi * G)
    # Bryan & Norman (1998)
    Oz = lambda z: Om * (1+z)**3 / (Om * (1.+z)**3+OL)
    Delta_c = lambda z: 18. * pi**2 + 82. * (Oz(z) - 1.) - 39. * (Oz(z) - 1.)**2
    rho_hat = 4. / 3. * pi * Delta_c(0.) * rho_c

    # 0.6 factor to account for different normalization in SHMR
    vc = sqrt(G * f(2.15) / f(cvir) * cvir / 2.15 * pow(rho_hat, 1./3.)) * power(mhalo / h, 1./3.)
    for i in range(len(vc)):
        vc[i] *= .6*vrot_vc_P12(vc[i])

    # print "%e %e" % (f(10.) * 2.15 / (f(2.15) * 10.), sqrt(2. * f(10.) * 2.15 / (f(2.15) * 10.)))

    # Kravtsov 2013
    # rho_200 = 4. / 3. * pi * 200. * rho_c
    r200 = 1e3 * power(mhalo / rho_hat, 1./3.)
    rs = []
    for r in r200:
        rs.append(10. ** normal(log10(0.015 * r), 0.25, 1)[0])
    rs = array(rs)
    js = rs * vc

    # jhalo - mhalo
    fe = cvir / 2. * (1.-1./power(1.+cvir, 2) - 2. * log(1.+cvir)/(1.+cvir)) / power(cvir/(1.+cvir)-log(1.+cvir), 2)
    # fe = f(2.15) / f(cvir) * cvir / 2.15
    j_halo = sqrt(2. / fe) * lambda_halo * sqrt(G * 1e3 * mhalo / r200) * r200
    # j_halo da Romanowsky & Fall (2012) eq. (14)
    # j_halo = 4.23e4 * lambda_halo * power(mhalo / 1e12, 2./3.)

    """
    ------------- Definition of B/T cuts
    """
    # bt_discs, bt_spirals, bt_ltg, bt_lents, bt_ells = 0.1, 0.2, 0.5, 0.8, 0.95
    bt_discs, bt_spirals, bt_ltg, bt_lents, bt_ells = 0.1, 0.25, 0.5, 0.65, 0.8
    # fj(Mhalo)
    def jstar_FR(mass, bf):
        if (bf >= 0.) & (bf <= bt_discs):
            # Sc
            j0, alpha, s = 3.29, 0.55, 0.18
        elif (bf > bt_discs) & (bf <= bt_spirals):
            # Sb
            j0, alpha, s = 3.21, 0.68, 0.15
        elif (bf > bt_spirals) & (bf <= bt_ltg):
            # Sa
            j0, alpha, s = 3.02, 0.64, 0.12
        elif (bf > bt_ltg) & (bf <= bt_lents):
            # S0
            j0, alpha, s = 3.05, 0.8, 0.22
        elif (bf > bt_lents) & (bf <= bt_ells):
            # fE
            j0, alpha, s = 2.875, 0.6, 0.2
        elif (bf > bt_ells) & (bf <= 1.):
            # sE
            j0, alpha, s = 2.73, 0.6, 0.2
        elif bf == -1:
            j0, alpha, s = 3. , 0.67, 0.22
        else:
            print (bf)
            raise ValueError("Problem in bt not in ]0,1]")

        return 10. ** normal(j0+alpha*(mass-11.), s, 1)[0]

    # fitting function to jb/jd estimated from Romanowsky & Fall (2012) data
    # jb/jd = 0.025 + B/T^2
    # jd = j_star / (1 + (jb/jd-1)B/T)
    # --> here jdisc = jstar / fb_func
    jb_over_jd_fit = lambda fb: 0.025 + fb**2
    fb_func = lambda fb: 1. + (jb_over_jd_fit(fb) - 1.) * fb

    jstar, jdisc, jbulge = [], [], []
    print ("generating jstar...")
    for i in range(len(mstar)):
        jstar.append(jstar_FR(log10(mstar[i]), bt[i]))
        jdisc.append(jstar[i] / fb_func(bt[i]))
        jbulge.append(jdisc[i] * jb_over_jd_fit(bt[i]))
    jstar, jdisc, jbulge = array(jstar), array(jdisc), array(jbulge)
    fj = jstar / j_halo  # jstar_FR(log10(0.158 * mhalo)) / j_halo

    # Tully Fisher
    Mstar_TF = lambda t: 10.**(-0.61+4.93*asarray(t))  # McGaugh & Schombert (2015), sTF @ 3.6um
    vstar_TF = lambda t: 10.**(0.61/4.93 + 1./4.93*asarray(t))
    # Mstar_TF = lambda t: 10.**(1.49+4.09*asarray(t))  # McGaugh & Schombert (2015), Baryonic-TF @ 3.6um
    # vstar_TF = lambda t: 10.**(-1.49/4.09 + 1./4.09*asarray(t))

    sstar_FJ = lambda t: 10.**(2.054 + 0.286 * asarray(t-10.))
    Mstar_FJ = lambda t: 10.**(-2.054/0.286 + 10. + 1./0.286 * asarray(t))

    # Shen et al. (2003)
    mass_size_etg = lambda mass: -5.54061 + 0.56*log10(mass)
    mass_size_ltg = lambda mass: -1. + 0.14*log10(mass) + 0.25 * log10(1. + mass/3.98e10)

    # Lange et al. (2015) (Tables 2-3, g-i colour cut, r-band Re)
    # mass_size_etg = lambda mass: log10(8.25e-5) + 0.44*log10(mass)
    # mass_size_ltg = lambda mass: log10(13.98e-3) + 0.25*log10(mass)
    # mass_size_ltg = lambda mass: -1. + 0.16*log10(mass) + 0.65 * log10(1. + mass/17.1e10)

    # Cappellari et al. 13, sigma from mass plane
    sigma_MP = lambda mass: 10.**(0.5*(log10(mass)-10.6-log10(10.**mass_size_etg(mass)/2.))+log10(130.))

    factor = 1.1  # V_circ(R_50) := factor * sigma(R_50)

    '''
    plt.figure()
    js[bt<bt_spirals] = 10.**mass_size_ltg(mstar[bt<bt_spirals]) * vstar_TF(log10(mstar[bt<bt_spirals]))
    js[bt>bt_ells] = 10.**mass_size_etg(mstar[bt>bt_ells]) * 0.33 * factor * sstar_FJ(log10(mstar[bt>bt_ells]))
    # js[bt>bt_ells] = 10.**mass_size_etg(mstar[bt>bt_ells]) * 0.33 * factor * sigma_MP(mstar[bt>bt_ells])
    plt.plot(log10(mstar[bt<bt_spirals]), log10(js[bt<bt_spirals]), 'b.', alpha=0.1)
    plt.plot(log10(mstar[bt>bt_ells]), log10(js[bt>bt_ells]), 'r.', alpha=0.1)
    ms_rf12_discs_str = "11.06 9.84 9.23 10.46 11.31 11.50 11.34 10.13 11.42 10.62 10.61 10.74 11.03 11.31 10.93 11.32 10.01 11.34 10.67 11.33 10.81 10.83 10.65 11.31 11.23 11.10 10.32 10.74 11.13 10.95 11.12 10.61 8.62 10.49 11.37 10.39 9.86 11.07 10.22 9.43 11.08 11.42 10.43 10.03 11.56 9.58 10.73 10.96 10.76 10.54 11.10 11.31 11.11 11.50 10.84 11.07 11.34 10.47 11.23 11.34 10.86 10.44 11.51 9.14 11.74 10.82 11.26"
    jt_rf12_discs_str = "2230 2290 770 790 190 260 480 550 1480 1930 4270 4280 2070 2230 340 360 1810 2900 630 820 1360 1450 990 1090 1210 1720 2580 3370 2230 2280 3150 3380 470 500 2220 2300 580 760 1110 1300 1250 1430 1520 1620 1030 1210 1400 2020 2070 2190 1070 1180 460 480 900 1000 2100 2380 920 1070 1680 2190 550 600 120 120 990 1040 4250 4470 1320 1350 160 180 930 1070 550 610 580 610 2240 2410 1060 1850 370 400 300 320 2030 2380 130 140 1220 1460 790 850 180 360 260 330 1650 1740 2440 2620 1280 1590 1430 1560 1300 1650 590 1030 1450 1580 570 580 2050 2150 2540 2780 710 830 960 1040 2360 2890 70 190 7560 8380 2010 2090 3140 3350"
    ms_rf12_ell_str = "10.97 11.94 10.52 11.13 9.79 10.68 11.34 10.75 11.05 11.26 11.66 10.18 9.97 10.35 10.79 10.76 10.55 11.33 10.26 11.04 10.24 10.95 10.50 10.96 10.09"
    jt_rf12_ell_str = "1270 3640 400 240 25 70 2330 1040 3100 630 680 7 130 680 1000 610 240 3360 270 1150 240 1700 160 210 110"
    ms_dvdb12_str = "9.484 9.631 9.784 9.941 10.103 10.270 10.442 10.618 10.796 10.979 11.164 11.352"
    js_dvdb12_str = "2.417 2.482 2.550 2.621 2.697 2.779 2.868 2.960 3.050 3.154 3.261 3.391"
    ms_rf12_discs = [float(s) for s in ms_rf12_discs_str.split()]
    jt_rf12_discs = [float(s) for s in jt_rf12_discs_str.split()]
    ms_rf12_ell = [float(s) for s in ms_rf12_ell_str.split()]
    jt_rf12_ell = [float(s) for s in jt_rf12_ell_str.split()]
    ms_dvdb12 = [float(s) for s in ms_dvdb12_str.split()]
    js_dvdb12 = [float(s) for s in js_dvdb12_str.split()]
    plt.plot(ms_rf12_discs, log10(jt_rf12_discs[::2]), 'bs')
    plt.plot(ms_rf12_ell, log10(jt_rf12_ell), 'rs')
    plt.plot(ms_dvdb12, js_dvdb12, 'k^', markersize=20)
    plt.plot(linspace(9,12), 3.18+0.52*(linspace(9,12)-11.),'k--')
    plt.xlabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\,j_\ast/km\,s^{-1}\,kpc$", fontsize=16)
    # plt.savefig('invEx_Mstar-jstar_wRF12.png', bbox_inches='tight')

    fig,ax = plt.subplots()
    mhalo_bin, fj_bin, e_fj_bin = median_bins(log10(mhalo), log10(js / j_halo), bt<bt_spirals, 19)
    plt.errorbar(mhalo_bin, fj_bin, yerr=e_fj_bin-fj_bin, fmt='o-', color='b')
    # plt.plot(log10(mhalo[bt<bt_spirals]), log10(js/j_halo)[bt<bt_spirals], 'b.', alpha=0.1)
    mhalo_bin, fj_bin, e_fj_bin = median_bins(log10(mhalo), log10(js / j_halo), bt>bt_ells, 17)
    plt.errorbar(mhalo_bin, fj_bin, yerr=e_fj_bin-fj_bin, fmt='o-', color='r')
    # plt.plot(log10(mhalo[bt>bt_ells]), log10(js/j_halo)[bt>bt_ells], 'r.', alpha=0.1)
    # mhalo_bin, fj_bin, e_fj_bin = median_bins(log10(mhalo), log10(js / j_halo), bt>0, 19)
    # plt.errorbar(mhalo_bin, fj_bin, yerr=e_fj_bin-fj_bin, fmt='--', color='k')
    ax.set_xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    ax.set_ylabel(r"$\log\rm\,f_j(M_h)\equiv j_\ast / j_h$", fontsize=16)
    # plt.savefig('invEx_Mhalo-fj.png', bbox_inches='tight')
    plt.show()
    print (j[2])
    '''

    # Re = jdisc / (vstar_TF(log10(mstar)))
    # Re[bt > 0.8] = jbulge[bt > 0.8] / (0.7 * vstar_TF(log10(mstar[bt > 0.8])))
    # Re[bt > 0.8] = jbulge[bt > 0.8] / (10.**(-1.06 + 0.32*log10(mstar[bt > 0.8])))
    Re = jstar / vc  # vstar_TF(log10(mstar))
    Re[bt > bt_ells] = jstar[bt > bt_ells] / (vc[bt > bt_ells] / factor)  # sstar_FJ(log10(mstar[bt > bt_ltg])))

    # Mo, Mao & White (1998)
    # lambda_halo_prime = lambda_halo * jdisc / j_halo  # fj
    lambda_halo_prime = lambda_halo * fj  # fj
    cvir = array(cvir)
    fac_sis_nfw = 1. / sqrt(cvir / 2. * (1.-1./power(1.+cvir, 2) - 2. * log(1. + cvir) / (1. + cvir)) /
                            power(cvir / (1. + cvir) - log(1. + cvir), 2))
    # md = mstar / mhalo
    # fac_sis_nfw *= pow(lambda_halo_prime / 0.1, -0.0 + 2.71 * md + 0.0047 / lambda_halo_prime) * \
    #                (1-3.*md+5.2*md*md) * (1-0.019*cvir * 0.00025*cvir*cvir + 0.52 / cvir)
    Rd = 1. / sqrt(2.) * lambda_halo_prime * r200 * fac_sis_nfw

    # Compute dark matter fractions assuming NFW profile and that Re=stellar half mass radius
    rs = r200 / cvir
    rho_0 = mhalo / rs**3 / 4. / pi / (log(1.+cvir)-cvir/(1.+cvir))
    mh_re = 4. * pi * rho_0 * rs**3 * (log((rs+Re)/rs) - Re / (Re+rs))
    fDM = mh_re / (mstar / 2. + mh_re)

    w = bt < bt_spirals
    Ms_nos = Mstar(0., mode='high', model='L12', no_scatter=True)
    ms_scatter = (log10(mstar) - log10([Ms_nos(mh) for mh in mhalo])) # / log10([Ms_nos(mh) for mh in mhalo])
    x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, w, 20, [68., 90.], bin_size=30)
    fj_scatter = (log10(array(fj)) - interp(log10(mhalo), log10(mhalo_bin), log10(fj_bin))) # / \
    #                     interp(log10(mhalo), log10(mhalo_bin), log10(fj_bin))

    '''
    corner_data = vstack([log10(mstar)[w], log10(mstar/mhalo)[w], log10(fj)[w], log10(fDM)[w], log10(Re)[w], log10(vc)[w],
                          log10(jstar)[w], log10(j_halo)[w], ms_scatter[w], fj_scatter[w]]).T
    print(corner_data.shape)
    corner.corner(corner_data, labels=[r"$\rm\log\,M_\ast/M_\odot$", r"$\rm\log\,M_\ast/M_h$",
                                       r"$\rm\log\,f_j$", r"$\rm \log\,f_{DM}$", r"$\rm\log\,R_e/kpc$",
                                       r"$\rm\log\,v_c/km/s$",
                                       r"$\rm \log\,j_\ast$", r"$\rm \log\,j_h$",
                                       r"$\sigma_{\log M_\ast-\log M_h}$",
                                       r"$\sigma_{\log f_j-\log M_h}$"],
                  quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12},
                  range=[.99, .99, .99, .99, .99, .99, .99, .99, .95, .95],
                  color='b')
    plt.savefig("corner_plot.pdf", bbox_inches='tight')
    print (j[2])
    '''

    """
    HDF5 file output
    """
    f = h5py.File("jstar-jhalo.hdf5", 'w')

    group = f.create_group("columns")

    dset_lambda = group.create_dataset("Lambda_Halo", data=lambda_halo)
    dset_mhalo = group.create_dataset("Halo_Mass", data=mhalo)
    dset_mstar = group.create_dataset("Galaxy_Mass", data=mstar)
    dset_jhalo = group.create_dataset("Halo_Specific_Angular_Momentum", data=j_halo)
    dset_jstar = group.create_dataset("Galaxy_Specific_Angular_Momentum", data=jstar)
    dset_cvir = group.create_dataset("Halo_Concentration", data=cvir)
    dset_fj = group.create_dataset("Retained_fraction_of_j", data=fj)
    dset_bt = group.create_dataset("Bulge_Fraction", data=bt)
    dset_vc = group.create_dataset("Circular_Velocity", data=vc)
    dset_vc = group.create_dataset("Velocity_Dispersion", data=vc/1.1)
    dset_Re = group.create_dataset("Effective_Radius", data=Re)
    dset_Rd = group.create_dataset("Disc_Scale_Radius", data=Rd)


    """
    Plots
    """

    # plt.plot(log10(mstar * bt), log10(jbulge), 'r.', alpha=0.005)
    # plt.plot(log10(mstar * bt * (1./bt - 1.)), log10(jdisc), 'b.', alpha=0.005)
    # plt.plot(linspace(9, 12), 3.28+0.67*(linspace(9, 12)-11.), 'k-', lw=3)
    # plt.plot(linspace(9, 12), 2.75+0.67*(linspace(9, 12)-11.), 'k--', lw=3)
    # plt.show()

    size_hist = 20
    sigs = [68., 90.]
    save_plots = False

    '''
    ----------- Mstar-Mhalo
    '''
    # Dutton et al. 2010 relations
    d10_msmh_ltg = lambda x: 10.**1.61 * (asarray(x) / 10.**10.4)**-0.5 * (0.5 + 0.5 * (asarray(x) / 10.**10.4))**0.5
    d10_msmh_etg = lambda x: 10.**1.97 * (asarray(x) / 10.**10.8)**-0.15 * (0.5 + 0.5 * (asarray(x) / 10.**10.8)**2)**0.5
    d10_ms_ltg = logspace(9.35, 11)
    d10_ms_etg = logspace(9.85, 11.4)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_ylabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    # ax.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylim([8, 11.75])
    ax.set_xlim([10.75, 13.5])

    if len(bt[bt<0])>0:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo/h, mstar/h**2, bt<0, 30, sigs)
    else:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo/h, mstar/h**2, bt > 0, 30, sigs)
    # plt.plot(log10(d10_msmh_ltg(d10_ms_ltg) * d10_ms_ltg), log10(d10_ms_ltg), 'b-')
    # plt.plot(log10(d10_msmh_etg(d10_ms_etg) * d10_ms_etg), log10(d10_ms_etg), 'r-')
    plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)
    plt.xlim()

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.set_ylabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    ax2.set_xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    ax2.set_xlim([10.75, 13.5])
    fig.subplots_adjust(hspace=0)

    if len(bt[bt<0])>0:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo, mstar/mhalo, bt<0, 30, sigs)
    else:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo, mstar/mhalo, bt > 0, 30, sigs)
    plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)

    if save_plots:
        plt.savefig('SHMR_all_ref.pdf', bbox_inches='tight')

    '''
    ----------- lambda-Mhalo
    '''
    fig = plt.figure()
    plt.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\lambda_{halo}$", fontsize=16)
    plt.plot(log10(mhalo), log10(lambda_halo), 'b.', alpha=0.0025)
    if len(bt[bt<0])>0:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo, lambda_halo, bt<0,  size_hist, sigs)
    else:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo, lambda_halo, bt > 0,  size_hist, sigs)
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')


    '''
    ----------- j_halo - M_halo
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\,j_h/km\,s^{-1}\,kpc$", fontsize=16)
    if len(bt[bt<0])>0:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo, j_halo, bt<0, size_hist, sigs)
    else:
        x, y, H, lev, _, _, _ = bins_and_hist(mhalo, j_halo, bt > 0, size_hist, sigs)
    # ax.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
    ax.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)

    ax.plot(linspace(10.75, 13), 3.28+0.67*(linspace(10.75, 13)-11.), 'k--', lw=3)
    # plt.plot([12.5, log10(0.158 * 10.**12.5)], [3.505, 3.7481], 'm-', lw=3)
    # ax.arrow(12.5, 3.505, -12.5+log10(0.08 * 10.**12.5), -3.505+3.28+0.67*(log10(0.08 * 10.**12.5)-11.)-0.045,
    #          head_width=0.075, head_length=0.05, fc='m', ec='m', lw=3)
    ax.arrow(12., 3.1704, -12.+log10(0.08 * 10.**12.), -3.1704+3.28+0.67*(log10(0.08 * 10.**12.)-11.)-0.045,
             head_width=0.075, head_length=0.05, fc='m', ec='m', lw=3)

    ax.text(11.15, 4., r"$f_{\rm baryon}\simeq 0.16$"+"\n"+r"$M_\ast=M_{\rm bar}/2$", color='m', fontsize=18)

    if save_plots:
        plt.savefig('jh-Mh_all_ref.pdf', bbox_inches='tight')

    '''
    ----------- fj(M_halo)
    '''
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax = host_subplot(111)
    ax2 = ax.twiny()

    ax.set_xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    ax.set_ylabel(r"$\log\rm\,f_j(M_h)\equiv j_\ast / j_h$", fontsize=16)
    ax2.set_xlabel(r"$\log\,M_\ast/M_\odot$", fontsize=16)

    if len(bt[bt<0])>0:
        x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, bt<0, size_hist, sigs)
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)

        # ------ Median relation
        x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, bt<0, size_hist, sigs, bin_size=15)
        plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='k', fmt='o')

        ax.set_xlim(10.5,13.5)
        ax2.set_xlim(10.5,13.5)
        mhx = logspace(10.5, 13.5, num=100)
        # Ax2 tick location in stellar mass
        mstar_tick_loc = arange(8.0, 11.5, 0.5)
    else:
        ax.text(0.1, 0.9, r"$\rm late-types$", fontsize=18, color='#151AB0', transform=ax.transAxes)
        # ax.text(0.1, 0.85, r"$\rm Sa-S0$", fontsize=18, color='#009603', transform=ax.transAxes)
        ax.text(0.1, 0.825, r"$\rm early-types$", fontsize=18, color='#BD000D', transform=ax.transAxes)

        # spirals
        w = bt < bt_spirals
        # plt.plot(log10(mhalo[w]), log10(fj[w]), 'b.', alpha=0.0025)
        x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, w, size_hist, sigs)
        # plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='b', fmt='o')
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
        ax.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Blues'), alpha=0.75)

        '''
        # Sa and lenticulars
        w = (bt >= bt_spirals) & (bt < bt_ells)
        # plt.plot(log10(mhalo[w]), log10(fj[w]), 'b.', alpha=0.0025)
        x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, w, size_hist, sigs)
        # plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='g', fmt='o')
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#AB9700')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greens'), alpha=0.75)
        '''

        # ellipticals
        w = bt > bt_ells
        # plt.plot(log10(mhalo[w]), log10(fj[w]), 'r.', alpha=0.0025)
        x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, w, size_hist, sigs)
        # plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='r', fmt='o')
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#BD000D')
        ax.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Reds'), alpha=0.75)

        # ------ Median relation
        # x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, bt > 0, size_hist, sigs, bin_size=30)
        x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, (bt < bt_spirals), size_hist, sigs, bin_size=30)
        # plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='b', fmt='o')
        ax.plot(log10(mhalo_bin)[:8], log10(fj_bin)[:8], 'bo-')
        x, y, H, lev, mhalo_bin, fj_bin, e_fj_bin = bins_and_hist(mhalo, fj, (bt > bt_ells), size_hist, sigs, bin_size=30)
        # plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='r', fmt='s')
        ax.plot(log10(mhalo_bin)[2:], log10(fj_bin)[2:], 'rs-')

        # second x-axis with Mstar
        ax.set_xlim(11,13.5)
        ax2.set_xlim(11,13.5)
        mhx = logspace(11, 13.5, num=100)
        # Ax2 tick location in stellar mass
        mstar_tick_loc = arange(9, 11.5, 0.5)

    Ms = Mstar(0., mode='high', model='L12', no_scatter=True)
    ms_mhx = array([Ms(m) for m in mhx])

    # Ax2 string tick definition
    ax2_tick_labels = ["%2.1f" % (m) for m in mstar_tick_loc]

    ax2_tick_loc = []
    for msx in mstar_tick_loc:
        ax2_tick_loc.append(log10(mhx[argmin(npabs( log10(ms_mhx) - msx))]))

    ax2.set_xticks(ax2_tick_loc)
    ax2.set_xticklabels(ax2_tick_labels)

    if save_plots:
        plt.savefig('fj-Mh_BT_ref.pdf', bbox_inches='tight')
    # plt.savefig('fj-Mh_BT_ref.pdf', bbox_inches='tight')

    '''
    ----------- Kravtsov
    '''
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,R_e/kpc$", fontsize=16)
    plt.xlabel(r"$\log\rm\,r_{\rm vir}/kpc$", fontsize=16)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm late-types$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    # ax.text(0.1, 0.85, r"$\rm Sa-S0$", fontsize=18, color='#009603', transform=ax.transAxes)
    ax.text(0.1, 0.825, r"$\rm early-types$", fontsize=18, color='#BD000D', transform=ax.transAxes)
    # ax.text(0.4, 0.15, r"$\rm Kravtsov Plot$", fontsize=20, transform=ax.transAxes)

    if len(bt[bt<0])>0:
        x, y, H, lev, _, _, _ = bins_and_hist(r200, Re, bt<0, size_hist, sigs)
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)

        # ------ Median relation
        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Re, bt<0, size_hist, sigs, bin_size=30)
        plt.errorbar(log10(r200_bin), log10(Rd_bin), yerr=abs(log10(Rd_bin)-log10(e_Rd_bin)), c='k', fmt='o')
    else:
        # spirals
        w = bt < bt_spirals
        # plt.plot(log10(r200)[w], log10(Re)[w], 'b.', alpha=0.0075)
        x, y, H, lev, _, _, _ = bins_and_hist(r200, Re, w, size_hist, sigs)
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Blues'), alpha=0.75)

        '''
        # Sa and lenticulars
        w = (bt >= bt_spirals) & (bt < bt_ells)
        # plt.plot(log10(r200)[w], log10(Re)[w], 'b.', alpha=0.0075)
        x, y, H, lev, _, _, _ = bins_and_hist(r200, Re, w, size_hist, sigs)
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#AB9700')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greens'), alpha=0.75)
        '''

        # ellipticals
        w = bt > bt_ells
        # plt.plot(log10(r200)[w], log10(Re)[w], 'r.', alpha=0.0075)
        x, y, H, lev, _, _, _ = bins_and_hist(r200, Re, w, size_hist, sigs)
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#BD000D')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Reds'), alpha=0.75)

        # ------ Median relation
        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Re, (bt < bt_spirals), size_hist, sigs, bin_size=30)
        # plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='b', fmt='o')
        ax.plot(log10(r200_bin)[:8], log10(Rd_bin)[:8], 'bo-')
        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Re, (bt > bt_ells), size_hist, sigs, bin_size=30)
        # plt.errorbar(log10(mhalo_bin), log10(fj_bin), yerr=abs(log10(fj_bin)-log10(e_fj_bin)), c='r', fmt='s')
        ax.plot(log10(r200_bin)[2:], log10(Rd_bin)[2:], 'rs-')

    plt.plot(log10(linspace(125, 800)), log10(0.015 * linspace(125, 800)), 'k-', lw=2)
    plt.plot(log10(linspace(125, 800)), log10(0.015 * linspace(125, 800))+0.5, 'k--', lw=2)
    plt.plot(log10(linspace(125, 800)), log10(0.015 * linspace(125, 800))-0.5, 'k--', lw=2)
    plt.xlim([log10(125.), log10(800.)])

    if save_plots:
        plt.savefig('Re-r200_BT_ref.pdf', bbox_inches='tight')

    '''
    ----------- Mo, Mao & White Rd-r200
    '''
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,R_d/kpc$", fontsize=16)
    plt.xlabel(r"$\log\rm\,r_{\rm vir}/kpc$", fontsize=16)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm Sc-Sb$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    ax.text(0.1, 0.85, r"$\rm Sa-S0$", fontsize=18, color='#009603', transform=ax.transAxes)
    # ax.text(0.4, 0.15, r"$\rm Mo, Mao & White Plot$", fontsize=20, transform=ax.transAxes)

    if len(bt[bt<0])>0:
        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Rd, bt<0, size_hist, sigs)
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)

        # ------ Median relation
        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Rd, bt<0, size_hist, sigs, bin_size=30)
        plt.errorbar(log10(r200_bin), log10(Rd_bin), yerr=abs(log10(Rd_bin)-log10(e_Rd_bin)), c='k', fmt='o')
    else:
        # spirals
        w = bt < bt_spirals
        # plt.plot(log10(r200[w]), log10(Rd[w]), 'b.', alpha=0.01)
        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Rd, w, size_hist, sigs)
        # plt.errorbar(log10(r200_bin), log10(Rd_bin), yerr=abs(log10(Rd_bin)-log10(e_Rd_bin)), c='b', fmt='o')
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Blues'), alpha=0.75)


        # Sa and lenticulars
        w = (bt >= bt_spirals) & (bt < bt_lents)
        # plt.plot(log10(r200[w]), log10(Rd[w]), 'y.', alpha=0.01)
        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Rd, w, size_hist, sigs)
        # plt.errorbar(log10(r200_bin), log10(Rd_bin), yerr=abs(log10(Rd_bin)-log10(e_Rd_bin)), c='g', fmt='o')
        # plt.contour(x, y, log10(H).T, levels=lev, colors='#AB9700')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greens'), alpha=0.75)

        x, y, H, lev, r200_bin, Rd_bin, e_Rd_bin = bins_and_hist(r200, Rd, bt < bt_lents, size_hist, sigs, bin_size=30)
        plt.errorbar(log10(r200_bin), log10(Rd_bin), yerr=abs(log10(Rd_bin)-log10(e_Rd_bin)), c='k', fmt='o')

    plt.plot(log10(linspace(110, 800)), log10(0.0112*linspace(110, 800)), 'k-', lw=2)
    plt.xlim([log10(110.), log10(800.)])

    if save_plots:
        plt.savefig('Rd-r200_ScS0_ref.pdf', bbox_inches='tight')

    '''
    ----------- sTFR
    '''
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    plt.xlabel(r"$\log\rm\,V_{rot}/km\,s^{-1}$", fontsize=16)
    plt.ylim(8, 11.5)
    plt.xlim(1.8, 2.5)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm late-types$", fontsize=18, color='#151AB0', transform=ax.transAxes)

    if len(bt[bt<0])>0:
        x, y, H, lev, vc_bin, mstar_bin, e_mstar_bin = bins_and_hist(vc, mstar, bt<0, size_hist, sigs)
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)

    else:
        # spirals
        w = bt < bt_spirals
        # plt.plot(log10(vc[w]), log10(mstar[w]), 'b.', alpha=0.0025)
        x, y, H, lev, vc_bin, mstar_bin, e_mstar_bin = bins_and_hist(vc, mstar, w, size_hist, sigs)
        # plt.errorbar(log10(vc_bin), log10(mstar_bin), yerr=abs(log10(mstar_bin)-log10(e_mstar_bin)), c='b', fmt='o')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Blues'))

    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4))), 'k-', lw=3)
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4)))+0.15, 'k--', lw=3)
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4)))-0.15, 'k--', lw=3)

    if save_plots:
        plt.savefig('sTF_BT_ref.pdf', bbox_inches='tight')


    if len(bt[bt<0])>0:
        pass

    else:
        '''
        ----------- Faber-Jackson
        '''
        fig = plt.figure()
        plt.ylabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
        plt.xlabel(r"$\log\rm\,\sigma/km\,s^{-1}$", fontsize=16)
        plt.ylim(9, 11.75)
        plt.xlim(1.8, 2.6)
        ax = fig.add_subplot(111)
        ax.text(0.1, 0.9, r"$\rm early-types$", fontsize=18, color='#BD000D', transform=ax.transAxes)

        # ellipticals
        w = bt > bt_ells
        # turn vc to sigma for ellipticals
        vc[w] /= factor
        # Dutton et al. 2010 correction
        # (vc[w])[log10(mstar[w]) > 10.5] *= 10.**(0.1-0.3*(log10(mstar[w])[log10(mstar[w]) > 10.5]-10.5))
        # (vc[w])[log10(mstar[w]) <= 10.5] *= 10.**0.1
        # plt.plot(log10(mstar[w]), log10(vc[w]), 'r.', alpha=0.0025)
        x, y, H, lev, vc_bin, mstar_bin, e_mstar_bin = bins_and_hist(vc, mstar, w, size_hist, sigs)
        # plt.errorbar(log10(vc_bin), log10(mstar_bin), yerr=abs(log10(mstar_bin)-log10(e_mstar_bin)), c='r', fmt='o')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Reds'))
        plt.plot(linspace(1.8, 2.6), log10(Mstar_FJ(linspace(1.8, 2.6))), 'k-', lw=3)
        plt.plot(linspace(1.8, 2.6), log10(Mstar_FJ(linspace(1.8, 2.6)))+0.1, 'k--', lw=3)
        plt.plot(linspace(1.8, 2.6), log10(Mstar_FJ(linspace(1.8, 2.6)))-0.1, 'k--', lw=3)

        if save_plots:
            plt.savefig('FJ_BT_ref.pdf', bbox_inches='tight')
        '''
        ----------- FP
        '''
        fig = plt.figure()
        plt.xlabel(r"$\log\rm\,M_\ast/kpc$", fontsize=16)
        plt.ylabel(r"$\rm\,10.6+2\log\,\sigma/130\,km\,s^{-1}+\log\,R_e/2\,kpc$", fontsize=16)
        plt.xlim(9, 11.75)
        plt.xlim(9, 12.)
        ax = fig.add_subplot(111)
        ax.text(0.1, 0.825, r"$\rm early-types$", fontsize=18, color='#BD000D', transform=ax.transAxes)

        # etg
        w = bt > bt_ells
        # fp
        fp = 10.6+2.*log10(vc[w]/factor/130.)+log10(Re[w]/2.)
        x, y, H, lev, mstar_bin, fp_bin, e_fp_bin = bins_and_hist(mstar[w], 10.**fp, mstar[w] > 0, size_hist, sigs)
        # plt.errorbar(log10(mstar_bin), fp_bin, yerr=0.5, c='r', fmt='o')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Reds'))
        plt.plot(linspace(9, 12), linspace(9, 12), 'k-', lw=3)
        plt.plot(linspace(9, 12), linspace(9, 12)+0.06, 'k--', lw=3)
        plt.plot(linspace(9, 12), linspace(9, 12)-0.06, 'k--', lw=3)

        if save_plots:
            plt.savefig('FP_BT_ref.pdf', bbox_inches='tight')

    '''
    ----------- Mass - size relation
    '''
    fig = plt.figure()
    plt.xlabel(r"$\log\rm\,M_\ast/kpc$", fontsize=16)
    plt.ylabel(r"$\log\rm\,R_e/kpc$", fontsize=16)
    plt.xlim(8, 11.75)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm late-types$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    # ax.text(0.1, 0.85, r"$\rm Sa-S0$", fontsize=18, color='#009603', transform=ax.transAxes)
    ax.text(0.1, 0.825, r"$\rm early-types$", fontsize=18, color='#BD000D', transform=ax.transAxes)

    if len(bt[bt<0])>0:
        x, y, H, lev, mstar_bin, Re_bin, e_Re_bin = bins_and_hist(mstar, Re, bt<0, size_hist, sigs)
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Greys'), alpha=0.75)


    else:
        # spirals
        w = bt < bt_ltg
        # plt.plot(log10(r200)[w], log10(Re)[w], 'b.', alpha=0.0075)
        x, y, H, lev, mstar_bin, Re_bin, e_Re_bin = bins_and_hist(mstar, Re, w, size_hist, sigs)
        # plt.errorbar(log10(mstar_bin), log10(Re_bin), yerr=abs(log10(Re_bin)-log10(e_Re_bin)), c='b', fmt='o')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Blues'), alpha=0.75)

        # ellipticals
        w = bt > bt_ltg
        # plt.plot(log10(r200)[w], log10(Re)[w], 'r.', alpha=0.0075)
        x, y, H, lev, mstar_bin, Re_bin, e_Re_bin = bins_and_hist(mstar, Re, w, size_hist, sigs)
        # plt.errorbar(log10(mstar_bin), log10(Re_bin), yerr=abs(log10(Re_bin)-log10(e_Re_bin)), c='r', fmt='o')
        plt.contourf(x, y, log10(H).T, levels=append(lev, log10(H).max()), cmap=plt.get_cmap('Reds'), alpha=0.75)

    plt.plot(linspace(9, 11.75), mass_size_etg(logspace(9, 11.75)), 'r-', lw=2)
    plt.plot(linspace(9, 11.75), mass_size_ltg(logspace(9, 11.75)), 'b--', lw=2)

    if save_plots:
        plt.savefig('Re-Ms_BT_ref.pdf', bbox_inches='tight')

    plt.show()


def plot_bf_distributions():

    # Bulge-fraction distribution
    btf = genfromtxt('bt_distr_mstarOLD.txt')

    for i in range(2, 6):
        for k in range(40, 50):
            bth = btf[k, 0]
            btf[k, i] /= ((bth-0.8)/0.2 * 8.) + 1.
    for i in range(2, 10):
        for k in range(0, 10):
            bth = btf[k, 0]
            btf[k, i] *= (-(bth-0.2)/0.2 * 2.) + 1.

    cmap = get_cmap('coolwarm')
    labs = [r"$9<\log\,M_\ast<9.25$", r"$9.25<\log\,M_\ast<9.5$", r"$9.5<\log\,M_\ast<9.75$", r"$9.75<\log\,M_\ast<10$",
            r"$10<\log\,M_\ast<10.25$", r"$10.25<\log\,M_\ast<10.5$", r"$10.5<\log\,M_\ast<10.75$",
            r"$10.75<\log\,M_\ast<11$", r"$\log\,M_\ast>11$"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(2, 11):
        c = linspace(0.01, 0.99, num=9)[i-2]
        ax.bar(btf[:, 0], btf[:, i], width=btf[:, 1]-btf[:, 0], color='none', edgecolor=cmap(c), lw=3)
        # ax.text(0.35, linspace(0.3, 0.9, num=9)[i-2], labs[i-2], fontsize=16, color=cmap(c), transform=ax.transAxes)
        ax.text(0.375, linspace(0.3, 0.89, num=9)[i-2], labs[i-2], fontsize=16, color=cmap(c), transform=ax.transAxes)

    ax.axvline(0.25, ymin=0, ymax=1, ls='--', c='k')
    # ax.axvline(0.5, ymin=0, ymax=.26, ls='--', c='k')
    # ax.axvline(0.5, ymin=0.8, ymax=1, ls='--', c='k')
    ax.axvline(0.8, ymin=0, ymax=1, ls='--', c='k')
    # ax.text(0.025, 0.9, r"$\rm late"
    ax.text(0.075, 0.85, r"$\rm late$" + "\n" +
                         r"$\rm types$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    # ax.text(0.325, 0.9, r"$\rm Sa$", fontsize=18, color='#008540', transform=ax.transAxes)
    # ax.text(0.625, 0.9, r"$\rm S0$", fontsize=18, color='#AB9700', transform=ax.transAxes)
    # ax.text(0.875, 0.9, r"$\rm early"
    ax.text(0.85, 0.85, r"$\rm early$" + "\n" +
                        r"$\rm types$", fontsize=18, color='#BD000D', transform=ax.transAxes)

    ax.set_xlabel(r"$\rm B/T$", fontsize=18)
    ax.set_ylabel(r"$\rm P(B/T)$", fontsize=18)
    ax.set_xlim([0, 1])
    plt.savefig("bt_hist_mstar_corr.pdf", bbox_inches='tight')
    plt.show()

def plot_shmrs():

    # uniform halo mass function
    mhalo = logspace(10.9, 14.5)

    Ms = Mstar(0., mode='low', model='B10', no_scatter=True)
    mstar_b10 = []

    print ("generating mstar...")
    for m in mhalo:
        mstar_b10.append(Ms(m))
    mstar_b10 = array(mstar_b10)

    mstar_vU16 = Mstar_vU(mhalo, mode='TF', no_scatter=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\rm\log M_h/M_\odot$", fontsize=16)
    ax.set_ylabel(r"$\rm\log M_\ast/M_\odot$", fontsize=16)
    ax.fill_between(log10(mhalo), log10(mstar_b10)+0.25, log10(mstar_b10)-0.25, facecolor='k', alpha=0.5)
    ax.fill_between(log10(mhalo), log10(mstar_vU16)+0.25, log10(mstar_vU16)-0.25, facecolor='r', alpha=0.5)
    ax.plot(log10(mhalo), log10(mstar_b10), 'k-', lw=2, label=r"$\rm Behroozi+2010$")
    ax.plot(log10(mhalo), log10(mstar_vU16), 'r--', lw=2, label=r"$\rm van\,Uitert+2016$")
    ax.legend(loc='best', frameon=False)
    plt.show()



if __name__ == "__main__":
    with catch_warnings():
        simplefilter("ignore")
        plot_angular_momentum_size_velocity()
        # plot_bf_distributions()
        # plot_shmrs()
