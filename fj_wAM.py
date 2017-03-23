__author__ = 'lposti'

from math import pi
from numpy import genfromtxt, power, log10, array, searchsorted, arange, interp, linspace,\
    asarray, log, exp, histogram2d, float32, percentile, sqrt, logspace, median, std, mean, append,\
    cumsum, sort, argmin, full_like, vstack, argsort, exp
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
from matplotlib.ticker import MultipleLocator
import corner


b10_dat = genfromtxt("rsf_msmh2.txt")
l12_dat = genfromtxt("rsf_msmh3.txt")
l12_mhalo, l12_mstar = 10.**l12_dat[:1000, 1], 10.**l12_dat[:1000, 0]
b10_mhalo, b10_mstar = 10.**b10_dat[:1000, 1], 10.**b10_dat[:1000, 0]
dot_dash_seq=[2,4,7,4]
long_dash_seq=[20,4]

# def median_bins(a, b, bin_size=11, bin_err=False):
#
#     # sort a
#     idx = argsort(a)
#     a = a[idx]
#
#     # construct bins of equal size
#     bins = [a[i*int(len(a)/bin_size):(i+1)*int(len(a)/bin_size)].min() for i in range(bin_size-1)]
#     bins.append(a.max())
#
#     a_bin = [median(a[i*int(len(a)/bin_size):(i+1)*int(len(a)/bin_size)]) for i in range(bin_size-1)]
#     b_bin = [median(b[idx][(a <= bins[i]) & (a > bins[i-1])]) for i in range(1, len(bins))]
#     e_b_bin = [percentile(b[idx][(a <= bins[i]) & (a > bins[i-1])], 80.) for i in range(1, len(bins))]
#
#     if bin_err:
#         return array(a_bin), array(b_bin), array(e_b_bin), array(bins)
#     else:
#         return array(a_bin), array(b_bin), array(e_b_bin)

def median_bins(a, b, bin_size=10, bin_err=False):

    bins = linspace(a.min(), a.max(), num=bin_size+1)

    a_bin, b_bin, e_b_bin, = [], [], []
    for i in range(len(bins)-1):
        w = (a>=bins[i]) & (a<bins[i+1])
        a_bin.append(median(a[w]))
        b_bin.append(median(b[w]))
        e_b_bin.append(b[w].std())

    if bin_err:
        return array(a_bin), array(b_bin), array(e_b_bin)+array(b_bin), array(a_bin)
    else:
        return array(a_bin), array(b_bin), array(e_b_bin)+array(b_bin)

def Mstar_D10_ltg(mstar):
    a, b = -0.5, 0
    x0, y0, g = 10.**10.4, 10.**1.61, 1.

    mh = []
    for m in mstar:
        y = 10.**normal(log10(y0 * (m/x0)**a * (0.5 + 0.5*(m/x0)**g)**((b-a)/g)), 0.2, 1)[0]
        mh.append(m*y*0.7)
    return array(mh)


def Mstar_D10_etg(mstar):
    a, b = -0.15, 0.85
    x0, y0, g = 10.**10.8, 10.**1.97, 2.

    mh = []
    for m in mstar:
        y = 10.**normal(log10(y0 * (m/x0)**a * (0.5 + 0.5*(m/x0)**g)**((b-a)/g)), 0.2, 1)[0]
        mh.append(m*y*0.7)
    return array(mh)


def Mstar_L12(mstar, no_scatter=False):

    mh = []
    for m in mstar:
        if no_scatter:
            mh.append(interp(m, l12_mstar, l12_mhalo))
        else:
            mh.append(normal(interp(m, l12_mstar, l12_mhalo), 0.19, 1)[0])

    return array(mh)


def Mstar_B10(mstar, no_scatter=False):

    mh = []
    for m in mstar:
        if no_scatter:
            mh.append(interp(m, b10_mstar, b10_mhalo))
        else:
            mh.append(normal(interp(m, b10_mstar, b10_mhalo), 0.15, 1)[0])

    return array(mh)


def Mstar_M13_msmh(mhalo):
    M1, N1 = 11.59, 0.0351
    b0, g0 = 1.376, 0.608

    ms = []
    for m in mhalo:
        y = 2.*N1*1./((m/10.**M1)**(-b0)+(m/10.**M1)**(g0))
        ms.append(m*y)
    return array(ms)


m13_mh = logspace(10,15,num=2000)
m13_ms = Mstar_M13_msmh(m13_mh)


def Mstar_M13(mstar, no_scatter=False):

    mh = []
    for m in mstar:
        if no_scatter:
            mh.append(interp(m, m13_ms, m13_mh))
        else:
            mh.append(normal(interp(m, m13_ms, m13_mh), 0.15, 1)[0])

    return array(mh)

def Mstar_vU16_msmh(mhalo):
    ms0, mh1 = 10.**10.58, 10.**10.97  # 10.**11.16, 10.**12.06
    b1, b2 = 7.5, 0.25  # 5.4, 0.15

    return ms0 * power(mhalo / mh1, b1) / power(1. + mhalo / mh1, b1-b2)

vU16_mh = logspace(10,15.5,num=2000)
vU16_ms = Mstar_vU16_msmh(vU16_mh)


def Mstar_vU16(mstar, no_scatter=False):

    mh = []
    for m in mstar:
        if no_scatter:
            mh.append(interp(m, vU16_ms, vU16_mh))
        else:
            mh.append(normal(interp(m, vU16_ms, vU16_mh), 0.15, 1)[0])

    return array(mh)


def g_RP15(x, a, d, g):
    # return d/(1.+exp(10.**(-x)))*(log10(1.+exp(x)))**g-log10(10.**(a*x)+1.)
    return d * (log10(1.+exp(x)))**g / (1.+exp(10.**(-x))) - log10(1.+10.**(a*x))


def Mstar_RP15_ltg_msmh(mhalo):
    leb, lM1 = -1.593, 11.58
    ab, db, gb = -1.5, 4.3, 0.4

    return 10.**(leb+lM1 + g_RP15(log10(mhalo)-lM1, ab, db, gb)-g_RP15(0., ab, db, gb))

def Mstar_RP15_etg_msmh(mhalo):
    leb, lM1 = -2.143, 11.367
    ab, db, gb = -2.858, 6., 0.3

    return 10.**(leb+lM1 + g_RP15(log10(mhalo)-lM1, ab, db, gb)-g_RP15(0., ab, db, gb))

RP15_mh = logspace(10,15.5,num=2000)
RP15_ms_ltg = Mstar_RP15_ltg_msmh(RP15_mh)
RP15_ms_etg = Mstar_RP15_etg_msmh(RP15_mh)

def Mstar_RP15_ltg(mstar):

    mh = []
    for m in mstar:
        # mh.append(interp(m, RP15_ms_ltg, RP15_mh))
        mh.append(normal(interp(m, RP15_ms_ltg, RP15_mh), 0.11, 1)[0])

    return array(mh)


def Mstar_RP15_etg(mstar):

    mh = []
    for m in mstar:
        # mh.append(interp(m, RP15_ms_etg, RP15_mh))
        mh.append(normal(interp(m, RP15_ms_etg, RP15_mh), 0.14, 1)[0])

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


def compute_galaxy_and_halo_properties(k, func, size):
    # Planck parameters
    h, Om, OL = 0.677, 0.31, 0.69

    if k==0:
        # mstar = array(10. ** uniform(9.35, 11., size), dtype=float32)
        mstar = array(10. ** uniform(8.75, 11.25, size), dtype=float32)
    elif k==1:
        # mstar = array(10. ** uniform(9.85, 11.4, size), dtype=float32)
        mstar = array(10. ** uniform(9., 12., size), dtype=float32)
    elif k==2:
        mstar = array(10. ** uniform(8.75, 11., size), dtype=float32)
    elif k==3:
        mstar = array(10. ** uniform(9., 12., size), dtype=float32)
    elif k==4:
        mstar = array(10. ** uniform(8.25, 11.5, size), dtype=float32)
    elif k==5:
        mstar = array(10. ** uniform(9., 12., size), dtype=float32)
    elif k==6:
        mstar = array(10. ** uniform(8.75, 11., size), dtype=float32)
    elif k==7:
        mstar = array(10. ** uniform(9., 12., size), dtype=float32)
    elif k==8:
        mstar = array(10. ** uniform(8.75, 11.4, size), dtype=float32)
    elif k==9:
        mstar = array(10. ** uniform(9., 12., size), dtype=float32)
    elif k==10:
        mstar = array(10. ** uniform(8.75, 11., size), dtype=float32)
    elif k==11:
        mstar = array(10. ** uniform(9., 12., size), dtype=float32)

    print ("generating halo masses...")
    mhalo = func(mstar)

    # DM halo spin parameter distribution
    # from Maccio', Dutton & van den Bosch (2008)
    lambda_halo = 10. ** normal(-1.466, 0.253, len(mhalo))

    # Dutton & Maccio' (2014)
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

    r200 = 1e3 * power(mhalo / rho_hat, 1./3.)
    vc = 0.9 * sqrt(G * f(2.15) / f(cvir) * cvir / 2.15 * pow(rho_hat, 1./3.)) * power(mhalo / h, 1./3.)

    # jhalo - mhalo
    fe = cvir / 2. * (1.-1./power(1.+cvir, 2) - 2. * log(1.+cvir)/(1.+cvir)) / power(cvir/(1.+cvir)-log(1.+cvir), 2)
    # fe = f(2.15) / f(cvir) * cvir / 2.15
    # j_halo = sqrt(2. / fe) * lambda_halo * sqrt(G * 1e3 * mhalo / r200) * r200
    # j_halo da Romanowsky & Fall (2012) eq. (14)
    j_halo = 4.23e4 * lambda_halo * power(mhalo / 1e12, 2./3.)

    return mstar, mhalo, lambda_halo, j_halo, vc, r200

def jstar_jhalo_direct():

    size = 50000

    # fj - M for different SHMRs
    fig11, ax11 = plt.subplots()
    fig12, ax12 = plt.subplots()
    fig13, ax13 = plt.subplots()
    fig14, ax14 = plt.subplots()
    fig15, ax15 = plt.subplots()
    fig16, ax16 = plt.subplots()
    ax22 = ax12.twiny()
    ax23 = ax13.twiny()
    ax24 = ax14.twiny()
    ax26 = ax16.twiny()

    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()

    # ax1.set_xlabel(r"$\rm\log \,M_h/M_\odot$", fontsize=16)
    # ax1.set_ylabel(r"$\rm\log \,f_j\equiv j_\ast/j_h$", fontsize=16)
    for aaxx in [ax11, ax12, ax13, ax14, ax15, ax16]:
        aaxx.set_xlabel(r"$\rm\log \,M_h/M_\odot$", fontsize=16)
        aaxx.set_ylabel(r"$\rm\log \,f_j\equiv j_\ast/j_h$", fontsize=16)
    for aaxx in [ax22, ax23, ax24, ax26]:
        aaxx.set_xlabel(r"$\rm\log \,M_\ast/M_\odot$", fontsize=16)

    ax2.set_xlabel(r"$\rm\log \,M_\ast/M_\odot$", fontsize=16)
    ax2.set_ylabel(r"$\rm\log \,V/km\,s^{-1}\,\,\,and\,\,\,\log \sigma/km\,s^{-1}$", fontsize=16)
    ax3.set_xlabel(r"$\rm\log \,M_\ast/M_\odot$", fontsize=16)
    ax3.set_ylabel(r"$\rm\log \,V_{rot}/V_{max}$", fontsize=16)
    ax4.set_xlabel(r"$\rm\log \,M_\ast/M_\odot$", fontsize=16)
    ax4.set_ylabel(r"$\rm\log \,R_e/kpc$", fontsize=16)
    ax5.set_xlabel(r"$\rm\log \,M_\ast/M_\odot$", fontsize=16)
    ax5.set_ylabel(r"$\rm\log \,R_e/(j_\ast/V_{rot})$", fontsize=16)
    ax6.set_xlabel(r"$\rm\log \,R_{200}/kpc$", fontsize=16)
    ax6.set_ylabel(r"$\rm\log \,R_e/kpc$", fontsize=16)

    rf12_ljs_lMs_ltg = lambda m: 10. ** normal(3.18+0.52*(log10(m)-11.), 0.19, 1)[0]
    rf12_ljs_lMs_etg = lambda m: 10. ** normal(2.73+0.6*(log10(m)-11.), 0.24, 1)[0]

    for k,func in enumerate([Mstar_D10_ltg, Mstar_D10_etg, Mstar_L12, Mstar_L12,
                             Mstar_M13, Mstar_M13, Mstar_B10, Mstar_B10,
                             Mstar_RP15_ltg, Mstar_RP15_etg, Mstar_vU16, Mstar_vU16]):

        mstar, mhalo, lambda_halo, j_halo, vc, r200 = compute_galaxy_and_halo_properties(k, func, size)

        jstar = []
        if k % 2 == 0:
            for m in mstar:
                jstar.append(rf12_ljs_lMs_ltg(m))
        elif k % 2 == 1:
            for m in mstar:
                jstar.append(rf12_ljs_lMs_etg(m))

        jstar = array(jstar)
        fj = jstar / j_halo

        vstar_TF = lambda t: 10.**(0.61/4.93 + 1./4.93*asarray(t))
        sstar_FJ = lambda t: 10.**(2.054 + 0.286 * asarray(t-10.))
        factor = 1.65  # V_circ(R_50) := factor * sigma(R_50)
        # Lange et al. (2015) (Tables 2-3, g-i colour cut, r-band Re)
        mass_size_etg = lambda mass: log10(8.25e-5) + 0.44*log10(asarray(mass))
        # mass_size_ltg = lambda mass: log10(27.72e-3) + 0.21*log10(asarray(mass))
        alpha_ltg, beta_ltg, gamma_ltg, m0_ltg = 0.15, 0.68, 0.11, 6.39e10
        mass_size_ltg = lambda mass: log10(gamma_ltg) + alpha_ltg * log10(mass) + (beta_ltg-alpha_ltg) * log10(1. + mass/m0_ltg)

        mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 9, bin_err=True)
        if k==0:
            #ax1.errorbar(mhalo_bin[1:-1], fj_bin[1:-1], yerr=e_fj_bin[1:-1]-fj_bin[1:-1],
            #             xerr=[[-e_mhalo_bin[i]+mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)],
            #                   [e_mhalo_bin[i+1]-mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)]],
            #             fmt='o', color='b')
            #ax1.errorbar(mhalo_bin, fj_bin*999, yerr=e_fj_bin, xerr=mhalo_bin,
            #             fmt='o', color='k', label=r"$\rm Dutton+10$")

            ax11.fill_between(mhalo_bin[5:-1], (fj_bin-(e_fj_bin-fj_bin))[5:-1], e_fj_bin[5:-1], edgecolor='none',
                             facecolor='b', alpha=0.2)
            pl=ax11.plot(mhalo_bin[5:-1], fj_bin[5:-1], 'b', lw=2)
            pl[0].set_dashes(long_dash_seq)
            pl=ax11.plot(mhalo_bin, fj_bin*999, 'k', lw=2, label=r"$\rm Dutton+10$")
            pl[0].set_dashes(long_dash_seq)
            ax11.text(14, -0.3, r"$\sigma_{f_j, \rm LTG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='b', fontsize=16)

            # errorbar on f_j
            # mean_err = (e_fj_bin-fj_bin).mean()
            # ax1.plot([11.25, 11.25], [-1.6-mean_err, -1.6+mean_err], 'k-', lw=3)

            mstar_bin, vrot_bin, e_vrot_bin = median_bins(log10(mstar), log10(vc), 30)
            # ax2.errorbar(mstar_bin, vrot_bin, yerr=e_vrot_bin-vrot_bin, fmt='o-', color='b')
            ax2.fill_between(mstar_bin, vrot_bin-(e_vrot_bin-vrot_bin), e_vrot_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            pl=ax2.plot(mstar_bin, vrot_bin, color='b', lw=2)
            pl[0].set_dashes(long_dash_seq)
            ax2.plot(linspace(9,11), log10(vstar_TF(linspace(9,11))), 'b--',
                label=r"$\rm TF\,(McGaugh\,&\,Schombert\,15)$")

            mstar_bin, vrotvmax_bin, e_vrotvmax_bin = median_bins(log10(mstar),
                    log10(vstar_TF(log10(mstar))/vc), 30)
            # ax3.errorbar(mstar_bin, vrotvmax_bin, yerr=e_vrotvmax_bin-vrotvmax_bin, fmt='o-', color='b')
            ax3.fill_between(mstar_bin, vrotvmax_bin-(e_vrotvmax_bin-vrotvmax_bin), e_vrotvmax_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            pl=ax3.plot(mstar_bin, vrotvmax_bin, color='b', lw=2)
            pl[0].set_dashes(long_dash_seq)

            re = 1.25 * jstar / vstar_TF(log10(mstar))
            mstar_bin, re_bin, e_re_bin = median_bins(log10(mstar), log10(re), 30)
            # ax4.errorbar(mstar_bin, re_bin, yerr=e_re_bin-re_bin, fmt='o-', color='b')
            ax4.fill_between(mstar_bin, re_bin-(e_re_bin-re_bin), e_re_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            pl=ax4.plot(mstar_bin, re_bin, color='b', lw=2)
            pl[0].set_dashes(long_dash_seq)
            ax4.plot(linspace(9,11), mass_size_ltg(logspace(9,11)), 'b--', label=r"$\rm Lange+15\,(LTG)$")

            mstar_bin, rerej_bin, e_rerej_bin = median_bins(log10(mstar),
                    mass_size_ltg(mstar)-log10(jstar / vstar_TF(log10(mstar))), 30)
            # ax5.errorbar(mstar_bin, rerej_bin, yerr=e_rerej_bin-rerej_bin, fmt='o-', color='b')
            ax5.fill_between(mstar_bin, rerej_bin-(e_rerej_bin-rerej_bin), e_rerej_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            pl=ax5.plot(mstar_bin, rerej_bin, color='b', lw=2)
            pl[0].set_dashes(long_dash_seq)

            r200_bin, re_bin, e_re_bin = median_bins(log10(r200), log10(re), 30)
            # ax6.errorbar(r200_bin, re_bin, yerr=e_re_bin-re_bin, fmt='o-', color='b')
            ax6.fill_between(r200_bin[5:-1], (re_bin-(e_re_bin-re_bin))[5:-1], e_re_bin[5:-1], edgecolor='none',
                             facecolor='b', alpha=0.2)
            pl=ax6.plot(r200_bin[5:-1], re_bin[5:-1], color='b', lw=2)
            pl[0].set_dashes(long_dash_seq)

        elif k==1:
            #ax1.errorbar(mhalo_bin[1:-1], fj_bin[1:-1], yerr=e_fj_bin[1:-1]-fj_bin[1:-1],
            #             xerr=[[-e_mhalo_bin[i]+mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)],
            #                   [e_mhalo_bin[i+1]-mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)]],
            #             fmt='o', color='r')
            ax11.fill_between(mhalo_bin[4:-1], (fj_bin-(e_fj_bin-fj_bin))[4:-1], e_fj_bin[4:-1], edgecolor='none',
                 facecolor='r', alpha=0.2)
            pl=ax11.plot(mhalo_bin[4:-1], fj_bin[4:-1], 'r', lw=2)
            pl[0].set_dashes(long_dash_seq)
            ax11.text(14, -0.5, r"$\sigma_{f_j, \rm ETG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='r', fontsize=16)

            mstar_bin, vrot_bin, e_vrot_bin = median_bins(log10(mstar), log10(vc/factor), 30)
            # ax2.errorbar(mstar_bin, vrot_bin, yerr=e_vrot_bin-vrot_bin, fmt='o-', color='r')
            ax2.fill_between(mstar_bin, vrot_bin-(e_vrot_bin-vrot_bin), e_vrot_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            pl=ax2.plot(mstar_bin, vrot_bin, color='r', lw=2)
            pl[0].set_dashes(long_dash_seq)
            ax2.plot(linspace(9.5,11.5), log10(sstar_FJ(linspace(9.5,11.5))), 'r-.', label=r"$\rm FJ\,(Gallazzi+06)$")

            mstar_bin, vrotvmax_bin, e_vrotvmax_bin = median_bins(log10(mstar),
                    log10(sstar_FJ(log10(mstar))/vc*factor), 30)
            # ax3.errorbar(mstar_bin, vrotvmax_bin, yerr=e_vrotvmax_bin-vrotvmax_bin, fmt='o-', color='r')
            ax3.fill_between(mstar_bin, vrotvmax_bin-(e_vrotvmax_bin-vrotvmax_bin), e_vrotvmax_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            pl=ax3.plot(mstar_bin, vrotvmax_bin, color='r', lw=2)
            pl[0].set_dashes(long_dash_seq)

            re = 4.5 * jstar / sstar_FJ(log10(mstar)) / factor
            mstar_bin, re_bin, e_re_bin = median_bins(log10(mstar), log10(re), 30)
            # ax4.errorbar(mstar_bin, re_bin, yerr=e_re_bin-re_bin, fmt='o-', color='r')
            ax4.fill_between(mstar_bin, re_bin-(e_re_bin-re_bin), e_re_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            pl=ax4.plot(mstar_bin, re_bin, color='r', lw=2)
            pl[0].set_dashes(long_dash_seq)
            ax4.plot(linspace(9.5,11.5), mass_size_etg(logspace(9.5,11.5)), 'r-.', label=r"$\rm Lange+15\,(ETG)$")

            mstar_bin, rerej_bin, e_rerej_bin = median_bins(log10(mstar),
                    mass_size_etg(mstar)-log10(jstar / sstar_FJ(log10(mstar))), 30)
            # ax5.errorbar(mstar_bin, rerej_bin, yerr=e_rerej_bin-rerej_bin, fmt='o-', color='r')
            ax5.fill_between(mstar_bin, rerej_bin-(e_rerej_bin-rerej_bin), e_rerej_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            pl=ax5.plot(mstar_bin, rerej_bin, color='r', lw=2)
            pl[0].set_dashes(long_dash_seq)

            r200_bin, re_bin, e_re_bin = median_bins(log10(r200), log10(re), 30)
            # ax6.errorbar(r200_bin, re_bin, yerr=e_re_bin-re_bin, fmt='o-', color='r')
            ax6.fill_between(r200_bin[4:-1], (re_bin-(e_re_bin-re_bin))[4:-1], e_re_bin[4:-1], edgecolor='none',
                             facecolor='r', alpha=0.2)
            pl=ax6.plot(r200_bin[4:-1], re_bin[4:-1], color='r', lw=2)
            pl[0].set_dashes(long_dash_seq)
            ax6.plot(linspace(2,3.25), log10(0.015*logspace(2,3.25)), 'k--', lw=2, label=r"$\rm Kravtsov\,13$")
            ax6.plot(linspace(2,3.25), log10(0.015*logspace(2,3.25))-0.25, 'k--')
            ax6.plot(linspace(2,3.25), log10(0.015*logspace(2,3.25))+0.25, 'k--')


        elif k==2:
            mstar_bin, vrot_bin, e_vrot_bin = median_bins(log10(mstar), log10(vc), 30)
            ax12.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            ax12.plot(mhalo_bin, fj_bin*999, 'k-', lw=2, label=r"$\rm Leauthaud+12$")
            ax12.plot(mhalo_bin, fj_bin, 'b-', lw=2)
            ax12.text(14, -0.3, r"$\sigma_{f_j, \rm LTG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='b', fontsize=16)
            ax2.plot(mstar_bin, vrot_bin, 'b-', lw=2)

            # re = 1.25 * jstar / vc
            re = 1.25 * jstar / vstar_TF(log10(mstar))
            mstar_bin, re_bin, e_re_bin = median_bins(log10(mstar), log10(re), 30)
            ax4.plot(mstar_bin, re_bin, 'b-', lw=2)

            r200_bin, re_bin, e_re_bin = median_bins(log10(r200), log10(re), 30)
            ax6.plot(r200_bin, re_bin, 'b-', lw=2)

        elif k==3:
            mstar_bin, vrot_bin, e_vrot_bin = median_bins(log10(mstar), log10(vc*0.5*vrot_vc_P12(vc)), 30)
            ax12.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            ax12.plot(mhalo_bin, fj_bin, 'r-', lw=2)
            ax12.text(14, -0.5, r"$\sigma_{f_j, \rm ETG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='r', fontsize=16)
            ax2.plot(mstar_bin, vrot_bin, 'r-', lw=2)

            re = 4.5 * jstar / sstar_FJ(log10(mstar)) / factor
            mstar_bin, re_bin, e_re_bin = median_bins(log10(mstar), log10(re), 30)
            ax4.plot(mstar_bin, re_bin, 'r-', lw=2)

            r200_bin, re_bin, e_re_bin = median_bins(log10(r200), log10(re), 30)
            ax6.plot(r200_bin, re_bin, 'r-', lw=2)

        elif k==4:
            ax13.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            print (Mstar_M13_msmh(10.**mhalo_bin), (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin)
            ax13.plot(mhalo_bin, fj_bin*999, 'k--', lw=2, label=r"$\rm Moster+13$")
            ax13.plot(mhalo_bin, fj_bin, 'b--', lw=2)
            ax13.text(14, -0.3, r"$\sigma_{f_j, \rm LTG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='b', fontsize=16)
        elif k==5:
            ax13.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            ax13.plot(mhalo_bin, fj_bin, 'r--', lw=2)
            ax13.text(14, -0.5, r"$\sigma_{f_j, \rm ETG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='r', fontsize=16)
        elif k==6:
            ax14.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            ax14.plot(mhalo_bin, fj_bin*999, 'k-.', lw=2, label=r"$\rm Behroozi+13$")
            ax14.plot(mhalo_bin, fj_bin, 'b-.', lw=2)
            ax14.text(14, -0.3, r"$\sigma_{f_j, \rm LTG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='b', fontsize=16)
        elif k==7:
            ax14.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            ax14.plot(mhalo_bin, fj_bin, 'r-.', lw=2)
            ax14.text(14, -0.5, r"$\sigma_{f_j, \rm ETG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='r', fontsize=16)
        elif k==8:
            ax15.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            ax15.plot(mhalo_bin, fj_bin*999, 'k:', lw=2, label=r"$\rm Rodriguez$-$\rm Puebla+15$")
            ax15.plot(mhalo_bin, fj_bin, 'b:', lw=2)
            ax15.text(14, -0.3, r"$\sigma_{f_j, \rm LTG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='b', fontsize=16)
        elif k==9:
            ax15.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            ax15.plot(mhalo_bin, fj_bin, 'r:', lw=2)
            ax15.text(14, -0.5, r"$\sigma_{f_j, \rm ETG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='r', fontsize=16)
        elif k==10:
            #mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mstar), log10(fj), 40, bin_err=True)
            #ax1.plot(mhalo_bin, fj_bin*999, 'kx', lw=2, label=r"$\rm Rodriguez$-$\rm Puebla+15$")
            ax16.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='b', alpha=0.2)
            pl=ax16.plot(mhalo_bin, fj_bin*999, 'k', lw=2, label=r"$\rm van\,\, Uitert+16$")
            pl[0].set_dashes(dot_dash_seq)
            pl=ax16.plot(mhalo_bin, fj_bin, 'b', lw=2)
            pl[0].set_dashes(dot_dash_seq)
            ax16.text(14, -0.3, r"$\sigma_{f_j, \rm LTG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='b', fontsize=16)
        elif k==11:
            #mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mstar), log10(fj), 40, bin_err=True)
            #ax1.plot(mhalo_bin, fj_bin, 'mx', lw=2)
            ax16.fill_between(mhalo_bin, (fj_bin-(e_fj_bin-fj_bin)), e_fj_bin, edgecolor='none',
                             facecolor='r', alpha=0.2)
            pl=ax16.plot(mhalo_bin, fj_bin, 'r', lw=2)
            pl[0].set_dashes(dot_dash_seq)
            ax16.text(14, -0.5, r"$\sigma_{f_j, \rm ETG}=%1.2f$" % (e_fj_bin-fj_bin).mean(), color='r', fontsize=16)


    # ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    # ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    # ax1.set_ylim([-2.,0.2])
    # ax1.set_xlim([11,15.25])
    # ax1.set_ylim([-2.,0.])
    # ax1.set_xlim([8.25,12.])
    for aaxx in [ax11, ax12, ax13, ax14, ax15, ax16]:
        aaxx.xaxis.set_minor_locator(MultipleLocator(0.1))
        aaxx.yaxis.set_minor_locator(MultipleLocator(0.1))
        aaxx.set_ylim([-2.,0.2])
        aaxx.set_xlim([11,15.25])
        aaxx.legend(loc='upper right', frameon=False, fontsize=16, numpoints=1)

    mstar_tick_loc = arange(9, 12., 0.5)
    mstar_func = [Mstar_L12, Mstar_M13, Mstar_B10, Mstar_vU16]
    for k,aaxx in enumerate([ax22, ax23, ax24, ax26]):
        aaxx.set_xlim([11,15.25])
        # Ax2 string tick definition
        aaxx_tick_labels = ["%2.1f" % (m) for m in mstar_tick_loc]

        aaxx_tick_loc = []
        for msx in mstar_tick_loc:
            # at which logMh Mstar is msx
            aaxx_tick_loc.append(log10(mstar_func[k]([10. ** msx], no_scatter=True)[0]))

        aaxx.set_xticks(aaxx_tick_loc)
        aaxx.set_xticklabels(aaxx_tick_labels)

    ax6.set_xlim([2,3.5])
    ax2.set_ylim([1.8,3.])
    # ax1.legend(loc='upper right', frameon=False, fontsize=16, numpoints=1)
    # ax1.legend(loc='lower left', frameon=False, fontsize=16, numpoints=1)
    ax2.legend(loc='best', frameon=False, fontsize=16)
    ax4.legend(loc='best', frameon=False, fontsize=16)
    ax6.legend(loc='best', frameon=False, fontsize=16)

    #fig1.savefig("new_DvdB12_direct_fj-Mh.pdf", bbox_inches='tight')
    # fig11.savefig("new_D10_direct_fj-Mh.pdf", bbox_inches='tight')
    # fig12.savefig("new_L12_direct_fj-Mh.pdf", bbox_inches='tight')
    # fig13.savefig("new_M13_direct_fj-Mh.pdf", bbox_inches='tight')
    # fig14.savefig("new_B13_direct_fj-Mh.pdf", bbox_inches='tight')
    # fig15.savefig("new_RP15_direct_fj-Mh.pdf", bbox_inches='tight')
    # fig16.savefig("new_vU16_direct_fj-Mh.pdf", bbox_inches='tight')


    #fig2.savefig("new_DvdB12_direct_V-Ms.pdf")
    #fig3.savefig("new_DvdB12_direct_VrotVmax.pdf")
    #fig4.savefig("new_DvdB12_direct_Re-Ms.pdf")
    #fig5.savefig("new_DvdB12_direct_ReReJ.pdf")
    #fig6.savefig("new_DvdB12_direct_R200-Re.pdf")
    plt.show()


def jstar_jhalo_inverse():

    size = 100000
    # Tully Fisher
    Mstar_TF = lambda t: 10.**(-0.61+4.93*asarray(t))  # McGaugh & Schombert (2015), sTF @ 3.6um
    vstar_TF = lambda t: 10.**(0.61/4.93 + 1./4.93*asarray(t) + normal(0, 0.05, len(t)))
    # Mstar_TF = lambda t: 10.**(1.49+4.09*asarray(t))  # McGaugh & Schombert (2015), Baryonic-TF @ 3.6um
    # vstar_TF = lambda t: 10.**(-1.49/4.09 + 1./4.09*asarray(t))

    sstar_FJ = lambda t: 10.**(2.054 + 0.286 * asarray(t-10.) + normal(0, 0.1, len(t)))
    Mstar_FJ = lambda t: 10.**(-2.054/0.286 + 10. + 1./0.286 * asarray(t))

    # Shen et al. (2003)
    # mass_size_etg = lambda mass: -5.54061 + 0.56*log10(mass)
    # mass_size_ltg = lambda mass: -1. + 0.14*log10(mass) + 0.25 * log10(1. + mass/3.98e10)

    # Lange et al. (2015) (Tables 2-3, g-i colour cut, r-band Re)
    mass_size_etg = lambda mass: log10(8.25e-5) + 0.44*log10(asarray(mass)) + normal(0, 0.1, len(mass))
    mass_size_ltg = lambda mass: log10(13.98e-3) + 0.25*log10(asarray(mass)) + normal(0, 0.1, len(mass))
    # alpha_ltg, beta_ltg, gamma_ltg, m0_ltg = 0.15, 0.68, 0.11, 6.39e10
    # mass_size_ltg = lambda mass: log10(gamma_ltg) + alpha_ltg * log10(mass) + (beta_ltg-alpha_ltg) * log10(1. + mass/m0_ltg)

    # Cappellari et al. 13, sigma from mass plane
    sigma_MP = lambda mass: 10.**(0.5*(log10(mass)-10.6-log10(10.**mass_size_etg(mass)/2.))+log10(130.))

    factor = 1.65  # V_circ(R_50) := factor * sigma(R_50)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(r"$\rm\log \,M_h/M_\odot$", fontsize=16)
    ax2.set_ylabel(r"$\rm\log \,f_j\equiv j_\ast/j_h$", fontsize=16)
    ax1.set_xlabel(r"$\rm\log \,M_\ast/M_\odot$", fontsize=16)
    ax1.set_ylabel(r"$\rm\log \,j_\ast/kpc\,km\,s^{-1}$", fontsize=16)
    for k,func in enumerate([Mstar_L12, Mstar_L12, Mstar_D10_ltg, Mstar_D10_etg,
                             Mstar_M13, Mstar_M13, Mstar_B10, Mstar_B10,
                             Mstar_RP15_ltg, Mstar_RP15_etg, Mstar_vU16, Mstar_vU16]):

        mstar, mhalo, lambda_halo, j_halo, vc, r200 = compute_galaxy_and_halo_properties(k, func, size)

        if k==2:
            vrot = vstar_TF(log10(mstar))
            re = 10. ** mass_size_ltg(mstar)

            jstar = vrot * re
            rf12_discs = genfromtxt('rf12_discs.dat')
            rf12_discs_lmstar = rf12_discs[:, 14]
            rf12_discs_ljstar = log10(rf12_discs[:, 13])

            ax1.plot(rf12_discs_lmstar, rf12_discs_ljstar, 's', label=r"$\rm RF12\,\,spirals$",
                     markeredgecolor='c', markerfacecolor='None', markeredgewidth=2)

            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            #ax2.errorbar(mhalo_bin[1:-1], fj_bin[1:-1], yerr=e_fj_bin[1:-1]-fj_bin[1:-1],
            #             xerr=[[-e_mhalo_bin[i]+mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)],
            #                   [e_mhalo_bin[i+1]-mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)]],
            #             fmt='D', color='b')
            #ax2.errorbar(mhalo_bin, fj_bin*999, yerr=e_fj_bin, xerr=mhalo_bin,
            #             fmt='D', color='k', label=r"$\rm Dutton+10$")
            pl=ax2.plot(mhalo_bin[1:-1], fj_bin[1:-1], 'b', lw=2)
            pl[0].set_dashes(long_dash_seq)
            pl=ax2.plot(mhalo_bin, fj_bin*999, 'k', lw=2, label=r"$\rm Dutton+10$")
            pl[0].set_dashes(long_dash_seq)

            # errorbar on f_j
            mean_err = 0.85*(e_fj_bin-fj_bin).mean()
            ax2.plot([11.25, 11.25], [-1.75-mean_err, -1.75+mean_err], 'k-', lw=3)
        elif k==3:

            vrot = factor * sstar_FJ(log10(mstar))
            re = 10. ** mass_size_etg(mstar)

            jstar = .2 * vrot * re
            rf12_sph = genfromtxt('rf12_sph.dat', dtype=None)
            rf12_sph_lmstar = rf12_sph['f10']
            rf12_sph_ljstar = log10(rf12_sph['f9'] * 1.65)
            rf12_sph_ljstar[rf12_sph['f2']==b'S0'] += log10(1.21)-log10(1.65)
            ''' do not plot S0 '''
            rf12_sph_lmstar = rf12_sph_lmstar[rf12_sph['f2']!=b'S0']
            rf12_sph_ljstar = rf12_sph_ljstar[rf12_sph['f2']!=b'S0']
            ''' -------------- '''

            ax1.plot(rf12_sph_lmstar, rf12_sph_ljstar, 'o', alpha=0.75, label=r"$\rm RF12\,\,ellipticals$",
                     markeredgecolor='m', markerfacecolor='None', markeredgewidth=2)

            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            # ax2.errorbar(mhalo_bin[1:-1], fj_bin[1:-1], yerr=e_fj_bin[1:-1]-fj_bin[1:-1],
            #              xerr=[[-e_mhalo_bin[i]+mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)],
            #                    [e_mhalo_bin[i+1]-mhalo_bin[i] for i in range(1,len(mhalo_bin)-1)]],
            #              fmt='D', color='r')
            pl=ax2.plot(mhalo_bin[1:-1], fj_bin[1:-1], 'r', lw=2)
            pl[0].set_dashes(long_dash_seq)
        elif k==0:

            vrot = vstar_TF(log10(mstar))
            re = 10. ** mass_size_ltg(mstar)
            jstar = vrot * re

            mstar_bin, jstar_bin, e_jstar_bin = median_bins(log10(mstar), log10(jstar), 15)
            ax1.plot(mstar_bin[:-1], jstar_bin[:-1], 'b-', lw=2)

            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'b-', lw=2)
            ax2.plot(mhalo_bin, fj_bin*999, 'k-', lw=2, label=r"$\rm Leauthaud+12$")

        elif k==1:

            vrot = factor * sstar_FJ(log10(mstar))
            re = 10. ** mass_size_etg(mstar)
            jstar = .2 * vrot * re

            mstar_bin, jstar_bin, e_jstar_bin = median_bins(log10(mstar), log10(jstar), 15)
            ax1.plot(mstar_bin[:-1], jstar_bin[:-1], 'r-', lw=2)

            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'r-', lw=2)

        elif k==4:

            jstar = vstar_TF(log10(mstar)) * 10. ** mass_size_ltg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'b--', lw=2)
            ax2.plot(mhalo_bin, fj_bin*999, 'k--', lw=2, label=r"$\rm Moster+13$")

        elif k==5:

            jstar = .2 * factor * sstar_FJ(log10(mstar)) * 10. ** mass_size_etg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'r--', lw=2)

        elif k==6:

            jstar = vstar_TF(log10(mstar)) * 10. ** mass_size_ltg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'b-.', lw=2)
            ax2.plot(mhalo_bin, fj_bin*999, 'k-.', lw=2, label=r"$\rm Behroozi+13$")

        elif k==7:

            jstar = .2 * factor * sstar_FJ(log10(mstar)) * 10. ** mass_size_etg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'r-.', lw=2)

        elif k==8:

            jstar = vstar_TF(log10(mstar)) * 10. ** mass_size_ltg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'b:', lw=2)
            ax2.plot(mhalo_bin, fj_bin*999, 'k:', lw=2, label=r"$\rm Rodriguez$-$\rm Puebla+15$")

        elif k==9:

            jstar = .2 * factor * sstar_FJ(log10(mstar)) * 10. ** mass_size_etg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            ax2.plot(mhalo_bin, fj_bin, 'r:', lw=2)
        elif k==10:

            jstar = vstar_TF(log10(mstar)) * 10. ** mass_size_ltg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            pl=ax2.plot(mhalo_bin, fj_bin*999, 'k', lw=2, label=r"$\rm van \,\,Uitert+16$")
            pl[0].set_dashes(dot_dash_seq)
            pl=ax2.plot(mhalo_bin, fj_bin, 'b', lw=2)
            pl[0].set_dashes(dot_dash_seq)

        elif k==11:

            jstar = .2 * factor * sstar_FJ(log10(mstar)) * 10. ** mass_size_etg(mstar)
            fj = jstar / j_halo
            mhalo_bin, fj_bin, e_fj_bin, e_mhalo_bin = median_bins(log10(mhalo), log10(fj), 15, bin_err=True)
            pl=ax2.plot(mhalo_bin, fj_bin, 'r', lw=2)
            pl[0].set_dashes(dot_dash_seq)

    ax2.set_ylim([-2,0.25])
    ax2.set_xlim([11,15.25])
    ax1.legend(loc='best', frameon=False, numpoints=1)
    ax2.legend(loc='best', frameon=False, numpoints=1, fontsize=16)
    fig2.savefig("new_DvdB12_inverse_fj-Mh.pdf")
    # fig1.savefig("new_DvdB12_inverse_js-Ms.pdf")
    plt.show()

if __name__ == "__main__":
    jstar_jhalo_direct()
    #jstar_jhalo_inverse()
