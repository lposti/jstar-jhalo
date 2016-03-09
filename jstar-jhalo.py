__author__ = 'morpheus'

from math import pi
from numpy import genfromtxt, power, log10, array, searchsorted, arange, interp, linspace,\
    asarray, log, exp, histogram2d, float32, percentile, sqrt, logspace
from numpy.random import normal, uniform, choice
from warnings import simplefilter, catch_warnings
import matplotlib.pylab as plt
from matplotlib.cm import get_cmap


class Mstar(object):

    def __init__(self, z, mode='high'):
        self.d = genfromtxt("rsf_msmh3.txt")
        self.z = z
        self.mode = mode

        self._index_search = arange(399) * 1000
        self._z_file = self.d[self._index_search, 3]

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
        return 10. ** normal(log10(Ms_med), 0.25, 1)[0]


def Mstar_vU(mhalo, mode='def', no_scatter=False):

    if mode == 'def':
        # default
        mh1, ms0, b1, b2 = 10.**12.06, 10.**11.16, 5.4, 0.15
    elif mode == 'TF':
        # stellar Tully-Fisher
        mh1, ms0, b1, b2 = 10.**12.3, 10.**11.5, 2.25, 0.6
    elif mode == 'FR':
        # Fall & Kravtsov relations
        mh1, ms0, b1, b2 = 10.**12.06, 10.**10.9, 1.25, 0.4
    else:
        raise ValueError("mode not recognized!")

    if no_scatter:
        s = 0.001
    else:
        s = 0.14
    mhalo = asarray(mhalo)
    if mhalo[mhalo < 1e4] is []:
        mhalo = 10. ** mhalo

    ms = []
    for m in mhalo:
        ms.append(10. ** normal(log10(ms0 * power(m / mh1, b1) / power(1. + m / mh1, b1-b2)), s, 1)[0])

    return array(ms)


def plot_angular_momentum_size_velocity():

    size = 1000000
    # Planck parameters
    h, Om, OL = 0.677, 0.31, 0.69

    # uniform halo mass function
    mhalo = array(10. ** uniform(10.9, 14.5, size), dtype=float32)

    # read halo mass-function
    # mhf = genfromtxt('mVector_PLANCK-SMT_11-13.txt')
    mhf = genfromtxt('mVector_PLANCK-SMT_10.5-13.txt')
    # mhalo = array(choice(mhf[:, 0], p=mhf[:, 8]/sum(mhf[:, 8]), size=size), dtype=float32)


    Ms = Mstar(0., mode='low')
    mstar = []

    print "generating mstar..."
    for m in mhalo:
        mstar.append(Ms(m))
    mstar = array(mstar)
    '''
    mstar = Mstar_vU(mhalo, mode='FR')
    '''

    ms_lowlim, ms_highlim = 7., 11.75
    mhalo = mhalo[(log10(array(mstar)) > ms_lowlim) & (log10(array(mstar)) < ms_highlim)]
    mstar = mstar[(log10(array(mstar)) > ms_lowlim) & (log10(array(mstar)) < ms_highlim)]

    # Bulge-fraction distribution
    btf = genfromtxt('bt_distr_mstar.txt')

    bt = []
    for m in mstar:
        if log10(m) <= 9.25:
            bt.append(choice(btf[:, 0], p=btf[:, 2]/sum(btf[:, 2]), size=1)[0])
        if (log10(m) > 9.25) & (log10(m) <= 9.5):
            bt.append(choice(btf[:, 0], p=btf[:, 3]/sum(btf[:, 3]), size=1)[0])
        if (log10(m) > 9.5) & (log10(m) <= 9.75):
            bt.append(choice(btf[:, 0], p=btf[:, 4]/sum(btf[:, 4]), size=1)[0])
        if (log10(m) > 9.75) & (log10(m) <= 10.):
            bt.append(choice(btf[:, 0], p=btf[:, 5]/sum(btf[:, 5]), size=1)[0])
        if (log10(m) > 10.) & (log10(m) <= 10.25):
            bt.append(choice(btf[:, 0], p=btf[:, 6]/sum(btf[:, 6]), size=1)[0])
        if (log10(m) > 10.25) & (log10(m) <= 10.5):
            bt.append(choice(btf[:, 0], p=btf[:, 7]/sum(btf[:, 7]), size=1)[0])
        if (log10(m) > 10.5) & (log10(m) <= 10.75):
            bt.append(choice(btf[:, 0], p=btf[:, 8]/sum(btf[:, 8]), size=1)[0])
        if (log10(m) > 10.75) & (log10(m) <= 11.):
            bt.append(choice(btf[:, 0], p=btf[:, 9]/sum(btf[:, 9]), size=1)[0])
        if log10(m) > 11.:
            bt.append(choice(btf[:, 0], p=btf[:, 10]/sum(btf[:, 10]), size=1)[0])


    bt = array(bt)

    # DM halo spin parameter distribution
    # from Maccio', Dutton & van den Bosch (2008)
    lambda_halo = 10. ** normal(-1.466, 0.253, len(mhalo))

    # Dutton & Maccio' (2014)
    # b = -0.101 + 0.026 * array([0., 1.])
    # a = 0.52 + (0.905 - 0.52) * exp(-0.617 * array([0., 1.]) ** 1.21)
    b = -0.097 + 0.024 * array([0., 1.])
    a = 0.537 + (1.025 - 0.537) * exp(-0.718 * array([0., 1.]) ** 1.08)
    # scatter 0.11dex
    cvir = []
    print "generating concentrations..."
    for m in mhalo:
        cvir.append(10. ** normal(a[0] - b[0] * log10(m / (1e12 / h)), 0.11, 1)[0])

    # Circular velocity for NFW haloes
    G = 4.302e-3 * 1e-6  # Mpc Mo^-1 (km/s)^2
    f = lambda x: log(1.+asarray(x)) - asarray(x)/(1.+asarray(x))
    rho_c = 3. * (h * 100.)**2 / (8. * pi * G)
    # Bryan & Norman (1998)
    Oz = lambda z: Om * (1+z)**3 / (Om * (1.+z)**3+OL)
    Delta_c = lambda z: 18. * pi**2 + 82. * (Oz(z) - 1.) - 39. * (Oz(z) - 1.)**2
    rho_hat = 4. / 3. * pi * Delta_c(0.) * rho_c

    vc = sqrt(G * f(2.15) / f(cvir) * cvir / 2.15 * pow(rho_hat, 1./3.)) * power(mhalo / h, 1./3.)

    # Kravtsov 2013
    # rho_200 = 4. / 3. * pi * 200. * rho_c
    r200 = 1e3 * power(mhalo / rho_hat, 1./3.)
    rs = []
    for r in r200:
        rs.append(10. ** normal(log10(0.015 * r), 0.25, 1)[0])
    rs = array(rs)
    js = rs * vc

    # jhalo - mhalo
    # j_halo = sqrt(2) * lambda_halo * vc * 2.15 * r200 / cvir * sqrt(2./3. + power(array(cvir) / 21.5, 0.7))
    # j_halo da Romanowsky & Fall (2012) eq. (14)
    j_halo = 4.23e4 * lambda_halo * power(mhalo / 1e12, 2./3.)

    # fj(Mhalo)
    def jstar_FR(mass, bf):
        if (bf >= 0.) & (bf <= 0.1):
            # Sc
            j0, alpha, s = 3.29, 0.55, 0.18
        elif (bf > 0.1) & (bf <= 0.2):
            # Sb
            j0, alpha, s = 3.21, 0.68, 0.15
        elif (bf > 0.2) & (bf <= 0.5):
            # Sa
            j0, alpha, s = 3.02, 0.64, 0.12
        elif (bf > 0.5) & (bf <= 0.8):
            # S0
            j0, alpha, s = 3.05, 0.8, 0.22
        elif (bf > 0.8) & (bf <= .95):
            # fE
            j0, alpha, s = 2.875, 0.6, 0.2
        elif (bf > 0.95) & (bf <= 1.):
            # sE
            j0, alpha, s = 2.73, 0.6, 0.2
        else:
            print bf
            raise ValueError("Problem in bt not in ]0,1]")

        return 10. ** normal(j0+alpha*(mass-11.), s, 1)[0]

    # fitting function to jb/jd estimated from Romanowsky & Fall (2012) data
    # jb/jb = 0.025 + B/T^2
    # jd = j_star / (1 + (jb/jd-1)B/T)
    # --> here jdisc = jstar / fb_func
    jb_over_jd_fit = lambda fb: 0.025 + fb**2
    fb_func = lambda fb: 1. + (jb_over_jd_fit(fb) - 1.) * fb

    jstar, jdisc, jbulge = [], [], []
    print "generating jstar..."
    for i in range(len(mstar)):
        jstar.append(jstar_FR(log10(mstar[i]), bt[i]))
        jdisc.append(jstar[i] / fb_func(bt[i]))
        jbulge.append(jdisc[i] * jb_over_jd_fit(bt[i]))
    jstar, jdisc, jbulge = array(jstar), array(jdisc), array(jbulge)
    fj = jstar / j_halo  # jstar_FR(log10(0.158 * mhalo)) / j_halo

    # Kravtsov
    Mstar_TF = lambda t: 10.**(-0.61+4.93*asarray(t))  # McGaugh & Schombert (2015), sTF @ 3.6um
    vstar_TF = lambda t: 10.**(0.61/4.93 + 1./4.93*asarray(t))
    Re = jdisc / (vstar_TF(log10(mstar)))
    Re[bt > 0.8] = jbulge[bt > 0.8] / (0.7 * vstar_TF(log10(mstar[bt > 0.8])))

    # Mo, Mao & White (1998)
    lambda_halo_prime = lambda_halo * jdisc / j_halo  # fj
    cvir = array(cvir)
    fac_sis_nfw = 1. / sqrt(cvir / 2. * (1.-1./power(1.+cvir, 2) - 2. * log(1. + cvir) / (1. + cvir)) /
                            power(cvir / (1. + cvir) - log(1. + cvir), 2))
    Rd = 1. / sqrt(2.) * lambda_halo_prime * r200 * fac_sis_nfw

    """
    Plots
    """

    # plt.plot(log10(mstar * bt), log10(jbulge), 'r.', alpha=0.005)
    # plt.plot(log10(mstar * bt * (1./bt - 1.)), log10(jdisc), 'b.', alpha=0.005)
    # plt.plot(linspace(9, 12), 3.28+0.67*(linspace(9, 12)-11.), 'k-', lw=3)
    # plt.plot(linspace(9, 12), 2.75+0.67*(linspace(9, 12)-11.), 'k--', lw=3)
    # plt.show()

    size_hist = 20
    sigs = [80., 95.]
    bt_spirals, bt_ltg, bt_lents, bt_ells = 0.2, 0.5, 0.7, 0.9
    '''
    ----------- Mstar-Mhalo
    '''
    plt.ylabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    plt.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    plt.plot(log10(mhalo), log10(mstar), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mhalo).min(), log10(mhalo).max(), num=size_hist),\
                   linspace(log10(mstar).min(), log10(mstar).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mhalo), log10(mstar), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mhalo).min(), log10(mhalo).max(), num=H.shape[0]), \
           linspace(log10(mstar).min(), log10(mstar).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')

    '''
    ----------- lambda-Mhalo
    '''
    fig = plt.figure()
    plt.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\lambda_{halo}$", fontsize=16)
    plt.plot(log10(mhalo), log10(lambda_halo), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mhalo).min(), log10(mhalo).max(), num=size_hist),\
                   linspace(log10(lambda_halo).min(), log10(lambda_halo).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mhalo), log10(lambda_halo), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mhalo).min(), log10(mhalo).max(), num=H.shape[0]), \
           linspace(log10(lambda_halo).min(), log10(lambda_halo).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')

    '''
    ----------- j_halo - M_halo
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\,j_h/km\,s^{-1}\,kpc$", fontsize=16)
    ax.plot(log10(mhalo), log10(j_halo), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mhalo).min(), log10(mhalo).max(), num=size_hist),\
                   linspace(log10(j_halo).min(), log10(j_halo).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mhalo), log10(j_halo), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mhalo).min(), log10(mhalo).max(), num=H.shape[0]), \
           linspace(log10(j_halo).min(), log10(j_halo).max(), num=H.shape[1])
    ax.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
    ax.plot(linspace(11, 13), 3.28+0.67*(linspace(11, 13)-11.), 'k--', lw=3)
    # plt.plot([12.5, log10(0.158 * 10.**12.5)], [3.505, 3.7481], 'm-', lw=3)
    ax.arrow(12.5, 3.505, -12.5+log10(0.08 * 10.**12.5), -3.505+3.28+0.67*(log10(0.08 * 10.**12.5)-11.)-0.045,
             head_width=0.075, head_length=0.05, fc='m', ec='m', lw=3)
    ax.text(11.15, 3.875, r"$f_{\rm baryon}\simeq 0.16$"+"\n"+r"$M_\ast=M_{\rm bar}/2$", color='m', fontsize=18)

    '''
    ----------- fj(M_halo)
    '''
    fig = plt.figure()
    plt.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\,f_j(M_h)\equiv j_{\ast, FR} / j_h$", fontsize=16)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm Sc-Sb$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    ax.text(0.1, 0.85, r"$\rm Sa-S0$", fontsize=18, color='#AB9700', transform=ax.transAxes)
    ax.text(0.1, 0.8, r"$\rm Es$", fontsize=18, color='#BD000D', transform=ax.transAxes)

    # spirals
    w = bt < bt_spirals
    # plt.plot(log10(mhalo[w]), log10(fj[w]), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mhalo[w]).min(), log10(mhalo[w]).max(), num=size_hist),\
                   linspace(log10(fj[w]).min(), log10(fj[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mhalo[w]), log10(fj[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mhalo[w]).min(), log10(mhalo[w]).max(), num=H.shape[0]), \
           linspace(log10(fj[w]).min(), log10(fj[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')

    # Sa and lenticulars
    w = (bt >= bt_spirals) & (bt < bt_lents)
    # plt.plot(log10(mhalo[w]), log10(fj[w]), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mhalo[w]).min(), log10(mhalo[w]).max(), num=size_hist),\
                   linspace(log10(fj[w]).min(), log10(fj[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mhalo[w]), log10(fj[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mhalo[w]).min(), log10(mhalo[w]).max(), num=H.shape[0]), \
           linspace(log10(fj[w]).min(), log10(fj[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#AB9700')

    # ellipticals
    w = bt > bt_ells
    # plt.plot(log10(mhalo[w]), log10(fj[w]), 'r.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mhalo[w]).min(), log10(mhalo[w]).max(), num=size_hist),\
                   linspace(log10(fj[w]).min(), log10(fj[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mhalo[w]), log10(fj[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mhalo[w]).min(), log10(mhalo[w]).max(), num=H.shape[0]), \
           linspace(log10(fj[w]).min(), log10(fj[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#BD000D')
    plt.xlim([10.7, 13.5])

    '''
    ----------- Kravtsov
    '''
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,R_e/kpc$", fontsize=16)
    plt.xlabel(r"$\log\rm\,r_{200}/kpc$", fontsize=16)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm Sc-Sb$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    ax.text(0.1, 0.85, r"$\rm Sa-S0$", fontsize=18, color='#AB9700', transform=ax.transAxes)
    ax.text(0.1, 0.8, r"$\rm Es$", fontsize=18, color='#BD000D', transform=ax.transAxes)
    ax.text(0.4, 0.15, r"$\rm Kravtsov Plot$", fontsize=20, transform=ax.transAxes)

    # spirals
    w = bt < bt_spirals
    # plt.plot(log10(r200)[w], log10(Re)[w], 'b.', alpha=0.0075)
    x_bin, y_bin = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=size_hist),\
                   linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(r200[w]), log10(Re[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=H.shape[0]), \
           linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')

    # Sa and lenticulars
    w = (bt >= bt_spirals) & (bt < bt_lents)
    # plt.plot(log10(r200)[w], log10(Re)[w], 'b.', alpha=0.0075)
    x_bin, y_bin = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=size_hist),\
                   linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(r200[w]), log10(Re[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=H.shape[0]), \
           linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#AB9700')

    # ellipticals
    w = bt > bt_ells
    # plt.plot(log10(r200)[w], log10(Re)[w], 'r.', alpha=0.0075)
    x_bin, y_bin = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=size_hist),\
                   linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(r200[w]), log10(Re[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=H.shape[0]), \
           linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#BD000D')

    plt.plot(log10(linspace(110, 800)), log10(0.015 * linspace(110, 800)), 'k-', lw=2)
    plt.plot(log10(linspace(110, 800)), log10(0.015 * linspace(110, 800))+0.5, 'k--', lw=2)
    plt.plot(log10(linspace(110, 800)), log10(0.015 * linspace(110, 800))-0.5, 'k--', lw=2)
    plt.xlim([log10(110.), log10(800.)])

    '''
    ----------- Mo, Mao & White Rd-r200
    '''
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,R_d/kpc$", fontsize=16)
    plt.xlabel(r"$\log\rm\,r_{200}/kpc$", fontsize=16)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm Sc-Sb$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    ax.text(0.1, 0.85, r"$\rm Sa-S0$", fontsize=18, color='#AB9700', transform=ax.transAxes)
    ax.text(0.4, 0.15, r"$\rm Mo, Mao & White Plot$", fontsize=20, transform=ax.transAxes)

    # spirals
    w = bt < bt_spirals
    # plt.plot(log10(r200[w]), log10(Rd[w]), 'b.', alpha=0.01)
    x_bin, y_bin = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=size_hist),\
                   linspace(log10(Rd[w]).min(), log10(Rd[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(r200[w]), log10(Rd[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=H.shape[0]), \
           linspace(log10(Rd[w]).min(), log10(Rd[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')

    # Sa and lenticulars
    w = (bt >= bt_spirals) & (bt < bt_lents)
    # plt.plot(log10(r200[w]), log10(Rd[w]), 'y.', alpha=0.01)
    x_bin, y_bin = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=size_hist),\
                   linspace(log10(Rd[w]).min(), log10(Rd[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(r200[w]), log10(Rd[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(r200[w]).min(), log10(r200[w]).max(), num=H.shape[0]), \
           linspace(log10(Rd[w]).min(), log10(Rd[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#AB9700')
    plt.plot(log10(linspace(110, 800)), log10(0.0112*linspace(110, 800)), 'k-', lw=2)
    plt.xlim([log10(110.), log10(800.)])

    '''
    ----------- sTFR
    '''
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    plt.xlabel(r"$\log\rm\,V_{max}/km\,s^{-1}$", fontsize=16)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm Sc-Sb$", fontsize=18, color='#151AB0', transform=ax.transAxes)

    # spirals
    w = bt < bt_spirals
    plt.plot(log10(vc[w]), log10(mstar[w]), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(vc[w]).min(), log10(vc[w]).max(), num=size_hist),\
                   linspace(log10(mstar[w]).min(), log10(mstar[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(vc[w]), log10(mstar[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(vc[w]).min(), log10(vc[w]).max(), num=H.shape[0]), \
           linspace(log10(mstar[w]).min(), log10(mstar[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4))), 'k-', lw=3)
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4)))+0.15, 'k--', lw=3)
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4)))-0.15, 'k--', lw=3)

    '''
    ----------- Mass - size relation
    '''
    fig = plt.figure()
    plt.xlabel(r"$\log\rm\,M_\ast/kpc$", fontsize=16)
    plt.ylabel(r"$\log\rm\,R_e/kpc$", fontsize=16)
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, r"$\rm Sc-Sb-Sa$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    ax.text(0.1, 0.825, r"$\rm S0-Es$", fontsize=18, color='#BD000D', transform=ax.transAxes)

    # spirals
    w = bt < bt_ltg
    # plt.plot(log10(r200)[w], log10(Re)[w], 'b.', alpha=0.0075)
    x_bin, y_bin = linspace(log10(mstar[w]).min(), log10(mstar[w]).max(), num=size_hist),\
                   linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mstar[w]), log10(Re[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mstar[w]).min(), log10(mstar[w]).max(), num=H.shape[0]), \
           linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')

    # ellipticals
    w = bt > bt_ltg
    # plt.plot(log10(r200)[w], log10(Re)[w], 'r.', alpha=0.0075)
    x_bin, y_bin = linspace(log10(mstar[w]).min(), log10(mstar[w]).max(), num=size_hist),\
                   linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mstar[w]), log10(Re[w]), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mstar[w]).min(), log10(mstar[w]).max(), num=H.shape[0]), \
           linspace(log10(Re[w]).min(), log10(Re[w]).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#BD000D')

    mass_size_etg = lambda mass: -5.54061 + 0.56*log10(mass)
    mass_size_ltg = lambda mass: -1. + 0.14*log10(mass) + 0.25 * log10(1. + mass/3.98e10)
    plt.plot(linspace(9, 11.75), mass_size_etg(logspace(9, 11.75)), 'r-', lw=2)
    plt.plot(linspace(9, 11.75), mass_size_ltg(logspace(9, 11.75)), 'b--', lw=2)

    plt.show()


def plot_bf_distributions():

    # Bulge-fraction distribution
    btf = genfromtxt('bt_distr_mstar.txt')

    cmap = get_cmap('coolwarm')
    labs = [r"$9<\log\,M_\ast<9.25$", r"$9.25<\log\,M_\ast<9.5$", r"$9.5<\log\,M_\ast<9.75$", r"$9.75<\log\,M_\ast<10$",
            r"$10<\log\,M_\ast<10.25$", r"$10.25<\log\,M_\ast<10.5$", r"$10.5<\log\,M_\ast<10.75$",
            r"$10.75<\log\,M_\ast<11$", r"$\log\,M_\ast>11$"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(2, 11):
        c = linspace(0.01, 0.99, num=9)[i-2]
        ax.bar(btf[:, 0], btf[:, i], width=btf[:, 1]-btf[:, 0], color='none', edgecolor=cmap(c), lw=3)
        ax.text(0.35, linspace(0.3, 0.7, num=9)[i-2], labs[i-2], fontsize=16, color=cmap(c), transform=ax.transAxes)

    ax.axvline(0.2, ymin=0, ymax=1, ls='--', c='k')
    ax.axvline(0.5, ymin=0, ymax=.26, ls='--', c='k')
    ax.axvline(0.5, ymin=0.8, ymax=1, ls='--', c='k')
    ax.axvline(0.8, ymin=0, ymax=1, ls='--', c='k')
    ax.text(0.025, 0.9, r"$\rm Sc-Sb$", fontsize=18, color='#151AB0', transform=ax.transAxes)
    ax.text(0.325, 0.9, r"$\rm Sa$", fontsize=18, color='#008540', transform=ax.transAxes)
    ax.text(0.625, 0.9, r"$\rm S0$", fontsize=18, color='#AB9700', transform=ax.transAxes)
    ax.text(0.875, 0.9, r"$\rm Es$", fontsize=18, color='#BD000D', transform=ax.transAxes)

    ax.set_xlabel(r"$\rm B/T$", fontsize=18)
    ax.set_ylabel(r"$\rm P(B/T)$", fontsize=18)
    ax.set_xlim([0, 1])
    plt.show()

if __name__ == "__main__":
    with catch_warnings():
        simplefilter("ignore")
        plot_angular_momentum_size_velocity()
        # plot_bf_distributions()