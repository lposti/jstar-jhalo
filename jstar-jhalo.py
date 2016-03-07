__author__ = 'morpheus'

from math import pi
from numpy import genfromtxt, power, log10, array, searchsorted, arange, interp, linspace,\
    asarray, log, exp, histogram2d, float32, percentile, sqrt
from numpy.random import normal, uniform, choice
import matplotlib.pylab as plt


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

    size = 50000
    # Planck parameters
    h, Om, OL = 0.677, 0.31, 0.69

    # uniform halo mass function
    mhalo = array(10. ** uniform(11, 13, size), dtype=float32)

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

    # ms_lowlim = 9.
    # mhalo = mhalo[log10(array(mstar)) > ms_lowlim]
    # mstar = mstar[log10(array(mstar)) > ms_lowlim]

    # Bulge-fraction distribution
    btf = genfromtxt('bt_distr_mstar.txt')

    bt = []
    for m in mstar:
        if log10(m) <= 9.5:
            bt.append(choice(btf[:, 0], p=btf[:, 2]/sum(btf[:, 2]), size=1)[0])
        if (log10(m) > 9.5) & (log10(m) <= 10.):
            bt.append(choice(btf[:, 0], p=btf[:, 3]/sum(btf[:, 3]), size=1)[0])
        if (log10(m) > 10.) & (log10(m) <= 10.5):
            bt.append(choice(btf[:, 0], p=btf[:, 4]/sum(btf[:, 4]), size=1)[0])
        if (log10(m) > 10.5) & (log10(m) <= 11.):
            bt.append(choice(btf[:, 0], p=btf[:, 5]/sum(btf[:, 5]), size=1)[0])
        if log10(m) > 11.:
            bt.append(choice(btf[:, 0], p=btf[:, 6]/sum(btf[:, 6]), size=1)[0])

    bt = array(bt)

    # DM halo spin parameter distribution
    # from Maccio', Dutton & van den Bosch (2008)
    lambda_halo = []
    for m in mhalo:
        lambda_halo.append(10. ** normal(-1.466, 0.253, 1)[0])
    lambda_halo = array(lambda_halo)

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
    # vc = 6.72 * 1e-3 * power(mhalo, 1./3.) * sqrt(cvir / f(cvir))
    # print " compare: %e %e " % (6.72e-3, sqrt(G * f(2.15) / 2.15 * pow(rho_hat, 1./3.)) / h ** 0.333)

    # Kravtsov 2013
    rho_200 = 4. / 3. * pi * 200. * rho_c
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
    jstar_FR = lambda t: 10. ** normal(3.23+0.67*(t-11.), 0.2, 1)[0]
    jstar = []
    print "generating jstar..."
    for m in mstar:
        jstar.append(jstar_FR(log10(m)))
    jstar = array(jstar)
    fj = jstar / j_halo  # jstar_FR(log10(0.158 * mhalo)) / j_halo

    # Kravtsov
    # Mstar_TF = lambda t: 10.**(1.82+3.84*asarray(t))
    # vstar_TF = lambda t: 10.**(-1.82/3.84 + 1./3.84*asarray(t))
    Mstar_TF = lambda t: 10.**(-0.61+4.93*asarray(t))  # McGaugh & Schombert (2015), sTF @ 3.6um
    vstar_TF = lambda t: 10.**(0.61/4.93 + 1./4.93*asarray(t))
    Re = jstar / vstar_TF(log10(mstar))

    # Mo, Mao & White (1998)
    # mstar = 0.158 * mhalo
    md = (mstar / mhalo)  # 0.158
    lambda_halo_prime = lambda_halo * fj
    cvir = array(cvir)
    fac_sis_nfw = 1. / sqrt(cvir / 2. * (1.-1./power(1.+cvir, 2) - 2. * log(1. + cvir) / (1. + cvir)) /
                            power(cvir / (1. + cvir) - log(1. + cvir), 2))
    # fac_sis_nfw *= power(lambda_halo_prime / 0.1, -0.06 + 2.71 * md + 0.0047 / lambda_halo_prime) * \
    #                (1. - 3.*md + 5.2*md**2) * (1. - 0.019*cvir + 0.00025*cvir**2 + 0.52/cvir)
    Rd = 1. / sqrt(2.) * lambda_halo_prime * r200 * fac_sis_nfw

    lambda_halo_prime, md, cvir = lambda_halo.mean() * array(fj).mean(), array(mstar / mhalo).mean(), 20.

    '''
    fac_sis_nfw = 1. / sqrt(cvir / 2. * (1.-1./power(1.+cvir, 2) - 2. * log(1. + cvir) / (1. + cvir)) /
                            power(cvir / (1. + cvir) - log(1. + cvir), 2))
    # fac_sis_nfw *= power(lambda_halo_prime / 0.1, -0.06 + 2.71 * md + 0.0047 / lambda_halo_prime) * \
    #                (1. - 3.*md + 5.2*md**2) * (1. - 0.019*cvir + 0.00025*cvir**2 + 0.52/cvir)
    print lambda_halo.mean(), array(fj).mean(), array(mstar / mhalo).mean(), \
        0.707 * lambda_halo_prime * fac_sis_nfw
    '''

    """
    Plots
    """

    size_hist = 30
    sigs = [80., 95., 98.]
    # Mstar-Mhalo
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

    # lambda-Mhalo
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

    # j_halo - M_halo
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


    # fj(M_halo)
    fig = plt.figure()
    plt.xlabel(r"$\log\rm\,M_h/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\,f_j(M_h)\equiv j_{\ast, FR} / j_h$", fontsize=16)
    plt.plot(log10(mhalo), log10(fj), 'r.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mhalo).min(), log10(mhalo).max(), num=size_hist),\
                   linspace(log10(fj).min(), log10(fj).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mhalo), log10(fj), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mhalo).min(), log10(mhalo).max(), num=H.shape[0]), \
           linspace(log10(fj).min(), log10(fj).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#BD000D')

    # kravtsov
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,R_e/kpc$", fontsize=16)
    plt.xlabel(r"$\log\rm\,0.015\times r_{200}/kpc$", fontsize=16)
    plt.plot(log10(0.015 * r200), log10(Re), 'r.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(0.015 * r200).min(), log10(0.015 * r200).max(), num=size_hist),\
                   linspace(log10(Re).min(), log10(Re).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(0.015 * r200), log10(Re), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(0.015 * r200).min(), log10(0.015 * r200).max(), num=H.shape[0]), \
           linspace(log10(Re).min(), log10(Re).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#BD000D')
    plt.plot(log10(linspace(1.6, 10.)), log10(linspace(1.6, 10.)), 'k-', lw=2)
    plt.plot(log10(linspace(1.6, 10.)), log10(linspace(1.6, 10.))+0.5, 'k--', lw=2)
    plt.plot(log10(linspace(1.6, 10.)), log10(linspace(1.6, 10.))-0.5, 'k--', lw=2)

    # Mo, Mao & White Rd-r200
    fig = plt.figure()
    plt.ylabel(r"$\log\rm\,R_d/kpc$", fontsize=16)
    plt.xlabel(r"$\log\rm\,r_{200}/kpc$", fontsize=16)
    plt.plot(log10(r200), log10(Rd), 'g.', alpha=0.005)
    x_bin, y_bin = linspace(log10(r200).min(), log10(r200).max(), num=size_hist),\
                   linspace(log10(Rd).min(), log10(Rd).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(r200), log10(Rd), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(r200).min(), log10(r200).max(), num=H.shape[0]), \
           linspace(log10(Rd).min(), log10(Rd).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#035200')
    plt.plot(log10(linspace(100, 600)), log10(0.0112*linspace(100, 600)), 'k-', lw=2)

    # sTFR
    fig = plt.figure()
    plt.plot(log10(vc), log10(mstar), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(vc).min(), log10(vc).max(), num=size_hist),\
                   linspace(log10(mstar).min(), log10(mstar).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(vc), log10(mstar), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(vc).min(), log10(vc).max(), num=H.shape[0]), \
           linspace(log10(mstar).min(), log10(mstar).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
    # plt.plot(linspace(1.9, 2.4), 1.82+3.84*linspace(1.9, 2.4), 'k-', lw=3)
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4))), 'k-', lw=3)
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4)))+0.15, 'k--', lw=3)
    plt.plot(linspace(1.9, 2.4), log10(Mstar_TF(linspace(1.9, 2.4)))-0.15, 'k--', lw=3)
    plt.ylabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    plt.xlabel(r"$\log\rm\,V_{max}/km\,s^{-1}$", fontsize=16)

    # Fall
    fig = plt.figure()
    plt.plot(log10(mstar), log10(js), 'b.', alpha=0.0025)
    x_bin, y_bin = linspace(log10(mstar).min(), log10(mstar).max(), num=size_hist),\
                   linspace(log10(js).min(), log10(js).max(), num=size_hist)
    H, xe, ye = histogram2d(log10(mstar), log10(js), bins=(x_bin, y_bin))
    lev = percentile(log10(H), sigs)  # 1.6sigma, 2sigma, 2.5sigma
    x, y = linspace(log10(mstar).min(), log10(mstar).max(), num=H.shape[0]), \
           linspace(log10(js).min(), log10(js).max(), num=H.shape[1])
    plt.contour(x, y, log10(H).T, levels=lev, colors='#151AB0')
    # plt.plot(linspace(9, 12), 3.18+0.52*(linspace(9, 12)-11.), 'k-', lw=3)
    plt.plot(linspace(9, 12), 3.23+0.67*(linspace(9, 12)-11.), 'k--', lw=3)
    plt.xlabel(r"$\log\rm\,M_\ast/M_\odot$", fontsize=16)
    plt.ylabel(r"$\log\rm\,j_\ast/km\,s^{-1}\,kpc$", fontsize=16)
    plt.show()


if __name__ == "__main__":
    plot_angular_momentum_size_velocity()