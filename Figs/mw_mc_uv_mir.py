import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import astropy.units as u

from dust_extinction.averages import G21
from dust_extinction.parameter_averages import F19
from dust_extinction.averages import G03_SMCBar, G03_LMCAvg, G03_LMC2, GCC09_MWAvg
from measure_extinction.extdata import ExtData
from measure_extinction.merge_obsspec import _wavegrid


def plot_obsext(ax, obsext, alpha=1.0, color="k"):

    obsext_wave = obsext.waves["BAND"].value

    obsext_ext = obsext.exts["BAND"]
    obsext_ext_uncs = obsext.uncs["BAND"]

    gindxs_IRS = np.where(obsext.npts["IRS"] > 0)
    obsext_IRS_wave = obsext.waves["IRS"][gindxs_IRS].value
    obsext_IRS_ext = obsext.exts["IRS"][gindxs_IRS]
    obsext_IRS_uncs = obsext.uncs["IRS"][gindxs_IRS]

    ax.plot(
        obsext_IRS_wave,
        obsext_IRS_ext,
        f"{color}-",  # pcol[i] + psym[i],
        markersize=5,
        markeredgewidth=1.0,
        alpha=alpha,
        label="MWAvg",
    )

    # rebin IRS
    wrange = [5.0, 36.0]
    res = 25
    full_wave, full_wave_min, full_wave_max = _wavegrid(res, wrange)
    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)

    cwaves = obsext_IRS_wave
    cfluxes = obsext_IRS_ext
    cuncs = obsext_IRS_uncs
    for k in range(n_waves):
        (indxs,) = np.where((cwaves >= full_wave_min[k]) & (cwaves < full_wave_max[k]))
        if len(indxs) > 0:
            # weights = 1.0 / np.square(cuncs[indxs])
            weights = 1.0
            full_flux[k] += np.sum(weights * cfluxes[indxs])
            full_unc[k] += np.sum(1.0 / np.square(cuncs[indxs]))
            full_npts[k] += len(indxs)

    findxs = full_npts > 0
    full_flux[findxs] /= full_npts[findxs]
    full_unc[findxs] = np.sqrt(1.0 / full_unc[findxs])

    # ax.errorbar(
    #    full_wave[findxs],
    #    full_flux[findxs],
    #    yerr=full_unc[findxs],
    #    fmt="b-",  # pcol[i] + psym[i],
    #    markersize=5,
    #    markeredgewidth=1.0,
    #    alpha=1.0,
    # )


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 16

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 10))

    # mw
    Rvs = [2.0, 3.0, 5.0]
    mod_lam = np.logspace(np.log10(0.13), np.log10(2.4), num=500) * u.micron
    lstyles = [":", "--", "-."]
    for k, cRv in enumerate(Rvs):
        mwmod = F19(cRv)
        ax[0, 0].plot(
            mod_lam,
            mwmod(mod_lam),
            label=f"F19 MW Rv={cRv}",
            color="k",
            alpha=0.5,
            linestyle=lstyles[k],
        )
    mw1 = GCC09_MWAvg()
    ax[0, 0].plot(1.0 / mw1.obsdata_x, mw1.obsdata_axav, "b-", label="GCC09 MWAvg")

    obsext = ExtData()
    obsext.read(
        "data/all_ext_14oct20_diffuse_ave_POWLAW2DRUDE.fits"
    )
    plot_obsext(ax[0, 1], obsext, color="b")
    ax[0, 2].plot([obsext.g21_p50_fit["SIL1_CENTER"][0]], [obsext.g21_p50_fit["SIL1_FWHM"][0]],
                  "bo", label="MWAvg")
    ax[0, 1].legend(fontsize=0.8 * fontsize)
    ax[0, 2].legend(fontsize=0.8 * fontsize)

    # read in the individual curves and plot their fits
    xvals = np.arange(1.0, 30.0, 0.1) * u.micron
    ifiles = glob.glob("indiv/*.fits")
    npts = len(ifiles)
    center = np.zeros(npts)
    width = np.zeros(npts)
    for k, cfile in enumerate(ifiles):
        obsext = ExtData(filename=cfile)
        center[k] = obsext.g21_p50_fit["SIL1_CENTER"][0]
        width[k] = obsext.g21_p50_fit["SIL1_FWHM"][0]

        gmod = G21(scale=obsext.g21_p50_fit["SCALE"][0],
                   alpha=obsext.g21_p50_fit["ALPHA"][0],
                   sil1_amp=obsext.g21_p50_fit["SIL1_AMP"][0],
                   sil1_center=obsext.g21_p50_fit["SIL1_CENTER"][0],
                   sil1_fwhm=obsext.g21_p50_fit["SIL1_FWHM"][0],
                   sil1_asym=obsext.g21_p50_fit["SIL1_ASYM"][0],
                   sil2_amp=obsext.g21_p50_fit["SIL2_AMP"][0],
                   sil2_center=obsext.g21_p50_fit["SIL2_CENTER"][0],
                   sil2_fwhm=obsext.g21_p50_fit["SIL2_FWHM"][0],
                   sil2_asym=obsext.g21_p50_fit["SIL2_ASYM"][0])
        ax[0, 1].plot(xvals, gmod(xvals), "k-", alpha=0.25)
        # obsext.trans_elv_alav()
        # plot_obsext(ax[0, 1], obsext, alpha=0.1, color="r-")

    ax[0, 2].plot(center, width, "ko", alpha=0.25)

    # lmc
    lmc1 = G03_LMCAvg()
    lmc2 = G03_LMC2()
    ax[1, 0].plot(1.0 / lmc1.obsdata_x, lmc1.obsdata_axav, "g-", label="G03 LMCAvg")
    ax[1, 0].plot(1.0 / lmc2.obsdata_x, lmc2.obsdata_axav, label="G03 LMC2 (30 Dor)")

    ax[1, 1].plot(1.0 / lmc1.obsdata_x, lmc1.obsdata_axav, "c-", label="G03 LMCAvg")
    ax[1, 1].plot(1.0 / lmc2.obsdata_x, lmc2.obsdata_axav, label="G03 LMC2 (30 Dor)")
    ax[1, 1].text(
        10.0,
        0.05,
        "this proposal",
        fontsize=30,
        verticalalignment="center",
        horizontalalignment="center",
    )

    ax[1, 2].text(
        9.8,
        2.7,
        "this proposal",
        fontsize=30,
        verticalalignment="center",
        horizontalalignment="center",
    )

    # smc
    smc1 = G03_SMCBar()
    ax[2, 0].plot(1.0 / smc1.obsdata_x, smc1.obsdata_axav, "r-", alpha=0.5, label="G03 SMCBar")
    ax[2, 0].set_xlabel(r"$\lambda [\mu m]$")

    ax[2, 1].plot(1.0 / smc1.obsdata_x, smc1.obsdata_axav, label="G03 LMCBar")
    ax[2, 1].text(
        10,
        0.05,
        "this proposal",
        fontsize=30,
        verticalalignment="center",
        horizontalalignment="center",
    )
    ax[2, 1].set_xlabel(r"$\lambda [\mu m]$")

    ax[2, 2].text(
        9.8,
        2.7,
        "this proposal",
        fontsize=30,
        verticalalignment="center",
        horizontalalignment="center",
    )
    ax[2, 2].set_xlabel(r"$\lambda_{o1} [\mu m]$")

    for k in range(3):
        for l in range(2):
            ax[k, l].set_ylabel(r"$A(\lambda)/A(V)$")

        ax[k, 2].set_ylabel(r"$\gamma_{o1} [\mu m]$")

        ax[k, 0].set_xscale("log")
        ax[k, 0].set_xlim(0.1, 3.0)
        ax[k, 0].set_ylim(0.0, 7.0)
        ax[k, 0].xaxis.set_major_formatter(ScalarFormatter())
        # ax[k, 0].xaxis.set_minor_formatter(ScalarFormatter())

        ax[k, 1].set_xlim(5.0, 15.0)
        ax[k, 1].set_ylim(0.0, 0.1)
        # ax[k, 1].yaxis.tick_right()
        # ax[k, 1].yaxis.set_label_position("right")

        ax[k, 2].set_xlim(9.6, 10.0)
        ax[k, 2].set_ylim(1.0, 4.5)

        ax[k, 0].legend(fontsize=0.8 * fontsize)

    plt.tight_layout()

    basesave = "mw_mc_uv_mir"
    if args.png:
        fig.savefig(f"{basesave}.png")
    elif args.pdf:
        fig.savefig(f"{basesave}.pdf")
    else:
        plt.show()
