import argparse
import matplotlib.pyplot as plt
from astropy.table import Table

if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get the ULLYSES targets
    ulmc = Table.read("data/lmc_sil_sample.dat", format="ascii.commented_header")
    usmc = Table.read("data/smc_sil_sample.dat", format="ascii.commented_header")
    names = ["SMC", "LMC"]
    types = ["o", "s"]
    tags = {"smc": ["Avg", "Bump"], "lmc": ["Avg", "30Dor"]}

    fontsize = 18

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

    for k, ctab in enumerate((usmc, ulmc)):
        cname = names[k]
        ctype = types[k]
        extcols = ["r", "b"]
        for l, ctag in enumerate(tags[cname.lower()]):
            cecol = extcols[l]
            ext = [f"E{l+1}" in cnote for cnote in ctab["Notes"]]
            ax[k].plot(
                ctab["K"][ext],
                ctab["E(B-V)"][ext],
                f"{cecol}{ctype}",
                label=f"{cname} Ext {ctag}",
                alpha=0.5,
                markerfacecolor="none",
                markersize=10,
            )
        ext = [("J17" in cnote) or ("R20" in cnote) for cnote in ctab["Notes"]]
        ax[k].plot(
            ctab["K"][ext],
            ctab["E(B-V)"][ext],
            f"g{ctype}",
            label=f"{cname} DepPub",
            alpha=0.5,
            markerfacecolor="none",
            markersize=14,
        )
        ext = ["U" in cnote for cnote in ctab["Notes"]]
        ax[k].plot(
            ctab["K"][ext],
            ctab["E(B-V)"][ext],
            f"c{ctype}",
            label=f"{cname} ULYSSES",
            alpha=0.5,
            markerfacecolor="none",
            markersize=18,
        )

        ax[k].set_xlim(10.5, 14.2)
        ax[k].set_ylim(0.15, 0.50)
        ax[k].set_xlabel("K")
        ax[k].set_ylabel("E(B-V)")
        ax[k].legend(fontsize=0.8 * fontsize)

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    plt.tight_layout()

    basesave = "ebv_kmag_silsamp"
    if args.png:
        fig.savefig(f"{basesave}.png")
    elif args.pdf:
        fig.savefig(f"{basesave}.pdf")
    else:
        plt.show()
