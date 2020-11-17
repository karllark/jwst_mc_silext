import matplotlib.pyplot as plt

from astropy.table import Table

if __name__ == "__main__":
    # get the ULLYSES targets
    ulmc = Table.read("data/large-magellanic-cloud/large-magellanic-cloud_O7B5V.csv")
    usmc = Table.read("data/")

    gstypes = ulmc["E(B-V)"] > 0.15
    print(ulmc["Target"][gstypes])

    gstypes = usmc["E(B-V)"] > 0.15
    print(usmc["Target"][gstypes])

    # get MCExt targets (Gordon et al. 2003, 2021)
    lmcext = Table.read("data/lmcext_sample.dat", format="ascii.commented_header")
    smcext = Table.read("data/smcext_sample.dat", format="ascii.commented_header")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        ulmc["V"],
        ulmc["E(B-V)"],
        "bs",
        label="LMC ULLYSES (O7-B5)",
        markerfacecolor="none",
        alpha=0.5,
    )
    ax.plot(
        usmc["V"],
        usmc["E(B-V)"],
        "gs",
        label="SMC ULLYSES (O7-B5)",
        markerfacecolor="none",
        alpha=0.5,
    )
    ax.plot(
        lmcext["V"], lmcext["E(B-V)"], "bo", label="LMC Ext", markersize=11, alpha=0.5
    )
    ax.plot(
        smcext["V"], smcext["E(B-V)"], "go", label="SMC Ext", markersize=11, alpha=0.5
    )
    ax.set_xlabel("V")
    ax.set_ylabel("E(B-V)")
    ax.legend()

    plt.tight_layout()

    plt.show()
