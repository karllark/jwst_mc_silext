import numpy as np

from astroquery.vizier import Vizier
import astropy.units as u
from astropy.table import QTable


def get_irac_phot(filename, galname="smc"):

    jtable = {"lmc": "J/AJ/138/1003/table3", "smc": "J/AJ/140/416/table3"}
    kcol = {"lmc": "Ksmag", "smc": "Kmag"}

    format = None
    if "csv" not in filename:
        format = "ascii.commented_header"
    exttab = QTable.read(filename, format=format)

    if "Name" not in exttab.colnames:
        exttab.rename_column("Target", "Name")

    ntable = QTable()

    N = len(exttab)
    dtype = [
        ("Name", "U20"),
        ("E(B-V)", "float"),
        ("V", "float"),
        ("K", "float"),
        ("IRAC3", "float"),
        ("IRAC4", "float"),
        ("K-IRAC3", "float"),
        ("K-IRAC4", "float"),
    ]
    ntable = QTable(data=np.zeros(N, dtype=dtype))

    for i in range(N):
        cname = exttab["Name"][i]
        if cname[0:2] == "sk":
            cname = cname.replace("sk", "sk -").replace("d", " ")

        # get the IRAC photometry for the sources
        result = Vizier.query_object(
            cname, catalog=jtable[galname], radius=1 * u.arcsec
        )
        if len(result) > 0:
            k = result[jtable[galname]][kcol[galname]]
            i3 = result[jtable[galname]]["__5.8_"]
            i4 = result[jtable[galname]]["__8.0_"]
            if np.ma.is_masked(k):
                k = 0.0
            if np.ma.is_masked(i3):
                i3 = 0.0
            if np.ma.is_masked(i4):
                i4 = 0.0
            # print(cname, i3, i4)
        else:
            k = 0.0
            i3 = 0.0
            i4 = 0.0

        v = exttab["V"][i]
        ebv = exttab["E(B-V)"][i]

        if (i3 > 0) and (k > 0):
            ki3 = k - i3
        else:
            ki3 = 10.0
        if (i4 > 0) and (k > 0):
            ki4 = k - i4
        else:
            ki4 = 10.0
        ntable[i] = (cname, ebv, v, k, i3, i4, ki3, ki4)
    return ntable


if __name__ == "__main__":

    fnames = [
        "data/large-magellanic-cloud/large-magellanic-cloud_O7B5.csv",
        "data/lmcext_sample.dat",
        "data/small-magellanic-cloud/small-magellanic-cloud_O7B5.csv",
        "data/smcext_sample.dat",
    ]
    ftypes = ["lmc", "lmc", "smc", "smc"]
    onames = [
        "data/lmc_ulysses.fits",
        "data/lmc_uvext.fits",
        "data/smc_ulysses.fits",
        "data/smc_uvext.fits",
    ]

    for i, cfname in enumerate(fnames):
        ntable = get_irac_phot(cfname, galname=ftypes[i])
        ntable.write(onames[i], overwrite=True)
        grows = np.logical_and(
            np.logical_and(
                np.logical_or(ntable["IRAC4"] > 0.1, ntable["IRAC3"] > 0.1),
                ntable["E(B-V)"] > 0.15,
            ),
            np.logical_or(ntable["K-IRAC3"] < 0.3, ntable["K-IRAC3"] < 0.4)
        )
        print(onames[i])
        print(ntable[grows])
