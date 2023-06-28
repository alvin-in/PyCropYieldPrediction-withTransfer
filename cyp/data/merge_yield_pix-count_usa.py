import numpy as np
import pandas as pd

PATH = "/pycrop-yield-prediction/data"
YIELDFILE = "abs_yield_usa.csv"
COVERPIX = "usa_pix_counter.csv"
OUTFIlE = "abs_usa_yield_with_pix.csv"
cols = ["Program", "Year", "Period", "Week Ending", "Geo Level", "State", "State ANSI", "Ag District",
        "Ag District Code", "County", "County ANSI", "Zip Code", "Region", "watershed_code", "Watershed", "Commodity",
        "Data Item", "Domain", "Domain Category", "Value", "CV (%)", "mask_field_pixel"
        ]


def isNaN(num):
    return num != num


yield_csv = pd.read_csv(PATH + "\\" + YIELDFILE)
yield_np = yield_csv.to_numpy()

cov_pix_np = pd.read_csv(PATH + "\\" + COVERPIX).to_numpy()
covpix = np.zeros_like(yield_np[:, -1])
relative_covpix = np.zeros_like(yield_np[:, -1])

print('Appends cropland cover to csv. This could take a few seconds...')

for i in range(len(cov_pix_np[:, 0])):
    for j in range(len(yield_np[:, 0])):
        if not isNaN(yield_np[j, 6]) and not isNaN(yield_np[j, 10]):
            if cov_pix_np[i, 1] == yield_np[j, 1] and str(cov_pix_np[i, 2]) == str(int(yield_np[j, 6])) + "_" + str(int(yield_np[j, 10])):
                covpix[j] = cov_pix_np[i, 3]
                # if covpix[j] > 0:
                #     relative_covpix[j] = yield_np[j, 8]
                #     relative_covpix[j] /= cov_pix_np[i, 3]*25

yield_np = np.hstack((yield_np, covpix.reshape(-1, 1)))
# yield_np = np.hstack((yield_np, relative_covpix.reshape(-1, 1)))

yield_df = pd.DataFrame(yield_np, columns=cols)
yield_df.to_csv(PATH + "\\" + OUTFIlE)
