import numpy as np
import pandas as pd


PATH = "/pycrop-yield-prediction/data"
YIELDFILE = "soja-serie-1969-2019_clean_bu.csv"
COVERPIX = "year_pix_counter(2).csv"
OUTFIlE = "soja-serie-1969-2019(257).csv"
cols = ["cultivo_nombre", "anio", "campania", "provincia_nombre", "provincia_id", "departamento_nombre",
        "departamento_id", "superficie_sembrada_ha", "superficie_cosechada_ha", "produccion_tm", "redimiento_kgxha",
        "redimiento_buxacre", "mask_field_pixel"
        ]
col_with_names = ["provincia_nombre", "departamento_nombre"]

yield_csv = pd.read_csv(PATH+"\\"+YIELDFILE, encoding='1252')
for col in col_with_names:
    yield_csv[col] = yield_csv[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

yield_np = yield_csv.to_numpy()

print('Calculates bushles per acre...')
buxacre = np.zeros_like(yield_np[:, -1])
k = []
for i in range(len(yield_np[:, -1])):
    if yield_np[i, -1] == "SD":
        k.append(i)
    else:
        buxacre[i] = float(yield_np[i, -1]) * 0.01478

yield_np = np.hstack((yield_np, buxacre.reshape(-1, 1)))
for j in k:
    yield_np = np.delete(yield_np, j, 0)
    for i in range(len(k)):
        k[i] -= 1

cov_pix_np = pd.read_csv(PATH+"\\"+COVERPIX).to_numpy()
covpix = np.zeros_like(yield_np[:, -1])
relative_covpix = np.zeros_like(yield_np[:, -1])

print('Appends cropland cover to csv. This could take a few seconds...')

for i in range(len(cov_pix_np[:, 0])):
    for j in range(len(yield_np[:, 0])):
        if cov_pix_np[i, 1] == yield_np[j, 1] and cov_pix_np[i, 2] == yield_np[j, 6]:
            covpix[j] = cov_pix_np[i, 3]
            # if covpix[j] > 0:
            #     relative_covpix[j] = yield_np[j, 8]
            #     relative_covpix[j] /= cov_pix_np[i, 3]*25

yield_np = np.hstack((yield_np, covpix.reshape(-1, 1)))
# yield_np = np.hstack((yield_np, relative_covpix.reshape(-1, 1)))

yield_df = pd.DataFrame(yield_np, columns=cols)
yield_df.to_csv(PATH+"\\"+OUTFIlE)
