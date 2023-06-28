import numpy as np
import pandas as pd
import csv


# Helper methods
def isNaN(num):
    return num != num


def find_year_row(np_array, index_of_years, year):
    for row in range(len(np_array[:, index_of_years])):
        if int(np_array[row, index_of_years]) == year:
            return row


# Loads yield-csv (given by datos.gob.ar), encodes to UTF-8 and adds bushles per acre. Algins ID's to departamentos-csv.
# Output is a CSV-File, which get safed in input directory.
PATH = "H:/BA/pycrop-yield-prediction/data"
FILE = 'estimaciones_clean.csv'    # "soja-serie-1969-2019_clean_bu.csv"
OUTFIlE = 'yield_data_with2020.csv'     # "soja-serie-1969-2019(3).csv"
DEPARTAMENTOS = 'H:/BA/pycrop-yield-prediction/data/departamentos.csv'
COVERPIX = "year_pix_counter(2).csv"
pixcount = True
start_year = 2010
min_covpix = 2000
min_soyratio = 0
cols = ["cultivo_nombre", "anio", "campania", "provincia_nombre", "provincia_id", "departamento_nombre",
        "departamento_id", "superficie_sembrada_ha", "superficie_cosechada_ha", "produccion_tm", "redimiento_kgxha",
        "redimiento_buxacre", "mask_field_pixel", "soy_satfield_ratio"]
col_with_name = ["provincia_nombre", "departamento_nombre"]

yield_csv = pd.read_csv(PATH + "/" + FILE, encoding='1252')
# removes accents
for col in col_with_name:
    yield_csv[col] = yield_csv[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

yield_np = yield_csv.to_numpy()
buxacre = np.zeros_like(yield_np[:, -1])

# Deletes rows with yield entry (last row in input) "SD" and calculates bushles per acre.
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

# running second script (adjust_yield-csv_to_departamentos-csv.py):
# Algins ID's of yield-csv to departamentos-csv.
with open(DEPARTAMENTOS, mode='r') as dep:
    reader = csv.reader(dep)
    mydict = {rows[6].upper() + '_' + rows[4].upper(): rows[7] + '_' + rows[5] for rows in reader}

for i in range(len(yield_np[:, 3])):
    s_c = (yield_np[i, 3] + '_' + yield_np[i, 5]).upper()
    if mydict.get(s_c) is not None:
        yield_np[i, 6] = mydict[s_c].split('_')[1]

# Delete NaNs
for i in range(len(yield_np[:, 6])):
    if isNaN(yield_np[i, 6]):
        k.append(i)
for j in k:
    yield_np = np.delete(yield_np, j, 0)
    for i in range(len(k)):
        k[i] -= 1

# Merge pixel count and yield. Crops the Dataframe to [start_year:]
if pixcount:
    start_row = find_year_row(yield_np, 1, start_year)
    yield_np = yield_np[start_row:, :]
    cov_pix_np = pd.read_csv(PATH + "\\" + COVERPIX).to_numpy()
    covpix = np.zeros_like(yield_np[:, -1])
    relative_covpix = np.zeros_like(yield_np[:, -1])

    print('Appends cropland cover to csv. This could take a few seconds...')

    for i in range(len(cov_pix_np[:, 0])):
        for j in range(len(yield_np[:, 0])):
            if int(cov_pix_np[i, 1]) == int(yield_np[j, 1]) and int(cov_pix_np[i, 2]) == int(yield_np[j, 6]):
                covpix[j] = cov_pix_np[i, 3]
                if covpix[j] > 0:
                    relative_covpix[j] = yield_np[j, 8]
                    relative_covpix[j] /= cov_pix_np[i, 3] * 25

    yield_np = np.hstack((yield_np, covpix.reshape(-1, 1)))
    yield_np = np.hstack((yield_np, relative_covpix.reshape(-1, 1)))

    # Removes data with less than min_covpix field pixel given by satellite mask (here: COVERPIX).
    _rm_cov = []
    for i in range(len(covpix)):
        if covpix[i] < min_covpix:
            _rm_cov.append(i)
    for row in _rm_cov:
        yield_np = np.delete(yield_np, row, 0)
        for i in range(len(_rm_cov)):
            _rm_cov[i] -= 1

    # Removes data with less than min_soyratio soybean to field ratio.
    _rm_soyratio = []
    for i in range(len(yield_np)):
        if yield_np[i, -1] < min_soyratio:
            _rm_soyratio.append(i)
    for row in _rm_soyratio:
        yield_np = np.delete(yield_np, row, 0)
        for i in range(len(_rm_soyratio)):
            _rm_soyratio[i] -= 1

# Output
pd.DataFrame(yield_np, columns=cols).to_csv(PATH + "\\" + OUTFIlE, encoding='utf-8')
