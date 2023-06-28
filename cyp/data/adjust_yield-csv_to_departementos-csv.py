import pandas as pd
import csv


# open yield-CSV
inp = pd.read_csv("/pycrop-yield-prediction/data/soja-serie-1969-2019(3).csv")
inp = pd.DataFrame(inp)

# open State- and County-ID-CSV and transform to {state_name-county_name: state-id_county-id: }-dict
with open('/pycrop-yield-prediction/data/departamentos.csv', mode='r') as dep:
    reader = csv.reader(dep)
    mydict = {rows[6].upper()+'_'+rows[4].upper(): rows[7]+'_'+rows[5] for rows in reader}

# transform yield-CSV to numpy-array for easier manipulation
inp_np = inp.to_numpy()
for i in range(len(inp_np[:, 4])):
    s_c = (inp_np[i, 4]+'_'+inp_np[i, 6]).upper()
    if mydict.get(s_c) is not None:
        inp_np[i, 7] = mydict[s_c].split('_')[1]

pd.DataFrame(inp_np).to_csv("/pycrop-yield-prediction/data/soja-serie-1969-2019(4).csv")
