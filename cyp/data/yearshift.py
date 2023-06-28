import pandas as pd
import csv


# open yield-CSV
inp = pd.read_csv("H:\\BA\\pycrop-yield-prediction\\data\\soja-serie-1969-2019(3).csv")
inp = pd.DataFrame(inp)

inp_np = inp.to_numpy()
for i in range(len(inp_np[:, 2])):
    inp_np[i, 2] = int(inp_np[i, 2]) + 1

pd.DataFrame(inp_np).to_csv("H:\\BA\\pycrop-yield-prediction\\data\\soja-serie-1969-2019(5).csv")
