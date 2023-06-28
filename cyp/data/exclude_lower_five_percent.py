import pandas as pd
import numpy as np
import csv


def isNaN(num):
    return num != num


# open yield-CSV
inp = pd.read_csv("/pycrop-yield-prediction/data/soja-serie-1969-2019(3)_jahresshift_comp.csv")
inp_df = pd.DataFrame(inp)
inp_df = inp_df[7903:]  # from 2010

print(inp_df)

inp_np = inp_df.to_numpy()
yield_sum = np.sum(inp_np[:, 12])

my_dict = {}
for i in range(7903, 7903 + len(inp_df)):
    if not isNaN(inp_df["departamento_id"][i]):
        my_dict[int(inp_df["departamento_id"][i])] = 0

for key, value in my_dict.items():
    for i in range(7903, 7903 + len(inp_df)):
        if key == inp_df["departamento_id"][i]:
            my_dict[key] = my_dict[key] + inp_df["redimiento_buxacre"][i]

arr = []
for key, value in my_dict.items():
    arr.append((key, value))

dtype = [('key', float), ('value', float)]
np_dict = np.array(arr, dtype=dtype)
dict_sum = np.sum(np_dict['value'])
five_perc = dict_sum/20

np_sorted_dict = np.sort(np_dict, order='value')

sum_counter = 0
i = 0
while sum_counter <= five_perc:
    sum_counter += np_sorted_dict['value'][i]
    i += 1

print(sum_counter)
print(sum_counter/dict_sum)

dep_out = np_sorted_dict[:i]   # Bezirke mit weniger als 5% Gesamtertrag ab 2010
dep_in = np_sorted_dict[i:]
assert(len(dep_in)+len(dep_out) == len(np_sorted_dict))

new_arr = []
for j in range(len(inp_df)):
    for (key, value) in dep_in:
        if key == inp_df["departamento_id"][j+7903]:
            new_arr.append(inp_np[j])

new_df = pd.DataFrame(new_arr, columns=['Column1', 'cultivo_nombre', 'anio', 'campania', 'provincia_nombre',
                                        'provincia_id', 'departamento_nombre', 'departamento_id',
                                        'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm',
                                        'rendimiento_kgxha', 'redimiento_buxacre', 'departamento_field_pixel'])

#new_df.to_csv("H:\\BA\\pycrop-yield-prediction\\data\\soja-serie-1969-2019(10).csv")
