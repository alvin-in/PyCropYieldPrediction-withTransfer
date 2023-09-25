import geopandas as gpd
import numpy
import numpy as np
import matplotlib.pyplot as plt
import geemap
import ee
import torch
from bs4 import BeautifulSoup
from pathlib import Path
import matplotlib as mpl
import pandas as pd
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mapclassify
import matplotlib.colors as colors

# new
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.style.use('seaborn')

PATH = "H:\\BA\\pycrop-yield-prediction\\data"
YIELDFILE = "yield_data_with2020.csv"


def vis_one_year(year):
    ee.Initialize()

    county_region = ee.FeatureCollection("users/nikhilarundesai/cultivos_maiz_sembrada_1314")
    departamentos = ee.FeatureCollection("users/JASPR/Geography/AR/Departamentos")
    gdf = geemap.ee_to_geopandas(county_region)

    with open('H:/BA/pycrop-yield-prediction/data/departamentos.csv', mode='r') as inp:
        reader = csv.reader(inp)
        mydict = {rows[7] + '_' + rows[5]: rows[6].upper() + '_' + rows[4].upper() for rows in reader}
        # print(mydict)

    yield_csv = pd.read_csv(PATH + "\\" + YIELDFILE, encoding='utf8')
    yield_np = yield_csv.to_numpy()

    start_row = find_year_row(yield_csv, year)
    end_row = len(yield_csv) - 1
    if year < 2020:
        mid_row = find_year_row(yield_csv, year + 1)
    else:
        mid_row = end_row
    print(start_row, mid_row, end_row)

    # Initialize dict with 0 entries
    dict = {}
    for i in range(start_row, end_row):
        if not isNaN(yield_np[i, 5]) and not isNaN(yield_np[i, 7]):
            state, county = int(yield_np[i, 5]), int(yield_np[i, 7])

            state = str(state)
            county = str(county)
            dict[mydict[state + '_' + county]] = 0

    # print(yield_np[:, 5])
    for i in range(start_row, end_row):
        if not isNaN(yield_np[i, 5]) and not isNaN(yield_np[i, 7]):
            state, county = int(yield_np[i, 5]), int(yield_np[i, 7])

            state = str(state)
            county = str(county)
            dict[mydict[state + '_' + county]] += yield_np[i, -2]  # *0.03674
        else:
            print(yield_np[i, 5], yield_np[i, 7], yield_np[i, -2])  # *0.03674)

    for key, value in dict.items():
        dict[key] = int(dict[key] / 11)

    print(dict)
    print(len(dict))

    plot_map(dict, gdf, 'Gesamtertrag ' + str(year) + " bis 2020")

    # upload pred_dict and gp_dict to wandb
    pps = pd.DataFrame(dict.items(), columns=['Region', 'Error'])


def plot_map(pred_dict, gdf, title, ):
    plt.rcParams['figure.figsize'] = [6, 7]
    pred_arr = np.array(list(pred_dict.items()))
    provincia_arr = []
    for i in range(len(pred_arr[:, 0])):
        provincia, partido = pred_arr[i, 0].upper().split('_')  # partido = departemento
        pred_arr[i, 0] = partido
        provincia_arr.append(provincia)
    provincia_arr = np.array(provincia_arr).reshape(-1, 1)
    pred_arr = np.hstack((pred_arr, provincia_arr))
    # print(pred_arr)
    pdf = pd.DataFrame(pred_arr, columns=['partido', 'pred_err', 'provincia'])

    np_gdf = gdf.to_numpy()
    np_pdf = pdf.to_numpy()

    np_err = np.zeros([len(np_gdf[:, 1]), 1])
    for i in range(len(np_gdf[:, 1])):
        np_err[i] = numpy.nan
    # print(np_err)
    np_gdf = np.hstack((np_gdf, np_err))
    #  print(np_gdf)
    for g in range(len(np_gdf[:, 2])):
        for p in range(len(np_pdf[:, 0])):
            if np_pdf[p, 0] == np_gdf[g, 3] and np_pdf[p, 2] == np_gdf[g, 4]:
                np_gdf[g, 5] = np_pdf[p, 1]

    gdf = gpd.GeoDataFrame(np_gdf, columns=['geometry', 'campana', 'maizs', 'partido', 'provincia', 'pred_err'])
    gdf['pred2'] = gdf.pred_err
    gdf['pred2'] = gdf['pred2'].astype(float)
    print(gdf[gdf['pred2'] < 0]['pred2'])

    # gdf = pd.concat([gdf, pdf], axis=1)
    # gdf = gpd.GeoDataFrame(gdf)
    # gdf.to_csv("H:\\BA\\pycrop-yield-prediction\\data\\asdf.csv")

    fig, ax = plt.subplots(1, 1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_axis_off()
    # , scheme='user_defined', classification_kwds={'bins': [1, 200000, 400000, 1300000, 2000000]}

    gdf.plot(column='pred2', ax=ax, cmap='RdPu', edgecolor="black", legend=True, scheme='user_defined',
             missing_kwds={"color": "lightgrey", "edgecolor": "darkgrey", "hatch": "///", "label": "No data",
                           }, classification_kwds={'bins': [2000, 10000]})
    plt.rcParams.update({'font.size': 36})
    plt.title('Average number of cropland pixel over the 11 years from 2010 to 2020')
    plt.show()


def find_year_row(df, year):
    for i in range(len(df["anio"])):
        if int(df["anio"][i]) == year:
            return i


def isNaN(num):
    return num != num


vis_one_year(2010)
