import geopandas as gpd
import numpy
import numpy as np
import matplotlib.pyplot as plt
import geemap
import ee
import torch
from pathlib import Path
import pandas as pd
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wandb

plt.style.use('seaborn')


def vis_arg(years):
    for year in years:
        vis_one_year(year)


def vis_one_year(year):
    ee.Initialize()

    county_region = ee.FeatureCollection("users/nikhilarundesai/cultivos_maiz_sembrada_1314")
    gdf = geemap.ee_to_geopandas(county_region)

    # print(gdf)

    model = Path("H:\\BA\\pycrop-yield-prediction\\data\\arg_models\\cnn\\" + str(year) + "_1_32_gp.pth.tar")

    model_sd = torch.load(model, map_location="cpu")

    # model_dir = model.parents[0]

    real_values = model_sd["test_real"]
    pred_values = model_sd["test_pred"]
    gp = True
    try:
        gp_values = model_sd["test_pred_gp"]
    except KeyError:
        gp = False

    indices = model_sd["test_indices"]

    with open('H:/BA/pycrop-yield-prediction/data/departamentos.csv', mode='r') as inp:
        reader = csv.reader(inp)
        mydict = {rows[7] + '_' + rows[5]: rows[6].upper() + '_' + rows[4].upper() for rows in reader}
        # print(mydict)

    pred_err = pred_values - real_values
    pred_err_rel = (pred_values - real_values) / real_values
    pred_dict = {}
    pred_rel_dict = {}
    for idx, err in zip(indices, pred_err):
        state, county = idx

        state = str(state)
        county = str(county)

        pred_dict[mydict[state + '_' + county]] = err

    for idx, err in zip(indices, pred_err_rel):
        state, county = idx

        state = str(state)
        county = str(county)

        pred_rel_dict[mydict[state + '_' + county]] = err

    if gp:
        gp_pred_err = gp_values - real_values
        gp_pred_err_rel = (gp_values - real_values) / real_values
        gp_dict = {}
        gp_dict_rel = {}
        for idx, err in zip(indices, gp_pred_err):
            state, county = idx

            state = str(state)
            county = str(county)

            gp_dict[mydict[state + '_' + county]] = err

        for idx, err in zip(indices, gp_pred_err_rel):
            state, county = idx

            state = str(state)
            county = str(county)

            gp_dict_rel[mydict[state + '_' + county]] = err

    print(pred_dict)
    print(len(pred_dict))

    plot_map(pred_dict, gdf, 'CNN_' + str(year))
    plot_map(pred_rel_dict, gdf, 'CNN_relative_' + str(year), rel=True)
    if gp:
        plot_map(gp_dict, gdf, 'CNN_GP_' + str(year))
        plot_map(gp_dict_rel, gdf, 'CNN_GP_relative_' + str(year), rel=True)

    # upload pred_dict and gp_dict to wandb
    pps = pd.DataFrame(pred_dict.items(), columns=['Region', 'Error'])
    wandb.log({"pred_per_states_" + str(year): pps})
    if gp:
        pps_gp = pd.DataFrame(gp_dict.items(), columns=['Region', 'Error'])
        wandb.log({"pred_gp_per_states_" + str(year): pps_gp})


def plot_map(pred_dict, gdf, title, rel=False):
    plt.rcParams['figure.figsize'] = [10, 10]
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

    # gdf = pd.concat([gdf, pdf], axis=1)
    # gdf = gpd.GeoDataFrame(gdf)
    # gdf.to_csv("H:\\BA\\pycrop-yield-prediction\\data\\asdf.csv")

    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_axis_off()
    if not rel:
        gdf.plot(column='pred2', ax=ax, cmap='bwr', edgecolor="black", vmin=-30, vmax=30,
                 legend_kwds={'label': title}, legend=True, cax=cax,
                 missing_kwds={"color": "lightgrey", "edgecolor": "darkgrey", "hatch": "///",
                               "label": "Missing values", })
    else:
        gdf.plot(column='pred2', ax=ax, cmap='bwr', edgecolor="black", vmin=-1.0, vmax=1.0,
                 legend_kwds={'label': title}, legend=True, cax=cax,
                 missing_kwds={"color": "lightgrey", "edgecolor": "darkgrey", "hatch": "///",
                               "label": "Missing values", })
        gdf['pred2'] = gdf['pred2'].round(2)
        gdf['pred2'] = gdf['pred2'].replace(np.nan, " ")
        # gdf.apply(lambda x: ax.annotate(
        #     text=x['pred2'], xy=x.geometry.centroid.coords[0], ha='center', fontsize=7), axis=1)
    wandb.log({title: wandb.Image(plt)})
