import matplotlib.pyplot as plt
import geemap
import ee
import csv
import numpy as np

plt.style.use('seaborn')

PATH = "H:\\BA\\pycrop-yield-prediction\\data\\"
YIELDFILE = "usa_yield_with_pix2.csv"


def vis():
    ee.Initialize()

    county_region = ee.FeatureCollection("TIGER/2018/Counties")
    gdf = geemap.ee_to_geopandas(county_region)
    yield_dict = {}

    with open(PATH + YIELDFILE, mode='r') as inp:
        reader = csv.reader(inp)
        next(reader)
        for rows in reader:
            yield_dict[str(rows[8].zfill(2)) + '_' + str(rows[12].zfill(3)) + '_' + str(rows[3])] = \
                float(rows[21].replace(',', ''))

        # state-ansi_county-ansi_year -> 21 for yield, 23 for pix - depending on csv
        print(yield_dict)

    arr = []
    for i in range(len(gdf['STATEFP'])):
        sum = 0
        for j in range(2010, 2021):
            try:
                sum += yield_dict[str(gdf['STATEFP'][i]) + '_' + str(float(gdf['COUNTYFP'][i])) + '_' + str(j)]
            except:
                sum += 0
        arr.append(sum / 11)

    gdf['yield'] = arr
    gdf.replace(0, np.nan, inplace=True)
    print(gdf[gdf['yield'].notna()])

    plt.rcParams.update({'font.size': 40})
    fig, ax = plt.subplots(1, 1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_axis_off()
    # , scheme='user_defined', classification_kwds={'bins': [1, 200000, 400000, 1300000, 2000000]}

    gdf.plot(column='yield', ax=ax, cmap='YlGn', edgecolor="black", legend=True, scheme='quantiles',
             missing_kwds={"color": "lightgrey", "edgecolor": "darkgrey", "hatch": "///", "label": "0"})
    # classification_kwds={'bins': [1000, 2000, 5000, 10000]})
    plt.title('Average yield by area over the 11 years from 2010 to 2020 in bushels per acre')
    plt.show()


def find_year_row(df, year):
    for i in range(len(df["Year"])):
        if int(df["Year"][i]) == year:
            return i


def isNaN(num):
    return num != num


vis()
