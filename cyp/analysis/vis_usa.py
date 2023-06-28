import matplotlib.pyplot as plt
import geemap
import ee
import csv

plt.style.use('seaborn')

PATH = "H:\\BA\\pycrop-yield-prediction\\data\\"
YIELDFILE = "abs_usa_yield_with_pix.csv"


def vis():
    ee.Initialize()

    county_region = ee.FeatureCollection("TIGER/2018/Counties")
    gdf = geemap.ee_to_geopandas(county_region)
    yield_dict = {}

    with open(PATH + YIELDFILE, mode='r') as inp:
        reader = csv.reader(inp)
        next(reader)
        for rows in reader:
            yield_dict[str(rows[7].zfill(2)) + '_' + str(rows[11].zfill(3)) + '_' + str(rows[2])] =\
                float(rows[20].replace(',', ''))

        # state-ansi_county-ansi_year -> 20 for yield, 22 for pix
        # 2. Wert seltsamerweise float(?), daher cast des Vergleichs auf float
        print(yield_dict)

    arr = []
    for i in range(len(gdf['STATEFP'])):
        sum = 0
        for j in range(2010, 2020):
            try:
                sum += yield_dict[str(gdf['STATEFP'][i]) + '_' + str(float(gdf['COUNTYFP'][i])) + '_' + str(j)]
            except:
                sum += 0
        arr.append(sum/10)

    gdf['yield'] = arr
    print(gdf)

    plt.rcParams.update({'font.size': 40})
    fig, ax = plt.subplots(1, 1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_axis_off()
    # , scheme='user_defined', classification_kwds={'bins': [1, 200000, 400000, 1300000, 2000000]}

    gdf.plot(column='yield', ax=ax, cmap='YlGn', edgecolor="black", legend=True, scheme='quantiles')
             #classification_kwds={'bins': [600, 2000]})
    plt.title('Durchschnittliche Anzahl der Feldpixel der Jahre 2010 bis 2019')
    plt.show()


def find_year_row(df, year):
    for i in range(len(df["anio"])):
        if int(df["anio"][i]) == year:
            return i


def isNaN(num):
    return num != num


vis()
