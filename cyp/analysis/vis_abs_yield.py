import matplotlib as mpl
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
from scipy import stats

mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt


PATH = "H:\\BA\\pycrop-yield-prediction\\data"
YIELDFILE = "yield_data_with2020.csv"
YIELDFILE_USA = "usa_yield_with_pix2.csv"

def myfunc(x):
    return slope * x + intercept


yield_df = pd.read_csv(PATH + "\\" + YIELDFILE, encoding='utf8')
usa_yield_df = pd.read_csv(PATH + "\\" + YIELDFILE_USA)
sum_per_year = []
abs_sum_per_year = []
std = []
usa_sum = []
std_usa = []

x = np.array(range(2010, 2021))

for i in x:
    sum_per_year.append(yield_df[yield_df['anio'] == i]["redimiento_buxacre"].mean())
    std.append(yield_df[yield_df['anio'] == i]["redimiento_buxacre"].std())
    usa_sum.append(usa_yield_df[usa_yield_df['Year'] == i]['Value'].mean())
    std_usa.append(usa_yield_df[usa_yield_df['Year'] == i]['Value'].std())
    # abs_sum_per_year.append(yield_df[yield_df['anio'] == i]["produccion_tm"].sum() * 0.03674)

slope, intercept, r, p, std_err = stats.linregress(x, sum_per_year)
mymodel = list(map(myfunc, x))
slope, intercept, r, p, std_err = stats.linregress(x, usa_sum)
mymodel_usa = list(map(myfunc, x))

plt.rcParams.update({'font.size': 20})
fig, par1 = plt.subplots()
# par1 = ax.twinx()
width = 0.35
# ax.bar(x - width/2, std, width, label="std", color='darkgreen')
# par1.plot(x, sum_per_year, color='darkgreen', marker='o', linestyle='None')  # label="Average yield by area in bu/acre",
# y_formatter = ScalarFormatter(useOffset=False)
# ax.yaxis.set_major_formatter(y_formatter)
par1.set_xticks(x, x)
par1.set_ylabel("Average yield by area and standard deviation \n in bushels per acre")
# ax.set_ylabel("Average yield in bushels")
fig.tight_layout()
par1.set_xlabel("Year of the beginning of the season")
par1.errorbar(x+width/2, sum_per_year, yerr=std, fmt="o", color="deepskyblue", linewidth=4, label="Argentina")
par1.errorbar(x-width/2, usa_sum, yerr=std, fmt="o", color="red", linewidth=4, label="USA")
print(usa_sum)
plt.plot(x, mymodel, label="Argentina Trend", color="navy", linewidth=4)
plt.plot(x, mymodel_usa, label="USA Trend", color="maroon", linewidth=4)
for i in x:
    plt.axvspan(i-.33, i+.33, facecolor='lightgrey', alpha=0.5)
fig.legend()
plt.show()
