import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import numpy as np

PATH = "H:\\BA\\pycrop-yield-prediction\\data"
YIELDFILE = "yield_data_with2020.csv"

yield_df = pd.read_csv(PATH + "\\" + YIELDFILE, encoding='utf8')
sum_per_year = []
abs_sum_per_year = []

x = np.array(range(2010, 2021))

for i in x:
    sum_per_year.append(yield_df[yield_df['anio'] == i]["redimiento_buxacre"].mean())
    abs_sum_per_year.append(yield_df[yield_df['anio'] == i]["produccion_tm"].sum()*0.03674)

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots()
par1 = ax.twinx()
width = 0.35
ax.bar(x - width/2, abs_sum_per_year, width, label="Average yield in bu", color='darkgreen')
par1.bar(x + width/2, sum_per_year, width, label="Average yield by area in bu/acre", color='darkgrey')
# y_formatter = ScalarFormatter(useOffset=False)
# ax.yaxis.set_major_formatter(y_formatter)
ax.set_xticks(x, x)
par1.set_ylabel("Average yield by area in bushels per acre")
ax.set_ylabel("Average yield in bushels")
fig.legend()
fig.tight_layout()
ax.set_xlabel("Year")
plt.show()
