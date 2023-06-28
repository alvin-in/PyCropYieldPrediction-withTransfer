import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from pathlib import Path
import skgstat as skg
from scipy.spatial.distance import pdist

plt.style.use('seaborn')

years = range(2015, 2020)

colors = cm.rainbow(np.linspace(0, 1, len(years)))
all_coords = []
all_pred_err = []
all_gp_pred_err = []
for idx, year in enumerate(years):
    model = Path("H:\\BA\\pycrop-yield-prediction\\data\\arg_models\\cnn\\" + str(year) + "_1_32_gp.pth.tar")
    # model = Path("H:\\BA\\pycrop-yield-prediction\\data\\us_81-337_new_75perc\\cnn\\" + str(year) +
    #              "_1_32_gp.pth.tar")
    model_sd = torch.load(model, map_location="cpu")


    real_values = model_sd["test_real"]
    pred_values = model_sd["test_pred"]
    pred_err = abs(pred_values - real_values)
    all_pred_err.extend(pred_err)

    gp = True
    try:
        gp_values = model_sd["test_pred_gp"]
    except KeyError:
        gp = False

    if gp:
        gp_pred_err = abs(gp_values - real_values)
        all_gp_pred_err.extend(gp_pred_err)

    print(model_sd["test_loc"].shape, model_sd["test_years"].shape)
    coords = model_sd["test_loc"]
    # coords = model_sd["test_years"]
    # coords = np.concatenate((model_sd["test_loc"], model_sd["test_years"].reshape((-1, 1))), axis=1)
    all_coords.extend(coords)

print(all_coords)

V = skg.Variogram(all_coords, all_pred_err, normalize=False, n_lags=1000, maxlag=14,
                  model='gaussian', dist_func='euclidean')
V.estimator = 'cressie'
print(V.describe())
xdata = V.bins
ydata = V.experimental
#V.plot(grid=False)
# V.location_trend()
# V.distance_difference_plot(ax=None, plot_bins=False, show=True)
plt.xlabel("Räumliche Distanz in Grad")
plt.ylabel("Semivarianz")
plt.plot(xdata, ydata, '.')
# , color=colors[idx])

plt.xlim([0, 14])
# plt.ylim([0, 23])
# plt.legend(years, prop={'size': 16})
plt.show()

