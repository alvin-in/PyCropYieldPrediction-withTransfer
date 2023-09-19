import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

path_to_hist = Path("H:\\BA\\pycrop-yield-prediction\\data\\arg_im_out\\histogram_all_full.npz")
path_to_hist2 = Path("H:\\BA\\pycrop-yield-prediction\\data\\img_output\\histogram_all_full.npz")

with np.load(path_to_hist2) as hist:
    images = hist["output_image"]

fig, axs = plt.subplots(9)
for i in range(9):
    axs[i].axis('off')
    axs[i].imshow(images[0][i, :32, :].transpose(), cmap=plt.cm.gray)
plt.show()
