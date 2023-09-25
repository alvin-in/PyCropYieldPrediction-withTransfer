from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
import math
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from typing import Optional

from .utils import load_clean_yield_data as load
from .utils import get_tif_files


class YieldDataCleansing:
    """Take the exported, downloaded data and clean it.

    Specifically:
    - split the image collections into years
    - merge the temperature and reflection images
    - apply the mask, so only the farmland pixels are considered

    Parameters
    -----------
    mask_path: pathlib Path, default=Path('data/crop_yield-data_mask')
    """

    def __init__(
            self,
            mask_path=Path("data/crop_yield-data_mask")
            # mask_path=Path("I:/US-SAT-DS/crop_yield-data_mask")
            # Path("data/cover")  # USA: Path("data/crop_yield-data_mask")
    ):
        self.mask_path = mask_path

        self.tif_files = get_tif_files(self.mask_path)

        self.pix_count = []

    def process(self, num_years=11, out='usa_yield_with_pix2'):
        """
        Process all the data.

        Parameters
        ----------
        num_years: int, default=14
            How many years of data to create.
        out: String
            Name of resulting csv file
        """
        for filename in self.tif_files:
            process_county(
                self,
                filename,
                self.mask_path,
                num_years=num_years,
            )

        # print(self.counter)
        # mask_pix = []
        # for key, value in self.mask_pix_counter_dict:
        #    mask_pix.append([key, value])
        # USA:
        pd.DataFrame(self.pix_count, columns=['anio', 'id', 'pix_count']).to_csv(
            "data\\"+out+".csv", index=False)
        # pd.DataFrame(self.pix_count).to_csv("H:\\BA\\pycrop-yield-prediction\\data\\year_pix_counter(2).csv")


def check_for_tif_file(filepath: Path, prefix: str) -> Optional[Path]:
    """
    Returns a filepath if one exists, else returns None. This is useful
    because sometimes, Earth Engine exports files with characters added
    to the end of the filename, e.g. {intended_filename}{-more stuff}.tif
    """
    if (filepath / f"{prefix}.tif").exists():
        return filepath / f"{prefix}.tif"

    files_with_prefix = list(filepath.glob(f"{prefix}-*.tif"))
    if len(files_with_prefix) == 1:
        return files_with_prefix[0]
    elif len(files_with_prefix) == 0:
        return None
    elif len(files_with_prefix) > 1:
        print(f"Multiple files with prefix for {filepath / prefix}.tif")
        return None


def process_county(
        self,
        filename,
        mask_path,
        num_years,
):
    """
    Process and save county level data
    """
    # exporting.py saves the files in a "{state}_{county}.tif" format
    # the last 4 characters are always ".tif"
    locations = filename[:-4].split("_")

    # add another split for the county, since the tif filenames can
    # sometimes have additional strings at the end
    state, county = int(locations[0]), int(locations[1].split("-")[0])

    print(f"Processing {filename}")

    # check all the files exist:
    prefix = filename.split(".")[0].split("-")[0]
    mask_path = check_for_tif_file(mask_path, prefix)

    mask = np.transpose(
        np.array(gdal.Open(str(mask_path)).ReadAsArray(), dtype="uint16"),
        axes=(1, 2, 0),
    )
    # mask = mask[:, :, :14]

    # a value of 12 indicates farmland; everything else, we want to ignore
    mask[mask != 12] = 0
    mask[mask == 12] = 1

    # when exporting the image, we appended bands from many years into a single image for efficiency. We want
    # to split it back up now
    mask_list = divide_into_years(
        mask, bands=1, composite_period=365, num_years=num_years, extend=True, shift=True
    )

    # print(np.shape(mask_list))
    # print(mask_list)
    # print(mask_list[0][:, :])

    # fig, axs = plt.subplots(1)
    # axs.imshow(mask_list[0][:, :])
    # plt.show()

    start_year = 2010
    for i in range(0, num_years - 1):
        year = i + start_year
        # self.pix_count.append([str(year), str(county), np.sum(mask_list[i][:, :])])
        # USA:
        self.pix_count.append([str(year), str(state)+"_"+str(county), np.sum(mask_list[i][:, :])])


def divide_into_years(img, bands, composite_period, num_years=10, extend=False, shift=False):
    """
    Parameters
    ----------
    img: the appended image collection to split up
    bands: the number of bands in an individual image
    composite_period: length of the composite period, in days
    num_years: how many years of data to create.
    extend: boolean, default=False
        If true, and num_years > number of years for which we have data, then the extend the image
        collection by copying over the last image.
        NOTE: This is different from the original code, where the 2nd to last image is copied over
    shift: boolean, default=False
        If true, uses data at the beginning of the year before the season.
        If false, splits images by 2/3 (rounded up) of the year (between the seasons)
        Adaption for Argentina/southern hemisphere.

    Returns:
    ----------
    im_list: a list of appended image collections, where each element in the list is a year's worth
        of data
    """
    bands_per_year = bands * math.ceil(365 / composite_period)

    # if necessary, pad the image collection with the final image
    if extend:
        num_bands_necessary = bands_per_year * num_years
        while img.shape[2] < num_bands_necessary:
            img = np.concatenate((img, img[:, :, -bands:]), axis=2)

    image_list = []
    cur_idx = math.ceil(2 / 3 * math.ceil(365 / composite_period)) * bands
    if shift is True:
        cur_idx = 0
    for i in range(0, num_years - 1):
        image_list.append(img[:, :, cur_idx: cur_idx + bands_per_year])
        cur_idx += bands_per_year
    image_list.append(img[:, :, cur_idx:])
    return image_list
