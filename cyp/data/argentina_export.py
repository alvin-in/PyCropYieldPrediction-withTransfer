# edited from https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction/blob/master/README.md

import ee
import time
import argparse
import pandas as pd
from pathlib import Path
import os

IMG_COLLECTIONS = ['MODIS/061/MOD09A1', 'MODIS/061/MYD11A2', 'MODIS/006/MCD12Q1']
start_date = "2009-12-31"
end_date = "2022-12-31"
IMG_COLLECTION_BANDS = [[0, 1, 2, 3, 4, 5, 6], [0, 4], [0]]
IMG_COLLECTION_CODES = ['sat', 'temp', 'cover']

# "Boundary Filters": A rough bounding box for the entire country, to help GEE search for imagery faster
boundary_filter = [-74, -52, -54, -21]

# "Feature Collections": The path in Google Earth Engine to a shapefile table specifying a set of subdivisions of the country
FTR_COLLECTION = 'users/nikhilarundesai/cultivos_maiz_sembrada_1314'


# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer
def appendBand(current, previous):
    # Rename the band
    previous = ee.Image(previous)
    current = current.select(IMG_COLLECTION_BANDS[img_collection_index])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous, None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum


def export_to_drive(img, fname, folder, expregion, eeuser=None, scale=500):
    # print "export to cloud"
    expcoord = expregion.geometry().coordinates().getInfo()[0]
    expconfig = dict(description=fname, folder=folder, fileNamePrefix=fname, dimensions=None, region=expcoord,
                     scale=scale, crs='EPSG:4326', crsTransform=None, maxPixels=1e13)
    task = ee.batch.Export.image.toDrive(image=img.clip(expregion), **expconfig)
    task.start()
    while task.status()['state'] == 'RUNNING':
        print('Running...')
        time.sleep(10)
    print('Done.', task.status())


def load_clean_yield_data(yield_data_filepath):
    """
    Cleans the yield data by making sure any Nan values in the columns we care about
    are removed
    """
    important_columns = ["departamentos__provincia__nombre", "provincia_id", "departamentos__nombre", "departamento_id"]
    # ["Year", "State ANSI", "County ANSI", "Value"]  # change
    yield_data = pd.read_csv(yield_data_filepath).dropna(
        subset=important_columns, how="any"
    )
    return yield_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull MODIS data for specified countries and imagery types.")
    parser.add_argument("collection_name", choices=IMG_COLLECTION_CODES, help="Type of imagery to pull.")
    parser.add_argument("-t", "--target_folder", type=str,
                        help="Bucket folder where files will ultimately be moved. Checks if file already downloaded. "
                             "Enter empty string to just check bucket.")
    parser.add_argument("-s", "--scale", type=int, default=500,
                        help="Scale in meters at which to pull; defaults to 500")
    args = parser.parse_args()

    img_collection_index = IMG_COLLECTION_CODES.index(args.collection_name)
    image_collection = IMG_COLLECTIONS[img_collection_index]

    ee.Initialize()
    county_region = ee.FeatureCollection("users/nikhilarundesai/cultivos_maiz_sembrada_1314")

    imgcoll = ee.ImageCollection(image_collection) \
        .filterBounds(ee.Geometry.Rectangle(boundary_filter)) \
        .filterDate(start_date, end_date)
    img = imgcoll.iterate(appendBand)
    img = ee.Image(img)

    if img_collection_index != 1:  # temperature index <<< is this min max filtering needed?
        img_0 = ee.Image(ee.Number(0))
        img_5000 = ee.Image(ee.Number(5000))

        img = img.min(img_5000)
        img = img.max(img_0)
    ########
    feature_list = county_region.toList(1e5)
    feature_list_computed = feature_list.getInfo()

    # for file_names
    # yield_data_path = Path("H:\\BA\\pycrop-yield-prediction\\data\\departamentos.csv")
    data = load_clean_yield_data("H:\\BA\\pycrop-yield-prediction\\data\\departamentos.csv")[
        ["departamentos__provincia__nombre", "provincia_id", "departamentos__nombre", "departamento_id"]].values

    myDict = {}
    for i in range(len(data)):
        myDict[str(data[i][0]).lower()] = str(int(data[i][1])).lower()
        myDict[str(data[i][2]).lower()+'_'+str(data[i][0]).lower()] = str(int(data[i][3])).lower()
    # end of for file_names
    
    keys_with_issues = []
    count_already_downloaded = 0
    count_filtered = 0
    for idx, region in enumerate(feature_list_computed):
        subunit_key = region.get('properties').get('partido').lower() + "-" + region.get('properties').get(
            'provincia').lower()
        sub_file_name1 = myDict[region.get('properties').get('provincia').lower()]
        sub_file_name2 = myDict[
            region.get('properties').get('partido').lower()+'_'+region.get('properties').get('provincia').lower()]
        file_name = sub_file_name1 + '_' + sub_file_name2
        # file_name oberhalb bearbeitet, nicht getestet
        if args.target_folder is not None and \
                os.path.isfile(os.path.join(args.target_folder, file_name + '.tif')):
            print(subunit_key, 'already downloaded. Continuing...')
            count_already_downloaded += 1
            continue

        try:
            export_to_drive(img, file_name, args.collection_name, ee.Feature(region),
                            scale=args.scale)
        except KeyboardInterrupt:
            print('received SIGINT, terminating execution')
            break
        except Exception as e:
            print('issue with {}'.format(subunit_key))
            keys_with_issues.append(subunit_key)

    print('Successfully ordered',
          len(feature_list_computed) - len(keys_with_issues) - count_already_downloaded - count_filtered,
          'new tifs from GEE')
    print('Already had', count_already_downloaded)
    print('Failed to order', len(keys_with_issues))
    print('Filtered', count_filtered)
    print('There were issues with:\n\t' + ',\n\t'.join([(k[0] + ' (' + k[1] + ')') for k in keys_with_issues]))
