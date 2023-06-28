import os.path
from pathlib import Path

# Kurzes Script, um jeweils die Namen der TIFs in den Orndern sat, temp und cover einander anzugleichen.
PATH = "H:\\BA\\pycrop-yield-prediction\\data\\temp"


def get_tif_files(image_path):
    """
    Get all the .tif files in the image folder.

    Parameters
    ----------
    image_path: pathlib Path
        Directory to search for tif files
    Returns:
        A list of .tif filenames
    """
    files = []
    for dir_file in image_path.iterdir():
        if str(dir_file).endswith("tif"):
            files.append(str(dir_file.name))
    return files


tif_files = get_tif_files(Path(PATH))
for filename in tif_files:
    prefix = ""
    prefix_arr = filename.split("_")
    for i in range(2, len(prefix_arr), 1):
        prefix = prefix + prefix_arr[i]
        if i < len(prefix_arr) - 1:
            prefix = prefix + "_"

    old_file = os.path.join(PATH, filename)
    new_file = os.path.join(PATH, prefix)
    os.rename(old_file, new_file)

