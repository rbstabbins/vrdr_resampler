# Module for resampling CRISM VRDR data to the spectral response
# of a given instrument.
__author__ = "Roger Stabbins"
__copyright__ = "Copyright Info"
__credits__ = ["Roger Stabbins", "Joe McNeil"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "rstabbins"
__email__ = "r.stabbins@nhm.ac.uk"
__status__ = "development"

from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pandas as pd
import pdr
from pyproj import CRS
import rasterio as rio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from scipy.interpolate import interp1d
import time

# Product type string identifiers
# Product identifiers are prefixed with the instrument identifier,
# e.g. for CaSSIS 'c', the I/F Cube becomes 'cifcub'
IFCUB_STR = 'ifcub' # I/F Cube
SPCUB_STR = 'spcub' # Spectral Parameter (or Summary Product) Cube
# Browse products indicate the population over which the stretch parameter
# statistics have been evaluated.
# Browse product type IF is attached as suffix to the string identifier,
# e.g. for CaSSIS NPB tile-stretched browse product becomes 'cbtnpb'
BP_STR = {'tile': 'bt', 'quadrant': 'bq', 'global': 'bg'}
# BT_STR = 'bt' # Browse product 'Tile' stretched
# BQ_STR = 'bq' # Browse product 'Quadrant' stretched
# BG_STR = 'bg' # Browse product 'Global' stretched


def get_quadrant_dirs(vrdr_path: Path) -> List[Path]:
    """Get the list of quadrant directories in the VRDR storage.

    :param vrdr_path: Path to the VRDR storage directory
    :type vrdr_path: Path
    :return: List of quadrant directories
    :rtype: List[Path]
    """
    mc_paths = [d for d in vrdr_path.iterdir() 
               if d.is_dir() and d.name.startswith('mc')]
    mc_paths.sort()
    return mc_paths

def get_tile_lbls(mc_path: Path) -> List[Path]:
    """Get the list of VRDR tile .lbl files in a quadrant directory.

    :param mc_dir: Path to the quadrant directory
    :type mc_dir: Path
    :return: List of VRDR tile .lbl file paths
    :rtype: List[Path]
    """    
    lbl_paths = [f for f in mc_path.glob('*.lbl') if not f.name.startswith('.')]
    lbl_paths.sort()
    return lbl_paths

def get_tile_dirs(mc_path: Path) -> List[Path]:
    """Get the list of VRDR tile directories in a quadrant directory.

    :param mc_dir: Path to the quadrant directory
    :type mc_dir: Path
    :return: List of VRDR tile .lbl file paths
    :rtype: List[Path]
    """    
    tile_dirs = [d for d in mc_path.iterdir() if d.is_dir()]
    tile_dirs.sort()
    return tile_dirs

def prepare_output_dir(
        lbl_path: Path, 
        mc_path: Path, 
        output_path: Path,
        overwrite: bool=False) -> Path:
    """Prepare the output directory for the given tile and quadrant.

    :param lbl_path: Path to the .lbl file
    :type lbl_path: Path
    :param mc_path: Path to the quadrant directory
    :type mc_path: Path
    :param output_path: Path to the output directory
    :type output_path: Path
    :return: Path to the prepared output directory
    :rtype: Path
    """    
    output_dir = output_path / mc_path.stem / lbl_path.stem
    if output_dir.exists():
        if overwrite:
            print('Output directory exists, overwriting: ', output_dir)
        else:
            print('Output directory exists, skipping: ', output_dir)
            return Path()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_vrdr_tile(lbl_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load a CRISM VRDR tile from a .lbl file.

    :param lbl_path: Path to the .lbl file
    :type lbl_path: Path
    :return: A tuple containing the image cube (3D numpy array), band wavelengths 
        (1D numpy array), and no-data value (float)
    :rtype: Tuple[np.ndarray, np.array, float]
    """    
    vrral = pdr.read(lbl_path)
    band_names = vrral.metaget('BAND_NAME')
    cwls = np.array([float(band) for band in band_names])

    vrral.load('IMAGE', reload=True)
    cube = vrral.IMAGE
    no_data = vrral.metaget('IMAGE')['MISSING_CONSTANT']

    return cube, cwls, no_data

def load_channel_profiles(inst_path: Path) -> pd.DataFrame:
    """Load the channel information from the instrument CSV file.
    
    :param instrument: Instrument name (e.g., 'CaSSIS-2019')
    :type instrument: str
    :return: DataFrame containing the channel profiles indexed by wavelength
    :rtype: pd.DataFrame
    """

    if not inst_path.exists():
        raise FileNotFoundError(f"{inst_path.stem} CSV file not found: {inst_path}")
    
    inst_df = pd.read_csv(inst_path)

    inst_df = inst_df.set_index('wvls')

    return inst_df

def plot_channel_profiles(
        inst_df: pd.DataFrame,
        instrument: str
        ) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the channel profiles from the instrument DataFrame.
    
    :param inst_df: DataFrame containing the channel profiles indexed by 
        wavelength
    :type inst_df: pd.DataFrame
    :param instrument: Instrument name (e.g., 'CaSSIS-2019')
    :type instrument: str
    """        
    fig, ax = plt.subplots(figsize=(5, 4))
    for band in inst_df.columns:
        ax.plot(inst_df.index, inst_df[band], label=band, linestyle='-', marker='o', markersize=4)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Spectral Response')
    ax.set_title(f'{instrument} Channel Profiles')
    ax.legend()
    ax.grid()
    return fig, ax

def unpack_channel_profiles(
        inst_df: pd.DataFrame
        ) -> Tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Unpack the channel profiles DataFrame into arrays and lists.
    
    :param inst_df: DataFrame containing the channel profiles indexed by 
        wavelength
    :type inst_df: pd.DataFrame
    :return: Tuple containing band array (2D numpy array), array wavelengths 
        (1D numpy array), band list (list), and band center wavelengths (list)
    :rtype: Tuple[np.ndarray, np.ndarray, list, list]
    """
    band_arr = inst_df.to_numpy()
    arr_wvls = inst_df.index.to_numpy()
    band_list = inst_df.columns.tolist()
    band_cwls = np.round(
                        np.array(np.sum(band_arr*arr_wvls[:, np.newaxis], axis=0) 
                            / np.sum(band_arr, axis=0))
                        )

    return band_arr, arr_wvls, band_list, band_cwls

def interp_band_profile(
        band_profile: np.ndarray,
        band_wvls: np.ndarray,
        cube_wvls: np.ndarray
        ) -> np.ndarray:
    """Interpolate a band profile to the VRDR cube wavelengths, with -ve 
    value clipping
    
    :param band_profile: The band profile to interpolate (1D numpy array)
    :type band_profile: np.ndarray
    :param band_wvls: The wavelengths of the band profile (1D numpy array)
    :type band_wvls: np.ndarray
    :param cube_wvls: The wavelengths of the VRDR cube (1D numpy array)
    :type cube_wvls: np.ndarray
    :return: The interpolated band profile (1D numpy array)
    :rtype: np.ndarray
    """
    interp_func = interp1d(band_wvls, 
                           band_profile.flatten(), 
                           kind='linear', 
                           bounds_error=False,
                           fill_value=0.0)
    resampled_profile: np.ndarray = interp_func(cube_wvls)
    resampled_profile = np.clip(resampled_profile, 0, None)
    return resampled_profile

def resample_cube(
        band_arr: np.ndarray,
        band_wvls: np.ndarray,
        cube: np.ndarray,
        cube_wvls: np.ndarray,
        no_data: float
        ) -> np.ndarray:
    """Resample the VRDR cube to the instrument band spectral responses.
    
    :param band_arr: Array of band profiles (2D numpy array)
    :type band_arr: np.ndarray
    :param band_wvls: Wavelengths of the band profiles (1D numpy array)
    :type band_wvls: np.ndarray
    :param cube: VRDR image cube (3D numpy array)
    :type cube: np.ndarray
    :param cube_wvls: Wavelengths of the VRDR cube (1D numpy array)
    :type cube_wvls: np.ndarray
    :param no_data: No-data value in the VRDR cube
    :type no_data: float
    :return: Resampled image cube (3D numpy array)
    :rtype: np.ndarray
    """
    b_w = band_wvls
    c_w = cube_wvls
    b_a = band_arr
    resampled_profiles = [interp_band_profile(b_p, b_w, c_w) for b_p in b_a.T]

    # set no_data values to nan
    cube[cube == no_data] = np.nan

    band_cube = []
    cube_t =cube.transpose(1,2,0)
    for band in resampled_profiles:        
        band_image = np.dot(cube_t, band) / np.sum(band)
        band_cube.append(band_image)

    # set nan values back to no_data value
    band_cube = np.nan_to_num(band_cube, nan=no_data)

    return np.array(band_cube).astype(np.float32)

def stretch_image(image: np.ndarray, 
                  no_data: float,
                  pct_params: Tuple[float, float]=(0.01, 99.95),
                  st_params: Tuple[float, float]|None = None,
                  ) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Apply a linear stretch to the image based on the given percentiles.
    
    :param image: Input image (2D numpy array)
    :type image: np.ndarray
    :param no_data: No-data value in the image
    :type no_data: float
    :param lower_pct: Lower percentile for stretch (default: 0.01)
    :type lower_pct: float
    :param upper_pct: Upper percentile for stretch (default: 99.95)
    :type upper_pct: float
    :return: Contrast-stretched image (2D numpy array), and stretch parameters
    :rtype: Tuple[np.ndarray, Tuple[float, float]]
    """
    # treat no_data values as NaN
    image = image.astype(float)
    stretch_img = image.copy()
    stretch_img[image == no_data] = np.nan
    if st_params is None:
        lower, upper = np.nanpercentile(stretch_img, pct_params)  
    else:
        lower, upper = st_params      

    stretched = (stretch_img - lower) / (upper - lower)
    stretched = np.clip(stretched, 0.0, 1.0)

    # convert to 8-bit
    stretched = (np.rint(stretched * 254))
    stretched[np.isnan(stretch_img)] = 255
    
    return stretched.astype(np.uint8), (lower, upper)

def stretch_cube(
        cube: np.ndarray,
        no_data: float,
        pct_params: Tuple[float, float]=(0.01, 99.95),
        st_params: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
    """Apply contrast stretch to each band in a cube.

    :param cube: Input image cube (3D numpy array)
    :type cube: np.ndarray
    :param no_data: No-data value in the cube
    :type no_data: float
    :param lower_pct: Lower percentile for stretch (default: 0.01)
    :type lower_pct: float
    :param upper_pct: Upper percentile for stretch (default: 99.95)
    :type upper_pct: float
    :return: Contrast-stretched image cube (3D numpy array)
    :rtype: np.ndarray
    """
    no_data_locs = cube == no_data
    stcub = cube.astype(float)
    stcub[no_data_locs] = np.nan

    # get stretch parameters
    if st_params is None:
        lower, upper = np.nanpercentile(stcub, pct_params, axis=(1,2))
    else:
        lower, upper = st_params

    # apply stretch
    lower = lower[:, np.newaxis, np.newaxis]
    upper = upper[:, np.newaxis, np.newaxis]
    stcub = (stcub - lower) / (upper - lower)
    stcub = np.clip(stcub, 0, 1)
    stcub = (np.rint(stcub * 254))
    stcub[no_data_locs] = 255
    
    st_params = np.stack((lower.flatten(), upper.flatten()), axis=0)
    
    return np.array(stcub).astype(np.uint8), st_params

def log_st_params(
        st_params: np.ndarray,
        band_list: list[str],
        lbl_path: Path,
        out_dir: Path
    ) -> Path:
    """Log the stretch parameters in a tab-delimited text file

    :param st_params: Stretch parameters (min, max) for each band
    :type st_params: np.ndarray
    :param band_list: List of band names
    :type band_list: list[str]
    :param lbl_path: Path to the VRDR .lbl file
    :type lbl_path: Path
    :param out_dir: Output directory for the log file
    :type out_dir: Path
    :return: Path to the log file
    :rtype: Path
    """    
    # set the txt file to write out to
    out_stem = lbl_path.stem.replace('vrral', 'strtch')
    out_txt = out_dir / f'{out_stem}.log'

    with open(out_txt, 'w') as f:
        f.write('Band\tLower\tUpper\n')
        for i, band in enumerate(band_list):
            f.write(f'{band}\t{np.round(st_params[0,i], 6)}\t{np.round(st_params[1,i], 6)}\n')
    
    return out_txt

def write_ifcub_envi_hdr(
        band_list: list[str], 
        band_cwls: np.ndarray,
        no_data: float,
        default_bands: list[int],
        instrument_id: str,
        lbl_path: Path, 
        out_dir: Path
        ) -> Path:
    """Write an ENVI header file to accompany the GeoTIFF

    :param band_list: List of band names
    :type band_list: list
    :param band_cwls: List of band center wavelengths
    :type band_cwls: list
    :param no_data: No-data value for the bands
    :type no_data: float
    :param default_bands: List of default bands for display
    :type default_bands: list
    :param instrument_id: Instrument identifier string
    :type instrument_id: str
    :param lbl_path: Path to the VRDR .lbl file
    :type lbl_path: Path
    :param out_dir: Output directory for the ENVI header
    :type out_dir: Path
    :return: Path to the written ENVI header file
    :rtype: Path
    """    
    # read in the envi header as a text
    envi_header_path = lbl_path.with_suffix('.hdr')
    with open(envi_header_path, 'r') as f:
        envi_header_text = f.readlines()
    # parse the header into a dictionary
    envi_header_dict = OrderedDict()
    for line in envi_header_text:
        if '=' in line:
            key, value = line.split('=', 1)
            if key.strip() == 'band names':
                envi_header_dict['band names'] = ''
                continue
            if key.strip() == 'wavelength':                
                envi_header_dict['wavelength'] = ''
                continue
            envi_header_dict[key.strip()] = value.strip()
    # replace key header tags
    envi_header_dict['bands'] = str(len(band_list))
    envi_header_dict['band names'] = ', '.join(band_list)
    envi_header_dict['wavelength'] = ', '.join([f'{wvl}' for wvl in band_cwls])
    envi_header_dict['file type'] = 'TIFF'
    envi_header_dict['data ignore value'] = str(no_data)
    envi_header_dict['default bands'] = ', '.join(str(x) for x in default_bands)

    # prepare the output header path
    ifcub_id = instrument_id + IFCUB_STR
    cube_stem = lbl_path.stem.replace('vrral', ifcub_id)
    out_path = out_dir / (cube_stem + '.hdr')

    # write the dict to a text file, as a new envi header file
    with open(out_path, 'w') as f:
        f.write('ENVI\n')
        for key, value in envi_header_dict.items():
            if key in ['band names', 'wavelength','default bands']:
                f.write(f'{key} = {{')
                # write the values in lines of max 70 characters
                line = ''
                for item in value.split(', '):
                    if len(line) + len(item) + 2 > 70:
                        f.write(line[:-2] + '\n')
                        line = ''
                    line += item + ', '
                f.write(line[:-2] + '}\n')
            else:
                f.write(f'{key} = {value}\n')
    return out_path

def export_ifcub_geotiff(
        cube: np.ndarray,
        band_list: list[str],
        band_cwls: np.ndarray,
        no_data: float,
        instrument_id: str,
        lbl_path: Path,
        out_dir: Path,
        new_resolution: float | None = None # the new resolution in meters
    ) -> Path:
    """Export the cube as a GeoTIFF file with georeferencing from the 
    VRDR LBL file.
    
    :param cube: Image cube to export (3D numpy array)
    :type cube: np.ndarray
    :param band_list: List of band names
    :type band_list: list[str]
    :param band_cwls: Array of band center wavelengths
    :type band_cwls: np.ndarray
    :param no_data: No-data value in the cube
    :type no_data: float
    :param instrument_id: Instrument identifier string
    :type instrument_id: str
    :param lbl_path: Path to the VRDR .lbl file
    :type lbl_path: Path
    :param out_dir: Output directory for the GeoTIFF
    :type out_dir: Path
    :param new_resolution: New resolution in meters (optional)
    :type new_resolution: float | None
    :return: Path to the exported GeoTIFF file
    :rtype: Path    
    """

    # from the LBL file, get the PDS IMG file, from which the georeferencing
    # will be extracted
    pds_file = lbl_path.with_suffix('.img')

    # set the output path, replacing vrral product ID with the new 
    # [instrument]ifcub ID
    ifcub_id = instrument_id + IFCUB_STR
    cube_stem = lbl_path.stem.replace('vrral', ifcub_id)
    out_path = out_dir / (cube_stem + '.tif')

    # read the georeferencing info from the PDS IMG file
    pds_gdal = gdal.Open(str(pds_file))
    pds_transform = Affine.from_gdal(*pds_gdal.GetGeoTransform())
    pds_crs = CRS.from_wkt(pds_gdal.GetProjection())

    if new_resolution is not None:
        scale = pds_transform.a /new_resolution
        new_width = int(pds_gdal.RasterXSize * scale)
        new_height = int(pds_gdal.RasterYSize * scale)

        new_transform = Affine(
            new_resolution, pds_transform.b, pds_transform.c,
            pds_transform.d, -new_resolution, pds_transform.f
        )

        new_arr = np.empty((cube.shape[0], new_height, new_width), dtype=cube.dtype)

        for b_idx in range(cube.shape[0]):
            reproject(
                source=cube[b_idx, :, :],
                destination=new_arr[b_idx, :, :],
                src_transform=pds_transform,
                src_crs=pds_crs,
                dst_transform=new_transform,
                dst_crs=pds_crs,
                src_nodata=no_data,
                dst_nodata=no_data,
                resampling=Resampling.bilinear
            )
        
        cube = new_arr
        pds_transform = new_transform

    # prepare the output GeoTIFF
    with rio.open(out_path, 'w', 
                    driver='GTiff',
                    height=cube.shape[1], 
                    width=cube.shape[2],
                    count=cube.shape[0],
                    dtype=str(cube.dtype),
                    compress='LZW',
                    nodata=no_data,
                    crs=pds_crs, transform=pds_transform) as new_dataset:
        # write each band to the GeoTIFF
        for b_idx, band in enumerate(band_list):
            # write the band data
            new_dataset.write(cube[b_idx,:,:], b_idx + 1)
            # set the band name
            new_dataset.set_band_description(b_idx+1, band)
            # set the band wavelength tag
            new_dataset.update_tags(b_idx+1, WAVELENGTH=f'{band_cwls[b_idx]}nm')
    new_dataset.close()
    
    return out_path

def export_bp_geotiff(
        browse_product_data: np.ndarray,
        browse_product_id: str,
        no_data: float,
        band_list: list[str],
        instrument_id: str,
        lbl_path: Path,
        out_dir: Path,
        stretch_extent: Literal['tile','quadrant','global']= 'tile',
        new_resolution: float | None = None, # the new resolution in meters
        apply_new_res: bool = True
    ) -> Path:
    """Export a browse product as a GeoTIFF file with georeferencing 
    from the VRDR LBL file.
    
    :param browse_product_data: Browse product image data (3D numpy array)
    :type browse_product_data: np.ndarray
    :param browse_product_id: Browse product ID (e.g., 'NPB', 'RGB')
    :type browse_product_id: str
    :param no_data: No-data value in the browse product
    :type no_data: float
    :param band_list: List of band names
    :type band_list: list[str]
    :param instrument_id: Instrument identifier string
    :type instrument_id: str
    :param lbl_path: Path to the VRDR .lbl file
    :type lbl_path: Path
    :param out_dir: Output directory for the GeoTIFF
    :type out_dir: Path
    :param stretch_extent: Stretch extent for the browse product
    :type stretch_extent: Literal['tile','quadrant','global']
    :param new_resolution: New resolution in meters (optional)
    :type new_resolution: float | None
    :return: Path to the exported GeoTIFF file
    :rtype: Path
    """        
    pds_file = lbl_path.with_suffix('.img')

    # set the output path, replacing vrral product ID with the new 
    # [instrument]spcub ID    
    bp_id = instrument_id + BP_STR[stretch_extent] + browse_product_id.lower()
    browse_product_stem = lbl_path.stem.replace('vrral', bp_id)
    # check the outdir exists
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (browse_product_stem + '.tif')

    pds_gdal = gdal.Open(str(pds_file))
    pds_transform = Affine.from_gdal(*pds_gdal.GetGeoTransform())
    pds_crs = CRS.from_wkt(pds_gdal.GetProjection())    
    arr = browse_product_data    

    # if 1 band, expand dims duplicate band 3 times    
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=2)
        arr = np.repeat(arr, 3, axis=2)
        # expand band list by duplication
        band_list = band_list * 3

    if new_resolution is not None:
        scale = pds_transform.a / new_resolution

        new_transform = Affine(
            new_resolution, pds_transform.b, pds_transform.c,
            pds_transform.d, -new_resolution, pds_transform.f
        )

        if apply_new_res:
            print(f'Scale factor: {scale}')
            new_width = int(pds_gdal.RasterXSize * scale)
            new_height = int(pds_gdal.RasterYSize * scale)
            new_arr = np.empty((new_height, new_width,arr.shape[2]), dtype=arr.dtype)

            for b_idx in range(arr.shape[2]):
                reproject(
                    source=arr[:, :, b_idx],
                    destination=new_arr[:, :, b_idx],
                    src_transform=pds_transform,
                    src_crs=pds_crs,
                    dst_transform=new_transform,
                    dst_crs=pds_crs,
                    src_nodata=no_data,
                    dst_nodata=no_data,
                    resampling=Resampling.bilinear
                )
            arr = new_arr
        
        pds_transform = new_transform

    with rio.open(out_path, 'w', driver='GTiff',
                                height = arr.shape[0], width = arr.shape[1],
                                count=3, dtype=str(arr.dtype),
                                compress='LZW',
                                nodata=no_data,
                                crs=pds_crs, transform=pds_transform) as new_dataset:
        for band_idx, band in enumerate(band_list):
            new_dataset.write(arr[:,:,band_idx], band_idx + 1)    
            new_dataset.set_band_description(band_idx+1, band)            
    new_dataset.close()
    
    return out_path

def write_spcub_envi_hdr(
        sp_list: list[str], 
        no_data: float,
        default_bands: list[int],
        instrument_id: str,
        lbl_path: Path, 
        out_dir: Path) -> Path:
    """Write an ENVI header file for the SUmmary Product Stack

    :param sp_list: List of spectral parameter names
    :type sp_list: list[str]
    :param no_data: No-data value for the bands
    :type no_data: float
    :param default_bands: List of default bands for display
    :type default_bands: list[int]
    :param lbl_path: Path to the VRDR .lbl file
    :type lbl_path: Path
    :param out_dir: Output directory for the ENVI header
    :type out_dir: Path
    :return: Path to the written ENVI header file
    :rtype: Path
    """    
    envi_header_path = lbl_path.with_suffix('.hdr')
    # read in the envi header as a text file into a dictionary
    with open(envi_header_path, 'r') as f:
        envi_header_text = f.readlines()
    envi_header_dict = OrderedDict()
    for line in envi_header_text:
        if '=' in line:
            key, value = line.split('=', 1)
            if key.strip() == 'band names':
                envi_header_dict['band names'] = ''
                continue
            if key.strip() == 'wavelength':
                envi_header_dict['wavelength'] = ''
                continue
            envi_header_dict[key.strip()] = value.strip()

    envi_header_dict['bands'] = str(len(sp_list))
    envi_header_dict['band names'] = ', '.join(sp_list)
    # remove 'wavelength' entry
    if 'wavelength' in envi_header_dict:
        del envi_header_dict['wavelength']
    envi_header_dict['file type'] = 'TIFF'
    envi_header_dict['default bands'] = ', '.join(str(x) for x in default_bands)
    envi_header_dict['data ignore value'] = str(no_data)

    # replace vrral with cspcub
    # set the output path, replacing vrral product ID with the new 
    # [instrument]spcub ID
    spcub_id = instrument_id + SPCUB_STR
    cube_stem = lbl_path.stem.replace('vrral', spcub_id)
    out_path = out_dir / (cube_stem + '.hdr')

    # write the dict to a text file, as a new envi header file
    with open(out_path, 'w') as f:
        f.write('ENVI\n')
        for key, value in envi_header_dict.items():
            if key in ['band names', 'default bands']:
                f.write(f'{key} = {{')
                # write the values in lines of max 70 characters
                line = ''
                for item in value.split(', '):
                    if len(line) + len(item) + 2 > 70:
                        f.write(line[:-2] + '\n')
                        line = ''
                    line += item + ', '
                f.write(line[:-2] + '}\n')
            else:
                f.write(f'{key} = {value}\n')
    return out_path

def export_spcub_geotiff(
        spcub: np.ndarray,
        sp_ids: list[str],
        no_data: float,
        instrument_id: str,
        lbl_path: Path,
        out_dir: Path,
        new_resolution: float | None = None # the new resolution in meters
    ) -> Path:
    """Export a stack of single-channel spectral parameters as a GeoTIFF 
    file with georeferencing from the VRDR LBL file."""
    pds_file = lbl_path.with_suffix('.img')

    # set the output path, replacing vrral product ID with the new 
    # [instrument]spcub ID
    spcub_id = instrument_id + SPCUB_STR
    cube_stem = lbl_path.stem.replace('vrral', spcub_id)
    out_path = out_dir / (cube_stem + '.tif')

    pds_gdal = gdal.Open(str(pds_file))
    pds_transform = Affine.from_gdal(*pds_gdal.GetGeoTransform())
    pds_crs = CRS.from_wkt(pds_gdal.GetProjection())   

    if new_resolution is not None:
        scale = pds_transform.a /new_resolution
        new_width = int(pds_gdal.RasterXSize * scale)
        new_height = int(pds_gdal.RasterYSize * scale)

        new_transform = Affine(
            new_resolution, pds_transform.b, pds_transform.c,
            pds_transform.d, -new_resolution, pds_transform.f
        )

        new_arr = np.empty((spcub.shape[0],new_height, new_width), dtype=spcub.dtype)

        for b_idx in range(spcub.shape[0]):
            reproject(
                source=spcub[b_idx, :, :],
                destination=new_arr[b_idx, :, :],
                src_transform=pds_transform,
                src_crs=pds_crs,
                dst_transform=new_transform,
                dst_crs=pds_crs,
                src_nodata=no_data,
                dst_nodata=no_data,
                resampling=Resampling.bilinear
            )
        
        spcub = new_arr
        pds_transform = new_transform

    with rio.open(out_path, 'w', driver='GTiff',
                                height = spcub.shape[1], width = spcub.shape[2],
                                count=spcub.shape[0], dtype=str(spcub.dtype),
                                compress='LZW',
                                nodata=no_data,
                                crs=pds_crs, transform=pds_transform) as new_dataset:
        for sp_idx, sp in enumerate(sp_ids):
            new_dataset.write(spcub[sp_idx,:,:], sp_idx + 1)    
            new_dataset.set_band_description(sp_idx+1, sp)
    new_dataset.close()
    
    return out_path

# read a geotiff cube
def read_geotiff_cube(
        geotiff_path: Path
    ) -> Tuple[np.ndarray, float]:
    """Read a GeoTIFF cube and return the data array, transform, CRS, and 
    no-data value.

    :param geotiff_path: Path to the GeoTIFF file
    :type geotiff_path: Path
    :return: Tuple containing the image cube (3D numpy array), transform 
        (Affine), CRS (CRS), and no-data value (float)
    :rtype: Tuple[np.ndarray, Affine, CRS, float]
    """    
    # check the file exists
    if not geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {geotiff_path}")
    with rio.open(geotiff_path) as dataset:
        cube = dataset.read()
        no_data = dataset.nodata
    return cube, no_data

def read_geotiff_stretch_stats(
        geotiff_path: Path
    ) -> dict:
    """Read the statistics from a GeoTIFF file.

    :param geotiff_path: Path to the GeoTIFF file
    :type geotiff_path: Path
    :return: Dictionary containing statistics for each band
    :rtype: dict
    """    
    stats_list = []
    with rio.open(geotiff_path) as dataset:
        for band_idx in range(1, dataset.count + 1):
            stats = dataset.statistics(band_idx)  
            stretch_min = stats.mean - 4*stats.std
            if stretch_min < 0:
                stretch_min = stats.min           
            band_stats = stretch_min, stats.mean + 4*stats.std
            # band_stats = stats.min, stats.max
            stats_list.append(band_stats)
    return stats_list

def read_log_file(
        log_path: Path
    ) -> dict:
    """Read the stretch parameters from a log file.

    :param log_path: Path to the log file
    :type log_path: Path
    :return: Dictionary containing stretch parameters for each band
    :rtype: dict
    """    
    st_params = pd.read_csv(log_path, sep='\t', index_col=0)
    return st_params

def get_quadrant_stretch_params(
        tile_paths: List[Path]
    ) -> np.ndarray:
    """Get the quadrant-wide stretch parameters from the tile log files.

    :param tile_paths: List of tile directory paths
    :type tile_paths: List[Path]
    :return: Quadrant-wide stretch parameters (2D numpy array)
    :rtype: np.ndarray
    """
    # get quadrant-wide lower and upper stretch params
    lower_log = []
    upper_log = []
    for tile_path in tile_paths:
        log_file = [f for f in tile_path.glob('*strtch*.log')][0]        
        st_params = read_log_file(log_file).to_numpy().T
        lower_log.append(st_params[0])
        upper_log.append(st_params[1])
    quad_min = np.min(np.array(lower_log), axis=0)
    quad_max = np.max(np.array(upper_log), axis=0)
    quad_st_params = np.array([quad_min, quad_max])
    return quad_st_params