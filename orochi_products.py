# Module for generating OROCHI browse products from an OROCHI I/F cube
# Assumes input cube is np.array(7 x N x M),
# with band order of [390nm, 480nm, 550nm, 650nm, 730nm, 860nm, 950nm]
from typing import Callable
import numpy as np

BP_LIST = [
    'NPB', # (950, 730, 480) NIR, PAN, BLU
    'NRB', # (950, 850, 480) NIR, RED, BLU
    'NRP', # (950, 850, 730) NIR, RED, PAN
    'RPB', # (850, 730, 480) RED, PAN, BLU
    'VNA', # (730, 730, 730) Visible-Near-IR
    'RGB' # (650, 550, 480) RED, GREEN, BLUE
    ]
SP_LIST = [
    # SP derived from CRISM (Viviano-Beck et al 2014)
    # 'R730', # dust/ice
    # 'RBR', # (730,480) ratio, npFeOx, dust in atmosphere
    # 'BD550', # (480,550,650) fine-grained crystalline hematite
    # 'SH650', # (550,650,730) ferric minerals
    # 'SH730', # (650,730,860) ferric minerals
    # 'BD640', # (550,650,730) Ferric minerals
    # 'BD860',  # (730,860,950) crystalline ferric (hematite)
    # 'S950' # (860, 950) slope at 950nm, crystalline ferric minerals + LCP
    # Sp derived from CaSSIS (Tornabene et al 2017)
    'RPR', # (860,730) RED/PAN Ratio
    'PBR', # (730,480) PAN/BLU Ratio
    'PNR', # (730, 950) PAN/NIR Ratio
    'ICE-ATM', # (Mean[480-950]/(950/480)) Atmospheric Ice Index
    'ICE-H2O' # (Mean[480-730]/(950)) Water Ice Index
    ]
CP_LIST = [    
    'CBRC1', 
    'CBRC2']

def spectral_parameter_stack(oifcub: np.ndarray,
                             no_data: float
                             ) -> tuple[np.ndarray, list[str]]:
    """Make stack of all spectral parameters (summary products)
    for CaSSIS, as defined in this module

    :param cifcub: CaSSIS I/F Cube
    :type cifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: Stacked spectral parameters and their names
    :rtype: tuple[np.ndarray, list[str]]
    """    
    sp_arrays = []
    sp_list = []
    for sp_id in SP_LIST:
        sp_func = spectral_parameter(sp_id)
        sp_arr = sp_func(oifcub, no_data)
        sp_arrays.append(sp_arr)
        sp_list.append(sp_id)
    return np.stack(sp_arrays), sp_list

def spectral_parameter(sp_id: str) -> Callable[[np.ndarray, float], np.ndarray]:
    """Get the spectral parameter function for the given spectral parameter ID

    :param sp_id: Spectral parameter ID
    :type sp_id: str
    :raises ValueError: If the spectral parameter ID is unknown
    :return: Spectral parameter function
    :rtype: Callable
    """
    if sp_id == 'RPR':
        sp_func = orochi_sp_rpr
    elif sp_id == 'PBR':
        sp_func = orochi_sp_pbr
    elif sp_id == 'PNR':
        sp_func = orochi_sp_pnr
    elif sp_id == 'ICE-ATM':
        sp_func = orochi_sp_ice_atm
    elif sp_id == 'ICE-H2O':
        sp_func = orochi_sp_ice_h2o
    else:
        raise ValueError(f'Unknown CaSSIS spectral parameter ID: {sp_id}')
    return sp_func

def orochi_sp_rpr(oifcub: np.ndarray, no_data: float
                  ) -> np.ndarray:
    """RED/PAN Ratio: 860nm / 730nm

    :param oifcub: OROCHI I/F Cube
    :type oifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: RED/PAN Ratio, 'RPR' key
    :rtype: tuple[np.ndarray, list[str]]
    """    
    no_data_locs = oifcub == no_data
    sp_arr = np.divide(oifcub[5], oifcub[4], where=~no_data_locs[4])
    sp_arr[no_data_locs[4]] = no_data
    return sp_arr

def orochi_sp_pnr(oifcub: np.ndarray, no_data: float
                  ) -> np.ndarray:
    """PAN/NIR Ratio: 730nm / 950nm

    :param oifcub: OROCHI I/F Cube
    :type oifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: PAN/NIR Ratio, 'PNR' key
    :rtype: tuple[np.ndarray, list[str]]
    """    
    no_data_locs = oifcub == no_data
    sp_arr = np.divide(oifcub[4], oifcub[6], where=~no_data_locs[6])
    sp_arr[no_data_locs[6]] = no_data
    return sp_arr

def orochi_sp_pbr(oifcub: np.ndarray, no_data: float
                  ) -> np.ndarray:
    """PAN/BLU Ratio: 730nm / 480nm

    :param oifcub: OROCHI I/F Cube
    :type oifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: PAN/BLU Ratio, 'PBR' key
    :rtype: tuple[np.ndarray, list[str]]
    """    
    no_data_locs = oifcub == no_data
    sp_arr = np.divide(oifcub[4], oifcub[1], where=~no_data_locs[1])
    sp_arr[no_data_locs[1]] = no_data
    return sp_arr

def orochi_sp_ice_atm(oifcub: np.ndarray, no_data: float
                      ) -> np.ndarray:
    """MEAN[480-950]/(950/480)
    Atmospheric Ice Index

    :param oifcub: OROCHI I/F Cube
    :type oifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: Atmospheric Ice Index, 'ICE-ATM' key
    :rtype: tuple[np.ndarray, list[str]]
    """
    no_data_locs = oifcub == no_data
    sp_arr = np.divide(np.mean(oifcub[1:7,:,:], axis=0), (oifcub[6] / oifcub[1]),
                       where=~no_data_locs[3])
    sp_arr[no_data_locs[3]] = no_data
    return sp_arr

def orochi_sp_ice_h2o(oifcub: np.ndarray, no_data: float
                      )-> np.ndarray:
    """MEAN[B,P,R]/NIR
    Water Ice Index

    :param oifcub: OROCHI I/F Cube
    :type oifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: Water Ice Index, 'ICE-H2O' key
    :rtype: tuple[np.ndarray, list[str]]
    """
    no_data_locs = oifcub == no_data
    sp_arr = np.divide(np.mean(oifcub[0:5,:,:], axis=0), (oifcub[6]),
                       where=~no_data_locs[6])
    sp_arr[no_data_locs[6]] = no_data
    return sp_arr

# def orochi_sp_R730(oifcub: np.ndarray, no_data: float
#                   ) -> np.ndarray:
#     """R730: Reflectance at 730nm

#     :param oifcub: OROCHI I/F Cube
#     :type oifcub: np.ndarray
#     :param no_data: No data value
#     :type no_data: float
#     :return: Reflectance at 730nm, 'R730' key
#     :rtype: tuple[np.ndarray, list[str]]
#     """
#     sp_arr = oifcub[4]
#     return sp_arr

# def orochi_sp_RBR(oifcub: np.ndarray, no_data: float
#                     ) -> np.ndarray:
#     """RBR: (730nm / 480nm) Ratio

#     :param oifcub: OROCHI I/F Cube
#     :type oifcub: np.ndarray
#     :param no_data: No data value
#     :type no_data: float
#     :return: RBR spectral parameter
#     :rtype: tuple[np.ndarray, list[str]]
#     """
#     no_data_locs = oifcub == no_data
#     sp_arr = np.divide(oifcub[4], oifcub[1], where=~no_data_locs[1])
#     sp_arr[no_data_locs[1]] = no_data
#     return sp_arr

# def orochi_sp_BD550(oifcub: np.ndarray, no_data: float
#                     ) -> np.ndarray:
#     """BD550: Band Depth at 550nm

#     :param oifcub: OROCHI I/F Cube
#     :type oifcub: np.ndarray
#     :param no_data: No data value
#     :type no_data: float
#     :return: BD550 spectral parameter
#     :rtype: tuple[np.ndarray, list[str]]
#     """
#     no_data_locs = oifcub == no_data
#     beta = 

def browse_product(bp_id: str
                ) -> Callable[[np.ndarray], tuple[np.ndarray, list[str]]]:
    """Get the browse product function for the given browse product ID

    :param bp_id: Browse product ID
    :type bp_id: str
    :raises ValueError: If the browse product ID is unknown
    :return: Browse product function
    :rtype: Callable
    """    
    if bp_id == 'NPB':
        bp_func = orochi_bp_npb
    elif bp_id == 'NRB':
        bp_func = orochi_bp_nrb
    elif bp_id == 'NRP':
        bp_func = orochi_bp_nrp
    elif bp_id == 'RPB':
        bp_func = orochi_bp_rpb
    elif bp_id == 'VNA':
        bp_func = orochi_bp_vna
    elif bp_id == 'RGB':
        bp_func = orochi_bp_rgb
    else:
        raise ValueError(f'Unknown OROCHI browse product ID: {bp_id}')
    return bp_func

def orochi_bp_npb(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NPB: NIR, PAN, BLU == Red, Green, Blue

    :param st_cube: 8-bit stretched OROCHI I/F Cube
    :type st_cube: np.ndarray
    :return: NPB browse product
    :rtype: tuple[np.ndarray, list[str]]
    """    
    band_list = ['950nm', '730nm', '480nm']
    r_arr = st_cube[6]
    g_arr = st_cube[4]
    b_arr = st_cube[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def orochi_bp_nrb(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NRB: NIR, RED, BLU == Red, Green, Blue

    :param st_cube: 8-bit stretched OROCHI I/F Cube
    :type st_cube: np.ndarray
    :return: NRB browse product
    :rtype: tuple[np.ndarray, list[str]]
    """    
    band_list = ['950nm', '860nm', '480nm']
    r_arr = st_cube[6]
    g_arr = st_cube[5]
    b_arr = st_cube[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def orochi_bp_nrp(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NRP: NIR, RED, PAN == Red, Green, Blue

    :param st_cube: 8-bit stretched OROCHI I/F Cube
    :type st_cube: np.ndarray
    :return: NRP browse product
    :rtype: tuple[np.ndarray, list[str]]
    """
    band_list = ['950nm', '860nm', '730nm']
    r_arr = st_cube[6]
    g_arr = st_cube[5]
    b_arr = st_cube[4]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def orochi_bp_rpb(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """RPB: RED, PAN, BLU == Red, Green, Blue

    :param st_cube: 8-bit stretched OROCHI I/F Cube
    :type st_cube: np.ndarray
    :return: RPB browse product
    :rtype: tuple[np.ndarray, list[str]]
    """
    band_list = ['860nm', '730nm', '480nm']
    r_arr = st_cube[5]
    g_arr = st_cube[4]
    b_arr = st_cube[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def orochi_bp_vna(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """PAN: PAN, PAN, PAN == Red, Green, Blue

    :param st_cube: stretched OROHCI I/F Cube
    :type st_cube: np.ndarray
    :return: PAN browse product
    :rtype: tuple[np.ndarray, list[str]]
    """
    band_list = ['730nm', '730nm', '730nm']
    r_arr = st_cube[4]
    g_arr = st_cube[4]
    b_arr = st_cube[4]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def orochi_bp_rgb(st_cube_mod: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """RGB browse product

    :param st_cube: stretched cube
    :type st_cube: np.ndarray
    :return: RGB browse product
    :rtype: (np.ndarray, list)
    """    
    band_list = ['650nm', '550nm', '480nm']
    r_arr = st_cube_mod[3]
    g_arr = st_cube_mod[2]
    b_arr = st_cube_mod[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def composite_product(cp_id: str):
    if cp_id == 'CBRC1':
        cp_func = orochi_cp_cbrc1
    elif cp_id == 'CBRC2':
        cp_func = orochi_cp_cbrc2
    else:
        raise ValueError(f'Composite product {cp_id} not recognised.')    
    return cp_func

def orochi_cp_cbrc1(ifcub: np.ndarray, no_data: float) -> tuple[np.ndarray, list[str]]:
    """CBRC1: RPR, PBR, PNR == Red, Green, Blue

    :param ifcub: Input I/F cube
    :type ifcub: np.ndarray
    :param no_data: No-data value
    :type no_data: float
    :return: Composite spectral parameter cube and band list
    :rtype: Tuple[np.ndarray, list[str]]
    """    
    sp_list = ['RPR', 'PBR', 'PNR']        
    cp_arr = np.array([
        orochi_sp_rpr(ifcub, no_data),
        orochi_sp_pbr(ifcub, no_data),
        orochi_sp_pnr(ifcub, no_data)
    ])
    return cp_arr, sp_list

def orochi_cp_cbrc2(ifcub: np.ndarray, no_data: float) -> tuple[np.ndarray, list[str]]:
    """CBRC2: RPR, PBR, ICE == Red, Green, Blue

    :param ifcub: Input I/F cube
    :type ifcub: np.ndarray
    :param no_data: No-data value
    :type no_data: float
    :return: Composite spectral parameter cube and band list
    :rtype: Tuple[np.ndarray, list[str]]
    """
    no_data_locs = ifcub[0] == no_data
    sp_list = ['RPR', 'PBR', 'ICE-ATM']
    cp_arr = np.array([
        orochi_sp_rpr(ifcub, no_data),
        orochi_sp_pbr(ifcub, no_data),
        orochi_sp_ice_atm(ifcub, no_data)
        ])
    return cp_arr, sp_list