# Module for generating CaSSIS browse products from a CaSSIS I/F cube
# Assumes input cube is np.array(5 x N x M),
# with band order of [BLU, PAN, 'RED', 'NIR', 'sBLU']
# where the 5th sBLU channel is added to the standard 4-band cube ([BLU, PAN, 
# RED, NIR]) using the cassis_add_sblu function
from typing import Callable
import numpy as np

BP_LIST = ['NPB', 'NRB', 'NRP', 'RPB', 'PAN','RGB']
SP_LIST = ['RPR','PBR','PNR','ICE-ATM','ICE-H2O']
CP_LIST = ['CBRC1', 'CBRC2']

def spectral_parameter_stack(cifcub: np.ndarray,
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
        sp_arr = sp_func(cifcub, no_data)
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
        sp_func = cassis_sp_rpr
    elif sp_id == 'PBR':
        sp_func = cassis_sp_pbr
    elif sp_id == 'PNR':
        sp_func = cassis_sp_pnr
    elif sp_id == 'ICE-ATM':
        sp_func = cassis_sp_ice_atm
    elif sp_id == 'ICE-H2O':
        sp_func = cassis_sp_ice_h2o
    else:
        raise ValueError(f'Unknown CaSSIS spectral parameter ID: {sp_id}')
    return sp_func

def cassis_sp_rpr(cifcub: np.ndarray, no_data: float
                  ) -> np.ndarray:
    """RED/PAN Ratio

    :param cifcub: CaSSIS I/F Cube
    :type cifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: RED/PAN Ratio, 'RPR' key
    :rtype: tuple[np.ndarray, list[str]]
    """    
    no_data_locs = cifcub == no_data
    sp_arr = np.divide(cifcub[2], cifcub[1], where=~no_data_locs[1])
    sp_arr[no_data_locs[1]] = no_data
    return sp_arr

def cassis_sp_pnr(cifcub: np.ndarray, no_data: float
                  ) -> np.ndarray:
    """PAN/NIR Ratio

    :param cifcub: CaSSIS I/F Cube
    :type cifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: PAN/NIR Ratio, 'PNR' key
    :rtype: tuple[np.ndarray, list[str]]
    """    
    no_data_locs = cifcub == no_data
    sp_arr = np.divide(cifcub[1], cifcub[3], where=~no_data_locs[3])
    sp_arr[no_data_locs[3]] = no_data
    return sp_arr

def cassis_sp_pbr(cifcub: np.ndarray, no_data: float
                  ) -> np.ndarray:
    """PAN/BLU Ratio

    :param cifcub: CaSSIS I/F Cube
    :type cifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: PAN/BLU Ratio, 'PBR' key
    :rtype: tuple[np.ndarray, list[str]]
    """    
    no_data_locs = cifcub == no_data
    sp_arr = np.divide(cifcub[1], cifcub[0], where=~no_data_locs[0])
    sp_arr[no_data_locs[0]] = no_data
    return sp_arr

def cassis_sp_ice_atm(cifcub: np.ndarray, no_data: float
                      ) -> np.ndarray:
    """MEAN[B,P,R,N]/(NIR/BLU)
    Atmospheric Ice Index

    :param cifcub: CaSSIS I/F Cube
    :type cifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: Atmospheric Ice Index, 'ICE-ATM' key
    :rtype: tuple[np.ndarray, list[str]]
    """
    no_data_locs = cifcub == no_data
    sp_arr = np.divide(np.mean(cifcub, axis=0), (cifcub[3] / cifcub[0]),
                       where=~no_data_locs[3])
    sp_arr[no_data_locs[3]] = no_data
    return sp_arr

def cassis_sp_ice_h2o(cifcub: np.ndarray, no_data: float
                      )-> np.ndarray:
    """MEAN[B,P,R]/NIR
    Water Ice Index

    :param cifcub: CaSSIS I/F Cube
    :type cifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: Water Ice Index, 'ICE-H2O' key
    :rtype: tuple[np.ndarray, list[str]]
    """
    no_data_locs = cifcub == no_data
    sp_arr = np.divide(np.mean(cifcub[0:3,:,:], axis=0), (cifcub[3]),
                       where=~no_data_locs[3])
    sp_arr[no_data_locs[3]] = no_data
    return sp_arr

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
        bp_func = cassis_bp_npb
    elif bp_id == 'NRB':
        bp_func = cassis_bp_nrb
    elif bp_id == 'NRP':
        bp_func = cassis_bp_nrp
    elif bp_id == 'RPB':
        bp_func = cassis_bp_rpb
    elif bp_id == 'PAN':
        bp_func = cassis_bp_pan
    elif bp_id == 'RGB':
        bp_func = cassis_bp_rgb
    else:
        raise ValueError(f'Unknown CaSSIS browse product ID: {bp_id}')
    return bp_func

def cassis_add_sblu(cifcub: np.ndarray, no_data: float) -> np.ndarray:
    """Create and append a 'Synthetic-BLU' channel to the CaSSIS I/F Cube,
    as described by Tornabene et al 2017 doi:10.1007/s11214-017-0436-7

    :param cifcub: CaSSIS I/F Cube
    :type cifcub: np.ndarray
    :param no_data: No data value
    :type no_data: float
    :return: CaSSIS I/F Cube with synthetic BLU channel appended
    :rtype: np.ndarray
    """    
    # make synthetic blu channel
    # mask with no data
    no_data_mask = (cifcub[0] != no_data) & (cifcub[1] != no_data)
    sBLU = (2*cifcub[0]) - (0.3*cifcub[1])
    sBLU[~no_data_mask] = no_data
    # add sBLU to the cube
    cifcub_mod = np.append(cifcub, [sBLU], axis=0)
    return cifcub_mod

def cassis_bp_npb(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NPB: NIR, PAN, BLU == Red, Green, Blue

    :param st_cube: 8-bit stretched CaSSIS I/F Cube
    :type st_cube: np.ndarray
    :return: NPB browse product
    :rtype: tuple[np.ndarray, list[str]]
    """    
    band_list = ['NIR', 'PAN', 'BLU']
    r_arr = st_cube[3]
    g_arr = st_cube[1]
    b_arr = st_cube[0]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_nrb(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NRB: NIR, RED, BLU == Red, Green, Blue

    :param st_cube: 8-bit stretched CaSSIS I/F Cube
    :type st_cube: np.ndarray
    :return: NRB browse product
    :rtype: tuple[np.ndarray, list[str]]
    """    
    band_list = ['NIR', 'RED', 'BLU']
    r_arr = st_cube[3]
    g_arr = st_cube[2]
    b_arr = st_cube[0]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_nrp(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NRP: NIR, RED, PAN == Red, Green, Blue

    :param st_cube: 8-bit stretched CaSSIS I/F Cube
    :type st_cube: np.ndarray
    :return: NRP browse product
    :rtype: tuple[np.ndarray, list[str]]
    """
    band_list = ['NIR', 'RED', 'PAN']
    r_arr = st_cube[3]
    g_arr = st_cube[2]
    b_arr = st_cube[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_rpb(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """RPB: RED, PAN, BLU == Red, Green, Blue

    :param st_cube: 8-bit stretched CaSSIS I/F Cube
    :type st_cube: np.ndarray
    :return: RPB browse product
    :rtype: tuple[np.ndarray, list[str]]
    """
    band_list = ['RED', 'PAN', 'BLU']
    r_arr = st_cube[2]
    g_arr = st_cube[1]
    b_arr = st_cube[0]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_pan(st_cube: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """PAN: PAN, PAN, PAN == Red, Green, Blue

    :param st_cube: stretched CaSSIS I/F Cube
    :type st_cube: np.ndarray
    :return: PAN browse product
    :rtype: tuple[np.ndarray, list[str]]
    """
    band_list = ['PAN', 'PAN', 'PAN']
    r_arr = st_cube[1]
    g_arr = st_cube[1]
    b_arr = st_cube[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_rgb(st_cube_mod: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Special RGB browse product, that requires a modified st_cube
    that includes the synthetic BLU channel

    :param st_cube: stretched cube with synthetic BLU channel
    :type st_cube: np.ndarray
    :return: RGB browse product
    :rtype: (np.ndarray, list)
    """    
    band_list = ['PAN', 'BLU', 'sBLU']
    r_arr = st_cube_mod[1]
    g_arr = st_cube_mod[0]
    b_arr = st_cube_mod[4]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def composite_product(cp_id: str):
    if cp_id == 'CBRC1':
        cp_func = cassis_cp_cbrc1
    elif cp_id == 'CBRC2':
        cp_func = cassis_cp_cbrc2
    else:
        raise ValueError(f'Composite product {cp_id} not recognised.')    
    return cp_func

def cassis_cp_cbrc1(ifcub: np.ndarray, no_data: float) -> tuple[np.ndarray, list[str]]:
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
        cassis_sp_rpr(ifcub, no_data),
        cassis_sp_pbr(ifcub, no_data),
        cassis_sp_pnr(ifcub, no_data)
    ])
    return cp_arr, sp_list

def cassis_cp_cbrc2(ifcub: np.ndarray, no_data: float) -> tuple[np.ndarray, list[str]]:
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
        cassis_sp_rpr(ifcub, no_data),
        cassis_sp_pbr(ifcub, no_data),
        cassis_sp_ice_atm(ifcub, no_data)
        ])
    return cp_arr, sp_list