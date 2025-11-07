# Module for generating CaSSIS browse products from a CaSSIS I/F cube
# Assumes input cube is np.array(5 x N x M),
# with band order of [BLU, PAN, 'RED', 'NIR', 'sBLU']
# where the 5th sBLU channel is added to the normal 4-band cube
# ([BLU, PAN, 'RED', 'NIR'])
# using the cassis_add_sblu function
import numpy as np

BP_LIST = ['NPB', 'NRB', 'NRP', 'RPB', 'RGB']

def cassis_browse_product(id):
    if id == 'NPB':
        bp_func = cassis_bp_npb
    elif id == 'NRB':
        bp_func = cassis_bp_nrb
    elif id == 'NRP':
        bp_func = cassis_bp_nrp
    elif id == 'RPB':
        bp_func = cassis_bp_rpb
    elif id == 'PAN':
        bp_func = cassis_bp_pan
    elif id == 'RGB':
        bp_func = cassis_bp_rgb
    else:
        raise ValueError(f'Unknown CaSSIS browse product ID: {id}')
    return bp_func

def cassis_add_sblu(cube, no_data):
    # make synthetic blu channel
    # mask with no data
    no_data_mask = (cube[0] != no_data) & (cube[1] != no_data)
    sBLU = (2*cube[0]) - (0.3*cube[1])
    sBLU[~no_data_mask] = no_data
    # add sBLU to the cube
    cube = np.append(cube, [sBLU], axis=0)
    return cube

def cassis_bp_npb(st_cube):
    band_list = ['NIR', 'PAN', 'BLU']
    r_arr = st_cube[3]
    g_arr = st_cube[1]
    b_arr = st_cube[0]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_nrb(st_cube):
    band_list = ['NIR', 'RED', 'BLU']
    r_arr = st_cube[3]
    g_arr = st_cube[2]
    b_arr = st_cube[0]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_nrp(st_cube):
    band_list = ['NIR', 'RED', 'PAN']
    r_arr = st_cube[3]
    g_arr = st_cube[2]
    b_arr = st_cube[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_rpb(st_cube):
    band_list = ['RED', 'PAN', 'BLU']
    r_arr = st_cube[2]
    g_arr = st_cube[1]
    b_arr = st_cube[0]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_pan(st_cube):
    band_list = ['PAN', 'PAN', 'PAN']
    r_arr = st_cube[1]
    g_arr = st_cube[1]
    b_arr = st_cube[1]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list

def cassis_bp_rgb(st_cube):
    """Special RGB browse product, that requires a modified st_cube
    that includes the synthetic BLU channel

    :param st_cube: stretched cube with synthetic BLU channel
    :type st_cube: np.ndarray
    :return: RGB browse product
    :rtype: (np.ndarray, list)
    """    
    band_list = ['PAN', 'BLU', 'sBLU']
    r_arr = st_cube[1]
    g_arr = st_cube[0]
    b_arr = st_cube[4]

    browse_arr = np.array([r_arr, g_arr, b_arr])
    browse_arr = browse_arr.transpose((1, 2, 0))

    return browse_arr, band_list