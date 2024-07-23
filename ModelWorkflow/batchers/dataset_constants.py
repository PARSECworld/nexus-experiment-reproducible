DHS_COUNTRIES = ['brazil']

LSMS_COUNTRIES = ['brazil']


_SURVEY_NAMES_DHS_OOC_A = {
    'train': ['brazil_2010'],
    'val': [],
    'test': [],
}
_SURVEY_NAMES_DHS_OOC_B = {
    'train': ['brazil_2010'],
    'val': [],
    'test': [],
}
_SURVEY_NAMES_DHS_OOC_C = {
    'train': ['brazil_2010'],
    'val': [],
    'test': [],
}
_SURVEY_NAMES_DHS_OOC_D = {
    'train': ['brazil_2010'],
    'val': [],
    'test': [],
}
_SURVEY_NAMES_DHS_OOC_E = {
    'train': ['brazil_2010'],
    'val': [],
    'test': [],
}

SURVEY_NAMES = {  # TODO: rename to SURVEY_NAMES_DHS?
    'DHS_OOC_A': _SURVEY_NAMES_DHS_OOC_A,
    'DHS_OOC_B': _SURVEY_NAMES_DHS_OOC_B,
    'DHS_OOC_C': _SURVEY_NAMES_DHS_OOC_C,
    'DHS_OOC_D': _SURVEY_NAMES_DHS_OOC_D,
    'DHS_OOC_E': _SURVEY_NAMES_DHS_OOC_E
}

SURVEY_NAMES_LSMS = ['brazil_2010']

TOTAL_SIZE = 23675

SIZES = { 
    'DHS': {'train': TOTAL_SIZE, 'val': 0, 'test': 0, 'all': TOTAL_SIZE},
    'DHSNL': {'all': TOTAL_SIZE},
    'DHS_OOC_A': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_OOC_B': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_OOC_C': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_OOC_D': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_OOC_E': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_incountry_A': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_incountry_B': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_incountry_C': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_incountry_D': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
    'DHS_incountry_E': {'train': 18000, 'val': 6000, 'test': 6000, 'all': TOTAL_SIZE},
}

URBAN_SIZES = {
    'DHS': {'train': 3954, 'val': 1212, 'test': 1635, 'all': 6801},
    'DHS_OOC_A': {'train': 4264, 'val': 1221, 'test': 1316, 'all': 6801},
    'DHS_OOC_B': {'train': 4225, 'val': 1355, 'test': 1221, 'all': 6801},
    'DHS_OOC_C': {'train': 4010, 'val': 1436, 'test': 1355, 'all': 6801},
    'DHS_OOC_D': {'train': 3892, 'val': 1473, 'test': 1436, 'all': 6801},
    'DHS_OOC_E': {'train': 4012, 'val': 1316, 'test': 1473, 'all': 6801},
}

RURAL_SIZES = {
    'DHS': {'train': 8365, 'val': 2045, 'test': 2458, 'all': 12868},
    'DHS_OOC_A': {'train': 7533, 'val': 2688, 'test': 2647, 'all': 12868},
    'DHS_OOC_B': {'train': 7595, 'val': 2585, 'test': 2688, 'all': 12868},
    'DHS_OOC_C': {'train': 7790, 'val': 2493, 'test': 2585, 'all': 12868},
    'DHS_OOC_D': {'train': 7920, 'val': 2455, 'test': 2493, 'all': 12868},
    'DHS_OOC_E': {'train': 7766, 'val': 2647, 'test': 2455, 'all': 12868},
}

# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands

# Old means used by Ali
#_MEANS_DHS = {
#    'BLUE':  0.059183,
#    'GREEN': 0.088619,
#    'RED':   0.104145,
#    'SWIR1': 0.246874,
#    'SWIR2': 0.168728,
#    'TEMP1': 299.078023,
#    'NIR':   0.253074,
#    'DMSP':  4.005496,
#    'VIIRS': 1.096089,
#    # 'NIGHTLIGHTS': 5.101585, # nightlights overall
#}

# Means used in RafCarIgor Project
#_MEANS_DHS = {
# 'BLUE': 0.041884,
# 'GREEN': 0.067119,
# 'RED': 0.073044,
# 'SWIR1': 0.227129,
# 'SWIR2': 0.126073,
# 'TEMP1': 297.948739,
# 'NIR': 0.246790,
# 'DMSP': 1.805470,
# 'VIIRS': 0.0
#}

# Means used in RafCarIgor Paper - 50000 images
#_MEANS_DHS = {
# 'BLUE': 0.041970,
# 'GREEN': 0.067411,
# 'RED': 0.073357,
# 'SWIR1': 0.228420,
# 'SWIR2': 0.126632,
# 'TEMP1': 297.969432,
# 'NIR': 0.247321,
# 'DMSP': 0.575938,
# 'VIIRS': 0.0
#}

# Means used in RafCarIgor Paper - 30000 images
_MEANS_DHS = {
 'BLUE' : 0.041997,
 'GREEN': 0.067451,
 'RED'  : 0.073398,
 'SWIR1': 0.228551,
 'SWIR2': 0.126762,
 'TEMP1': 297.979406,
 'NIR'  : 0.247326,
 'DMSP' : 0.568886,
 'VIIRS': 0.0
}


_MEANS_DHSNL = {
    'BLUE':  0.063927,
    'GREEN': 0.091981,
    'RED':   0.105234,
    'SWIR1': 0.235316,
    'SWIR2': 0.162268,
    'TEMP1': 298.736746,
    'NIR':   0.245430,
    'DMSP':  7.152961,
    'VIIRS': 2.322687,
}

_MEANS_LSMS = {
    'BLUE':  0.062551,
    'GREEN': 0.090696,
    'RED':   0.105640,
    'SWIR1': 0.242577,
    'SWIR2': 0.165792,
    'TEMP1': 299.495280,
    'NIR':   0.256701,
    'DMSP':  5.105815,
    'VIIRS': 0.557793,
}

# Std_devs used in Ali
#_STD_DEVS_DHS = {
#    'BLUE':  0.022926,
#    'GREEN': 0.031880,
#    'RED':   0.051458,
#    'SWIR1': 0.088857,
#    'SWIR2': 0.083240,
#    'TEMP1': 4.300303,
#    'NIR':   0.058973,
#    'DMSP':  23.038301,
#    'VIIRS': 4.786354,
#    # 'NIGHTLIGHTS': 23.342916, # nightlights overall
#}

# Std_devs used in RafCarIgor Project
#_STD_DEVS_DHS = {
# 'BLUE': 0.017966,
# 'GREEN': 0.021222,
# 'RED': 0.033227,
# 'SWIR1': 0.069806,
# 'SWIR2': 0.056328,
# 'TEMP1': 2.128368,
# 'NIR': 0.041571,
# 'DMSP': 16.171133,
# 'VIIRS': 0.0
#}

# Std_devs used in RafCarIgor Paper - 50000 images
#_STD_DEVS_DHS = {
# 'BLUE': 0.018486,
# 'GREEN': 0.020820,
# 'RED': 0.032774,
# 'SWIR1': 0.069165,
# 'SWIR2': 0.055493,
# 'TEMP1': 2.110832,
# 'NIR': 0.041874,
# 'DMSP': 3.616887,
# 'VIIRS': 0.0
#}

_STD_DEVS_DHS = {
 'BLUE' : 0.017343,
 'GREEN': 0.020520,
 'RED'  : 0.032392,
 'SWIR1': 0.068925,
 'SWIR2': 0.055515,
 'TEMP1': 2.112616,
 'NIR'  : 0.042004,
 'DMSP' : 3.594086,
 'VIIRS': 0.0
}

_STD_DEVS_DHSNL = {
    'BLUE':  0.023697,
    'GREEN': 0.032474,
    'RED':   0.051421,
    'SWIR1': 0.095830,
    'SWIR2': 0.087522,
    'TEMP1': 6.208949,
    'NIR':   0.071084,
    'DMSP':  29.749457,
    'VIIRS': 14.611589,
}
_STD_DEVS_LSMS = {
    'BLUE':  0.023979,
    'GREEN': 0.032121,
    'RED':   0.051943,
    'SWIR1': 0.088163,
    'SWIR2': 0.083826,
    'TEMP1': 4.678959,
    'NIR':   0.059025,
    'DMSP':  31.688320,
    'VIIRS': 6.421816,
}

MEANS_DICT = {
    'DHS_OOC_A': _MEANS_DHS,
    'DHS': _MEANS_DHS,
    'DHSNL': _MEANS_DHSNL,
    'LSMS': _MEANS_LSMS,
}

STD_DEVS_DICT = {
    'DHS_OOC_A': _STD_DEVS_DHS,
    'DHS': _STD_DEVS_DHS,
    'DHSNL': _STD_DEVS_DHSNL,
    'LSMS': _STD_DEVS_LSMS,
}