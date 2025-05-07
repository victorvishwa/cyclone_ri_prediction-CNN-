# This file defines important configurations, including data sources and model parameters.

# API endpoints and model parameters
DATA_SOURCES = {
    'satellite': {
        'url': 'https://rammb-slider.cira.colostate.edu/data/json/goes-16/full_disk/geocolor/latest_times.json',
        'image_pattern': 'https://rammb-slider.cira.colostate.edu/data/imagery/{date}/goes-16---full_disk/geocolor/{time}/00_{resolution}x{resolution}.jpg',
        'region_of_interest': (60, 100, 0, 30)  # lon_min, lon_max, lat_min, lat_max (Indian Ocean)
    },
    'best_track': {
        'ibtracs': 'https://www.ncei.noaa.gov/data/international-best-track-archive/ibtracs/v04r00/access/csv/ibtracs.last3years.list.v04r00.csv',
        'active_storms': 'https://www.nhc.noaa.gov/gtwo.php?basin=io'  # Indian Ocean active storms
    }
}

MODEL_CONFIG = {
    'input_shape': (256, 256, 3),
    'ri_threshold': 0.7,
    'location_weight': 0.3,
    'min_wind_speed': 35,
    'basin': 'NI'
}