#!/bin/bash

set -e -o pipefail

python -m wsfr_download bulk data_download/hindcast_train_config_snotel.yml
python -m wsfr_download bulk data_download/hindcast_test_config_snotel.yml

python -m wsfr_download bulk data_download/hindcast_train_config_usgs_streamflow.yml
python -m wsfr_download bulk data_download/hindcast_test_config_usgs_streamflow.yml

python download_USBR_reservoir_inflow.py
