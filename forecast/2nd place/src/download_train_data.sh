#!/bin/bash

python -m wsfr_download bulk data_download/hindcast_train_config_snotel.yml
python -m wsfr_download bulk data_download/hindcast_test_config_snotel.yml

python -m wsfr_download bulk data_download/hindcast_train_config_usgs_streamflow.yml
python -m wsfr_download bulk data_download/hindcast_test_config_usgs_streamflow.yml