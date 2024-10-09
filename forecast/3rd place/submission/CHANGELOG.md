# Forecast Stage Submission

## General

Our solution is the extension of Hindcast stage with additional PDSI model

1. Get the latest available data by removing NA value (set -999999 in prior)
2. Scale SWE to 0-max, relax data completeness at least 65%
3. Ensemble 4 models variant x 3 seeds x 3 validation years = 36 models

## Changelog

### 2024-05-23

* Log raw dataframe output from USBR