# MODEL DEPLOYMENT (FORECAST STAGE)
python train_cv_fcst_mm.py -c pdsi_swe_k5_s3_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd
python train_cv_fcst_mm.py -c pdsi_swe_k9_s1_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd
python train_cv_fcst_mm.py -c swe_k5_s3_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd
python train_cv_fcst_mm.py -c swe_k9_s1_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd

python train_cv_fcst_mm.py -c pdsi_swe_k5_s3_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 1024
python train_cv_fcst_mm.py -c pdsi_swe_k9_s1_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 1024
python train_cv_fcst_mm.py -c swe_k5_s3_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 1024
python train_cv_fcst_mm.py -c swe_k9_s1_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 1024

python train_cv_fcst_mm.py -c pdsi_swe_k5_s3_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 3024
python train_cv_fcst_mm.py -c pdsi_swe_k9_s1_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 3024
python train_cv_fcst_mm.py -c swe_k5_s3_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 3024
python train_cv_fcst_mm.py -c swe_k9_s1_dp -f True -e 2011 2015 -v 2020 2021 2022 -t 2023 -d _prd -s 3024