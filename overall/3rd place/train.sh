set -e -o pipefail

python train_cv.py -c pdsi_swe_era5_s51 -f True -e 2011 2015 -d /afs_base_syc -dm main -ch True
python train_cv.py -c base_swe -f True -e 2011 2015 -d /afs_base_syc -dm main
python train_cv.py -c swe_ua -f True -e 2011 2015 -d /afs_base_syc -dm main
python train_cv.py -c pdsi_swe_s51 -f True -e 2011 2015 -d /afs_base_syc -dm main
python train_cv.py -c pdsi_swe_era5 -f True -e 2011 2015 -d /afs_base_syc -dm main
python train_cv.py -c ngm_ua -f True -e 2011 2015 -d /afs_ngm_syc -dm main
python train_cv.py -c ngm_pdsi_ua_s51 -f True -e 2011 2015 -d /afs_ngm_syc -dm main
python train_cv.py -c ngm_pdsi_era5_s51 -f True -e 2011 2015 -d /afs_ngm_syc -dm main
python train_cv.py -c ngm_pdsi_ua_era5_s51 -f True -e 2011 2015 -d /afs_ngm_syc -dm main -ch True
