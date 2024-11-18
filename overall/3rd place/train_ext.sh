set -e -o pipefail

# Extended cross-validation
python train_cv.py -c pdsi_swe_era5_s51 -f True -e 2011 2015 -d /afs_base_syc -dm main_ext -ch True -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c base_swe -f True -e 2011 2015 -d /afs_base_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c swe_ua -f True -e 2011 2015 -d /afs_base_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c pdsi_swe_s51 -f True -e 2011 2015 -d /afs_base_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c pdsi_swe_era5 -f True -e 2011 2015 -d /afs_base_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c ngm_ua -f True -e 2011 2015 -d /afs_ngm_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c ngm_pdsi_ua_s51 -f True -e 2011 2015 -d /afs_ngm_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c ngm_pdsi_era5_s51 -f True -e 2011 2015 -d /afs_ngm_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
python train_cv.py -c ngm_pdsi_ua_era5_s51 -f True -e 2011 2015 -d /afs_ngm_syc -dm main_ext -v 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003
