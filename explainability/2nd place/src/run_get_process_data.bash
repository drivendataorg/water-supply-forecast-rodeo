#!/bin/bash -l
#
#SBATCH --job-name=get_data
#SBATCH --account=project_2002138
#SBATCH --time=8:00:00
#SBATCH --partition=fmi
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -c 4
#SBATCH --mem=185G
#SBATCH -o get_data.out





set -e -o pipefail

# Additional USGS data
for start_year in {1890..2020..10}
do
    end_year=$((start_year + 9))

    for year in $(seq $start_year $end_year)
    do
        echo $year
        python get_neighborhood_streamflow.py $year &
    done
    wait
done



# ECMWF data
for year in {1981..2024}
do
    for month in '01' '02' '03' '04'
    do
        echo $year $month
        python get_seasonal_from_cds.py $year $month &
    done
    echo ' '
    wait

    for month in '05' '06' '07' '12'
    do
        echo $year $month
        python get_seasonal_from_cds.py $year $month &
    done
    echo ' '
    wait
done


# Process the teleconnection indices
python preprocess_teleindices.py &

# Download and Preprocess various data sources
for year in {1971..2024}
do
    echo $year
    python preprocess_ecmwf.py $year &
    python get_process_SWANN.py $year &
    python preprocess_snotel.py $year &
    python preprocess_cdec.py $year &
    python preprocess_pdsi.py $year &
    wait
done






end=`date +%s`; runtime=`date -d@$((end-start)) -u +"%H:%M:%S"`
echo 'Finished! Total run time: '$runtime


echo ' '
