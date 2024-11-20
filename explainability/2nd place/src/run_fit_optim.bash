#!/bin/bash -l
#
#SBATCH --job-name=fit_optim
#SBATCH --account=xxx
#SBATCH --time=06:00:00
#SBATCH --partition=yyy
#SBATCH --mem=185G
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5


issue_month='01'            # '01 # '02' # '03' # '04' # '05' # '06' # '07'
experiment='loocv'          # 'kfold3' # 'loocv'
optimize='False'            # 'False' # 'True'
use_preopt_results='False'  # 'False' # 'True'
fit_models='False'          # 'False' # 'True'
fit_shap='False'            # 'False' # 'True'
use_slurm=0                 # 0 # 1

code_dir='/home/kamarain/prize-winner-template/src'
data_dir='/media/kamarain/varmuus/streamflow_data'

# Initialize conda
#source /home/kamarain/miniconda3/etc/profile.d/conda.sh
# eval "$(/fmi/scratch/project_2002138/miniconda/bin/conda shell.bash hook)"

# eval "$(/home/kamarain/miniconda3/bin/conda shell.bash hook)"
# conda activate watersupply_kamarain

#export PATH="/fmi/projappl/project_2002138/miniconda/bin:$PATH"
#export LD_LIBRARY_PATH="/fmi/projappl/project_2002138/miniconda/lib"

export OMP_NUM_THREADS=1





# Start the stopwatch
start=`date +%s`




sites=(
 'detroit_lake_inflow'
 'owyhee_r_bl_owyhee_dam'
 'pueblo_reservoir_inflow'
 'fontenelle_reservoir_inflow'
 'weber_r_nr_oakley'
 'san_joaquin_river_millerton_reservoir'
 'merced_river_yosemite_at_pohono_bridge'
 'american_river_folsom_lake'
 'colville_r_at_kettle_falls'
 'stehekin_r_at_stehekin'
 'virgin_r_at_virtin'
 'skagit_ross_reservoir'
 'boysen_reservoir_inflow'
 'pecos_r_nr_pecos'
 'hungry_horse_reservoir_inflow'
 'snake_r_nr_heise'
 'sweetwater_r_nr_alcova'
 'missouri_r_at_toston'
 'animas_r_at_durango'
 'yampa_r_nr_maybell'
 'libby_reservoir_inflow'
 'boise_r_nr_boise'
 'green_r_bl_howard_a_hanson_dam'
 'taylor_park_reservoir_inflow'
 'dillon_reservoir_inflow'
 'ruedi_reservoir_inflow'
)




for site in "${sites[@]}"
do
    for issue_day in "01" "08" "15" "22"
    do
        issue_time=$issue_month-$issue_day
        if [ $use_slurm -eq 1 ]; then
            echo 'Fitting with slurm'
            srun -N1 -n1 -c5 --mem=23G python $code_dir/fit_optim_qregress_multi.py $experiment $site $issue_time $optimize $use_preopt_results $fit_models $fit_shap $code_dir $data_dir &
        fi

        if [ $use_slurm -eq 0 ]; then
            echo 'Fitting without slurm'
            python $code_dir/fit_optim_qregress_multi.py $experiment $site $issue_time $optimize $use_preopt_results $fit_models $fit_shap $code_dir $data_dir &
        fi
    done
    if [ $use_slurm -eq 0 ]; then
        wait
    fi
done
wait










# Collect results and clean temp files
python $code_dir/combine_validation_results.py $data_dir $experiment $issue_month


rm $data_dir/metrics_${experiment}_*_${issue_month}-*.csv
rm $data_dir/crosval_${experiment}_*_${issue_month}-*.csv




end=`date +%s`; runtime=`date -d@$((end-start)) -u +"%H:%M:%S"`
echo 'Finished! Total run time: '$runtime


echo ' '
