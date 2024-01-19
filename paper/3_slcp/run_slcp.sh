#PBS -N slcp
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/ltu-ili/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=128,mem=256gb
#PBS -t 1-29

source /home/mattho/.bashrc
source /home/mattho/data/anaconda3/etc/profile.d/conda.sh
export OPENBLAS_NUM_THREADS=16

WDIR=${HOME}/git/ltu-ili/paper/3_slcp
csv_file="./configs/batch.csv"
cd ${WDIR}

row_number=${PBS_ARRAYID}
delimiter=","

# Read the row from the CSV file
IFS=$delimiter read -r obs N inf <<< $(sed -n "${row_number}p" "$csv_file")

# Print the environment variables
echo "obs: $obs"
echo "N: $N"
echo "inf: $inf"

if [[ $inf == *"pydelfi"* ]]; then
    conda activate ili-pydelfi
else
    conda activate ili-torch
fi

echo "Running training..."
python run.py --obs ${obs} --N ${N} --inf ${inf}

echo "Running metrics..."
conda activate ili-torch
python metrics.py --obs ${obs} --N ${N} --inf ${inf}

echo "Done."
