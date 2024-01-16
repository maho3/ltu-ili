#PBS -N slcp
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/ltu-ili/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=08:00:00
#PBS -l nodes=1:has1gpu:ppn=4,mem=8gb
#PBS -t 1-9

source /home/mattho/.bashrc
source /home/mattho/data/anaconda3/etc/profile.d/conda.sh

WDIR=${HOME}/git/ltu-ili/paper/3_slcp
csv_file="./configs/batch.csv"
cd ${WDIR}

row_number=${PBS_ARRAYID}
delimiter=","

# Read the row from the CSV file
IFS=$delimiter read -r obs N seq inf <<< $(sed -n "${row_number}p" "$csv_file")

# Print the environment variables
echo "obs: $obs"
echo "N: $N"
echo "seq: $seq"
echo "inf: $inf"

if [[ $inf == *"pydelfi"* ]]; then
    conda activate ili-pydelfi
else
    conda activate ili-torch
fi

# echo "Running ${model}..."
python run.py --obs ${obs} --N ${N} --seq ${seq} --inf ${inf}

echo "Done."
