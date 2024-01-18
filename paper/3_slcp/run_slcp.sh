#PBS -N slcp
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/ltu-ili/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=08:00:00
#PBS -l nodes=1:has1gpu:ppn=8,mem=16gb
#PBS -t 91-240

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

echo "Running training..."
python run.py --obs ${obs} --N ${N} --inf ${inf}

echo "Running metrics..."
conda activate ili-torch
python metrics.py --obs ${obs} --N ${N} --inf ${inf}

echo "Done."
