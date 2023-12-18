#PBS -N slcp
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/ltu-ili/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=08:00:00
#PBS -l nodes=1:has1gpu:ppn=4,mem=8gb
#PBS -t 1-90

source /home/mattho/.bashrc
source /home/mattho/data/anaconda3/etc/profile.d/conda.sh

WDIR=${HOME}/git/ltu-ili/paper/3_slcp
csv_file="./configs/record.csv"
cd ${WDIR}

row_number=${PBS_ARRAYID}
delimiter=","

# Read the row from the CSV file
IFS=$delimiter read -r data inf val <<< $(sed -n "${row_number}p" "$csv_file")

# Print the environment variables
echo "data: $data"
echo "inf: $inf"
echo "val: $val"

if [[ $inf == *"pydelfi"* ]]; then
    conda activate ili-pydelfi
else
    conda activate ili-sbi
fi

# echo "Running ${model}..."
python run.py --data $data --inf $inf --val $val

echo "Done."
