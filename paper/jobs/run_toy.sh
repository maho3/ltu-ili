#PBS -N toy
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/ltu-ili/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=8:00:00
#PBS -l nodes=1:has1gpu:ppn=20,mem=64gb
#PBS -t 0-3

source /home/mattho/.bashrc
source /home/mattho/data/anaconda3/etc/profile.d/conda.sh

WDIR=${HOME}/git/ltu-ili/paper/wdir/
CFGDIR=${HOME}/git/ltu-ili/paper/configs/toy_model/
i=${PBS_ARRAYID}

cd ${WDIR}

models=(npe nle nre pydelfi)
model=${models[i]}

if [ ${model} == "pydelfi" ]; then
    conda activate ili-pydelfi
else
    conda activate ili-sbi
fi
echo "Running ${model}..."
python ${CFGDIR}/run.py --model ${model} --cfgdir ${CFGDIR}

echo "Done."
