#!/bin/bash -l
# Batch script to run a GPU job on Legion under SGE.
# 0. Force bash as the executing shell.
#$ -S /bin/bash
# 1. Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=47:00:00
# 3. Request 60 gigabytes of RAM (must be an integer)
#$ -l mem=60G
# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G
# 5. Set the name of the job.
#$ -N Tree2Seq_FullScale_TF25_LR0001
# 6. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/uczcdra/Scratch/tree-autoencoder
# 7. Your work *must* be done in $TMPDIR 
cd $TMPDIR
# 8. load the cuda module (in case you are running a CUDA program
module unload compilers mpi
module load compilers/gnu/4.9.2
module load compilers/intel/2020/release
module load cuda/10.1.243/gnu-4.9.2
module load cudnn/7.5.0.56/cuda-10.1
# 9. Run the application - the line below is just a random example.
#virtualenv envs/skipgram_syns_env
echo 'MYRIAD JOB DETAILS:'
echo '$PATH:'
echo $PATH
echo 'Available compilers'
module avail compilers
echo '$HOME'
echo $HOME
export PYTHONPATH=$HOME/Scratch/tree-autoencoder/myriad_venv/bin/
export PATH=$HOME/Scratch/tree-autoencoder/myriad_venv/bin/:$PATH
source "$HOME/Scratch/tree-autoencoder/myriad_venv/bin/activate" --always-copy
which python
/usr/bin/time --verbose python $HOME/Scratch/tree-autoencoder/tree2seq.py # this prints the usage of the program
# 10. Preferably, tar-up (archive) all output files onto the shared scratch area
tar zcvf $HOME/Scratch/tree-autoencoder/myriad_output/files_from_job_$JOB_ID.tar.gz $TMPDIR
# Make sure you have given enough time for the copy to complete!
