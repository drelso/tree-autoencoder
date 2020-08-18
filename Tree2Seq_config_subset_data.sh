#!/bin/bash -l
# Batch script to run a GPU job on Legion under SGE.
# 0. Force bash as the executing shell.
#$ -S /bin/bash
# 1. Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:00
# 3. Request 12 gigabytes of RAM (must be an integer)
#$ -l mem=12G
# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G
# 5. Set the name of the job.
#$ -N Tree2Seq_config_subset_data
# 6. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/uczcdra/Scratch/tree-autoencoder
# 7. Your work *must* be done in $TMPDIR 
cd $TMPDIR
# 8. load the cuda module (in case you are running a CUDA program
# module unload compilers mpi
# module load compilers/gnu/4.9.2
# module load cuda/9.0.176-patch4/gnu-4.9.2
# 9. Run the application - the line below is just a random example.
#virtualenv envs/skipgram_syns_env
echo 'MYRIAD JOB DETAILS:'
echo '$PATH:'
echo $PATH
echo 'Available compilers'
module avail compilers
echo '$HOME'
echo $HOME
#export PYTHONPATH=/home/uczcdra/python_src/Python-3.7.4
#export PATH=/home/uczcdra/python_src/Python-3.7.4:$PATH
export PYTHONPATH=$HOME/python_src/bin
export PATH=$HOME/python_src/bin:$PATH
#source "$HOME/Scratch/sculpting-word-embeddings/envs/skipgram_syns_env_Scratch/bin/activate" --always-copy
#!/usr/bin/env python3
/usr/bin/time --verbose python3 $HOME/Scratch/tree-autoencoder/tree2seq_wird-embs_batch.py # this prints the usage of the program
# 10. Preferably, tar-up (archive) all output files onto the shared scratch area
tar zcvf $HOME/Scratch/myriad_output/files_from_job_$JOB_ID.tar.gz $TMPDIR
# Make sure you have given enough time for the copy to complete!