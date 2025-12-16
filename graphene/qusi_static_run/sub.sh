#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclude=ca[0302,0405-0406,0503,0506,0606,0803,0902-0903,1001,1005,1901,2003],cc[0501,0603,0904],pcb[0802,0906,1303,1402,1501],pcc1205,pcd[0301,0405,0506,0605-0606,0902,1005,1103,1201,1303,1305,1502],pce[0303,0407,0602,0703,0803,1005,1101,1106,1203,2006]
# source /public1/soft/modules/module.sh
# module load anaconda/3-Python-3.8.3-phonopy-phono3py
# source activate lmp-dpmd
source ~/shipengjie/01-app/deepmd-kit/bin/activate
export TF_INTER_OP_PARALLELISM_THREADS=0
export TF_INTRA_OP_PARALLELISM_THREADS=0
# mpiexec.hydra -n 32 lmp -in in.graphene
python run_2.py
#cat  /proc/cpuinfo
 
