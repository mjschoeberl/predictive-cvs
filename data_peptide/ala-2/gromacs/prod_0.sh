#!/bin/sh -f

#PBS -l nodes=1:ppn=1
#PBS -l walltime=800:00:00
#PBS -M mschoeberl@gmail.com
#PBS -m abe
#PBS -d /home/markus/projects/peptide/ala-2/2_production/

echo -n "this script is running on: "
hostname -f
echo "selected queue: ${PBS_QUEUE}"
date

sleep 1s
ulimit -a
export LD_LIBRARY_PATH=/opt/openmpi/1.4.5/gcc/lib64/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/markus/lib/openmpi/lib64:$LD_LIBRARY_PATH
/home/markus/software/gromacs-5.0.4/build/bin/gmx mdrun -s prod_0.tpr -c prodconf_0.gro -o prod_0.trr -e prod_0.edr
# /usr/lib64/mpi/gcc/openmpi/bin/mpirun lmp_mpi < in.spce
