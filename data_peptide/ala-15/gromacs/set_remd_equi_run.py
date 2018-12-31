# Write replica exchange scripts for Gromacs
# ALA-15 peptide
# Author: Markus Schoeberl, mschoeberl@gmail.com

import numpy as np

from subprocess import call
import os

def qsub( fname, command, index ):
  
  nodeMax = 12
  nodes = 1
  ppn = 1
  
  if not bEqui:
    nodes = 5
    ppn = 5
  
  sCont = "#!/bin/sh -f\n\n"
  sCont = sCont + "#PBS -l nodes=" + str(nodes) + ":ppn=" + str(ppn) + "\n"
  iNodeToUse = index%nodeMax #(index/nodeMax)*nodeMax + index%nodeMax
  #sCont = sCont + "#PBS -l nodes=node" + str(iNodeToUse).zfill(2) + ":ppn=" + str(ppn) + "\n"
  #sCont = sCont + "#PBS -l nodes=node06" + ":ppn=" + str(ppn) + "+node07" + ":ppn=" + str(ppn) + "+node08" + ":ppn=" + str(ppn) + "+node09" + ":ppn=" + str(ppn) + "+node10" + ":ppn=" + str(ppn) + "\n"
  sCont = sCont + "#PBS -N GMX_MD_15_20_REP\n"
  sCont = sCont + "#PBS -l walltime=800:00:00\n"
  sCont = sCont + "#PBS -M mschoeberl@gmail.com\n"
  sCont = sCont + "#PBS -m abe\n"
  sCont = sCont + "#PBS -d "+ workingDirCluster + "\n\n"
  sCont = sCont + "echo -n \"this script is running on: \"\n"
  sCont = sCont + "hostname -f\n"
  sCont = sCont + "echo \"selected queue: ${PBS_QUEUE}\"\n"
  sCont = sCont + "date\n\n"
  sCont = sCont + "sleep 1s\n"
  sCont = sCont + "ulimit -a\n"
  sCont = sCont + "export LD_LIBRARY_PATH=/opt/openmpi/1.4.5/gcc/lib64/:$LD_LIBRARY_PATH\n"
  # command here
  sCont = sCont + command + "\n"
  
  writeFile( sCont, fname )
  
def mdRunREMD( replicas ):
  
  # choose steps between replicas
  stepBrep = 2000
  
  tprName = "remd_.tpr"
  groOut = "remd_.gro"
  trrOut = "remd_.xtc"
  
  if (bCluster==True):
    gmx = gromacsGMXClusterMpi
    
    sCommand = "/opt/openmpi/1.4.5/gcc/bin/mpirun -np " + str(replicas) + " " + gmx + " mdrun -s remd_.tpr -multi " + str(replicas) + " -replex " + str(stepBrep) # " mdrun -s " + tprName + " -c " + groOut + " -o " + trrOut
    
    # write qsub file
    qsubFname = workingDirCluster + "remd.sh"
    
    qsub(qsubFname, sCommand, 0)
    qsubCommand = "qsub " + qsubFname
    #os.system(qsubCommand)
    
  else:
    gmx = gromacsGMXmpi
    sCommand = sCommand = "mpirun -np " + str(replicas) + " " + gmx + " mdrun -s remd_.tpr -multi " + str(replicas) + " -replex " + str(stepBrep) # " mdrun -s " + tprName + " -c " + groOut + " -o " + trrOut
    os.system(sCommand)
    
def mdRun( step ):
  
  tprName = "equi_"+str(step)+".tpr"
  groOut = "equiconf_"+str(step)+".gro"
  trrOut = "equi_"+str(step)+".trr"
  edrName = "equi_"+str(step)+".edr"
  
  if (bCluster==True):
    gmx = gromacsGMXCluster
    
    sCommand = gmx + " mdrun -s " + tprName + " -c " + groOut + " -o " + trrOut + " -e " + edrName
    
    # write qsub file
    qsubFname = workingDirCluster + "equi_"+str(step)+".sh"
    
    qsub(qsubFname, sCommand, step)
    qsubCommand = "qsub " + qsubFname
    os.system(qsubCommand)
    
  else:
    gmx = gromacsGMX
    sCommand = sCommand = gmx + " mdrun -s " + tprName + " -c " + groOut + " -o " + trrOut + " -e " + edrName
    os.system(sCommand)
  

def createTprFile( step ):
  
  if bEqui:
    mdpName = "equi_"+str(step)+".mdp"
    tprName = "equi_"+str(step)+".tpr"
    edrName = "equi_"+str(step)+".edr"
    groName = "ala_15_alpha.gro"#equibrillation.gro"
    topName = "ala_15_alpha.top"
  else:
    mdpName = "remd_"+str(step)+".mdp"
    tprName = "remd_"+str(step)+".tpr"
    edrName = "remd_"+str(step)+".edr"
    groName = "equiconf_"+str(step)+".gro"
    print step
    
    topName = "alpha_ala_15.top"
  
  if (bCluster==True):
    gmx = gromacsGMXCluster
    
    sCommand = gmx + " grompp -f " + mdpName + " -c " + groName + " -p " + topName + " -o " + tprName + " -e " + edrName
    
    # write qsub file
    qsubFname = workingDirCluster + "equi_"+str(step)+".sh"
    os.system(sCommand)
    #qsub(qsubFname, sCommand)

  else:
    gmx = gromacsGMX
    sCommand = gmx + " grompp -f " + mdpName + " -c " + groName + " -p " + topName + " -o " + tprName + " -e " + edrName
    os.system(sCommand)
  

  #call(sCommand)

  

def writeFile( sToWrite, sFileName ):
  text_file = open(sFileName, "w")
  text_file.write(sToWrite)
  text_file.close()

def writeGramcsInputFile( step, temp ):
  
  
  
  dt = 0.001
  if not(bTest):
    nsteps = 100000000
    #nsteps = 2500000
  else:
    nsteps = 25000
  writingFreq = 1000
  
  if bEqui:
    fName = "equi_"+str(step)+".mdp"
  else:
    fName = "remd_"+str(step)+".mdp"
      
  
  fCont = "#################################### INPUT ####################################\n"
  fCont = fCont + ";ld_seed     = RAND      ; Use random seed from WESTPA\n"
  fCont = fCont + "################################# INTEGRATOR ##################################\n"
  fCont = fCont + "integrator  = sd        ; Langevin thermostat\n"
  fCont = fCont + "dt          = " + str(dt) +"     ; Timestep (ps)\n"
  fCont = fCont + "nsteps      = " + str(nsteps) + "    ;25000000   ; Simulation duration (timesteps)\n"
  fCont = fCont + "nstcomm     = 250       ; Center of mass motion removal interval\n"
  fCont = fCont + "comm_mode   = ANGULAR   ;linear    ; Center of mass motion removal mode (angular: also rotation removed)\n"
  fCont = fCont + "################################## ENSEMBLE ###################################\n"
  fCont = fCont + "ref_t       = " + str(temp) + "      ; System temperature (K)\n"
  fCont = fCont + "tcoupl      = andersen\n"
  fCont = fCont + "tau_t       = 2.0       ; Thermostat time constant (ps)\n"
  fCont = fCont + "tc_grps     = system    ; Apply thermostat to complete system\n"
  fCont = fCont + "############################## IMPLICIT SOLVENT ###############################\n"
  fCont = fCont + "implicit_solvent = GBSA ; Generalized Born implicit solvent\n"
  fCont = fCont + "gb_algorithm     = HCT  ; Hawkins-Cramer-Truhlar radii calculation\n"
  fCont = fCont + "rgbradii         = 0.0  ; Cutoff for Born radii calculation (A)\n"
  fCont = fCont + "########################### NONBONDED INTERACTIONS ############################\n"
  fCont = fCont + "cutoff_scheme = group   ; Method of managing neighbor lists\n"
  fCont = fCont + "pbc           = no      ; Periodic boundary conditions disabled\n"
  fCont = fCont + "coulombtype   = cut-off ; Calculate coulomb interactions using cutoff\n"
  fCont = fCont + "rcoulomb      = 0.0     ; Coulomb cutoff of infinity\n"
  fCont = fCont + "vdw_type      = cut-off ; Calculate van der Waals interactions using cutoff\n"
  fCont = fCont + "rvdw          = 0.0     ; Van der Waals cutoff of infinity\n"
  fCont = fCont + "rlist         = 0.0     ; Neighbor list cutoff\n"
  fCont = fCont + "nstlist       = 0       ; Do not update neighbor list\n"
  fCont = fCont + "################################### OUTPUT ####################################\n"
  fCont = fCont + "nstlog        = " + str(writingFreq) + "     ; Log output interval (timesteps)\n"
  fCont = fCont + "nstenergy     = " + str(writingFreq) + "     ; Energy output interval (timesteps)\n"
  fCont = fCont + "nstcalcenergy = " + str(writingFreq) + "     ; Energy calculation interval (timesteps)\n"
  fCont = fCont + "nstxout       = " + str(writingFreq) + "     ; Trajectory output interval (timesteps)\n"
  fCont = fCont + "nstvout       = " + str(writingFreq) + "     ; Velocity outout interval (timesteps)\n"
  fCont = fCont + "nstfout       = " + str(writingFreq) + "     ; Force output interval (timesteps)\n"
  fCont = fCont + "\n"
  fCont = fCont + "constraints           = h-bonds\n"
  fCont = fCont + "constraint-algorithm  = lincs\n"
  fCont = fCont + "lincs-iter  = 2\n"
  fCont = fCont + "unconstrained-start   = no   ; Do not constrain the start configuration\n"
  fCont = fCont + ";shake_tol             = 0.0001\n"
  
  writeFile(fCont,fName)
  

bCluster = True
bEqui = False
bTest = False

workingDirCluster = "/home/markus/projects/peptide/ala-15/production/"
gromacsGMXCluster = "/home/markus/software/gromacs-5.0.4/build/bin/gmx"
gromacsGMXClusterMpi = "/home/markus/software/gromacs-5.0.4/build/bin/gmx_mpi"
workingDir = ""
gromacsGMX = "/home/schoeberl/software/gromacs-5.0.4/build/bin/gmx"
gromacsGMXmpi = "/home/schoeberl/software/gromacs-5.0.4/build/bin/gmx_mpi"

N = 21
k = 0.04
T0 = 270

I = np.arange(0,N)
T = T0 * np.exp(k * I)

print 'Simulated temperatures: '
print T

for i in I:
  writeGramcsInputFile(i,T[i])
  createTprFile(i)
  if bEqui:
    mdRun(i)
    
    
if not bEqui:
  mdRunREMD(N)
  
