# -*- coding: utf-8 -*
from __future__ import unicode_literals

import numpy as np

import argparse, os
import sys

import MDAnalysis


from joblib import Parallel, delayed
import multiprocessing


## not displying the plots
import matplotlib as mpl
mpl.use('Agg')

## for font size adoption
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 20}
rc('font', **font)
#rc('text', usetex=True)
leg = {'fontsize': 18}#,
          #'legend.handlelength': 2}
rc('legend', **leg)

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.colors import LogNorm
#import matplotlib.mlab as mlab


def getAbsCoordinates(xyz):
  _xyzAbs = np.zeros([ xyz.shape[0]+1, xyz.shape[1] ])
  
  # number of residues
  nACE = 1
  nALA = 15
  nNME = 1
  
  ACEleng = 6
  ALAleng = 10
  NMEleng = 6
  
  # go through every residue
  aACE = np.zeros([nACE*ACEleng,3])
  aALA = np.zeros([nALA*ALAleng,3])
  aNME = np.zeros([nNME*NMEleng,3])
  
  # 1HH3 = CH3 + (1HH3 - CH3)
  aACE[0,:] = xyz[0,:]
  # CH3 = 0
  #aACE[1,:] = 0
  # 2HH3 = CH3 + (2HH3 - CH3)
  aACE[2,:] = xyz[1,:]
  # 3HH3 = CH3 + (3HH3 - CH3)
  aACE[3,:] = xyz[2,:]
  # C = CH3 + (C - CH3)
  aACE[4,:] = xyz[3,:]
  # O = C + (O - C)
  aACE[5,:] = aACE[4,:] + xyz[4,:]
  
  # first N coordinate
  aALA[0,:] = aACE[4,:] + xyz[5,:]
  
  for iALA in range(0,nALA):
    # N = C + (N - C)
    if iALA > 0:
      aALA[iALA*ALAleng + 0,:] = aALA[ iALA*ALAleng - 2, : ] + xyz[ ACEleng + iALA*ALAleng - 1,:]
    # H = N + (H - N)
    aALA[iALA*ALAleng + 1,:] = aALA[ iALA*ALAleng + 0, : ] + xyz[ ACEleng + iALA*ALAleng + 0,:]
    # CA = N + (CA - N)
    aALA[iALA*ALAleng + 2,:] = aALA[ iALA*ALAleng + 0, : ] + xyz[ ACEleng + iALA*ALAleng + 1,:]
    # HA = CA + (HA - CA)
    aALA[iALA*ALAleng + 3,:] = aALA[ iALA*ALAleng + 2, : ] + xyz[ ACEleng + iALA*ALAleng + 2,:]
    # CB = CA + (CB - CA)
    aALA[iALA*ALAleng + 4,:] = aALA[ iALA*ALAleng + 2, : ] + xyz[ ACEleng + iALA*ALAleng + 3,:]
    # HB1 = CB + (HB1 - CB)
    aALA[iALA*ALAleng + 5,:] = aALA[ iALA*ALAleng + 4, : ] + xyz[ ACEleng + iALA*ALAleng + 4,:]
    # HB2 = CB + (HB2 - CB)
    aALA[iALA*ALAleng + 6,:] = aALA[ iALA*ALAleng + 4, : ] + xyz[ ACEleng + iALA*ALAleng + 5,:]
    # HB3 = CB + (HB3 - CB)
    aALA[iALA*ALAleng + 7,:] = aALA[ iALA*ALAleng + 4, : ] + xyz[ ACEleng + iALA*ALAleng + 6,:]
    # C = CA + (C - CA)
    aALA[iALA*ALAleng + 8,:] = aALA[ iALA*ALAleng + 2, : ] + xyz[ ACEleng + iALA*ALAleng + 7,:]
    # O = C + (O - C)
    aALA[iALA*ALAleng + 9,:] = aALA[ iALA*ALAleng + 8, : ] + xyz[ ACEleng + iALA*ALAleng + 8,:]
  
  # Last part
  # N = C + (N - C)
  aNME[0 , :] = aALA[nALA*ALAleng - 2,:] + xyz[ ACEleng + nALA*ALAleng - 1,:]
  # H = N + (H - N)
  aNME[1 , :] = aNME[0,:] + xyz[ ACEleng + nALA*ALAleng + 0,:]
  # CH3 = N + (CH3 - N)
  aNME[2 , :] = aNME[0,:] + xyz[ ACEleng + nALA*ALAleng + 1,:]
  # 1HH3 = CH3 + (1HH3 - CH3)
  aNME[3 , :] = aNME[2,:] + xyz[ ACEleng + nALA*ALAleng + 2,:]
  # 2HH3 = CH3 + (2HH3 - CH3)
  aNME[4 , :] = aNME[2,:] + xyz[ ACEleng + nALA*ALAleng + 3,:]
  # 3HH3 = CH3 + (2HH3 - CH3)
  aNME[5 , :] = aNME[2,:] + xyz[ ACEleng + nALA*ALAleng + 4,:]
  
  _xyzAbs[0:(ACEleng),:] = aACE
  _xyzAbs[ACEleng:(ACEleng+nALA*ALAleng),:] = aALA
  _xyzAbs[(ACEleng+nALA*ALAleng):,:] = aNME
  
  return _xyzAbs

def getCartesian(rphitheta):
  
  rphithetaShape = rphitheta.shape
  
  #print rphithetaShape
  _xyz = np.zeros(rphithetaShape)

  _xyz[:,0] = rphitheta[:,0]*np.cos(rphitheta[:,2])*np.sin(rphitheta[:,1])
  _xyz[:,1] = rphitheta[:,0]*np.sin(rphitheta[:,2])*np.sin(rphitheta[:,1])
  _xyz[:,2] = rphitheta[:,0]*np.cos(rphitheta[:,1])
  
  xyzAbs = getAbsCoordinates(xyz=_xyz)

  return xyzAbs

def printList():
  print txtList
  print xtcList
  print rmsdList
  print ramaList
  print gyrateList
  print plotSuffixList

def updateList():
  if bReadPredFileDirect:
    txtList.append(predFile)
    txtListW.append(predFileW)
    txtListX.append(predFileX)
    xtcList.append(xtcFile)
    rmsdList.append(sPredPath + 'rmsd_' + sFileNamePred + '.xvg')
    ramaList.append(sPredPath + 'rama_' + sFileNamePred + '.xvg')
    gyrateList.append(sPredPath + 'gyrate_' + sFileNamePred + '.xvg')
    plotSuffixList.append('_' + sFileNamePred)
  else:
    txtList.append(predFile)
    txtListW.append(predFileW)
    txtListX.append(predFileX)
    xtcList.append(xtcFile)
    rmsdList.append(workingDir+'rmsd_iter_'+str(iteration)+'.xvg')
    ramaList.append(workingDir+'rama_iter_'+str(iteration)+'.xvg')
    gyrateList.append(workingDir+'gyrate_iter_'+str(iteration)+'.xvg')
    plotSuffixList.append('_' + str(iteration) )
  
def updateListPost():
  txtList.append(predFile)
  xtcList.append(xtcFile)
  rmsdList.append(workingDir+'rmsd_iter_'+str(iteration)+'_postS_'+str(postS)+'.xvg')
  ramaList.append(workingDir+'rama_iter_'+str(iteration)+'_postS_'+str(postS)+'.xvg')
  gyrateList.append(workingDir+'gyrate_iter_'+str(iteration)+'_postS_'+str(postS)+'.xvg')
  plotSuffixList.append('_' + str(iteration) + '_postS_' + str(postS) )
  
def updateListPostFlex(_midNameExtention):

  if bReadPredFileDirect:
    txtList.append(predFile)
    xtcList.append(xtcFile)
    rmsdList.append(sPredPath + 'rmsd_' + sFileNamePred + _midNameExtention + '.xvg')
    ramaList.append(sPredPath + 'rama_' + sFileNamePred + _midNameExtention + '.xvg')
    gyrateList.append(sPredPath + 'gyrate_' + sFileNamePred + _midNameExtention + '.xvg')
    plotSuffixList.append('_' + sFileNamePred + _midNameExtention)
  else:
    txtList.append(predFile)
    xtcList.append(xtcFile)
    rmsdList.append(workingDir+'rmsd_iter_'+str(iteration)+_midNameExtention+'.xvg')
    ramaList.append(workingDir+'rama_iter_'+str(iteration)+_midNameExtention+'.xvg')
    gyrateList.append(workingDir+'gyrate_iter_'+str(iteration)+_midNameExtention+'.xvg')
    plotSuffixList.append('_' + str(iteration) + _midNameExtention )
  
def removeFiles(listRm):
  sCommand = 'rm'
  for i in range(0,len(listRm)):
    sCommand = sCommand + ' ' + listRm[i]
  os.system(sCommand)

def rmsd(inputXTC, outputRMSD, groF='alpha_shell.gro'):
  sCommand = 'printf \'1\\n\' | '+ gmx + " rmsdist -f " + inputXTC + " -s " + groF + " -o " + outputRMSD #+ " -select " + str(1)
  os.system(sCommand)

def rama(inputXTC, outputRAMA, tprF='remd_5.tpr'):
  sCommand = gmx + " rama -f " + inputXTC + " -s " + tprF + " -o " + outputRAMA
  os.system(sCommand)
  
def gyrate(inputXTC, outputGYRATE, tprF='remd_5.tpr'):
  sCommand = 'printf \'1\\n\' | '+ gmx + " gyrate -f " + inputXTC + " -s " + tprF + " -o " + outputGYRATE
  os.system(sCommand)

def credIntervalBoundaries( _binsHightAll, _credInterval):
   _binsHightAllSort = np.sort( _binsHightAll, axis=1 )
   _dim = _binsHightAllSort.shape[0]

def createLineFromBin(_bin,_val):
  _x = np.zeros(2*_bin.shape[0])
  _y = np.zeros(2*_bin.shape[0])
  _y[0] = 0.
  _y[-1] = 0.
  
  _count = 0
  _county = 1
  
  print "bin"
  print _bin
  
  for i in range(0,_bin.shape[0]):
    _x[_count] = _bin[i]
    _x[_count+1] = _bin[i]
    _count = _count + 2
    
  print "Val"
  print _val

  for j in range(1,_val.shape[0]):
    _y[_county] = _val[j]
    _y[_county+1] = _val[j]
    _county = _county + 2
    
  return _x, _y
  

def binningTruth():
  rmsdTrue = np.loadtxt(rmsdTrueXVG, delimiter='  ', skiprows=skiprmsd)[:, 1]*10 # for angstrom conversion
  _nRmsdTrue, _binsRmsdTrue, _patches = plt.hist(rmsdTrue, rmsdNumBins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
  plt.close()

  rgTrue = np.loadtxt(gyrateTrueXVG, skiprows=skipgyrate)[:,1]*10 # for angstrom conversion
  _nGyrateTrue, _binsGyrateTrue, _patches = plt.hist(rgTrue, rgNumBins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
  plt.close()

  return _binsRmsdTrue, _binsGyrateTrue

def plotRMSDuq(_n, _bins, _m, _outputPlot):
  fname = _outputPlot
  if bPlotTruth:
    rmsdTrue = np.loadtxt(rmsdTrueXVG, delimiter='  ', skiprows=skiprmsd)[:,1]*10 # for angstrom conversion
    rmsdTrueMean = np.mean(rmsdTrue)
    rmsdTrueMedian = np.median(rmsdTrue)
    
  if bPlotPosterior:
    rmsdPost = np.loadtxt(rmsdPostXVG, delimiter='  ', skiprows=skiprmsd)[:,1]*10 # for angstrom conversion
    rmsdPostMean = np.mean(rmsdPost)
    rmsdPostMedian = np.median(rmsdPost)
  
  num_bins = rmsdNumBins
  # the histogram of the data
  # use the same bin size as for the posterior plots
  #_nT, _binsT, _patches = plt.hist(rmsdTrue, num_bins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
  _nT, _binsT, _patches = plt.hist(rmsdTrue, bins=_bins[0,:], normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
  plt.close()
  
  if bPlotPosterior:
    _nPost, _binsPost, _patches = plt.hist(rmsdPost, bins=_bins[0,:], normed=1, color='b',facecolor='blue', histtype='step', linewidth=2, alpha=0.8, label=r'$q(X)$')    
    plt.close()
    _nP0 = np.zeros(1,dtype=_nPost.dtype)
    _nPc = np.hstack((_nP0,_nPost))
 
  _nT0 = np.zeros(1,dtype=_nT.dtype)
  _nTc = np.hstack((_nT0,_nT))
  

  
  _ps = len(xtcList)
  if _ps > 1:
    _bPost = True
  else:
    _bPost = False
  
  # chec if actual samples are available
  if _bPost:
    _n0 = np.zeros((_n.shape[0],1),dtype=_n.dtype)
    _nc = np.hstack((_n0,_n))
  else:   
    _n0 = np.zeros((_n.shape[0],1),dtype=_n.dtype)
    _nc = np.hstack((_n0,_n))
    print _n0
    print _nc
  
  # sort the bins
  _nSort = np.sort( _nc, axis=0 )
  _mSort = np.sort( _m, axis=0 )
  
  print "nsort"
  print _nSort
  print _n0
  
  print _binsT
  print _nTc
  
  fig, ax = plt.subplots(1)
  truth, = ax.plot(_binsT[:], _nTc, lw=2, label='Reference', color='black', drawstyle='steps')
  mapest, = ax.plot(_bins[0,:], _nc[0,:], lw=2, label='MAP', color='red', drawstyle='steps', dashes=[6, 2])

  if bPlotPosterior:
    post, = ax.plot(_binsPost[:], _nPc, lw=2, label=r'Using $q(X)$', color='blue', drawstyle='steps')
  #low, = ax.plot(_bins[0,:],  _nSort[quant5,:], lw=2, ls='--', label='low', color='black', alpha=0.7, drawstyle='steps')
  #top, = ax.plot(_bins[0,:], _nSort[quant95,:], lw=2, ls='--', label='top', color='black', alpha=0.7, drawstyle='steps')
  
  if _bPost:
    # create lines
    x5, y5 = createLineFromBin(_bin=_bins[0,:],_val = _nSort[quant5,:])
    x95, y95 = createLineFromBin(_bin=_bins[0,:],_val = _nSort[quant95,:])
    
    print "create lines"
    print x5
    print x95
    print y5
    print y95
    
    ax.fill_between(x5, y5, y95, lw=1, label=r'Posterior mean $\pm 3 \sigma$',facecolor='red', alpha=0.2 )
    filling = mpatches.Rectangle((0, 0), 1, 1, fc="r", alpha=0.2, label=r'5% - 95% Credible interval')
    #ax.legend(handles=[truth, mapest,filling ])#, loc='upper right')
    ax.legend(handles=[truth, mapest,filling ], loc='upper center', bbox_to_anchor=(0.5, -0.14),
          fancybox=False, shadow=False, ncol=3, frameon=False)
    if bPosterior:
      ax.legend(handles=[truth, post, mapest,filling ], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
    else:
      ax.legend(handles=[truth, mapest,filling ], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
  else:
    if bPosterior:
      ax.legend(handles=[truth, post, mapest], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
    else:
      ax.legend(handles=[truth, mapest ], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
  #ax.legend(handles=[truth, mapest ], loc='upper right')

  #ax.axvline(_m[0,0], c='r', ls='--')
  #ax.axvline(_mSort[quant5,0], c='r', ls=':')
  #ax.axvline(_mSort[quant95,0], c='r', ls=':')
  
  if bLoadHight:
    ax.set_ylim(0,pMaxRmsd)
  
  ax.set_ylabel('$p(RMSD)$')
  ax.set_xlabel('$RMSD [\mathrm{\AA}]$')
  ax.grid()
  #plt.legend()
  #plt.xlim([1,5])
  #plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
  for fformat in outputformat:
    plt.savefig(fname + fformat, bbox_inches='tight')
  plt.close()
  
def plotRMSDmi(inputXVG , n, bins, m, outputPlot):
  fname = outputPlot
  
  data = np.loadtxt(inputXVG, delimiter='  ', skiprows=skiprmsd)
  rmsd = data[:,1]*10
  
  if bPlotTruth:
    rmsdTrue = np.loadtxt(rmsdTrueXVG, delimiter='  ', skiprows=skiprmsd)[:,1]*10
    rmsdTrueMean = np.mean(rmsdTrue)
    rmsdTrueMedian = np.median(rmsdTrue)
  
  num_bins = rmsdNumBins
  
  # sort the bins
  nSort = np.sort( n, axis=1 )
  mSort = np.sort( m, axis=1 )
  
  plt.figure()
  if bPlotTruth:
    plt.hist(rmsdTrue, num_bins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
    plt.axvline(x=rmsdTrueMean, c='b', ls='--', lw=2)
    n, bins, patches = plt.hist(rmsd, num_bins, normed=1, color='r', facecolor='black', histtype='step', linewidth=2, alpha=0.8, label='MAP')
    plt.axvline(x=m[0,0], c='r', lw=2)
    plt.axvline(x=mSort[-1,0], c='r', ls='--', lw=2)
    plt.axvline(x=mSort[0,0], c='r', ls='--', lw=2)
  else:
    n, bins, patches = plt.hist(rmsd, num_bins, normed=1, facecolor='green', histtype='step',alpha=0.5)
  plt.xlabel('$RMSD [\mathrm{\AA}]$')
  plt.ylabel('$p(RMSD)$')
  plt.grid()
  plt.legend()
  #plt.xlim([1,5])
  #plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
  for fformat in outputformat:
    plt.savefig(fname + fformat, bbox_inches='tight')
  plt.close()

def plotRMSD(inputXVG, outputPlot, _binsGiven = None):
  fname = outputPlot
  data = np.loadtxt(inputXVG, delimiter='  ', skiprows=skiprmsd)
  rmsd = data[:,1]*10
  rmsdMean = np.mean(rmsd)
  rmsdMedian = np.median(rmsd)
  
  if bPlotTruth:
    rmsdTrue = np.loadtxt(rmsdTrueXVG, delimiter='  ', skiprows=skiprmsd)[:,1]*10
    rmsdTrueMean = np.mean(rmsdTrue)
    rmsdTrueMedian = np.median(rmsdTrue)
  
  num_bins = rmsdNumBins
  # the histogram of the data
  
  plt.figure()
  if bPlotTruth:
    if _binsGiven is None:
      plt.hist(rmsdTrue, num_bins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
    else:
      plt.hist(rmsdTrue, bins=_binsGiven, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
    plt.axvline(x=rmsdTrueMean, c='b', ls='--', lw=2)
    if _binsGiven is None:
      n, bins, patches = plt.hist(rmsd, num_bins, normed=1, color='r', facecolor='black', histtype='step', linewidth=2, alpha=0.8, label='MAP')
    else:
      n, bins, patches = plt.hist(rmsd, bins=_binsGiven, normed=1, color='r', facecolor='black', histtype='step', linewidth=2, alpha=0.8, label='MAP')
    plt.axvline(x=rmsdMean, c='r', lw=2)
  else:
    if _binsGiven is None:
      n, bins, patches = plt.hist(rmsd, num_bins, normed=1, facecolor='green', histtype='step',alpha=0.5)
    else:
      n, bins, patches = plt.hist(rmsd, bins=_binsGiven, normed=1, facecolor='green', histtype='step',alpha=0.5)
  plt.xlabel('$RMSD [\mathrm{\AA}]$')
  plt.ylabel('$p(RMSD)$')
  plt.grid()
  plt.legend()
  #plt.xlim([1,5])
  #plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
  if _binsGiven is None:
    for fformat in outputformat:
      plt.savefig(fname + fformat, bbox_inches='tight')
  plt.close()
  
  return n, bins, rmsdMean, rmsdMedian
  
def plotRAMA(inputXVG, outputPlot):  
  fname = outputPlot
  data = np.loadtxt(inputXVG, delimiter='  ', skiprows=skiprama, usecols=range(0,2))
  data = np.vstack([data, np.array([179.99, -179.99])])
  x = data[:,0]
  y = data[:,1]
  #plt.hexbin(data[:,0],data[:,1])
  plt.hist2d(x,y,bins=80, norm=LogNorm(),normed=True)
  plt.xlabel(r'$\phi$ [°]')
  plt.ylabel(r'$\psi$ [°]')
  plt.xlim([-180, 180])
  plt.ylim([-180, 180])
  plt.colorbar(format='%.1e')
  plt.title('Ramachandran Plot')
  for fformat in outputformat:
    plt.savefig(fname + '_log' + fformat, bbox_inches='tight')
  plt.close()
  
  plt.hist2d(x,y,bins=80, normed=True)
  plt.xlabel(r'$\phi$ [°]')
  plt.ylabel(r'$\psi$ [°]')
  plt.xlim([-180, 180])
  plt.ylim([-180, 180])
  plt.colorbar(format='%.1e')
  plt.title('Ramachandran Plot')
  for fformat in outputformat:
    plt.savefig(fname + fformat, bbox_inches='tight')
  plt.close()

def plotGYRATEuq(_n, _bins, _m, _outputPlot):
  fname = _outputPlot
  if bPlotTruth:
    rgTrue = np.loadtxt(gyrateTrueXVG, skiprows=skipgyrate)[:,1]*10 # for angstrom conversion
    rgTrueMean = np.mean(rgTrue)
    rgTrueMedian = np.median(rgTrue)
  
  if bPlotPosterior:
    rgPost = np.loadtxt(gyratePostXVG, skiprows=skipgyrate)[:,1]*10 # for angstrom conversion
    rgPostMean = np.mean(rgPost)
    rgPostMedian = np.median(rgPost)
  
  num_bins = rgNumBins
  # the histogram of the data
  # use the same bin size as for the posterior plots
  #_nT, _binsT, _patches = plt.hist(rgTrue, num_bins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
  _nT, _binsT, _patches = plt.hist(rgTrue, bins=_bins[0, :], normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
  plt.close()
  
  if bPlotPosterior:
    _nPost, _binsPost, _patches = plt.hist(rgPost, bins=_bins[0, :], normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='$q(X)$')
    plt.close()
    _nP0 = np.zeros(1,dtype=_nPost.dtype)
    _nPc = np.hstack((_nP0,_nPost))
   
  _nT0 = np.zeros(1,dtype=_nT.dtype)
  _nTc = np.hstack((_nT0,_nT))
    
  _ps = len(xtcList)
  if _ps > 1:
    _bPost = True
  else:
    _bPost = False
  
  # chec if actual samples are available
  if _bPost:
    _n0 = np.zeros((_n.shape[0],1),dtype=_n.dtype)
    _nc = np.hstack((_n0,_n))
  else:
    _n0 = np.zeros((_n.shape[0],1),dtype=_n.dtype)
    _nc = np.hstack((_n0,_n))
  
  # sort the bins
  _nSort = np.sort( _nc, axis=0 )
  _mSort = np.sort( _m, axis=0 )
  
  fig, ax = plt.subplots(1)
  truth, = ax.plot(_binsT[:], _nTc, lw=2, label='Reference', color='black', drawstyle='steps')
  mapest, = ax.plot(_bins[0,:], _nc[0,:], lw=2, label='MAP', color='red', drawstyle='steps', dashes=[6, 2])
  if bPlotPosterior:
    post, = ax.plot(_binsPost[:], _nPc, lw=2, label=r'Using $q(X)$', color='blue', drawstyle='steps')
  #low, = ax.plot(_bins[0,:],  _nSort[quant5,:], lw=2, ls='--', label='low', color='black', alpha=0.7, drawstyle='steps')
  #top, = ax.plot(_bins[0,:], _nSort[quant95,:], lw=2, ls='--', label='top', color='black', alpha=0.7, drawstyle='steps')
  
  if _bPost:
    # create lines
    x5, y5 = createLineFromBin(_bin=_bins[0,:],_val = _nSort[quant5, :])
    x95, y95 = createLineFromBin(_bin=_bins[0,:],_val = _nSort[quant95, :])
    
    ax.fill_between(x5, y5, y95, lw=1, label=r'Posterior mean $\pm 3 \sigma$',facecolor='red', alpha=0.2 )
    filling = mpatches.Rectangle((0, 0), 1, 1, fc="r", alpha=0.2, label=r'5% - 95% Credible interval')
    #ax.legend(handles=[truth, mapest,filling ], loc='upper right')
    if bPosterior:
      ax.legend(handles=[truth, post, mapest,filling ], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
    else:
      ax.legend(handles=[truth, mapest,filling ], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
  else:
    if bPosterior:
      ax.legend(handles=[truth, post, mapest], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
    else:
      ax.legend(handles=[truth, mapest ], loc='upper center', bbox_to_anchor=(0.5, -0.14),
		fancybox=False, shadow=False, ncol=3, frameon=False)
  #ax.legend(handles=[truth, mapest ], loc='upper right')

  #ax.axvline(_m[0,0], c='r', ls='--')
  #ax.axvline(_mSort[quant5,0], c='r', ls=':')
  #ax.axvline(_mSort[quant95,0], c='r', ls=':')
  
  if bLoadHight:
    ax.set_ylim(0,pMaxRg)
  
  ax.set_ylabel('$p(Rg)$')
  ax.set_xlabel('$Rg [\mathrm{\AA}]$')
  ax.grid()
  #plt.legend()
  #plt.xlim([1,5])
  #plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
  for fformat in outputformat:
    plt.savefig(fname + fformat, bbox_inches='tight')
  plt.close()

def plotGYRATEmi(inputXVG, n, bins, m, outputPlot):
  fname = outputPlot
  data = np.loadtxt(inputXVG, skiprows=skipgyrate)
  rg = data[:,1]*10 # for angstrom conversion
  
  if bPlotTruth:
    rgTrue = np.loadtxt( gyrateTrueXVG, skiprows=skipgyrate)[:,1]*10
    rgTrueMean = np.mean(rgTrue)
    rgTrueMedian = np.median(rgTrue)
  
  num_bins = rgNumBins

  # sort the bins
  nSort = np.sort( n, axis=1 )
  mSort = np.sort( m, axis=1 )
  
  plt.figure()
  if bPlotTruth:
    plt.hist(rgTrue, num_bins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
    plt.axvline(x=rgTrueMean, c='b', lw=2)
    n, bins, patches = plt.hist(rg, num_bins, normed=1, color='r', facecolor='black', histtype='step', linewidth=2, alpha=0.8, label='MAP')
    plt.axvline(x=m[0,0], c='r', lw=2)
    plt.axvline(x=mSort[-1,0], c='r', ls='--', lw=2)
    plt.axvline(x=mSort[0,0], c='r', ls='--', lw=2)
  else:
    n, bins, patches = plt.hist(rg, num_bins, normed=1, facecolor='green', alpha=0.5)
  
  plt.xlabel('$Rg [\mathrm{\AA}]$')
  plt.ylabel('$p(Rg)$')
  plt.grid()
  plt.legend()
  #plt.xlim([4,16])
  #plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
  for fformat in outputformat:
    plt.savefig(fname + fformat, bbox_inches='tight')
  plt.close()

def plotGYRATE(inputXVG, outputPlot, _binsGiven = None):
  fname = outputPlot
  data = np.loadtxt(inputXVG, skiprows=skipgyrate)
  rg = data[:,1]*10
  rgMean = np.mean(rg)
  rgMedian = np.median(rg)
  
  if bPlotTruth:
    rgTrue = np.loadtxt( gyrateTrueXVG, skiprows=skipgyrate)[:,1]*10 # for angstrom conversion
    rgTrueMean = np.mean(rgTrue)
    rgTrueMedian = np.median(rgTrue)
  
  num_bins = rgNumBins
  # the histogram of the data
  
  plt.figure()
  if bPlotTruth:
    if _binsGiven is None:
      plt.hist(rgTrue, num_bins, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
    else:
      plt.hist(rgTrue, bins=_binsGiven, normed=1, color='b',facecolor='red', histtype='step', linewidth=2, alpha=0.8, label='Reference')
    plt.axvline(x=rgTrueMean, c='b', ls='--', lw=2)
    if _binsGiven is None:
      n, bins, patches = plt.hist(rg, num_bins, normed=1, color='r', facecolor='black', histtype='step', linewidth=2, alpha=0.8, label='MAP')
    else:
      n, bins, patches = plt.hist(rg, bins=_binsGiven, normed=1, color='r', facecolor='black', histtype='step', linewidth=2, alpha=0.8, label='MAP')
    plt.axvline(x=rgMean, c='r', lw=2)
  else:
    if _binsGiven is None:
      n, bins, patches = plt.hist(rg, num_bins, normed=1, facecolor='green', histtype='step',alpha=0.5)
    else:
      n, bins, patches = plt.hist(rg, bins=_binsGiven, normed=1, facecolor='green', histtype='step',alpha=0.5)
  plt.xlabel('$Rg [\mathrm{\AA}]$')
  plt.ylabel('$p(Rg)$')
  plt.grid()
  plt.legend()
  #plt.xlim([1,5])
  #plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
  if _binsGiven is None:
    for fformat in outputformat:
      plt.savefig(fname + fformat, bbox_inches='tight')
  plt.close()
  
  return n, bins, rgMean, rgMedian
  
def convertPredTOxtcGiven(inputF, outputF, trjXTC, pdbF = 'alpha_shell.pdb'):
  dataPred = np.loadtxt(inputF)

  print pdbF
  print trjXTC

  U = MDAnalysis.Universe(pdbF, trjXTC)

  nAtoms = U.trajectory.n_atoms

  protein = U.select_atoms("protein")
  writer = MDAnalysis.Writer(outputF, protein.n_atoms)
  
  iTs = dataPred.shape[1]
  iCount = 0
  dataCoord = np.zeros((nAtoms, 3))
  
  tt = U.trajectory.ts
  
  for tsCurr in range(0,iTs):
    ts_data = dataPred[:,iCount]
    for i in range(0, nAtoms):
      dataCoord[i, 0] = ts_data[i*3]
      dataCoord[i, 1] = ts_data[i*3+1]
      dataCoord[i, 2] = ts_data[i*3+2]
      
    np.copyto(tt._pos, dataCoord)
    writer.write(tt)
    iCount = iCount +1
    #print "write timestep %t", iCount
  writer.close()
  print "successfully written!"
  
  return True

def convertdataTOxtc(dataPred, outputF, trjXTC, pdbF = 'alpha_shell.pdb'):
  
  U = MDAnalysis.Universe(pdbF, trjXTC)
  nAtoms = U.trajectory.n_atoms
  protein = U.select_atoms("protein")
  writer = MDAnalysis.Writer(outputF, protein.n_atoms)
  
  iTs = dataPred.shape[1]
  iCount = 0
  dataCoord = np.zeros( (nAtoms,3) )
  
  tt = U.trajectory.ts
  
  for tsCurr in range(0,iTs):
    ts_data = dataPred[:,iCount]
    for i in range(0,nAtoms):
      dataCoord[i,0] = ts_data[i*3]
      dataCoord[i,1] = ts_data[i*3+1]
      dataCoord[i,2] = ts_data[i*3+2]
    
    if bAngData:
      dataCoordToStore = getCartesian(dataCoord)
    else:
      dataCoordToStore = dataCoord
    np.copyto(tt._pos, dataCoordToStore)
    writer.write(tt)
    iCount = iCount +1
    #print "write timestep %t", iCount
  writer.close()
  print "successfully written!"
  
  return True

def readW(path):
  a = np.loadtxt(path+'W_'+str(iteration)+'.txt')
  return a

def readWcov(path):
  _covW = list()
  for i in range(0,dimFine):
    _fileNameCov = path+'W_'+ str(iteration) +'_cov_iDf_'+str(i)+'.txt'
    _covW.append(np.loadtxt(path+'W_'+ str(iteration) +'_cov_iDf_'+str(i)+'.txt') )
  return _covW

def readMu(path):
  a = np.loadtxt(path+'mu_'+str(iteration)+'.txt')
  return a

def readSigSq(path):
  a = np.loadtxt(path+'SigmaDiag_'+str(iteration)+'.txt')
  return a

def drawWsample():
  _sampleW = np.zeros([dimFine, dimCoarse])
  for _i in range(0, dimFine):
    _sampleW[_i,:] = np.random.multivariate_normal(W[_i,:], covW[_i])
  return _sampleW

def predictxGivenX(fileX, fileW):
  sampleX = np.loadtxt(fileX)
  numSamplesX = sampleX.shape[1]
  
  if bSMC:
    weightX = np.loadtxt(fileW)
    indexSMC = np.random.multinomial(numSamplesX*samplesPerX, weightX)
  else:
    indexSMC = np.zeros(numSamplesX)
    indexSMC = indexSMC + samplesPerX
  
  samplex = np.zeros( [dimFine, numSamplesX * samplesPerX] )
  
  #if bSMC:
  accum = np.zeros(indexSMC.shape[0], dtype=int)
  accum[0] = 0
  for i in range(1, indexSMC.shape[0]):
    accum[i] = accum[i-1] + indexSMC[i]
  
  counter = 0
  for i in range(0, numSamplesX):
    if indexSMC[i] > 0:
      mean = mu + np.inner(W, sampleX[:,i])
      for j in range(0, int(indexSMC[i])):
	for k in range(0, dimFine):
	  if bZeroVarPrediction:
	    samplex[ k, counter ] = mean[k]
	  else:
	    samplex[ k, counter ] = np.random.normal( mean[k], sigma[k] )
	#samplex[ k, accum[i]:accum[i]+indexSMC[i] ] = np.random.normal( mean[k], sigma[k], indexSMC[i] )
        counter = counter + 1

  return samplex

def predictxGivenXPostW(fileX, fileW):
  sampleX = np.loadtxt(fileX)
  numSamplesX = sampleX.shape[1]
  
  if bSMC:
    weightX = np.loadtxt(fileW)
    indexSMC = np.random.multinomial(numSamplesX*samplesPerX, weightX)
  else:
    indexSMC = np.zeros(numSamplesX)
    indexSMC = indexSMC + samplesPerX
  
  samplex = np.zeros( [dimFine, numSamplesX * samplesPerX] )
  
  #if bSMC:
  accum = np.zeros(indexSMC.shape[0], dtype=int)
  accum[0] = 0
  for i in range(1, indexSMC.shape[0]):
    accum[i] = accum[i-1] + indexSMC[i]
  _Ws = drawWsample()
  counter = 0
  for i in range(0, numSamplesX):
    if indexSMC[i] > 0:
      mean = mu + np.inner(_Ws, sampleX[:,i])
      for j in range(0, int(indexSMC[i])):
        for k in range(0, dimFine):
          if bZeroVarPrediction:
            samplex[ k, counter ] = mean[k]
          else:
            samplex[ k, counter ] = np.random.normal( mean[k], sigma[k] )
            #samplex[ k, accum[i]:accum[i]+indexSMC[i] ] = np.random.normal( mean[k], sigma[k], indexSMC[i] )
          counter  = counter + 1

  return samplex

def parallelPredictionPost(i):
  
  if not bPostW and bPostThetaC:
    if bReadPredFileDirect:
      _predFile = predFilePrefix + '_postS_' + str(i - 1) + '.txt'
      _predFileX = predFilePrefixX + '_postS_' + str(i - 1) + '.txt'
      _predFileW = predFilePrefixW + '_postS_' + str(i - 1) + '.txt'
      _xtcFile = sPredPath + 'protein_' + sFileNamePred + '_postS_' + str(i - 1) + '.xtc'
    else:
      _predFile = predFilePrefix + str(iteration) + '_postS_' + str(i-1) + '.txt'
      _predFileX = predFilePrefixX+str(iteration) + '_postS_' + str(i-1) + '.txt'
      _predFileW = predFilePrefixW+str(iteration) + '_postS_' + str(i-1) + '.txt'
      _xtcFile =  workingDir + 'protein_' + str(iteration)+ '_postS_' + str(i-1) +'.xtc'
  elif bPostW and bPostThetaC :
    _indPostS = int(i-1)/int(iPosWperTheta)
    _indPostSw = (i-1)%int(iPosWperTheta)
    
    _predFile = predFilePrefix + str(iteration) + '_postS_' + str(_indPostS) + '_sW_'+ str(_indPostSw) + '.txt'
    _predFileX = predFilePrefixX+str(iteration) + '_postS_' + str(_indPostS) + '.txt'
    _predFileW = predFilePrefixW+str(iteration) + '_postS_' + str(_indPostS) + '.txt'
    _xtcFile =  workingDir + 'protein_' + str(iteration)+ '_postS_' + str(_indPostS) + '_sW_'+ str(_indPostSw) +'.xtc'
  elif bPostW and not bPostThetaC :
    _indPostS = int(i-1)/int(iPosWperTheta)
    _indPostSw = (i-1)%int(iPosWperTheta)
    
    _predFile = predFilePrefix + str(iteration) + '_postS_' + str(_indPostS) + '_sW_'+ str(_indPostSw) + '.txt'
    # use samples of the MAP estimate of \theta_c
    _predFileX = predFilePrefixX+str(iteration)+'.txt'
    _predFileW = predFilePrefixW+str(iteration)+'.txt'
    _xtcFile =  workingDir + 'protein_' + str(iteration)+ '_postS_' + str(_indPostS) + '_sW_'+ str(_indPostSw) +'.xtc'
  else:
    print 'Error in parallelPredictionPost()!'
    quit()
  if bPredict:
    if not bPostW:
      _datax = predictxGivenX(fileX = _predFileX, fileW = _predFileW)
    else:
      _datax = predictxGivenXPostW(fileX = _predFileX, fileW = _predFileW)
    convertdataTOxtc(dataPred  = _datax, outputF = _xtcFile, trjXTC= folderPDB+trjFname, pdbF = folderPDB+pdbFname)
  else:
    convertPredTOxtcGiven(inputF = _predFile, outputF = _xtcFile, trjXTC= folderPDB+trjFname,  pdbF = folderPDB+pdbFname)

def parallelPostPlot(i):
  
  rmsd(inputXTC=xtcList[i], outputRMSD=rmsdList[i], groF=folderPDB+groFname)
  rama(inputXTC=xtcList[i], outputRAMA=ramaList[i], tprF=folderPDB+tprFname)
  gyrate(inputXTC=xtcList[i], outputGYRATE=gyrateList[i], tprF=folderPDB+tprFname)
  
  #if bBinningTruth:
  #  binuseRg = binsTrueGyrate
  #  binuseRmsd = binsTrueRmsd
  #else:
  binuseRg = binsRg[0,:]
  binuseRmsd = binsRmsd[0,:]
  
  _nRmsd, _binsRmsd, _mRmsd0, _mRmsd1 = plotRMSD(inputXVG=rmsdList[i], outputPlot=sPredPath+'plots/rmsd'+plotSuffixList[i], _binsGiven = binuseRmsd)
  #plotRAMA(inputXVG=ramaList[i], outputPlot=workingDir+'plots/rama'+plotSuffixList[i])
  _nRg, _binsRg, _mRg0, _mRg1 = plotGYRATE(inputXVG=gyrateList[i], outputPlot=sPredPath+'plots/gyrate'+plotSuffixList[i], _binsGiven = binuseRg)
  
  return _nRmsd, _binsRmsd, _mRmsd0, _mRmsd1, _nRg, _binsRg, _mRg0, _mRg1
  

"""parsing and configuration"""
def parse_args():
  desc = "Predicting peptide properties from txt file"
  parser = argparse.ArgumentParser(description=desc)

  parser.add_argument('--workingDir', type=str, default=os.getcwd()+'/',
                        help='Working directory of run')#, required=True)
  parser.add_argument('--iter', type=int, default=1,
                        help='Iteration to analyze. This is just for coupling with CGpeptide.')
  parser.add_argument('--postSamp', type=int, default=0,
                        help='Amount of posterior samples for theta_c to be considered.')
  parser.add_argument('--iSMC', type=int, default=0, help='Amount of SMC particles, in case we use SMC.')
  parser.add_argument('--iPredict', type=int, default=0, help='Amount of prediction samples.')
  parser.add_argument('--nCores', type=int, default=1, help='Amount of threads used for prediction.')
  parser.add_argument('--iN', type=int, default=526, help='Amount of training data.')
  parser.add_argument('--iDimCoarse', type=int, default=2, help='Coarse dimension of CG model.')
  parser.add_argument('--fileNamePred', type=str, default='',
                      help='If not employed with CGpeptide software, this file name should contain samples x (column).')
  parser.add_argument('--referenceDirectory', type=str, default=os.getcwd()+'/',
                      help='Parent directory which contains reference files for peptide.')
  parser.add_argument('--nameResults', type=str, default='',
                      help='Name of results.')
  parser.add_argument('--cluster', type=int, default=0,
                      choices=[0, 1, 2],
                      help='Select if running on local machine (0), cluster muc (1), or cluster nd (2).',
                      required=False)
  parser.add_argument('--predFilePath', type=str, default='',
                      help='Specify the path of the prediction file.')
  parser.add_argument('--conformation', type=str, default='m',
                      help='select conformation if we train just for a single mode.',
                      choices=['m', 'a', 'b1', 'b2'])
  parser.add_argument('--cleanFiles', type=int, default=1, help='Clean files after calculations performed.')
  parser.add_argument('--peptide', type=str, default='ala_2', choices=['ala_2', 'ala_15'],
                      help='Choose different peptides to be treated.')
  parser.add_argument('--dataCollected', type=str, default='assembled', choices=['assembled', 'random'],
                      help='Either using a randomly subsampled dataset or an assembled dataset which keeps the statistics between the modes. This is only relevant for the ala_2 case.')
  parser.add_argument('--compareTwoPred', type=str, default=None,
                      help='Compare two different predictions within one plot. Enter the name of the second'
                           ' file, without extension, here.')
  parser.add_argument('--quantile', type=float, default=None, help='Specify the desired quantile for UQ.')
  return parser.parse_args()

#"""main"""
#def main():

"""
Example usage / cmd line

python estimate_properties.py --fileNamePred samples --predFilePath prediction/propteinpropcal/results/separate_1000_ang_auggrouped/sep_b_64_c_1_z_10/ --conformation m --cluster 0

If files are in the folder:

python estimate_properties.py --fileNamePred 'samples_a_0' --cluster 0 --conformation a

"""

# parse arguments
args = parse_args()
if args is None:
    exit()

# name of peptide
sPepName = args.peptide

# set standard values
bSMC = True
bPlotTruth = True

# number of bins for histograms
rmsdNumBins = 22
rgNumBins = 22

# get working dir
workingDir = args.workingDir

iteration = args.iter
postSamp = args.postSamp
if postSamp > 0:
  bPostThetaC = True
else:
  bPostThetaC = False

iSMC = args.iSMC
if iSMC == 0:
  bSMC = False
else:
  bSMC = True

iPredict = args.iPredict
if iPredict == 0:
  bPredict = False
else:
  bPredict = True
nCores = args.nCores
# even for one, activate parallel computing
bParallel = True

iN = args.iN
iDimCoarse = args.iDimCoarse

bAngData = False

# get name of file containing x samples directly. If not given do own predictions
sFileNamePred = args.fileNamePred
sLabelNameRes = args.nameResults
# check if string is not empty, then do not use iteration name convention
if sFileNamePred:
  bReadPredFileDirect = True

# reference folder with solutions and bins
sRefFolder = args.referenceDirectory

# get which conformation we are looking at
selconf = args.conformation

sPredPath = args.predFilePath
if not sPredPath:
  sPredPath = workingDir

bZeroVarPrediction = False

if sPepName == 'ala_2':
  bBinningTruth = False
  bLoadBinning = True
  bLoadHight = True
else:
  bBinningTruth = False
  bLoadBinning = True
  bLoadHight = True

bLocalFolderRefs = True
sFolderCluster = 'data_peptide/'
sCurrentDir = os.getcwd()+'/'

if sPepName == 'ala_2':
  sRefFolerFiles = 'filesALA2'
else:
  sRefFolerFiles = 'filesALA15'

if not bLocalFolderRefs:
  sBinRg = sFolderCluster + "expBinRg.txt"
  sBinRmsd = sFolderCluster + "expBinRmsd.txt"
  sPmaxRg = sFolderCluster + "expPmaxRg.txt"
  sPmaxRmsd = sFolderCluster + "expPmaxRmsd.txt"
else:
  sBinRg = os.path.join(sRefFolder, sRefFolerFiles + '/bins/' + "expBinRg.txt")
  sBinRmsd = os.path.join(sRefFolder, sRefFolerFiles + '/bins/' + "expBinRmsd.txt")
  sPmaxRg = os.path.join(sRefFolder, sRefFolerFiles + '/bins/' + "expPmaxRg.txt")
  sPmaxRmsd = os.path.join(sRefFolder, sRefFolerFiles + '/bins/' + "expPmaxRmsd.txt")

if bLoadHight:
  pMaxRmsd = np.loadtxt(sPmaxRmsd)[1]
  pMaxRg =  np.loadtxt(sPmaxRg)[1]

if bLoadBinning == True:
  bBinningTruth = True

if sPepName == 'ala_2':
  iALAtype = 2
elif sPepName == 'ala_15':
  iALAtype = 15
else:
  print 'Peptide not supported.'
  quit()

icluster = args.cluster
outputformat = ['.pdf']
bCleanFiles = bool(args.cleanFiles)
samplesPerX = 15

bPostW = False
iPosWperTheta = 30

if bPostW:
  # in case we sample \theta_c either
  if bPostThetaC:
    postSampTot = postSamp*iPosWperTheta
  else:
    postSampTot = iPosWperTheta
else:
  postSampTot = postSamp


if args.quantile is None:
  quantile = 0.05
else:
  quantile = args.quantile

if postSampTot > 50:
  quant5 = int(postSampTot*quantile)
  quant95 = int(postSampTot*(1-quantile))
  if quant95 == postSampTot:
    quant95 -= 1
else:
    raise ValueError('Use more posterior samples.')

bPlotPosterior = False
bPosterior = False
iSampPerDp = 50

sExtZeroVar = ''
if bZeroVarPrediction:
  sExtZeroVar = '_zv'

# storage for histogram rmsd and gyrate
nRmsd = np.zeros( [postSampTot+1, rmsdNumBins] )
binsRmsd = np.zeros( [postSampTot+1, rmsdNumBins+1] )
nRg = np.zeros( [postSampTot+1, rgNumBins] )
binsRg = np.zeros( [postSampTot+1, rgNumBins+1] )
mRmsd = np.zeros( [postSampTot+1,2])
mRg = np.zeros( [postSampTot+1,2])

gromacsGMXCluster = "gromacs-5.0.4/build/bin/gmx"
gromacsGMXND = "gmx"
gromacsGMX = "gromacs-5.0.4/build/bin/gmx"
demuxNb = "gromacs-5.0.4/scripts/demux.pl"
demuxCluster = "gromacs-5.0.4/scripts/demux.pl"

if iALAtype == 15:
  #pdbFname = 'alpha_shell.pdb'
  #groFname = 'alpha_shell.gro'
  #trjFname = 'traj5.xtc'
  #tprFname = 'remd_5.tpr'
  if selconf == 'm':
    sConform = 'protein_dataset_ala_15_20000'
  elif selconf == 'a':
    sConform = 'alpha_data_10000_sub_2000'
  elif selconf == 'b1':
    sConform = 'beta1_data_10000_sub_2000'
  elif selconf == 'b2':
    sConform = 'beta2_data_10000_sub_2000'

  folderPDBlokal = 'data_peptide/ala-15/'
  folderPDBcluster = 'data_peptide/ala-15/'
  pdbFname = 'alpha_shell.pdb'
  groFname = 'alpha_shell.gro'

  gyrateTrueCluster = 'data_peptide/ala-2/2_production/gyrate_' + sConform + '.xvg'
  rmsdTrueCluster = 'data_peptide/ala-2/2_production/rmsd_' + sConform + '.xvg'
  gyrateTrueLoc = os.path.join(sRefFolder, sRefFolerFiles + '/refsolution/' + 'gyrate_' + sConform + '.xvg')
  rmsdTrueLoc = os.path.join(sRefFolder, sRefFolerFiles + '/refsolution/' + 'rmsd_' + sConform + '.xvg')

else:
  if args.dataCollected is 'assembled':
    if selconf == 'm':
      sConform = 'mixed_data_10000_m_1527'
    elif selconf == 'a':
      sConform = 'alpha_data_10000_sub_2000'
    elif selconf == 'b1':
      sConform = 'beta1_data_10000_sub_2000'
    elif selconf == 'b2':
      sConform = 'beta2_data_10000_sub_2000'
  else:
    if selconf == 'm':
      sConform = 'dataset_13334'
    else:
      print 'Error: In the random case with ALA2 only the dataset relying on all modes is valid. Choose the m conformation option'

  folderPDBlokal = 'data_peptide/ala-2/'
  folderPDBcluster = 'data_peptide/ala-2/'
  pdbFname = 'ala2_adopted.pdb'
  groFname = 'ala2_adopted.gro'

  gyrateTrueCluster = 'data_peptide/ala-2/2_production/gyrate_'+sConform+'.xvg'
  rmsdTrueCluster = 'data_peptide/ala-2/2_production/rmsd_'+sConform+'.xvg'
  gyrateTrueLoc = os.path.join(sRefFolder, sRefFolerFiles + '/refsolution/' + 'gyrate_' + sConform + '.xvg')
  rmsdTrueLoc = os.path.join(sRefFolder, sRefFolerFiles + '/refsolution/' + 'rmsd_' + sConform + '.xvg')

# select which machine the calculation is performed
if icluster==0:
  gmx = gromacsGMX
  demux = demuxNb

  folderPDB = os.path.join(sRefFolder, sRefFolerFiles + '/reftraj/')

  gyrateTrueXVG = gyrateTrueLoc
  rmsdTrueXVG = rmsdTrueLoc

  if sPepName == 'ala_2':
    trjFname = 'prod_0.trr'
    tprFname = 'prod_0.tpr'
  else:
    trjFname = 'traj5.trr'
    tprFname = 'remd_5.tpr'

  skiprmsd = 16; skipgyrate = 25; skiprama = 32

elif icluster==1:
  gmx = gromacsGMXCluster
  demux = demuxCluster
  folderPDB = folderPDBcluster
  gyrateTrueXVG = gyrateTrueCluster
  rmsdTrueXVG = rmsdTrueCluster

  trjFname = '2_production/prod_0.trr'
  tprFname = '2_production/prod_0.tpr'

  skiprmsd = 16; skipgyrate = 25; skiprama = 32

elif icluster==2:
  gmx = gromacsGMXND
  demux = demuxNb

  folderPDB = os.path.join(sRefFolder, sRefFolerFiles + '/reftraj/')
  gyrateTrueXVG = gyrateTrueLoc
  rmsdTrueXVG = rmsdTrueLoc


  if sPepName == 'ala_2':
    trjFname = 'prod_0.trr'
    tprFname = 'prod_0.tpr'
  else:
    trjFname = 'traj5.trr'
    tprFname = 'remd_5.tpr'

  #skiprmsd = 17; skipgyrate = 26; skiprama = 33
  skiprmsd = 18; skipgyrate = 27; skiprama = 34

else:
  print 'Error: Machine is not specified!'
  quit()

## List collecting all xtc input files
xtcList = list()
txtList = list()
txtListX = list()
txtListW = list()
rmsdList = list()
ramaList = list()
gyrateList = list()
plotSuffixList = list()


if bPredict:
  W = readW(path=workingDir)
  mu = readMu(path=workingDir)
  sigSq = readSigSq(path=workingDir)
  sigma = np.zeros( sigSq.shape )
  dimFine = W.shape[0]
  dimCoarse = W.shape[1]
  
  if bPostW:
    covW = readWcov(path=workingDir)
  
  for i in range(0, dimFine ):
    sigma[i] = np.sqrt(sigSq[i])
    
########################################
## Treating first the MAP estimate

directory = workingDir + 'plots'
if not os.path.exists(directory):
    os.makedirs(directory)

# in case CGpeptide not used, label the files according to desired label name
if bReadPredFileDirect:
  predFilePrefix = sPredPath + sFileNamePred
  predFile = predFilePrefix + '.txt'
  predFilePrefixX = sPredPath + 'predicted_MAP_X_iter_'
  predFileX = predFilePrefixX + str(iteration) + '.txt'
  predFilePrefixW = sPredPath + 'predicted_MAP_X_weight_iter_'
  predFileW = predFilePrefixW + str(iteration) + '.txt'
  xtcFile = sPredPath + 'protein_' + sFileNamePred + '.xtc'

  plotFolder = sPredPath+'/plots'
  if not os.path.exists(plotFolder):
    os.makedirs(plotFolder)
else:
  predFilePrefix = workingDir + 'predicted_MAP_x_iter_'
  predFile = predFilePrefix + str(iteration) + '.txt'
  predFilePrefixX = workingDir + 'predicted_MAP_X_iter_'
  predFileX = predFilePrefixX + str(iteration) + '.txt'
  predFilePrefixW = workingDir + 'predicted_MAP_X_weight_iter_'
  predFileW = predFilePrefixW + str(iteration) + '.txt'
  xtcFile = workingDir + 'protein_' + str(iteration) + '.xtc'
  plotFolder = workingDir+'/plots'
  if not os.path.exists(plotFolder):
    os.makedirs(plotFolder)

updateList()


if bPredict:
  datax = predictxGivenX(fileX = predFileX, fileW = predFileW)
  convertdataTOxtc(dataPred  = datax, outputF = xtcFile, trjXTC= folderPDB+trjFname, pdbF = folderPDB+pdbFname)
  np.savetxt(predFile, datax)
else:
  convertPredTOxtcGiven(inputF = predFile, outputF = xtcFile, trjXTC= folderPDB+trjFname,  pdbF = folderPDB+pdbFname)


########################################
## Using posterior for prediction
if bPosterior:
  muPost = np.zeros([iDimCoarse,iN])
  sigSqDiagPost = np.zeros([iDimCoarse,iN])
  Xpost = np.zeros([iDimCoarse,iSampPerDp*iN])
  weightPost = np.ones(iSampPerDp*iN)
  weightPost = weightPost*(1.0/(iSampPerDp*iN))

  predFileXpost = predFilePrefixX+str(iteration)+'_posterior.txt'
  predFileWpost = predFilePrefixW+str(iteration)+'_posterior.txt'
  xtcFilePost = workingDir+'protein_'+str(iteration)+'_posterior.xtc'

  rmsdPostXVG = workingDir+'rmsd_iter_'+str(iteration)+'_posterior.xvg'
  ramaPostXVG = workingDir+'rama_iter_'+str(iteration)+'_posterior.xvg'
  gyratePostXVG = workingDir+'gyrate_iter_'+str(iteration)+'_posterior.xvg'
  
  for i in range(0,iN):
    muFile = workingDir+'bvi_dp_'+str(i)+'_iter_'+str(iteration)+'_mu.txt'
    sigFile = workingDir+'bvi_dp_'+str(i)+'_iter_'+str(iteration)+'_sigsq.txt'
    muPost[:,i] = np.loadtxt(muFile)
    sigSqDiagPost[:,i] = np.loadtxt(sigFile)

  for i in range(0,iN):
    for j in range(0,iSampPerDp):
      for k in range(0,iDimCoarse):
        Xpost[k,i*iSampPerDp+j] = np.random.normal(muPost[k,i],np.sqrt(sigSqDiagPost[k,i]))
  np.savetxt(predFileXpost, Xpost)
  np.savetxt(predFileWpost, weightPost)

  datax = predictxGivenX(fileX = predFileXpost, fileW = predFileWpost)
  convertdataTOxtc(dataPred = datax, outputF = xtcFilePost, trjXTC = folderPDB+trjFname, pdbF = folderPDB+pdbFname)

  rmsd(inputXTC=xtcFilePost, outputRMSD=rmsdPostXVG, groF=folderPDB+groFname)
  rama(inputXTC=xtcFilePost, outputRAMA=ramaPostXVG, tprF=folderPDB+tprFname)
  gyrate(inputXTC=xtcFilePost, outputGYRATE=gyratePostXVG, tprF=folderPDB+tprFname)
  plotRMSD(inputXVG=rmsdPostXVG, outputPlot=sPredPath+'plots/rmsd_' + str(iteration) +'_posterior')
  plotRAMA(inputXVG=ramaPostXVG, outputPlot=sPredPath+'plots/rama_' + str(iteration) +'_posterior')
  plotGYRATE(inputXVG=gyratePostXVG, outputPlot=sPredPath+'plots/gyrate_' + str(iteration) +'_posterior')


#######################################
## Treating posterior samples

# in case CGpeptide not used, label the files according to desired label name
if not bPostW:
  for postS in range(0, postSamp):
    # for direct usage of the trajectory x, not z.
    if bReadPredFileDirect:
      predFile = predFilePrefix + '_postS_' + str(postS) + '.txt'
      predFileX = predFilePrefixX + '_postS_' + str(postS) + '.txt'
      predFileW = predFilePrefixW + '_postS_' + str(postS) + '.txt'
      xtcFile = sPredPath + 'protein_' + sFileNamePred + '_postS_' + str(postS) + '.xtc'
    # in this case we only have z given, we need to predict x with p(x|z).
    else:
      predFile = predFilePrefix + str(iteration) + '_postS_' + str(postS) + '.txt'
      predFileX = predFilePrefixX+str(iteration) + '_postS_' + str(postS) + '.txt'
      predFileW = predFilePrefixW+str(iteration) + '_postS_' + str(postS) + '.txt'
      xtcFile =  workingDir + 'protein_' + str(iteration)+ '_postS_' + str(postS) +'.xtc'
    updateListPostFlex( _midNameExtention='_postS_' + str(postS) )
      #updateListPost()
else:
  if bPostThetaC:
    for postS in range(0, postSamp):
      for postSw in range(0, iPosWperTheta):
        predFile = predFilePrefix + str(iteration) + '_postS_' + str(postS) + '_sW_'+ str(postSw) +'.txt'
        predFileX = predFilePrefixX+str(iteration) + '_postS_' + str(postS) +'.txt'
        predFileW = predFilePrefixW+str(iteration) + '_postS_' + str(postS) +'.txt'
        xtcFile =  workingDir + 'protein_' + str(iteration)+ '_postS_' + str(postS) + '_sW_'+ str(postSw) +'.xtc'
        updateListPostFlex(  _midNameExtention='_postS_' + str(postS) + '_sW_'+ str(postSw) )
  else:
    for postSw in range(0, iPosWperTheta):
      predFile = predFilePrefix + str(iteration) + '_postS_' + str(0) + '_sW_'+ str(postSw) +'.txt'
      # use here the samples of the MAP estimate of \theta_c
      predFileX = predFilePrefixX+str(iteration)+'.txt'
      predFileW = predFilePrefixW+str(iteration)+'.txt'
      ############################
      xtcFile =  workingDir + 'protein_' + str(iteration)+ '_postS_' + str(0) + '_sW_'+ str(postSw) +'.xtc'
      updateListPostFlex( _midNameExtention='_postS_' + str(0) + '_sW_'+ str(postSw) )
  # print list for checking purposes
  #printList()

if postSampTot > 0:
  # make another loop for parallel implementation
  inputs = range(1, postSampTot + 1)
  #num_cores = multiprocessing.cpu_count()
  results = Parallel(n_jobs=nCores)(delayed(parallelPredictionPost)(i) for i in inputs)


# make the binning optimal for the Truth
if bBinningTruth:
  if bLoadBinning == False:
    binsTrueRmsd, binsTrueGyrate = binningTruth()
  else:
    binsTrueRmsd = np.loadtxt(sBinRmsd)
    binsTrueGyrate = np.loadtxt(sBinRg)

# plot map estimate first
i = 0

rmsd(inputXTC=xtcList[i], outputRMSD=rmsdList[i], groF=folderPDB+groFname)
rama(inputXTC=xtcList[i], outputRAMA=ramaList[i], tprF=folderPDB+tprFname)
gyrate(inputXTC=xtcList[i], outputGYRATE=gyrateList[i], tprF=folderPDB+tprFname)

if bBinningTruth:
  nRmsd[i,:], binsRmsd[i,:], mRmsd[i,0], mRmsd[i,1] = plotRMSD(inputXVG=rmsdList[i], outputPlot=sPredPath+'plots/rmsd'+plotSuffixList[i], _binsGiven = binsTrueRmsd)
else:
  nRmsd[i,:], binsRmsd[i,:], mRmsd[i,0], mRmsd[i,1] = plotRMSD(inputXVG=rmsdList[i], outputPlot=sPredPath+'plots/rmsd'+plotSuffixList[i])
#np.savetxt('binrmsd.txt', binsRmsd[i,:])

if bBinningTruth:
  nRg[i,:], binsRg[i,:], mRg[i,0], mRg[i,1] = plotGYRATE(inputXVG=gyrateList[i], outputPlot=sPredPath+'plots/gyrate'+plotSuffixList[i], _binsGiven = binsTrueGyrate)
else:
  nRg[i,:], binsRg[i,:], mRg[i,0], mRg[i,1] = plotGYRATE(inputXVG=gyrateList[i], outputPlot=sPredPath+'plots/gyrate'+plotSuffixList[i])
  
plotRAMA(inputXVG=ramaList[i], outputPlot=sPredPath+'plots/rama'+plotSuffixList[i])

inputs = range(1, len(xtcList))
results = Parallel(n_jobs=nCores)(delayed(parallelPostPlot)(i) for i in inputs)

for i in range(1, len(xtcList)):
  nRmsd[i,:] = results[i-1][0]
  binsRmsd[i, :] = results[i-1][1]
  mRmsd[i, 0] = results[i-1][2]
  mRmsd[i, 1] = results[i-1][3]
  nRg[i, :] = results[i-1][4]
  binsRg[i, :] = results[i-1][5]
  mRg[i, 0] = results[i-1][6]
  mRg[i, 1] = results[i-1][7]


# in case CGpeptide not used, label the files according to desired label name
if bReadPredFileDirect:
  np.savetxt(workingDir + 'predicted_nRmsd_' + sFileNamePred + '.txt', nRmsd)
  np.savetxt(workingDir + 'predicted_binsRmsd_' + sFileNamePred + '.txt', binsRmsd)
  np.savetxt(workingDir + 'predicted_mRmsd_' + sFileNamePred + '.txt', mRmsd)
  np.savetxt(workingDir + 'predicted_nRg_' + sFileNamePred + '.txt', nRg)
  np.savetxt(workingDir + 'predicted_binsRg_' + sFileNamePred + '.txt', binsRg)
  np.savetxt(workingDir + 'predicted_mRg_' + sFileNamePred + '.txt', mRg)

  # export binning
  np.savetxt("expBinRg.txt", binsRg[0, :])
  np.savetxt("expBinRmsd.txt", binsRmsd[0, :])

  #plotRMSDmi(inputXVG=rmsdList[0], n=nRmsd, bins=binsRmsd, m=mRmsd, outputPlot=workingDir + 'plots/rmsd_' + sFileNamePred + '_mi' + sExtZeroVar)
  #plotGYRATEmi(inputXVG=gyrateList[0], n=nRg, bins=binsRg, m=mRg, outputPlot=workingDir + 'plots/gyrate_' + sFileNamePred + '_mi' + sExtZeroVar)
  plotRMSDuq(_n=nRmsd, _bins=binsRmsd, _m=mRmsd, _outputPlot=sPredPath + 'plots/rmsd_' + sFileNamePred + '_uq' + sExtZeroVar)
  plotGYRATEuq(_n=nRg, _bins=binsRg, _m=mRg, _outputPlot=sPredPath + 'plots/gyrate_' + sFileNamePred + '_uq' + sExtZeroVar)
else:
  np.savetxt(sPredPath+'predicted_nRmsd_'+ str(iteration)+'.txt',nRmsd)
  np.savetxt(sPredPath+'predicted_binsRmsd_'+ str(iteration)+'.txt',binsRmsd)
  np.savetxt(sPredPath+'predicted_mRmsd_'+ str(iteration)+'.txt',mRmsd)
  np.savetxt(sPredPath+'predicted_nRg_'+ str(iteration)+'.txt',nRg)
  np.savetxt(sPredPath+'predicted_binsRg_'+ str(iteration)+'.txt',binsRg)
  np.savetxt(sPredPath+'predicted_mRg_'+ str(iteration)+'.txt',mRg)

  # export binning
  np.savetxt("expBinRg.txt", binsRg[0, :])
  np.savetxt("expBinRmsd.txt", binsRmsd[0, :])

  #plotRMSDmi(inputXVG=rmsdList[0], n=nRmsd, bins=binsRmsd, m=mRmsd, outputPlot=workingDir + 'plots/rmsd_' + str(iteration) + '_mi' + sExtZeroVar)
  #plotGYRATEmi(inputXVG=gyrateList[0], n=nRg, bins=binsRg, m=mRg, outputPlot=workingDir + 'plots/gyrate_' + str(iteration) + '_mi' + sExtZeroVar)
  plotRMSDuq(_n=nRmsd, _bins=binsRmsd, _m=mRmsd, _outputPlot=sPredPath + 'plots/rmsd_' + str(iteration) + '_uq' + sExtZeroVar)
  plotGYRATEuq(_n=nRg, _bins=binsRg, _m=mRg, _outputPlot=sPredPath + 'plots/gyrate_' + str(iteration) + '_uq' + sExtZeroVar)

if bCleanFiles:
  #removeFiles(listRm=txtList)
  #removeFiles(listRm=xtcList)
  removeFiles(listRm=rmsdList)
  removeFiles(listRm=ramaList)
  removeFiles(listRm=gyrateList)

#if __name__ == '__main__':
#  main()
