
import os
#from subprocess import call
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def convertToFullCartersianCoordinates(data, dofsnp=np.array([18, 19, 20, 24, 25, 43]), dtype=int, x_dim_red=60, x_dim_original=66):

    dofs = dofsnp

    batch_size = data.shape[0]
    data_ext = np.zeros([batch_size, x_dim_original])

    data_ext[:, 0:dofs[0]] = data[:, 0:dofs[0]]
    data_ext[:, dofs[2]+1:dofs[3]] = data[:, dofs[0]:dofs[0]+3]
    data_ext[:, dofs[4]+1:dofs[5]] = data[:, dofs[0]+3:dofs[0]+3+(dofs[5]-dofs[4]-1)]
    data_ext[:, dofs[5]+1:] = data[:, dofs[0]+3+(dofs[5]-dofs[4]-1):]

    return data_ext

def convert_given_representation(samples, coordrep, unitgiven=1., bredcoord=False):

    convertfactor = unitgiven

    if coordrep == 'ang':
        samplesout = convertangulardataset(samples.T)
    elif coordrep == 'ang_augmented':
        samplesout = convertangularaugmenteddataset(samples.T)
    elif coordrep == 'ang_auggrouped':
        samplesout = convertangularaugmenteddataset(samples.T, bgrouped=True, convertfactor=convertfactor)
    else:
        if bredcoord:
            samples = convertToFullCartersianCoordinates(data=samples)
        samplesout = samples.T / convertfactor

    return samplesout

def getcolorcodeALA15(ramapath, N, ssize=5):
    """ Get color coding for ALA-15 1527 dataset. """

    from analyse_ala_15 import AngleCategorizer

    nResidues = 15
    #angles = np.loadtxt('rama_dataset_ala_15.xvg', skiprows=32, usecols=range(0, 2), delimiter='  ')
    angles = np.loadtxt(os.path.join(ramapath, 'rama_dataset_ala_15_1500.xvg'), skiprows=32, usecols=range(0, 2), delimiter='  ')
    nSamples = angles.shape[0]/15
    angles.resize(nSamples, nResidues, 2)
    angCat = AngleCategorizer(angles)
    angCat.categorize()
    angCat.countConfigurations()
    colInd = angCat.getColorMatrix()
    alphaInd = angCat.getAlphaVals()

    marker = list()
    patchlist = list()

    marker.append('o')
    marker.append('o')
    marker.append('o')

    import matplotlib.patches as mpatches
    patchlist.append(mpatches.Patch(color='black', label=r'$\alpha$'))
    patchlist.append(mpatches.Patch(color='blue', label=r'$\beta$-1'))
    patchlist.append(mpatches.Patch(color='red', label=r'$\beta$-2'))

    alpha = plt.scatter(0, 1, c='k', marker=marker[0], s=ssize, label=r'$\alpha$')
    beta1 = plt.scatter(0, 1, c='b', marker=marker[1], s=ssize, label=r'$\beta\textnormal{-}1$')
    beta2 = plt.scatter(0, 1, c='r', marker=marker[2], s=ssize, label=r'$\beta\textnormal{-}2$')
    plt.close()

    patchlist = [alpha, beta1, beta2]

    return colInd, marker, patchlist, alphaInd


def getcolorcode1527(ssize=5):
    """ Get color coding for ALA-2 1527 dataset. """

    iA = 29
    iB1 = 932
    iB2 = 566
    colInd = list()
    marker = list()
    patchlist = list()

    marker.append('o')
    marker.append('v')
    marker.append('x')

    for i in range(0, iA):
        colInd.append('k')
    for i in range(0, iB1):
        colInd.append('b')
    for i in range(0, iB2):
        colInd.append('r')


    import matplotlib.patches as mpatches
    patchlist.append(mpatches.Patch(color='black', label=r'$\alpha$'))
    patchlist.append(mpatches.Patch(color='blue', label=r'$\beta$-1'))
    patchlist.append(mpatches.Patch(color='red', label=r'$\beta$-2'))

    alpha = plt.scatter(0, 1, c='k', marker=marker[0], s=ssize, label=r'$\alpha$')
    beta1 = plt.scatter(0, 1, c='b', marker=marker[1], s=ssize, label=r'$\beta\textnormal{-}1$')
    beta2 = plt.scatter(0, 1, c='r', marker=marker[2], s=ssize, label=r'$\beta\textnormal{-}2$')
    plt.close()

    patchlist = [alpha, beta1, beta2]

    return colInd, marker, patchlist


def estimateProperties(samples_name, cluster, datasetN, pathofsamples=None, postS=0, nCores=2, peptide='ala_2'):

    command = ''

    if cluster == True:
        command += 'python /afs/crc.nd.edu/user/m/mschoebe/Private/projects/ganpepvae/estimate_properties.py'
        command += ' --referenceDirectory ' + '/afs/crc.nd.edu/user/m/mschoebe/Private/data/data_peptide/'
        command += ' --cluster ' + '2'
        command += ' --postS ' + str(postS)
        command += ' --nCores ' + str(nCores)
    else:
        #command += 'pyenv activate work; '
        command += 'python /home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/prediction/propteinpropcal/estimate_properties.py'
        command += ' --referenceDirectory ' + '/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/prediction/propteinpropcal/'

    if pathofsamples is not None:
        command += ' --predFilePath ' + pathofsamples + '/'

    command += ' --dataCollected random'

    command += ' --fileNamePred ' + samples_name
    command += ' --conformation ' + 'm'
    command += ' --peptide ' + peptide
    #--cluster 2 --postS 500 --nCores 24
    #os.system(command)
    #os.system(command)
    #call(['bash','pyenv activate work', command], shell=True)

    f = open(pathofsamples+'/est_prop.sh', 'w')
    #f.write('#!/bin/bash')

    if cluster == True:
        f.write('#!/bin/bash\n')
        f.write('#$ -N est_prop_' + samples_name + '\n')
        f.write('#$ -pe smp ' + str(nCores) + '\n')
        f.write('#$ -q debug\n\n')
        f.write('source activate work\n')
        f.write('module load gromacs\n\n')
    # $ -N est_prop
    # $ -pe smp 24
    # $ -q debug
    f.write(command)
    f.close()
    os.chmod(pathofsamples+'/est_prop.sh', 0o777)

    # this is for comparison with real dataset

    command = ''

    if cluster == True:
        command += 'python /afs/crc.nd.edu/user/m/mschoebe/Private/projects/ganpepvae/estimate_properties_compare.py'
        command += ' --referenceDirectory ' + '/afs/crc.nd.edu/user/m/mschoebe/Private/data/data_peptide/'
        command += ' --cluster ' + '2'
        command += ' --postS ' + str(postS)
        command += ' --nCores ' + str(nCores)
    else:
        #command += 'pyenv activate work; '
        command += 'python /home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/prediction/propteinpropcal/estimate_properties_compare.py'
        command += ' --referenceDirectory ' + '/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/prediction/propteinpropcal/'

    if pathofsamples is not None:
        command += ' --predFilePath ' + pathofsamples + '/'

    command += ' --dataCollected random'

    command += ' --compareRefData dataset_' + str(datasetN)

    command += ' --fileNamePred ' + samples_name
    command += ' --conformation ' + 'm'
    command += ' --peptide ' + peptide
    #--cluster 2 --postS 500 --nCores 24
    #os.system(command)
    #os.system(command)
    #call(['bash','pyenv activate work', command], shell=True)

    f = open(pathofsamples+'/est_prop_compare.sh', 'w')
    #f.write('#!/bin/bash')
    f.write(command)
    f.close()
    os.chmod(pathofsamples+'/est_prop_compare.sh', 0o777)


def getAbsCoordinates(xyz):
    _xyzAbs = np.zeros([xyz.shape[0] + 1, xyz.shape[1]])

    # number of residues
    nACE = 1
    nALA = 1
    nNME = 1

    ACEleng = 6
    ALAleng = 10
    NMEleng = 6

    # go through every residue
    aACE = np.zeros([nACE * ACEleng, 3])
    aALA = np.zeros([nALA * ALAleng, 3])
    aNME = np.zeros([nNME * NMEleng, 3])

    # 1HH3 = CH3 + (1HH3 - CH3)
    aACE[0, :] = xyz[0, :]
    # CH3 = 0
    # aACE[1,:] = 0
    # 2HH3 = CH3 + (2HH3 - CH3)
    aACE[2, :] = xyz[1, :]
    # 3HH3 = CH3 + (3HH3 - CH3)
    aACE[3, :] = xyz[2, :]
    # C = CH3 + (C - CH3)
    aACE[4, :] = xyz[3, :]
    # O = C + (O - C)
    aACE[5, :] = aACE[4, :] + xyz[4, :]

    # first N coordinate
    aALA[0, :] = aACE[4, :] + xyz[5, :]

    for iALA in range(0, nALA):
        # N = C + (N - C)
        if iALA > 0:
            aALA[iALA * ALAleng + 0, :] = aALA[iALA * ALAleng - 2, :] + xyz[ACEleng + iALA * ALAleng - 1, :]
        # H = N + (H - N)
        aALA[iALA * ALAleng + 1, :] = aALA[iALA * ALAleng + 0, :] + xyz[ACEleng + iALA * ALAleng + 0, :]
        # CA = N + (CA - N)
        aALA[iALA * ALAleng + 2, :] = aALA[iALA * ALAleng + 0, :] + xyz[ACEleng + iALA * ALAleng + 1, :]
        # HA = CA + (HA - CA)
        aALA[iALA * ALAleng + 3, :] = aALA[iALA * ALAleng + 2, :] + xyz[ACEleng + iALA * ALAleng + 2, :]
        # CB = CA + (CB - CA)
        aALA[iALA * ALAleng + 4, :] = aALA[iALA * ALAleng + 2, :] + xyz[ACEleng + iALA * ALAleng + 3, :]
        # HB1 = CB + (HB1 - CB)
        aALA[iALA * ALAleng + 5, :] = aALA[iALA * ALAleng + 4, :] + xyz[ACEleng + iALA * ALAleng + 4, :]
        # HB2 = CB + (HB2 - CB)
        aALA[iALA * ALAleng + 6, :] = aALA[iALA * ALAleng + 4, :] + xyz[ACEleng + iALA * ALAleng + 5, :]
        # HB3 = CB + (HB3 - CB)
        aALA[iALA * ALAleng + 7, :] = aALA[iALA * ALAleng + 4, :] + xyz[ACEleng + iALA * ALAleng + 6, :]
        # C = CA + (C - CA)
        aALA[iALA * ALAleng + 8, :] = aALA[iALA * ALAleng + 2, :] + xyz[ACEleng + iALA * ALAleng + 7, :]
        # O = C + (O - C)
        aALA[iALA * ALAleng + 9, :] = aALA[iALA * ALAleng + 8, :] + xyz[ACEleng + iALA * ALAleng + 8, :]

    # Last part
    # N = C + (N - C)
    aNME[0, :] = aALA[nALA * ALAleng - 2, :] + xyz[ACEleng + nALA * ALAleng - 1, :]
    # H = N + (H - N)
    aNME[1, :] = aNME[0, :] + xyz[ACEleng + nALA * ALAleng + 0, :]
    # CH3 = N + (CH3 - N)
    aNME[2, :] = aNME[0, :] + xyz[ACEleng + nALA * ALAleng + 1, :]
    # 1HH3 = CH3 + (1HH3 - CH3)
    aNME[3, :] = aNME[2, :] + xyz[ACEleng + nALA * ALAleng + 2, :]
    # 2HH3 = CH3 + (2HH3 - CH3)
    aNME[4, :] = aNME[2, :] + xyz[ACEleng + nALA * ALAleng + 3, :]
    # 3HH3 = CH3 + (2HH3 - CH3)
    aNME[5, :] = aNME[2, :] + xyz[ACEleng + nALA * ALAleng + 4, :]

    _xyzAbs[0:(ACEleng), :] = aACE
    _xyzAbs[ACEleng:(ACEleng + nALA * ALAleng), :] = aALA
    _xyzAbs[(ACEleng + nALA * ALAleng):, :] = aNME

    return _xyzAbs


def getCartesian(rphitheta, dataaugmented=False):
    rphithetaShape = rphitheta.shape

    if dataaugmented:
        _xyz = np.zeros([rphithetaShape[0], 3])
        r = rphitheta[:, 0]
        sinphi = rphitheta[:, 1]
        cosphi = rphitheta[:, 2]
        sintheta = rphitheta[:, 3]
        costheta = rphitheta[:, 4]
        _xyz[:, 0] = r * costheta * sinphi
        _xyz[:, 1] = r * sintheta * sinphi
        _xyz[:, 2] = r * cosphi
    else:
        _xyz = np.zeros(rphithetaShape)
        _xyz[:, 0] = rphitheta[:, 0] * np.cos(rphitheta[:, 2]) * np.sin(rphitheta[:, 1])
        _xyz[:, 1] = rphitheta[:, 0] * np.sin(rphitheta[:, 2]) * np.sin(rphitheta[:, 1])
        _xyz[:, 2] = rphitheta[:, 0] * np.cos(rphitheta[:, 1])

    xyzAbs = getAbsCoordinates(xyz=_xyz)

    return xyzAbs

def convertangulardataset(data):

    #outname = 'samples.txt'
    #data = np.loadtxt('dataset_mixed_1527_ang.txt')

    dim = data.shape[0]
    n = data.shape[1]

    datacatout = np.zeros([dim+3, n])

    for j in range(0, n):
        sample = data[:,j]
        rphitheta = np.zeros([dim/3, 3])
        for i in range(0, rphitheta.shape[0]):
            rphitheta[i, 0] = sample[i * 3 + 0]
            rphitheta[i, 1] = sample[i * 3 + 1]
            rphitheta[i, 2] = sample[i * 3 + 2]

        datacoord = getCartesian(rphitheta=rphitheta)
        datacoordvec = np.reshape(datacoord, sample.shape[0] + 3)
        datacatout[:, j] = np.copy(datacoordvec)

    return datacatout
    #np.savetxt(outname, datacatout)

def convertangularaugmenteddataset(data, bgrouped=False, convertfactor=1.):

    #outname = 'samples.txt'
    #data = np.loadtxt('dataset_mixed_1527_ang.txt')

    dim = data.shape[0]
    n = data.shape[1]

    # specify the size of one coordinate point: here (r, sin \theta, cos \theta, sin \psi, cos \psi)
    sizeofcoord = 5
    nparticles = int(dim / sizeofcoord + 1)
    ncoordtuples = nparticles - 1

    datacatout = np.zeros([nparticles * 3, n])

    if bgrouped:
        dataUse = np.zeros(data.shape)
        # sorted dataset r1 r2 r3 r4 , sin sin sin
        # temporary dataset for
        r = data[0 * ncoordtuples:1 * ncoordtuples, :]
        sinphi = data[1 * ncoordtuples:2 * ncoordtuples, :]
        cosphi = data[2 * ncoordtuples:3 * ncoordtuples, :]
        sintheta = data[3 * ncoordtuples:4 * ncoordtuples, :]
        costheta = data[4 * ncoordtuples:5 * ncoordtuples, :]
        for i in range(0, ncoordtuples):
            dataUse[i * sizeofcoord + 0, :] = r[i, :] / convertfactor
            dataUse[i * sizeofcoord + 1, :] = sinphi[i, :]
            dataUse[i * sizeofcoord + 2, :] = cosphi[i, :]
            dataUse[i * sizeofcoord + 3, :] = sintheta[i, :]
            dataUse[i * sizeofcoord + 4, :] = costheta[i, :]
    else:
        dataUse = np.copy(data)

    for j in range(0, n):
        sample = dataUse[:, j]
        rphithetaaugmented = np.zeros([int(dim/sizeofcoord), sizeofcoord])
        for i in range(0, rphithetaaugmented.shape[0]):
            rphithetaaugmented[i, 0] = sample[i * sizeofcoord + 0]
            rphithetaaugmented[i, 1] = sample[i * sizeofcoord + 1]
            rphithetaaugmented[i, 2] = sample[i * sizeofcoord + 2]
            rphithetaaugmented[i, 3] = sample[i * sizeofcoord + 3]
            rphithetaaugmented[i, 4] = sample[i * sizeofcoord + 4]

        datacoord = getCartesian(rphitheta=rphithetaaugmented, dataaugmented=True)
        datacoordvec = np.reshape(datacoord, nparticles * 3)
        datacatout[:, j] = np.copy(datacoordvec)

    return datacatout
    #np.savetxt(outname, datacatout)