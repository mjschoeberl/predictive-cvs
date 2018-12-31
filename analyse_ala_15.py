
import numpy as np
import matplotlib.pyplot as plt

class AngleCategorizer:
    def __init__(self, angles):

        self.angles = angles
        self.N = angles.shape[0]
        self.Nres = angles.shape[1]
        self.Nconf = 3
        self.alphaMin = 0.2
        self.countConf = np.zeros([self.N, self.Nconf])

        # selection alpha
        self.phiMinAlpha = -70
        self.phiMaxAlpha = -50
        self.psiMinAlpha = -60
        self.psiMaxAlpha = -40

        # selection beta1
        self.phiMaxAbsBeta1 = 40
        self.phiMeanBeta1 = -140
        self.psiMaxAbsBeta1 = 50
        self.psiMeanBeta1 = 155

        # selection beta2
        self.phiMaxAbsBeta2 = 40
        self.phiMeanBeta2 = -60
        self.psiMaxAbsBeta2 = 60
        self.psiMeanBeta2 = 150

    def categorize(self):

        # select trajectories in alpha configuration
        self.indexAlpha = np.where(
            (self.angles[:, :, 0] >= self.phiMinAlpha) & (self.angles[:, :, 0] <= self.phiMaxAlpha) & (self.angles[:, :, 1] >= self.psiMinAlpha) & (
                    self.angles[:, :, 1] <= self.psiMaxAlpha))

        # select trajectories in beta1 configuration
        indSmall = np.where(self.angles[:, :, 1] < 0)
        psiForBeta = np.copy(self.angles[:, :, 1])
        psiForBeta[indSmall] = psiForBeta[indSmall] + 360.

        self.indexBeta1 = np.where((abs(self.angles[:, :, 0] - self.phiMeanBeta1) <= self.phiMaxAbsBeta1) & (
                    abs(psiForBeta[:] - self.psiMeanBeta1) < self.psiMaxAbsBeta1))

        # select trajectories in beta2 configuration
        self.indexBeta2 = np.where((abs(self.angles[:, :, 0] - self.phiMeanBeta2) <= self.phiMaxAbsBeta2) & (
                    abs(psiForBeta[:] - self.psiMeanBeta2) < self.psiMaxAbsBeta2))

    def getIndices(self, strcategory='alpha'):

        if hasattr(self, 'indexAlpha'):
            if 'alpha' in strcategory:
                return self.indexAlpha
            elif 'beta1' in strcategory:
                return self.indexBeta1
            elif 'beta2' in strcategory:
                return self.indexBeta2
        else:
            print 'Please categorize trajectory first.'

    def countOneConfiguration(self, indicesArray, confID):

        if indicesArray==None:
            print 'Please specify indices by categorization.'
        else:
            for i in range(self.N):
                self.countConf[i, confID] = indicesArray[0][indicesArray[0]==i].shape[0]

    def countConfigurations(self):

        self.countOneConfiguration(indicesArray=self.indexAlpha, confID=0)
        self.countOneConfiguration(indicesArray=self.indexBeta1, confID=1)
        self.countOneConfiguration(indicesArray=self.indexBeta2, confID=2)
        #self.countConf = self.countConf/float(self.Nres)
        self.countCategorizedResiduesPerSamples =  np.asfarray(self.countConf.sum(axis=1))
        self.countConf = self.countConf/np.asfarray(self.countCategorizedResiduesPerSamples.reshape([self.N, 1]))
        self.alphaVal = (self.countCategorizedResiduesPerSamples/self.Nres + self.alphaMin)/(self.alphaMin+1.0)
        #return self.countConf

    def getColors(self, N):

        import colorsys
        HSV = [(x * 1.0 / N, 1., 1.) for x in range(N)]
        RGB = map(lambda x: colorsys.hsv_to_rgb(*x), HSV)
        return RGB

    def getColorMatrix(self):

        #RGB = self.getColors(self.Nconf)
        RGB = ([0., 0., 0.], [0., 0., 1.], [1., 0., 0.])

        #testmat = np.array([[1.,0.,0],[0.,1.,0],[0.,0.,1.],[0.5,.5,0]])

        col = np.inner(self.countConf, np.array(RGB).T)
        return col

    def getAlphaVals(self):
        return self.alphaVal

    def testPlot(self):

        f, ax = plt.subplots(1)
        col = self.getColorMatrix()
        ax.scatter(np.arange(self.N), np.repeat(0.5, self.N), c=col, s=200)
        plt.show(f)

def main():
    nResidues = 15
    #angles = np.loadtxt('rama_dataset_ala_15.xvg', skiprows=32, usecols=range(0, 2), delimiter='  ')
    angles = np.loadtxt('/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/prediction/propteinpropcal/rama_dataset_ala_15_10000.xvg', skiprows=32, usecols=range(0, 2), delimiter='  ')
    nSamples = angles.shape[0]/15
    angles.resize(nSamples, nResidues, 2)

    angles = angles[0:4,:,:]

    angCat = AngleCategorizer(angles)
    angCat.categorize()
    print angCat.getIndices(strcategory='beta1')

    angCat.countConfigurations()
    angCat.testPlot()


if __name__== '__main__':
    main()
