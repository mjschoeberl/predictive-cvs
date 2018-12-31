from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import os

import matplotlib


matplotlib.use('Agg')

font = {'weight' : 'normal',
        'size'   : 16}

#font = {'weight' : 'normal',
#        'size'   : 5}

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt


def colorpointsgaussian(x, nsamples, name_colmap=''):

    from scipy.stats import multivariate_normal
    x_dim = x.shape[1]
    var = multivariate_normal(mean=np.zeros(x_dim), cov=np.eye(x_dim))
    p = var.pdf(x)

    pmin = p.min()
    pmax = p.max()

    pscaled = (p - pmin) / (pmax - pmin)

    cm = getattr(matplotlib.cm, name_colmap)
    cmap = cm(pscaled)

    return cmap


class VAEparent(nn.Module):
    def __init__(self, args, x_dim, bfixlogvar):
        super(VAEparent, self).__init__()

        self.bplotdecoder = False
        self.bplotencoder = False
        self.bgetlogvar = False

        self.bfixlogvar = bfixlogvar

        self.x_dim = x_dim
        self.z_dim = args.z_dim

        self.listenc = []
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()

    def get_encoding_decoding_variance(self, x):

        mu, logvar = self.encode(x)
        btemp = self.bgetlogvar
        self.bgetlogvar = True
        mu_pred, logvar_pred = self.decode(mu)
        self.bgetlogvar = btemp

        var_decoder = logvar_pred.exp()
        var_encoder = logvar.exp()
        l2norm_var_dec = var_decoder.norm()
        l2norm_var_enc = var_encoder.norm()

        return {'var_encoder': var_encoder, 'var_decoder': var_decoder, 'norm_enc': l2norm_var_enc.data.numpy(), 'norm_dec': l2norm_var_dec.data.numpy()}

    def plotlatentrep(self, x, z_dim, path, postfix='', iter=-1, x_curr=0, y_curr=0, nprov=False, normaltemp=0, x_train=None, peptide='ala_2', data_dir=None):

        baddactfctannotation = False
        sizedataset = x.shape[0]

        mu, logvar = self.encode(x)

        munp = mu.data.cpu().numpy()

        ssize = 20

        # get the color code, markers, and legend addons
        if peptide is 'ala_2':
            from utils_peptide import getcolorcode1527
            colcode, markers, patchlist = getcolorcode1527(ssize=ssize)
        else:
            from utils_peptide import getcolorcodeALA15
            colcode, markers, patchlist, alphaPerSample = getcolorcodeALA15(ramapath=os.path.join(data_dir, 'ala-15'),
                                                                            ssize=ssize, N=sizedataset)

        if z_dim == 2:

            #fontloc = {'weight': 'normal', 'size': 10}
            #matplotlib.rc('font', **fontloc)

            plt.figure(1)
            f, ax = plt.subplots()

            iA = 29
            iB1 = 932
            iB2 = 566

            # plot N(0,I)
            #n_samples_normal = iA + iB1 + iB2
            n_samples_normal = 4000
            # Plot some samples from p(z)?
            if not nprov:
                normal = np.random.randn(n_samples_normal, 2)
            else:
                normal = normaltemp

            # This is deprecated.
            if False:
                normalpatch = ax.scatter(normal[:, 0], normal[:, 1], c='g', marker='.', s=ssize, alpha=alpha,
                                     label=r'$\boldsymbol{z} \sim \mathcal N (\boldsymbol{0},\boldsymbol{I})$')
                #h,l= ax.get_legend_handles_labels()
                patchlist.append(normalpatch)

            if peptide is 'ala_2':
                # Modify scatter points according their atomistic conformation vor visualization purposes.
                x, y = munp[0:iA, 0], munp[0:iA, 1]
                ax.scatter(x, y, c=colcode[0:iA], marker=markers[0], s=ssize)
                x, y = munp[iA:iA + iB1, 0], munp[iA:iA + iB1, 1]
                ax.scatter(x, y, c=colcode[iA:iA+iB1], marker=markers[1], s=ssize)
                x, y = munp[iA + iB1:iA + iB1 + iB2, 0], munp[iA + iB1:iA + iB1 + iB2, 1]
                ax.scatter(x, y, c=colcode[iA+iB1:iA+iB1+iB2], marker=markers[2], s=ssize)
            else:
                # In case of ALA15 the color coding is obtained according remarks in **paper**.
                x, y = munp[:, 0], munp[:, 1]
                [ax.scatter(x[i], y[i], c=colcode[i, :], s=10, alpha=alphaPerSample[i]) for i in range(sizedataset)]

            if baddactfctannotation:
                # This would add a text field with activation functions used.

                # List of encoder activation functions
                an = []
                an.append(ax.annotate('Encoder activations:', xy=(-2., 2.7), xycoords="data",
                      va="center", ha="center"))
                an.append(ax.annotate(self.listenc[0], xy=(1, 0.5), xycoords=an[0],  # (1,0.5) of the an1's bbox
                                      xytext=(20, 0), textcoords="offset points",
                                      va="center", ha="left",
                                      bbox=dict(boxstyle="round", fc="None")))
                for i in range(1, len(self.listenc)):
                    an.append(ax.annotate(self.listenc[i], xy=(1, 0.5), xycoords=an[i], # (1,0.5) of the an1's bbox
                      xytext=(20, 0), textcoords="offset points",
                      va="center", ha="left",
                      bbox=dict(boxstyle="round", fc="None"),
                      arrowprops=dict(arrowstyle="<-")))

            if x_train is not None:

                # encode the training data
                mu_train, logvar_train = self.encode(x_train)
                munp_train = mu_train.data.cpu().numpy()
                leng_train = munp_train.shape[0]
                # plot the training data
                if iter >= 0:
                    rnd = False
                    a_training_data = 0.6
                    col_training_data = 'C4'
                else:
                    rnd = True
                    a_training_data = 0.7
                    col_training_data = 'y'

                train_patch = ax.scatter(munp_train[:, 0], munp_train[:, 1],
                                         c=col_training_data, marker='d', s=ssize*0.9, alpha=a_training_data,
                                         label=r'Training Data')
                patchlist.append(train_patch)

            #ax.set_ylim([-3, 3])
            #ax.set_xlim([-3, 3])
            ax.set_xlabel(r'$z_1$')
            ax.set_ylabel(r'$z_2$')
            ax.grid(ls='dashed')
            ax.set_axisbelow(True)
            #ax.legend(handles=patchlist, loc=1)
            if x_train is None:
                ax.legend(handles=patchlist, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      fancybox=False, shadow=False, ncol=4)
            else:
                ax.legend(handles=patchlist, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          fancybox=False, shadow=False, ncol=3)

            if postfix == '' and iter < 0:
                ax.set_ylim([-4, 4])
                ax.set_xlim([-4, 4])

                ticksstep = 1.
                ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                ax.xaxis.set_ticks(ticks)
                ax.yaxis.set_ticks(ticks)

                f.savefig(path+'/lat_rep.pdf', bbox_inches='tight')#, transparent=True)
            elif postfix == '' and iter >= 0:
                ax.scatter(x_curr, y_curr, c='y', marker='*', s=ssize*35)
                ax.set_ylim([-4, 4])
                ax.set_xlim([-4, 4])

                ticksstep = 1.
                ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                ax.xaxis.set_ticks(ticks)
                ax.yaxis.set_ticks(ticks)

                f.savefig(path + '/lat_rep_vis_' + str(iter) + '.png', bbox_inches='tight')  # , transparent=True)
                return normal
            else:
                ax.set_ylim([-3.5, 3.5])
                ax.set_xlim([-3.5, 3.5])
                f.savefig(path + '/lat_rep' + postfix +'.png', bbox_inches='tight')  # , transparent=True)
            plt.close()
        elif peptide is 'ala_15':

            f, ax = plt.subplots(nrows=z_dim-1, ncols=z_dim-1, sharey=True, sharex=True)

            # This title is just valid if we use no training data different from the test data.
            if x_train is None:
                f.suptitle(r'AEVB: Encoded representation of training data: $\boldsymbol{\mu}(\boldsymbol{x}^{(i)})$')

            iA = 29
            iB1 = 932
            iB2 = 566

            # plot N(0,I)
            n_samples_normal = 4000
            if not nprov:
                normal = np.random.randn(n_samples_normal, z_dim)
            else:
                normal = normaltemp

            #if x_train is None:
            # deprecated
            if False:
                for i in range(z_dim-1):
                    for j in range(i, z_dim-1):
                        if not i == (j + 1):
                            normalpatch = ax[i, j].scatter(normal[:, i], normal[:, j+1], c='g', marker='.', s=ssize, alpha=alpha,
                                         label=r'$\boldsymbol{z} \sim \mathcal N (\boldsymbol{0},\boldsymbol{I})$')
                #h,l= ax.get_legend_handles_labels()
                patchlist.append(normalpatch)

            if peptide is 'ala_2':
                x, y = munp[0:iA, 0], munp[0:iA, 1]
                ax.scatter(x, y, c=colcode[0:iA], marker=markers[0], s=ssize)
                x, y = munp[iA:iA + iB1, 0], munp[iA:iA + iB1, 1]
                ax.scatter(x, y, c=colcode[iA:iA+iB1], marker=markers[1], s=ssize)
                x, y = munp[iA + iB1:iA + iB1 + iB2, 0], munp[iA + iB1:iA + iB1 + iB2, 1]
                ax.scatter(x, y, c=colcode[iA+iB1:iA+iB1+iB2], marker=markers[2], s=ssize)
            else:
                for i in range(z_dim-1):
                    for j in range(i, z_dim-1):
                        if not i == (j + 1):
                            x, y = munp[:, i], munp[:, j+1]
                            #[(x * 1.0 / N, 1., 1.) for x in range(N)]
                            if z_dim > 4:
                                ax[i, j].scatter(x, y, c=colcode, s=10)
                            else:
                                [ax[i, j].scatter(x[l], y[l], c=colcode[l, :], s=10, alpha=alphaPerSample[l]) for l in range(sizedataset)]
                            #ax.scatter(x, y, c=colcode, s=10)
            # TODO IMPLEMENT THIS FOR VAE
            if False and baddactfctannotation:
                # list of encoder activation functions
                an = []
                an.append(ax.annotate('Encoder activations:', xy=(-2., 2.7), xycoords="data",
                      va="center", ha="center"))
                an.append(ax.annotate(self.listenc[0], xy=(1, 0.5), xycoords=an[0],  # (1,0.5) of the an1's bbox
                                      xytext=(20, 0), textcoords="offset points",
                                      va="center", ha="left",
                                      bbox=dict(boxstyle="round", fc="None")))
                for i in range(1, len(self.listenc)):
                    an.append(ax.annotate(self.listenc[i], xy=(1, 0.5), xycoords=an[i], # (1,0.5) of the an1's bbox
                      xytext=(20, 0), textcoords="offset points",
                      va="center", ha="left",
                      bbox=dict(boxstyle="round", fc="None"),
                      arrowprops=dict(arrowstyle="<-")))
                # va="center", ha="left",

            #ax.set_ylim([-3, 3])
            #ax.set_xlim([-3, 3])
            for i in range(z_dim - 1):
                for j in range(z_dim - 1):
                    if not i==(j+1):
                        ax[i, j].set_xlabel(r'$z_%d$' % i)
                        ax[i, j].set_ylabel(r'$z_%d$' % j)
                        ax[i, j].set_xlim([-5, 5])
                        ax[i, j].set_ylim([-5, 5])
                        ax[i, j].grid(ls='dashed')

            if postfix == '' and iter < 0:
                #ax.set_ylim([-4, 4])
                #ax.set_xlim([-4, 4])
                #ticksstep = 1.
                #ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                #ax.xaxis.set_ticks(ticks)
                #ax.yaxis.set_ticks(ticks)
                f.savefig(path+'/lat_rep.pdf', bbox_inches='tight')#, transparent=True)
            elif postfix == '' and iter >= 0:
                #ax.scatter(x_curr, y_curr, c='y', marker='*', s=ssize*35)
                #ax.set_ylim([-4, 4])
                #ax.set_xlim([-4, 4])
                f.savefig(path + '/lat_rep_vis_' + str(iter) + '.png', bbox_inches='tight')  # , transparent=True)
                return normal
            else:
                #ax.set_ylim([-3.5, 3.5])
                #ax.set_xlim([-3.5, 3.5])
                f.savefig(path + '/lat_rep' + postfix +'.png', bbox_inches='tight')  # , transparent=True)
            plt.close()
        else:
            print('Warining: Representation of data in latent space not possible: z_dim is no 2')


class VAEmod(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar):
        super(VAEmod, self).__init__(args, x_dim, bfixlogvar)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.zeros(x_dim), requires_grad=True)

        h1_dim = 50
        h11_dim = 100
        h12_dim = 100

        # encoder
        self.enc_fc10 = nn.Linear(x_dim, h12_dim)
        self.enc_fc11 = nn.Linear(h12_dim, h11_dim)
        self.enc_fc12 = nn.Linear(h11_dim, h1_dim)
        self.enc_fc21 = nn.Linear(h1_dim, self.z_dim)
        self.enc_fc22 = nn.Linear(h1_dim, self.z_dim)

        # decoder
        self.dec_fc30 = nn.Linear(self.z_dim, h1_dim)
        self.dec_fc31 = nn.Linear(h1_dim, h11_dim)
        self.dec_fc32 = nn.Linear(h11_dim, h12_dim)
        self.dec_fc4 = nn.Linear(h12_dim, x_dim)

        if not hasattr(self, 'dec_logvar'):
            self.dec_fc5 = nn.Linear(h12_dim, x_dim)

    def encode(self, x):

        self.listenc = ['selu', 'selu', 'logsig']

        if True:
            h10 = self.selu(self.enc_fc10(x))
            h11 = self.selu(self.enc_fc11(h10))
            h1 = F.logsigmoid(self.enc_fc12(h11))

        return self.enc_fc21(h1), self.enc_fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        h30 = self.tanh(self.dec_fc30(z))
        h31 = self.tanh(self.dec_fc31(h30))
        h32 = self.tanh(self.dec_fc32(h31))

        mu = self.dec_fc4(h32)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            logvar = self.dec_fc5(h32)

        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar