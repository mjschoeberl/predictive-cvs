# -*- coding: utf-8 -*-

from __future__ import print_function
import utils
import torch
import time
import os
import pickle
import datetime

import torch._utils
# maintaining upward compatibility
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils_peptide import convert_given_representation
from utils_peptide import estimateProperties


class ARDprior:
    '''Helper class for calculating contributions from the ARD prior.

    This class provides calculates log p(\theta) while \theta being the parameters
    the ARD prior is applied. This assumes an independent precision for every \theta.
    See Eq. (20) in **paper_link**.
    '''
    def __init__(self, a0, paramlist, model, gpu):
        self.model = model
        self.paramlist = paramlist
        self.a0 = a0
        self.b0 = a0
        self.gpu = gpu

    def getlogpiorARD(self):
        i = 0
        if self.gpu:
            totalsumlogptheta = torch.autograd.Variable(torch.zeros(1)).cuda()
        else:
            totalsumlogptheta = torch.autograd.Variable(torch.zeros(1))

        for paramitem in self.paramlist:
            par = paramitem['params']

            psqu = par.pow(2.)
            denominator = psqu.mul(0.5) + self.b0
            nominator = torch.zeros_like(denominator)
            nominator.fill_(self.a0 + 0.5)
            expectau = nominator.div(denominator)
            logptheta = expectau.mul(psqu)
            logptheta.mul_(-0.5)
            sumlogptheta = logptheta.sum()

            totalsumlogptheta.add_(sumlogptheta)

        return totalsumlogptheta

class PseudoGibbs:
    '''Helper calss for the implementation of Metropolis-within-Gibbs Sampler.

    This class contains Algorithm 2 in **paper_link**.
    The original reference is https://arxiv.org/abs/1802.04826.

    '''
    def __init__(self, x_init, z_init, model):
        self.x_init = x_init
        self.z_init = z_init
        self.model = model
        self.model.bgetlogvar = True

        self.n_skip = 10
        self.n_init = 5000

    def sampleposterior(self, x):
        mu, logvar = self.model.encode(x)
        std = logvar.mul(0.5).exp_()
        post = torch.distributions.Normal(mu, std)
        sample = post.sample()
        return sample

    def samplepredictive(self, z):
        self.model.bgetlogvar = True
        mu, logvar = self.model.decode(z)
        std = logvar.mul(0.5).exp_()
        pred = torch.distributions.Normal(mu, std)
        x = pred.sample()
        return x

        mu, logvar = self.model.encode(x)
        post = torch.distributions.Normal(mu, logvar.exp_())
        sample = post.sample()
        return sample

    def evallogprobposterior(self, x, z):
        mu, logvar = self.model.encode(x)
        std = logvar.mul(0.5).exp_()
        post = torch.distributions.Normal(mu, std)
        logprob = post.log_prob(z).sum()
        return logprob

    def evallogprobcondpred(self, x, z):
        self.model.bgetlogvar = True
        mu, logvar = self.model.decode(z)
        std = logvar.mul(0.5).exp_()
        post = torch.distributions.Normal(mu, std)
        logprob = post.log_prob(x).sum()
        return logprob

    def evallogprobprior(self, z):
        mu = torch.zeros_like(z)
        scale = torch.ones_like(z)
        prior = torch.distributions.Normal(mu, scale)
        logprob = prior.log_prob(z).sum()
        return logprob

    def calcacceptanceratio(self, ztm1, ztprop, xtm1):

        p_xtm1_given_ztprop = self.evallogprobcondpred(xtm1, ztprop)
        p_xtm1_given_ztm1 = self.evallogprobcondpred(xtm1, ztm1)

        p_ztprop = self.evallogprobprior(ztprop)
        p_ztm1 = self.evallogprobprior(ztm1)

        q_ztm1_given_xtm1 = self.evallogprobposterior(xtm1, ztm1)
        q_ztprop_given_xtm1 = self.evallogprobposterior(xtm1, ztprop)

        ratio_pxgz = p_xtm1_given_ztprop - p_xtm1_given_ztm1
        ratio_pz = p_ztprop - p_ztm1
        ratio_qzgx = q_ztm1_given_xtm1 - q_ztprop_given_xtm1

        logroh = ratio_pxgz + ratio_pz + ratio_qzgx

        return logroh.exp()

    def getboolofvariable(self, bytetensor):
        res = bool(bytetensor[0])
        return res

    def sample(self, N):

        n_tot = self.n_init + N*self.n_skip
        n_accepted = 0

        xtm1 = self.x_init
        ztm1 = self.z_init

        x_samples = xtm1

        for i in range(n_tot):
            ztprop = self.sampleposterior(xtm1)
            rhot = self.calcacceptanceratio(ztm1, ztprop, xtm1)
            rhottensor = rhot.data[0]

            if rhottensor > 1.:
                zt = ztprop
                n_accepted += 1
            else:
                r = torch.rand(1)
                if rhottensor > r[0]:
                    zt = ztprop
                    n_accepted += 1
                else:
                    zt = ztm1

            xt = self.samplepredictive(zt)

            s = i - self.n_init
            if s > 1 and s % self.n_skip == 0:
                x_samples = torch.cat((x_samples, xt), 0)

            ztm1 = zt
            xtm1 = xt

        accept_ratio = n_accepted/float(n_tot)
        print(accept_ratio)

        return x_samples


class MVN:
    '''Helper class for multivariate Gaussian.
    '''
    def __init__(self, mean, cov):
        self.mean = mean.copy()
        self.cov = cov.copy()
    def sample(self):
        return np.random.multivariate_normal(self.mean, self.cov)

class UQ:
    '''Helper class for options with regard to uncertainty quantification of
    the model parameters.
    '''
    def __init__(self, bdouq=False, bcalchess=False, blayercov=False, buqbias=False):
        self.bdouq = bdouq
        self.npostsamples = 100
        self.bhessavailable = bcalchess
        self.blayercov = blayercov
        self.buqbias = buqbias

def checkandcreatefolder(dir):
    '''This function checks if a directory exists, if not it is created augmented by a date and time dependent postfix.

    :param dir: Complete path of the directory
    :return: Actual directory created.
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
        diraug = dir
    else:
        datetimepostfix = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        diraug = os.path.join(dir, datetimepostfix)
        os.makedirs(diraug)
    return diraug

def count_parameters(model):
    '''This function counts the parameters of a PyTorch model.

    :param model: PyTorch model.
    :return: Number of parameters requiring gradient.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TensorDatasetDataOnly(torch.utils.data.Dataset):
    """Dataset wrapping only data tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
    """

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

class VAEpeptide(object):
    def __init__(self, args):
        '''Initialize the VAE object.
        :param args: Arguments containing settings.
        '''

        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = bool(args.gpu_mode) and torch.cuda.is_available()

        self.model_name = args.mod_type
        self.c = args.clipping  # clipping value
        self.n_critic = args.n_critic  # the number of iterations of the critic per generator iteration
        self.z_dim = args.z_dim
        self.n_samples = args.samples_pred
        self.bClusterND = bool(args.clusterND)
        self.output_postfix = args.outPostFix
        self.angulardata = args.useangulardat
        self.autoencvarbayes = bool(args.AEVB)
        self.L = args.L  # amount of eps ~ p(eps) = N(0,1)
        self.n_samples_per_mu = args.samples_per_mean  # if 0, just use mean prediction: x = mu(z)
        self.lambdaexpprior = args.exppriorvar

        # Employ Metropolis-within-Gibbs Sampler
        self.exactlikeli = bool(args.exactlikeli)

        # perform UQ
        bqu = bool(args.npostS)
        self.uqoptions = UQ(bdouq=bqu, bcalchess=True, blayercov=False, buqbias=bool(args.uqbias))
        self.uqoptions.npostsamples = args.npostS
        # \sigma_\theta depending on z or not. If not, optimized as parameter
        self.bfixlogvar = bool(args.sharedlogvar)

        # Check if a trained model should be loaded
        self.filemodel = args.loadtrainedmodel
        self.bloadmodel = bool(self.filemodel)

        # Visualize training process?
        self.bvislatent_training = True
        self.bvismean_and_samples = False

        # ARD prior
        if args.ard > 0.:
            self.bard = True
            self.arda0 = args.ard
        else:
            self.bard = False
            self.arda0 = 0.

        # We can only sample if p(x|z) is defined like in Autoencoding Var. Bayes.
        if not self.autoencvarbayes:
            self.n_samples_per_mu = 0

        # Specify input data for e.g. angular. This is not scope of the work **paper_link** and therefore not content of the code.
        if False:
            self.x_dim = (22 - 1) * 3
            print('Error: Model for %s not implemented yet.' % self.angulardata)
            quit()
        elif self.angulardata == 'ang_augmented':
            self.x_dim = (22 - 1) * 5
            print('Error: Model for %s not implemented yet.' % self.angulardata)
            quit()
        elif self.angulardata == 'ang_auggrouped':
            self.x_dim = (22 - 1) * 5

            if self.autoencvarbayes:
                from VAEmodel import VAEmodauggrouped as VAEmod
            else:
                print('Not covered in **paper**.')
                quit()
        else:
            if 'ala_15' in self.dataset:
                self.x_dim = 162 * 3
            else:
                self.x_dim = 66
            if self.autoencvarbayes:
                from VAEmodel import VAEmod
            else:
                print('Not covered in **paper**.')
                quit()

        # seed the calculation for testing
        if not args.seed == 0:
            torch.manual_seed(args.seed)
            if bool(args.gpu_mode):
                torch.cuda.manual_seed(args.seed)

        # pre-sepcify foldername variable for dataset
        foldername = self.dataset
        predictprefix = ''

        # Select dataset
        self.selectdataset()

        # Specify as model_name the general kind of dataset: mixed or separate
        self.postfix_setting = self.dataset + '_z_' + str(self.z_dim) + '_' + str(self.batch_size) + '_' + str(self.epoch)
        self.predprefix = predictprefix

        # Saving directory
        working_dir = os.getcwd()
        tempdir = os.path.join(working_dir, self.result_dir, self.model_name, foldername, self.output_postfix)
        self.output_dir = checkandcreatefolder(dir=tempdir)

        # Initialize VAE model
        self.vaemodel = VAEmod(args, self.x_dim, self.bfixlogvar)

        # This should be changed for later versions of PyTorch
        if self.gpu_mode:
            self.vaemodel.cuda()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.vaemodel.parameters(), lr=1e-3)

        # Initialize ARD prior
        if self.bard:
            self.ardprior = ARDprior(self.arda0, self.getdecweightlist(), self.vaemodel, self.gpu_mode)

    def getweightlist(self):
        '''This function obtains a parameter list of the model.
        :return: Dict list of parameters.
        '''

        weight_list = []
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                weight_list.append({'name': name, 'id': id, 'params': param})
                #pclone = param.clone()
                #params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                print(name)  # , param.data
            id = id + 1
        return weight_list

    def getdecweightlist(self):
        '''This function obtains a parameter list of the decoding part of the model.
        :return: Dict list of parameters.
        '''

        decoding_weight_list = []
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                # UQ only for decoding network
                if 'dec_' in name:
                    # check if we want to uq bias uncertainty
                    if not ('.bias' in name) and not ('logvar' in name):
                        decoding_weight_list.append({'name': name, 'id': id, 'params': param})
                        print(name)  # , param.data
                    id = id + 1
        return decoding_weight_list

    def loss_function_autoencvarbayes(self, recon_mu, recon_logvar, x, mu, logvar, x_dim=784):
        ''' This function computes the objective according Autoencoding Variational Bayes of D. Kingma and M. Welling, 2014.

        :param recon_mu: Reconstructed mean of p(x|z) with z is the encoded data.
        :param recon_logvar: Reconstructed variance of p(x|z) with z is the encoded data.
        :param x: Data.
        :param mu: Mean of encoded x (data).
        :param logvar: Variance of encoded x (data).
        :param x_dim: Dim(x_i) of a single datum.
        :return:
        '''

        pointwiseMSEloss = 0.5 * F.mse_loss(recon_mu, x.view(-1, x_dim), size_average=False, reduce=False)

        # Maug is here the augmentet bacht size: explicitly: M*L while L indicates the sample for reparametrization \epsilon ~ p(\epsilon)
        Maug = pointwiseMSEloss.shape[0]

        sigsq = recon_logvar.exp()
        weight = sigsq.reciprocal()
        logvarobjective = 0.5 * recon_logvar.sum()
        pointwiseWeightedMSEloss = pointwiseMSEloss.mul(weight)
        # This implies the contribution from a Gaussian p(x|z) with the diagonal covariance entries sigsq.
        WeightedMSEloss = pointwiseWeightedMSEloss.sum()

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.train_hist['kl_qp'].append(KLD)

        # Prior on predictive variance
        # This is not used in the **paper_link**, thus not relevant.
        psigsqlamb = self.lambdaexpprior
        # employ prior if desired
        if psigsqlamb > 0.:
            lamb = torch.FloatTensor(1)
            lamb.fill_(psigsqlamb)

            if self.gpu_mode:
                lambvariable = Variable(lamb.cuda())
            else:
                lambvariable = Variable(lamb)
            loglamb = lambvariable.log()

            # minus here becuase of minimization; expression stems from max log-likelihood
            logpriorpvarexpanded = - (loglamb.expand_as(sigsq) - sigsq.mul(psigsqlamb))

            logpriorvarsum = logpriorpvarexpanded.sum()
            logpriorvar = logpriorvarsum.div(Maug)
        else:
            logpriorvar = torch.zeros_like(KLD)

        # return (WeightedMSEloss + KLD); logpriorvar is zero - since inactive for the paper.
        loss = (logvarobjective + WeightedMSEloss + KLD + logpriorvar)

        # chekc if ARD prior is active, if so, add its contribution to the loss.
        if self.bard:
            ardcontrib = self.ardprior.getlogpiorARD()
            ardcontrib.mul_(float(Maug)/self.N)
            loss.add_(-ardcontrib[0])

        # Can be removed since it runs stable.
        lossnp = loss.data.cpu().numpy()
        if lossnp != lossnp:
            print('Error: Loss is NaN')
        return loss

    def trainepoch(self, epoch):
        '''Train one epoch.

        :param epoch: Epoch number
        '''

        self.vaemodel.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.data_loader):

            # Copy the data tensor for using more eps ~ p(eps) samples Eq. (7) in AEVB paper
            L = self.L
            dataaug = data.clone()
            for l in range(L - 1):
                dataaug = torch.cat((dataaug, data), 0)

            data = Variable(dataaug)
            # can be removed for newer PyTorch versions
            if self.gpu_mode:
                data = data.cuda()

            # Set gradient to zero.
            self.optimizer.zero_grad()

            # This is actually the only version used in the **paper**: self.autoencvarbayes is True.
            if self.autoencvarbayes:
                recon_batch, mu, logvar = self.vaemodel(data)
                recon_mu = recon_batch[0]
                recon_logvar = recon_batch[1]

                loss = self.loss_function_autoencvarbayes(recon_mu, recon_logvar, data, mu, logvar, x_dim=self.x_dim)
            else:
                ## Other options not considered
                print('Use self.autoencvarbayes = True, other options are not in accordance with the paper and thus not included here.')
                quit()

            # loss.backward(retain_graph=True)
            loss.backward()
            train_loss += loss.data[0]

            self.optimizer.step()

            # Just some monitoring options.
            log_interval = 20
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.data_loader.dataset),
                           100. * batch_idx / len(self.data_loader),
                           loss.data[0] / len(data)))

        self.train_hist['Total_loss'].append(loss.data[0] / len(data))

    def samplePosterior(self, npostsamples):
        ''' This function samples p(\theta|X)
        and produces a trajectory for every \theta_i by sampling the generative model p(x|\theta_i).
        :param npostsamples: Amount of posterior samples p(\theta|X) and thus trajectories stored.
        '''

        # Compute the Hessian based on the converged MAP estimate.
        pert_param_list, params_dec_copy, hess_list = self.getHessian()

        # Obtain p(\theta|X)
        list_normals = []
        id = 0
        for hess in hess_list:
            if not hess['fullcov']:
                var = hess['diaghessian'].reciprocal()
                scale = var.sqrt()
                mean = params_dec_copy[id]['params']
                list_normals.append(torch.distributions.Normal(mean.data, scale))
            else:
                if not hess['parisvector']:
                    shape = params_dec_copy[id]['params'].shape()
                    elements = shape[0]*shape[1]
                    mean = params_dec_copy[id]['params'].reshape(elements)
                    cov = hess['diaghessian'].inverse()
                    meannp = mean.data.cpu().numpy()
                    covnp = cov.cpu().numpy()
                    list_normals.append(MVN(meannp, covnp))
                else:
                    mean = params_dec_copy[id]['params']
                    cov = hess['diaghessian'].inverse()
                    meannp = mean.data.cpu().numpy()
                    covnp = cov.cpu().numpy()
                    list_normals.append(MVN(meannp, covnp))
            id += 1

        # Perform the actual sampling process.
        for i in range(npostsamples):
            id = 0
            for parlistitem in pert_param_list:
                hess = hess_list[id]
                if not (parlistitem['name'] == 'dec_logvar'):
                    if not hess['fullcov']:
                        sample = list_normals[id].sample()
                        parlistitem['params'].data.set_(sample)
                    else:
                        if not hess['parisvector']:
                            shape = params_dec_copy[id]['params'].shape()

                            samplenp = list_normals[id].sample()
                            samplevec = torch.from_numpy(samplenp)
                            sample = samplevec.resize(shape[0], shape[1])
                            parlistitem['params'].data.set_(sample)

                        else:
                            samplenp = list_normals[id].sample()
                            sample = torch.from_numpy(samplenp).float()
                            parlistitem['params'].data.set_(sample)

                id += 1

            # Make predictions given the current posterior sample \theta ~ p(\theta|X)
            # and store it in a file.
            self.gen_samples(n_samples=self.n_samples, postsampid=i)

    def getHessian(self):
        '''This function computes the diagonal entries of the Hessian matrix of the
        decoding NN parameters.
        :return pert_param_list, params_dec_copy, hess_list
        '''

        blayercov = self.uqoptions.blayercov
        self.uqoptions.bhessavailable = True

        bvartemp = self.vaemodel.bgetlogvar
        self.vaemodel.bgetlogvar = True

        data_loader_hessian_approx = DataLoader(TensorDatasetDataOnly(self.data_tensor), batch_size=self.N,
                                                batch_sampler=None,
                                                shuffle=False, **self.kwargsdatloader)

        for index, data in enumerate(data_loader_hessian_approx):
            data = Variable(data)
            if self.gpu_mode:
                data = data.cuda()

            # Resetting any gradient
            self.optimizer.zero_grad()

            if self.autoencvarbayes:
                recon_batch, mu, logvar = self.vaemodel(data)
                recon_mu = recon_batch[0]
                recon_logvar = recon_batch[1]

                # print(np.exp(recon_logvar.data.cpu().numpy()))
                loss = self.loss_function_autoencvarbayes(recon_mu, recon_logvar, data, mu, logvar, x_dim=self.x_dim)
            else:
                recon_batch, mu, logvar = self.vaemodel(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, x_dim=self.x_dim)

            # Calculate gradient
            loss.backward(retain_graph=True)

        # Identify the parameters the second derivative is required
        pert_param_list = []
        params_dec_copy = []
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                # UQ only for decoding network
                if 'dec_' in name:
                    # check if we want to uq bias uncertainty
                    if '.bias' in name and self.uqoptions.buqbias:
                        pert_param_list.append({'name': name, 'id': id, 'params': param})
                        pclone = param.clone()
                        params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                        print(name)  # , param.data
                    else:
                        pert_param_list.append({'name': name, 'id': id, 'params': param})
                        pclone = param.clone()
                        params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                        print(name)  # , param.data
                    id = id + 1

        hess_list = []

        # for group in param_groups:
        #    for p in group['params'][2*nLayerEnc:]:
        for parentry in pert_param_list:
            p = parentry['params']

            if parentry['name'] == 'dec_logvar':
                blayercov = False
            else:
                blayercov = self.uqoptions.blayercov

            grad_params = torch.autograd.grad(loss, p, create_graph=True)
            # hv = torch.autograd.grad(g000, p, create_graph=True)
            hess_params = torch.zeros_like(grad_params[0])

            # print(hess_params.size())
            bfullcov = False
            bparisvector = False
            dim_grad = grad_params[0].dim()
            if dim_grad == 1:
                bparisvector = True
                if blayercov:
                    bfullcov = True
                    size = grad_params[0].size()
                    elements = size[0]
                    unrolled_grad_params = grad_params[0]

                    hess_params = torch.autograd.Variable(torch.torch.zeros(elements, elements))
                    for i in range(elements):
                        # gives the row of the hessian
                        hessrow = torch.autograd.grad(unrolled_grad_params[i], p, retain_graph=True)[0].resize(elements)
                        hess_params[i, :] = hessrow
                else:
                    bfullcov = False
                    for i in range(grad_params[0].size(0)):
                        hess_params[i] = torch.autograd.grad(grad_params[0][i], p, retain_graph=True)[0][i]
            else:
                bparisvector = False
                #if blayercov:
                # TODO sparse storage of matrix required needed
                if False:
                    bfullcov = True
                    size = grad_params[0].size()
                    elements = size[0]*size[1]
                    unrolled_grad_params = grad_params[0].resize(elements)

                    hess_params = torch.autograd.Variable(torch.torch.zeros(elements, elements))
                    for i in range(elements):
                        # gives the row of the hessian
                        hessrow = torch.autograd.grad(unrolled_grad_params[i], p, retain_graph=True)[0].resize(elements)
                        hess_params[i, :] = hessrow
                else:
                    bfullcov = False
                    for i in range(grad_params[0].size(0)):
                        for j in range(grad_params[0].size(1)):
                            hess_params[i, j] = torch.autograd.grad(grad_params[0][i][j], p, retain_graph=True)[0][i, j]

            if not bfullcov:
                hess_params[hess_params < 0.5] = 10.*self.N #1.e5
            hess_list.append({'name': parentry['name'], 'id': parentry['id'], 'diaghessian': hess_params.data, 'fullcov': bfullcov, 'parisvector': bparisvector})
            np.savetxt(os.path.join(self.output_dir, parentry['name']+'_' + str(self.N) + '.txt'), hess_params.data.cpu().numpy())

        self.vaemodel.bgetlogvar = bvartemp

        return pert_param_list, params_dec_copy, hess_list


    def train(self):
        '''This functions is the main training function.
        '''

        self.train_hist = {}
        self.train_hist['Total_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['kl_qp'] = []

        # Store intermediate steps
        intermediate = False

        # If the model from previous training process is loaded, no need to train it.
        if not self.bloadmodel:
            print('Training started.')
            start_time = time.time()

            for epoch in range(1, self.epoch + 1):
                epoch_start_time = time.time()

                # Train single epoch
                self.trainepoch(epoch)

                # visualize intermediate latent space
                self.visLatentTraining(epoch)

                # Make intermediate predictions during the training procedure
                if intermediate:
                    # sample the prior
                    sample = Variable(torch.randn(64, self.z_dim))
                    # This can be removed in newer PyTorch versions.
                    if self.gpu_mode:
                        sample = sample.cuda()
                    # Decode the samples
                    sample = self.vaemodel.decode(sample).cpu()
                    # The step below does not affect the prediction, but in case of providing Cartesian
                    # coordinates as done in the **paper**.
                    sampleout = convert_given_representation(samples=sample, coordrep=self.angulardata)
                    np.savetxt(self.output_dir + '/samples' + self.predprefix + '.txt', sampleout)

                # store training time for one epoch
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            # Training has been completed, save the trained model.
            torch.save(self.vaemodel, self.output_dir + '/model.pth')

            # Store total training data
            self.train_hist['total_time'].append(time.time() - start_time)
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                            self.epoch,
                                                                            self.train_hist['total_time'][0]))
            print("Training finish!... save training results")
            print("Final KLD loss %.3f" % (self.train_hist['kl_qp'][-1]))

            # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
            #                          self.epoch)

            utils.loss_plot(self.train_hist,
                            self.output_dir,
                            self.model_name + self.predprefix, bvae=True)

        else:
            # Load the vae model which was trained before for making predictions
            self.vaemodel = torch.load(self.filemodel)

        # Count active parameters
        if self.bard:
            nonactiveparams = self.countzeroweights(paramlist=self.ardprior.paramlist, threshold=0.0001)
        else:
            nonactiveparams = self.countzeroweights(paramlist=self.getdecweightlist(), threshold=0.0001)
        nap = np.ones(1)*nonactiveparams
        np.savetxt(self.output_dir + '/nonactiveparams.txt', nap)

        # Model has been trained
        self.vaemodel.eval()

        # Generate samples for predictions (MAP estimate)
        self.gen_samples(self.n_samples)

        # Calculate the hessian if we require UQ
        if self.uqoptions.bdouq:
            self.samplePosterior(self.uqoptions.npostsamples)


    def vis_phipsilatent(self, path):
        '''Visualize the phi and psi landscape for the learned latent representation.
        This function creates a grid and corresponding to the CVs make predictions x.
        :param path: string, provide path where to save the prediction.
        :return:
        '''

        # Create grid with numpy
        x = np.linspace(-4., 4., 101)
        y = np.linspace(-4., 4., 101)
        X, Y = np.meshgrid(x, y)
        Xvec = X.flatten()
        Yvec = Y.flatten()

        # Convert numpy array to torch
        Xtorch = torch.from_numpy(Xvec).float()
        Xtorch.unsqueeze_(1)
        Ytorch = torch.from_numpy(Yvec).float()
        Ytorch.unsqueeze(1)
        # Not required for newer PyTorch
        if self.gpu_mode:
            samples_z = Variable(torch.cat((Xtorch, Ytorch), 1).cuda())
        else:
            samples_z = Variable(torch.cat((Xtorch, Ytorch), 1))

        # Decode the CVs z to x
        torchsamples = self.vaemodel.decode(samples_z)
        # Convert to numpy
        samples = torchsamples[0].data.cpu().numpy()

        # Convert the samples if they are in the angular format - this function
        # does not show any effect when using a Cartesian representation as in the **paper**.
        samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata)
        np.savetxt(path + '/samples_vis_phipsi' + self.predprefix + '.txt', samplesout)

    def countzeroweights(self, paramlist, threshold=0.0001):
        '''This function counts the inactive weights of the decoding neural network.
        :param paramlist: Torch parameter list of parameters which should be considered.
        :param threshold: Threshold considering parameter as inactive.
        :return: Amount of inactive parameters.
        '''
        counter = 0
        for paramitem in paramlist:
            par = paramitem['params']

            abspar = par.abs()
            abspardat = abspar.data
            smaller = abspardat[abspardat < threshold]
            if smaller.dim() > 0:
                counter = counter + int(smaller.size()[0])

        return counter

    def vis_realizations(self):
        '''This function visualizes a specific amount of realizations per prediction of the decoded data points.
        For showing that the variance of p(x|z) captures outer Hydrogen atoms as noise.
        :return:
        '''

        ic = 0
        # load the
        for batch_idx, data in enumerate(self.data_loader):
            if ic == 0:
                data_vis = data

        x = Variable(data_vis)
        n_samples = x.shape[0]
        # Encode the data set into latent space
        muenc, log = self.vaemodel.encode(x)

        # Decode to mean and variance of predictions
        self.vaemodel.bgetlogvar = True
        if self.gpu_mode:
            samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(muenc).gpu()
        else:
            samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(muenc)

        mu = samplesTorchmu.data.cpu().numpy()
        vartorch = samplesTorchlogvar.exp()
        var = vartorch.data.cpu().numpy()

        # Init storage for total amount of samples.
        nsamples_per_dp = 300
        nsamples_per_dp_tot = nsamples_per_dp + 1
        n_samples_tot = n_samples * nsamples_per_dp_tot
        samples_aevb = np.zeros([n_samples_tot, self.x_dim])

        for n in range(n_samples):
            samples_aevb[n * nsamples_per_dp_tot, :] = mu[n, :]

            samples_aevb[n * nsamples_per_dp_tot + 1: (n + 1) * nsamples_per_dp_tot, :] = np.random.multivariate_normal(
                mu[n, :], np.diag(var[n, :]), nsamples_per_dp)
        self.vaemodel.bgetlogvar = False

        # Only takes action when non Cartesian coordinates are used - not the case here.
        samplesoutaevb = convert_given_representation(samples=samples_aevb, coordrep=self.angulardata)
        np.savetxt(self.output_dir + '/samples_aevb' + self.predprefix + '_vis_mean_samples_' + '.txt', samplesoutaevb)

    def vis_latentpredictions(self, yb, ny, path):
        '''This function predicts atomistic configurations along a provided path in the CV space. Those z are mapped
        to full atomistic configurations. Currently just ny and path are required. Only visualization.
        :param ny: Number of points in y direction.
        :param path: String of path for storing the prediction.
        :return:
        '''

        # This allows to plot the latent representation and augment it with an indicator at the current position in the latent space
        # This is only valid for ALA2
        bVisualizeStar = False
        if bVisualizeStar == True:
            ny = 251
        bShowTraining = True
        if bShowTraining:
            xt = Variable(self.data_tensor)
        else:
            xt = None
        bRnd = False

        #y = torch.linspace(yb[0], yb[1], ny)

        # y coordinates
        y = torch.linspace(-4, 4, ny)
        #y1 = torch.linspace(-4, 0, ny*3)
        #y2 = torch.linspace(0, 4, ny*2)
        #nges =  5*ny
        #y = torch.cat((y1,y2))

        # x coordinates
        x = torch.zeros(ny)

        # For different plots experiments visualizing a path defined in the latent space.
        # This path was used for ALA15 path in Fig. 14 in **paper**.
        if False:
            x = torch.linspace(-1, 3, ny)
            y12 = -2.*x
            y13 = -2./3.*x
            y = torch.zeros_like(y12)
            y.copy_(y12)
            y[20:] = y13[20:]

        # Check if gpu mode is active - drop for newer PyTorch.
        if self.gpu_mode:
            y = y.cuda()
            x = x.cuda()

        # summarize x and y in torch variable
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        samples_z = Variable(torch.cat((y, x), 1))

        # This is for showing a marker at the current position in the latent space.
        # E.g. for visualizing atomistic configurations for given CVs.
        if bVisualizeStar:
            for i in range(0, ny):
                xnp_curr = samples_z[i, 0].data.numpy()
                ynp_curr = samples_z[i, 1].data.numpy()
                if i==0:
                    n = self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir,
                                            iter=i, x_curr=xnp_curr, y_curr=ynp_curr, nprov=False, x_train=xt)
                else:
                    n = self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir,
                                            iter=i, x_curr=xnp_curr, y_curr=ynp_curr, nprov=True, normaltemp=n, x_train=xt)

        torchsamples = self.vaemodel.decode(samples_z)
        samples = torchsamples[0].data.cpu().numpy()

        # Convert the samples if they are in the angular format - not relevant for this **paper**
        samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata)
        np.savetxt(path + '/samples_vis_latent' + self.predprefix + '.txt', samplesout)

    def gen_samples(self, n_samples=4000, postsampid=None):
        '''This function samples predicted atomistic configurations involving the
        generative part in p(z) and p(x|z) while the variance is considererd.
        :param n_samples: Amount of required samples.
        :param postsampid: This parameter is automatically set in case UQ tasks are performed.
        :return:
        '''
        if postsampid == None and 'ala_15' not in self.dataset:
            self.postprocessing(n_samples=4000, postsampid=None)
        elif postsampid == None and 'ala_15' in self.dataset:
            self.postprocessingALA15(postsampid=None)

        # Saving samples with postfix - important for UQ
        if postsampid == None:
            postsamplepostfix = ''
        else:
            postsamplepostfix = '_postS_' + str(postsampid)

        # Convert the samples if GPU mode is active - deprecated for more
        # recent PyTorch versions
        if self.gpu_mode:
            sample_z_ = Variable(
                torch.randn((n_samples, self.z_dim)).cuda(), volatile=True)
            z_init = Variable(
                torch.randn((1, self.z_dim)).cuda(), volatile=True)
        else:
            sample_z_ = Variable(torch.randn((n_samples, self.z_dim)),
                                 volatile=True)
            z_init = Variable(torch.randn((1, self.z_dim)),
                              volatile=True)

        # Utilize pseudo gibbs algorithm as provided in Mattei, 2018. This algorithm corrects for the approximate
        # posterior
        if self.exactlikeli:
            self.vaemodel.bgetlogvar = True
            initmu, initlogvar = self.vaemodel.decode(z_init)
            initstd = initlogvar.mul(0.5).exp_()
            pinit = torch.distributions.Normal(initmu, initstd)
            x_init = pinit.sample()

            pgibs = PseudoGibbs(x_init, z_init, self.vaemodel)
            samples_aevb_gibbs = pgibs.sample(n_samples * self.n_samples_per_mu)
            samplesnp_aevb_gibbs = samples_aevb_gibbs.data.cpu().numpy()
            if self.autoencvarbayes:
                samplesoutaevbgibbs = convert_given_representation(samples=samplesnp_aevb_gibbs,
                                                                   coordrep=self.angulardata)
                np.savetxt(self.output_dir + '/samples_aevb_gibbs' + self.predprefix + '_' + self.postfix_setting + postsamplepostfix + '.txt',
                           samplesoutaevbgibbs)
                del samplesoutaevbgibbs
            self.vaemodel.bgetlogvar = False

        # Prediction for Variational Autoencoder. I.e. sample p(z), and project to x directly with \mu(z).
        # No probabilistic model employed for the mapping in this case.
        # This does not apply for the **paper**
        if not self.autoencvarbayes or self.n_samples_per_mu == 0:
            if self.gpu_mode:
                samplesTorchmu = self.vaemodel.decode(sample_z_)#.gpu()
            else:
                samplesTorchmu = self.vaemodel.decode(sample_z_)  # .cpu()

            # Samples are the means directly
            samples = samplesTorchmu.data.cpu().numpy()
            # convert the samples if they are in the angular format
            samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata)
            np.savetxt(self.output_dir + '/samples' + self.predprefix + '_' + self.postfix_setting + postsamplepostfix + '.txt', samplesout)

        # Prediction for Auto-Encoding Variational Bayes. I.e. sample p(z),
        # given those SAMPLE p(x|z) = N(\mu, \sigma^2). This corresponds to the actual AEVB algorithm and is Bayesian.
        else:
            # Provide at least one sample per mean prediction.
            if self.n_samples_per_mu == 0:
                aevb_samples_per_mu = 1
            else:
                aevb_samples_per_mu = self.n_samples_per_mu

            # In case of AEVB one requires the mean and variance of the predictive model p(x|z). Enable this here.
            self.vaemodel.bgetlogvar = True # This was incorrect in the original VAE implementation of Kingma,
            # one needs to account for the variance in p(x|z) in the predictions.
            if self.gpu_mode:
                samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(sample_z_) #.gpu()
            else:
                samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(sample_z_)  # .cpu()

            # TODO do not convert here to numpy for sampling from gaussian but use instead the torch implementation
            # of the Normal distribution. Those should be inculded in PyTorch version >=0.4.1
            # but not in 0.3.0
            mu = samplesTorchmu.data.cpu().numpy()
            vartorch = samplesTorchlogvar.exp()
            var = vartorch.data.cpu().numpy()

            # Initialize storage of samples.
            n_samples_tot = n_samples * aevb_samples_per_mu
            samples_aevb = np.zeros([n_samples_tot, self.x_dim])

            # sample the p(x|z) for different CVs z and its corresponding \mu(z), \sigma(z).
            for n in range(n_samples):
                samples_aevb[n * aevb_samples_per_mu:(n + 1) * aevb_samples_per_mu, :] = np.random.multivariate_normal(
                    mu[n, :], np.diag(var[n, :]), aevb_samples_per_mu)
            self.vaemodel.bgetlogvar = False

            # Store the predictions
            # The following does not has any effect for Cartesian coordinates.
            samplesoutaevb = convert_given_representation(samples=samples_aevb, coordrep=self.angulardata)
            np.savetxt(self.output_dir + '/samples_aevb' + self.predprefix + '_' + self.postfix_setting + postsamplepostfix + '.txt',
                       samplesoutaevb)

        # On local machine only: Estimate properties directly.
        if postsamplepostfix == '':
            samples_name = 'samples_aevb' + self.predprefix + '_' + self.postfix_setting + postsamplepostfix
            estimateProperties(samples_name=samples_name, cluster=self.bClusterND, datasetN=self.N, pathofsamples=self.output_dir, postS=self.uqoptions.npostsamples, peptide=self.name_peptide)

    def visLatentTraining(self, epoch):
        '''This visualizes the training process
        :param epoch: Epoch just for labelling required.
        :return:
        '''
        # Visualize intermediate steps, i.e. the latent embedding and the ELBO
        if self.bvislatent_training and not epoch % 200:
            if hasattr(self.vaemodel, 'plotlatentrep'):
                if self.gpu_mode:
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide)
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_dat_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide,  x_train=Variable(self.data_tensor).cuda())
                else:
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide)
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_dat_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide, x_train=Variable(self.data_tensor))

                utils.loss_plot(self.train_hist,
                                self.output_dir,
                                self.model_name + self.predprefix + '_' + str(epoch), bvae=True, bintermediate=True)

    def postprocessingALA15(self, postsampid=None):
        '''This function provides predictions given the trained model. In the case of \dim(z) = 2,
        further visualizations are issued automatically.
        :param n_samples: Amount of requred samples of z \sim p(z)
        :param postsampid: Do no specify this, it is just required internally for sampling the posterior of the decoding
        parametrization.
        :return:
        '''
        if hasattr(self.vaemodel, 'plotlatentrep') and postsampid == None:
            if self.gpu_mode:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                            path=self.output_dir, data_dir=self.data_dir, peptide=self.name_peptide)
            else:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir, x_train=Variable(self.data_tensor), data_dir=self.data_dir, peptide=self.name_peptide)

        # Visualize latent representation if and only if z_dim = 2
        if True and self.z_dim == 2 and postsampid == None:
            yborder = np.array([4., -4.])
            # Create predictions for the latent representation
            self.vis_latentpredictions(yb=yborder, ny=81, path=self.output_dir)
            # Visualize the phi-psi landscape given the CVs - so for not possible
            # for ALA15
            #self.vis_phipsilatent(path=self.output_dir)


    def postprocessing(self, n_samples=4000, postsampid=None):
        '''This function provides predictions given the trained model. In the case of \dim(z) = 2,
        further visualizations are issued automatically.
        :param n_samples: Amount of required samples of z \sim p(z)
        :param postsampid: Do no specify this, it is just required internally for sampling the posterior of the decoding
        parametrization.
        :return:
        '''

        # Visualize latent representation if and only if z_dim = 2
        if True and self.z_dim == 2 and postsampid == None and self.name_peptide == 'ala_2':
            yborder = np.array([4., -4.])
            # create predictions for the latent representation
            self.vis_latentpredictions(yb=yborder, ny=81, path=self.output_dir)
            # Visualize the phi-psi landscape given the CVs
            self.vis_phipsilatent(path=self.output_dir)

        # Visualize mapping between the different layers
        # TODO Potentially also for VAE not only GAN.
        if hasattr(self.vaemodel, 'plotdecoder') and postsampid == None and self.name_peptide == 'ala_2':
            self.vaemodel.plotdecoder(n_samples=500, z_dim=self.z_dim)
        else:
            print('No visualization for decoder available.')

        # Visualize the mapping from input to latent space
        # Not available for VAE.
        if False and hasattr(self.vaemodel, 'plotencoder') and postsampid == None:
            data_loader_visualization = DataLoader(TensorDatasetDataOnly(self.data_tensor),
                                                   batch_size=1527,
                                                   shuffle=False, **self.kwargsdatloader)

            for index, data in enumerate(data_loader_visualization):
                data = Variable(data)
                if self.gpu_mode:
                    data = data.cuda()
                self.vaemodel.plotencoder(x=data, z_dim=self.z_dim, strindex=str(index))
        elif postsampid == None:
            print('No visualization for encoder available.')

        if hasattr(self.vaemodel, 'plotlatentrep') and postsampid == None:
            if self.gpu_mode:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                            path=self.output_dir)
            else:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir, x_train=Variable(self.data_tensor))

            # store the variances of the test dataset
            varout = self.vaemodel.get_encoding_decoding_variance(x=Variable(self.data_tensor_vis_1527))
            temp_norm = np.append(varout['norm_enc'], varout['norm_dec'])
            np.savetxt(self.output_dir + '/normvar_enc_dec.txt', temp_norm)

        # visualize realizations along the z_1 or z_2 axis to show that the variance captures
        if False:
            self.vis_realizations()

    def selectdataset(self):
        '''This function is a helper function for loading the desired dataset.
        :return:
        '''

        # Initialize peptide name - will be adjusted according dataset. This is the default value.
        self.name_peptide = 'ala_2'

        # Data path
        if self.bClusterND:
            data_dir = '---'
        else:
            scwd = os.getcwd()
            data_dir = os.path.join(scwd, 'data_peptide')
            #data_dir = '---'
        self.data_dir = data_dir

        # Is using angular data set, add postfix of the data
        # This is not scope of the work **paper_link** and therefore not content of the code.
        if self.angulardata == 'ang':
            angpostfix = '_ang'
        elif self.angulardata == 'ang_augmented':
            angpostfix = '_ang_augmented'
        elif self.angulardata == 'ang_auggrouped':
            angpostfix = '_ang_auggrouped'
        else:
            angpostfix = ''

        if self.dataset == 'm_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_1527' + angpostfix + '.txt').T)
            self.N = 1527
        elif self.dataset == 'b1b2_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_b1b2_1527' + angpostfix + '.txt').T)
        elif self.dataset == 'ab1_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_ab1_1527' + angpostfix + '.txt').T)
        elif self.dataset == 'ab2_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_ab2_1527' + angpostfix + '.txt').T)
        elif self.dataset == 'm_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_4004' + angpostfix + '.txt').T)
            self.N = 4004
        elif self.dataset == 'm_102':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_102' + angpostfix + '.txt').T)
            self.N = 102
        elif self.dataset == 'm_262':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_262' + angpostfix + '.txt').T)
            self.N = 262
        elif self.dataset == 'm_52':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_52' + angpostfix + '.txt').T)
            self.N = 52
        elif self.dataset == 'ma_10':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_10' + angpostfix + '.txt').T)
            self.N = 10
        elif self.dataset == 'ma_50':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_50' + angpostfix + '.txt').T)
            self.N = 50
        elif self.dataset == 'ma_100':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_100' + angpostfix + '.txt').T)
            self.N = 100
        elif self.dataset == 'ma_200':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_200' + angpostfix + '.txt').T)
            self.N = 200
        elif self.dataset == 'ma_500':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_500' + angpostfix + '.txt').T)
            self.N = 500
        elif self.dataset == 'ma_1000':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_1000' + angpostfix + '.txt').T)
            self.N = 1000
        elif self.dataset == 'ma_1500':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_1500' + angpostfix + '.txt').T)
            self.N = 1500
        elif self.dataset == 'ma_4000':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_4000' + angpostfix + '.txt').T)
            self.N = 4000
        elif self.dataset == 'ma_13334':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_13334' + angpostfix + '.txt').T)
            self.N = 13334
        elif self.dataset == 'b1b2_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_b1b2_4004' + angpostfix + '.txt').T)
        elif self.dataset == 'ab1_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_ab1_4004' + angpostfix + '.txt').T)
        elif self.dataset == 'ab2_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_ab2_4004' + angpostfix + '.txt').T)
        elif self.dataset == 'samples':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_samples' + angpostfix + '.txt').T)
        elif self.dataset == 'm_526':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_526' + angpostfix + '.txt').T)
            self.N = 526
        elif self.dataset == 'm_1001':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_1001' + angpostfix + '.txt').T)
            self.N = 1001
        elif self.dataset == 'm_10437':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_mixed_10537' + angpostfix + '.txt').T)
        elif self.dataset == 'a_1000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_alpha_10000_sub_1000' + angpostfix + '.txt').T)
            foldername = 'separate_1000'
            predictprefix = '_a'
        elif self.dataset == 'a_10000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_alpha_10000' + angpostfix + '.txt').T)
            foldername = 'separate_10000'
            predictprefix = '_a'
        elif self.dataset == 'b1_1000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_beta1_10000_sub_1000' + angpostfix + '.txt').T)
            foldername = 'separate_1000'
            predictprefix = '_b1'
        elif self.dataset == 'b1_10000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_beta1_10000' + angpostfix + '.txt').T)
            foldername = 'separate_10000'
            predictprefix = '_b1'
        elif self.dataset == 'b2_1000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_beta2_10000_sub_1000' + angpostfix + '.txt').T)
            foldername = 'separate_1000'
            predictprefix = '_b2'
        elif self.dataset == 'b2_10000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_beta2_10000' + angpostfix + '.txt').T)
            foldername = 'separate_10000'
            predictprefix = '_b2'
        elif self.dataset == 'a_500':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_alpha_10000_sub_500' + angpostfix + '.txt').T)
            foldername = 'separate_500'
            predictprefix = '_a'
        elif self.dataset == 'b1_500':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_beta1_10000_sub_500' + angpostfix + '.txt').T)
            foldername = 'separate_500'
            predictprefix = '_b1'
        elif self.dataset == 'b2_500':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-2/dataset_beta2_10000_sub_500' + angpostfix + '.txt').T)
            foldername = 'separate_500'
            predictprefix = '_b2'
        elif self.dataset == 'm_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15' + angpostfix + '.txt').T)
            self.N = 2000
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_100_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_100' + angpostfix + '.txt').T)
            self.N = 100
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_200_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_200' + angpostfix + '.txt').T)
            self.N = 200
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_300_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_300' + angpostfix + '.txt').T)
            self.N = 300
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_500_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_500' + angpostfix + '.txt').T)
            self.N = 500
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_1500_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_1500' + angpostfix + '.txt').T)
            self.N = 1500
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_3000_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_3000' + angpostfix + '.txt').T)
            self.N = 3000
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_5000_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_5000' + angpostfix + '.txt').T)
            self.N = 5000
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_10000_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_10000' + angpostfix + '.txt').T)
            self.N = 10000
            self.name_peptide = 'ala_15'

        print('dataset size: {}'.format(data_tensor.size()))

        self.kwargsdatloader = {'num_workers': 2,
                                'pin_memory': True} if torch.cuda.is_available() else {}

        self.data_tensor = data_tensor
        self.data_loader = DataLoader(TensorDatasetDataOnly(data_tensor),
                                      batch_size=self.batch_size,
                                      shuffle=True, **self.kwargsdatloader)

        # for visualization purposes
        if self.dataset == 'm_1527':
            self.data_tensor_vis_1527 = self.data_tensor
        elif 'ala_15' not in self.dataset:
            self.data_tensor_vis_1527 = torch.Tensor(
                np.loadtxt(data_dir + '/ala-2/dataset_mixed_1527' + angpostfix + '.txt').T)
        elif 'ala_15' in self.dataset:
            self.data_tensor_vis_1527 = torch.Tensor(
                np.loadtxt(data_dir + '/ala-15/dataset_ala_15_1500' + angpostfix + '.txt').T)
