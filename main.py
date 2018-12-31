from __future__ import print_function

import argparse
import os

from VAE_up import VAEpeptide


"""get choices for implemented dataset"""
def get_choices():

    choice_list = []
    #['mnist', 'fashion-mnist', 'celebA', 'samples', 'm_526', 'm_10437',
    #               'a_500', 'b1_500', 'b2_500', 'a_1000', 'b1_1000', 'b2_1000',
    #               'a_10000', 'b1_10000', 'b2_10000', 'var_gauss', 'ala_2']

    choice_list.append('ma_50')
    choice_list.append('ma_100')
    choice_list.append('ma_200')
    choice_list.append('ma_500')
    #choice_list.append('ma_1500')
    #choice_list.append('ma_4000')
    #choice_list.append('ma_13334')
    #choice_list.append('m_ala_15')
    #choice_list.append('m_100_ala_15')
    #choice_list.append('m_200_ala_15')
    choice_list.append('m_300_ala_15')
    #choice_list.append('m_500_ala_15')
    choice_list.append('m_1500_ala_15')
    choice_list.append('m_3000_ala_15')
    choice_list.append('m_5000_ala_15')
    #choice_list.append('m_10000_ala_15')
    #choice_list.append('m_20000_ala_15')


    #for strN in ['1527', '4004']:
    #    choice_list.append('m_'+strN)
    #    choice_list.append('b1b2_' + strN)
    #    choice_list.append('ab1_' + strN)
    #    choice_list.append('ab2_' + strN)
    return choice_list

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of Predictive Collective Variable Discovery with Deep Bayesian Models"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--mod_type', type=str, default='VAE',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP',
                                 'DRAGAN', 'LSGAN', 'WGAN_peptide', 'GAN_peptide', 'VAE', 'VARjoint'],
                        help='The type of model to be trained.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=get_choices(),
                        help='The name of dataset. For PCVs, ma_* for ALA2 and m_*_ala_15 for ALA15 works.')
    parser.add_argument('--epoch', type=int, default=2000, help='The number of epochs to run.')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0001)
    parser.add_argument('--lrD', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=int, default=0)
    parser.add_argument('--clusterND', type=int, default=0, help='Irrelevant option for pubilc.')
    parser.add_argument('--outPostFix', type=str, default='')
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--clipping', type=float, default=0.01)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--samples_pred', type=int, default=4000)
    parser.add_argument('--useangulardat', type=str, default='no',
                        choices=['no', 'ang', 'ang_augmented', 'ang_auggrouped'], help='Irrelevant for PCVs since not applied.')
    parser.add_argument('--seed', type=int, default=3251,
                    help='random seed (default: 0), 0 for no seed.')
    parser.add_argument('--AEVB', type=int, default=1,
                    help='Use Auto-Encoding Variational Bayes. If not, formulation relates to adversarial learning.')
    parser.add_argument('--Z', type=int, default=1,
                        help='Relevant for variational approach. Amount of samples from p(z).')
    parser.add_argument('--L', type=int, default=1,
                    help='Samples from eps ~ p(eps) for VAE.')
    parser.add_argument('--samples_per_mean', type=int, default=3,
                    help='Amount of predictive samples for p(x|z) = N(mu(z), sigma(z)). If 0, mean prediction is used: mu(z).')
    parser.add_argument('--npostS', type=int, default=0, help='Amount of posterior samples.')
    parser.add_argument('--uqbias', type=int, default=1, help='Quantify uncertainty of bias terms in network.')
    parser.add_argument('--exppriorvar', type=float, default=0., help='lambda of exp(-lambda theta. If 0, no prior employed')
    parser.add_argument('--sharedlogvar', type=int, default=1,
                        help='Sharing the logvariance instead of cosidering a variance dpendent on the decoding network.')
    parser.add_argument('--sharedencoderlogvar', type=int, default=0,
                        help='Sharing the logvariance of the ENCODER, instead of cosidering a variance dpendent on the encoding network. This only applies for VARJ not VAE.')
    parser.add_argument('--loadtrainedmodel', type=str, default='',
                        help='Provide the path including file of an already trained model for doing predictions.')
    parser.add_argument('--ard', type=float, default=0., help='Value of a0 for ARD prior. If 0. then no ARD prior is applyed.')
    parser.add_argument('--exactlikeli', type=int, default=0, help='Perform leveraging the likelihood.')
    parser.add_argument('--outputfreq', type=int, default=500, help='Output frequency during the optimization process.')
    parser.add_argument('--x_dim', type=int, default=2, help='Just for variational approach - not for PCVs since not applied. Test to predict gaussian of dim x_dim.')
    parser.add_argument('--assignrandW', type=int, default=0,
                        help='Just for variational approach. Assign uniformly random variables to reference W.')
    parser.add_argument('--freeMemory', type=int, default=0,
                        help='Just for variational approach. Free memory during estimation of the loss function.')
    parser.add_argument('--stepSched', type=int, default=1, help='Use step scheduler module druing optimization. No effect for PCVs since not applied.')
    parser.add_argument('--betaVAE', type=float, default=1., help='Beta value for enforcing beta * KL(q(z|x) || p(z)). See https://openreview.net/pdf?id=Sy2fzU9gl. No effect for PCVs since not applied.')
    parser.add_argument('--separateLearningRate', type=int, default=0, help='This applies to separate learning rates between NN parameters and the parameters for the variances. Applies only if en- or decoding variance is modeled as parameter. No effect for PCVs since not applied.')
    parser.add_argument('--redDescription', type=int, default=0, help='Only relevant for reverse var. approach. This removes 6 DOFs from x to implicitly remove the rigid body motion. No effect for PCVs since not applied.')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # declare instance for model
    if args.mod_type == 'GAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'CGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'ACGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'infoGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'EBGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'WGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'WGAN_peptide':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'GAN_peptide':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'WGAN_GP':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'DRAGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'LSGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'BEGAN':
        print('Not content of Predictive CV Discovery.')
    elif args.mod_type == 'VAE':
        modt = VAEpeptide(args)
    elif args.mod_type == 'VARjoint':
        print('Not content of Predictive CV Discovery.')
    else:
        raise Exception("[!] There is no option for " + args.mod_type)

        # launch the graph in a session
    modt.train()
    print(" [*] Training finished!")

    # visualize learned generator
    # gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
