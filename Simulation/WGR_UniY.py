import sys
import os
current_dir = os.getcwd()  #use to import the defined functions
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)  

"""
incase the above code does not work, you can use the absolute path instead
sys.path.append(r".\")
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from utils.basic_utils import setup_seed
from data.SimulationData import DataGenerator 
from utils.training_utils import train_WGR_fnn
from utils.evaluation_utils import L1L2_MSE_mean_sd_G, MSE_quantile_G_uniY
from models.generator import generator_fnn
from models.discriminator import discriminator_fnn

import argparse


if 'ipykernel_launcher.py' in sys.argv[0]:  #if not work in jupyter, you can delete this part
    import sys
    sys.argv = [sys.argv[0]] 


parser = argparse.ArgumentParser(description='Implementation of WGR for M1')

parser.add_argument('--Xdim', default=100, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=1, type=int, help='dimensionality of Y')
parser.add_argument('--model', default='M2', type=str, help='model')

parser.add_argument('--noise_dim', default=5, type=int, help='dimensionality of noise vector')
parser.add_argument('--noise_dist', default='gaussian', type=str, help='distribution of noise vector')

parser.add_argument('--train', default=5000, type=int, help='size of train dataset')
parser.add_argument('--val', default=1000, type=int, help='size of validation dataset')
parser.add_argument('--test', default=1000, type=int, help='size of test dataset')

parser.add_argument('--train_batch', default=128, type=int, metavar='BS', help='batch size while training')
parser.add_argument('--val_batch', default=100, type=int, metavar='BS', help='batch size while validation')
parser.add_argument('--test_batch', default=100, type=int, metavar='BS', help='batch size while testing')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train')
parser.add_argument('--reps', default=100, type=int, help='number of replications')

args = parser.parse_args()

print(args)

# Set seed 
setup_seed(1234)
reps=(args.reps,)
seed = torch.randint(0, 10000, reps)  

def main():
    test_G_mean_sd = []  # Initialize as list
    test_G_quantile = []  # Initialize as list

    for k in range(args.reps):
        print('============================ REPLICATION ==============================')
        print(k, seed[k])

        setup_seed(seed[k].detach().numpy().item())

        # Generate data 
        data_gen = DataGenerator(args)
        DATA = data_gen.generate_data(args.model)

        train_X, train_Y = DATA['train_X'], DATA['train_Y']
        val_X, val_Y = DATA['val_X'], DATA['val_Y']
        test_X, test_Y = DATA['test_X'], DATA['test_Y']

        # Create TensorDatasets and initialize a DataLoaders
        train_dataset = TensorDataset( train_X.float(), train_Y.float() )
        loader_train = DataLoader(train_dataset , batch_size=args.train_batch, shuffle=True)

        val_dataset = TensorDataset( val_X.float(), val_Y.float() )
        loader_val = DataLoader(val_dataset , batch_size=args.val_batch, shuffle=True)

        test_dataset = TensorDataset( test_X.float(), test_Y.float() )
        loader_test  = DataLoader(test_dataset , batch_size=args.test_batch, shuffle=True)

        global G_net, D_net, trained_G, trained_D

        # Define generator network and discriminator network
        G_net = generator_fnn(Xdim=args.Xdim, Ydim=args.Ydim, noise_dim=args.noise_dim, hidden_dims = [64, 32])
        D_net = discriminator_fnn(input_dim=args.Xdim+args.Ydim, hidden_dims = [64, 32])
        
        # Initialize RMSprop optimizers
        D_solver = optim.Adam(D_net.parameters(), lr=0.001, betas=(0.9, 0.999))
        G_solver = optim.Adam(G_net.parameters(), lr=0.001, betas=(0.9, 0.999))  

        # Training
        trained_G, trained_D = train_WGR_fnn(D=D_net, G=G_net, D_solver=D_solver, G_solver=G_solver, loader_train = loader_train, 
                                             loader_val=loader_val, noise_dim=args.noise_dim, Xdim=args.Xdim, Ydim=args.Ydim, 
                                             batch_size=args.train_batch, save_path='./', model_type=args.model, device='cpu', num_epochs=300)
        
        # To calcualte the value of criterion of selecting m
        mean_sd_result = L1L2_MSE_mean_sd_G(G = trained_G,  test_size = args.train, noise_dim=args.noise_dim, Xdim=args.Xdim, 
                                    batch_size=100,  model_type=args.model, loader_dataset = loader_test )
        
        # Calculate the MSE of conditional quantiles at different levels.
        test_G_quantile = MSE_quantile_G_uniY(G = trained_G, loader_dataset = loader_test , noise_dim=args.noise_dim, Xdim=args.Xdim,
                                      test_size = args.test,  batch_size=args.test_batch, model_type=args.model)
        
        test_G_mean_sd.append(np.array(mean_sd_result))
        test_G_quantile.append(quantile_result.copy() if isinstance(quantile_result, np.ndarray) else np.array(list(quantile_result)))
        
    print("L1 error, L2 error, MSE(mean), MSE(sd):", test_G_mean_sd)
    print("MSE(quantile) at level {0.05, 0.25, 0.50, 0.75, 0.95}:", test_G_quantile)

    #saving the results as csv
    test_G_mean_sd_csv = pd.DataFrame(np.array(test_G_mean_sd) )
    test_G_quantile_csv = pd.DataFrame(np.array(test_G_quantile) )

    test_G_mean_sd_csv.to_csv("./test_G_mean_sd_"+str(args.model)+"_d"+str(args.Xdim)+".csv")
    test_G_quantile_csv.to_csv("./test_G_quantile_"+str(args.model)+"_d"+str(args.Xdim)+".csv")


#run
main()
