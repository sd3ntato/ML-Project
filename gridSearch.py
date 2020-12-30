import pandas as pd
import numpy as np
from itertools import permutations
from itertools import product
from MLP import MLP, cross_entropy
from scipy.special import softmax
from copy import deepcopy

Nh_monk=[10,25,50]
Nls_monk=1

Nh_blind = [8,12,20]
Nls_blind = [2, 3]

def combinations(Nh,Nls):
    x=[]
    for Nl in Nls :
        x.append(list(permutations(Nh_monk,Nls_monk)))
    
    combinedList=[*x[0],*x[1]]
    return combinedList

def get_configurations(params):
    # w_init = [i["weights_init"] for i in params]
    # w_scale = [i["weights_scale"] for i in params]
    x = [ dict( zip( params[0], v) ) for v in product( *params[0].values() ) ]
    return x


def read_and_split_dataset( k, filename='monks-1.train', id_col=8 ):
    # leggo dataset
    data = pd.read_csv( filename, sep=' ', index_col=id_col ) # read dataset
    data = data.drop(columns=data.columns[0]) # drop index column
    data = data.to_numpy() 
    folds = np.array_split(data, k) # splits dataset into k folds
    """
    deve diventare one-hot
    data[data==0] = data[data==0]-1
    print("AAAAAAAAA")
    print(data)
    print("AAAAAAAAA")
    """
    return folds

def get_fold(folds,i):
    train_set = np.concatenate(folds[:i] + folds[i+1:])
    val_set=folds[i]
    tr_x,tr_y = train_set[:,1:] , train_set[:,:1] # to_categorical(tr_y).reshape(-1,2,1)
    val_x,val_y = val_set[:,1:] , val_set[:,:1]    
    return tr_x, tr_y, val_x, val_y


def k_fold_CV(k, n_init, params, max_epochs, tresh):
    folds = read_and_split_dataset(k=4) # read monks-1 train and split it into 4 folds
    configurations = get_configurations(params) # given the params grid, get all the possible configurations
    best_error = np.inf # error given by configuraion that gives the best error atm
    best_conf = None #  configuraion that gives the best error atm
    # we try all the configurations, for each of them we compute error of n_init MLP 
    # initializations on all of the folds.
    print(f'testing {len(configurations)} configurations')
    for c in configurations:
        Nh = c['hidden_units']; a = c['alpha']; eta = c['learning_rate']; l = c['lambda']; f = c['activation']; w_r = c['weights_range']; w_s = c['weights_scale']
        for _ in range(n_init):
            best_error_init = np.inf # best error given by some initialization of an MLP with this configuration
            n = MLP(Nh = Nh, Nu = 6, Ny=2, f = f, f_out=softmax, w_range=w_r, w_scale=w_s, loss=cross_entropy, error=cross_entropy) # inizializzo matrice pesi
            init_w = deepcopy( n.w ) # salvo una copia dei pesi iniziali
            val_error = [None]*k # we save validation error for each fold
            # train and test the network on each fold
            for i in range(k):
                tr_x, tr_y, val_x, val_y = get_fold(folds,i) # get data form i-th fold
                n.w = init_w # reset weights to initial values
                n.train( tr_x, tr_y, eta, a=a, l=l, val_x=val_x, val_y=val_y, max_epochs=max_epochs, tresh=tresh, epoch_f=n.epoch_batch_BP, shuffle_data=False, measure_interval=10 ) # train the network 
                val_error[i] = n.test(val_x, val_y) # test network on this fold and save the resulting error
            val_error = np.mean(val_error) # validation medio sui k fold
            if val_error < best_error_init:
                best_error_init = val_error
        # best_error_init errore minimo che ho trovato con questa configurazione
        if best_error_init < best_error:
            best_error = best_error_init
            best_conf = c
    return best_conf, best_conf
        
