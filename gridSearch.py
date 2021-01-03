import pandas as pd
import numpy as np
from itertools import permutations
from itertools import product
from MLP import MLP, cross_entropy, to_categorical
from scipy.special import softmax
from IPython.display import clear_output


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
    x = [ dict( zip( params[0], v) ) for v in product( *params[0].values() ) ]
    return x

def read_dataset( filename='monks-1.train', id_col=8 ):
    # leggo dataset
    data = pd.read_csv( filename, sep=' ', index_col=id_col ) # read dataset
    data = data.drop(columns=data.columns[0]) # drop index column
    data = data.to_numpy() 
    return data

def get_fold(folds,i):
    train_set = np.concatenate(folds[:i] + folds[i+1:])
    val_set = folds[i]  
    return train_set, val_set

def k_fold_CV(data, params, k=4, n_init=10, max_epochs=300, tresh=.1, measure_interval=10, xy=None):
    folds = np.array_split(data, k)  #split data into 4 folds
    configurations = get_configurations(params) # given the params grid, get all the possible configurations
    best_error = np.inf # error given by configuraion that gives the best error atm
    best_conf = None #  configuraion that gives the best error atm
    # we try all the configurations, for each of them we compute error of n_init MLP 
    # initializations on all of the folds.
    print(f'testing {len(configurations)} configurations')
    for idx_c,c in enumerate(configurations):
        best_error_init = np.inf  # best error given by some initialization of an MLP with this configuration
        print(f'testing configuration {c}, {idx_c}/{len(configurations)}')
        for n in range(n_init):
            print(f'initialization {n}')
            n = MLP(Nh = c['hidden_units'], Nu = c['Nu'], Ny= c['Ny'], f = c['activation'], f_out=c['f_out'], w_range=c['weights_range'], w_scale=c['weights_scale'], loss=c['loss'], error=c['error']) # inizializzo matrice pesi
            init_w = np.copy( n.w ) # salvo una copia dei pesi iniziali
            val_error = [None]*k # we save validation error for each fold
            # train and test the network on each fold
            for i in range(k):
                print(f'fold {i}')
                train_set, val_set = get_fold(folds,i) # get data form i-th fold
                tr_x, tr_y = xy(train_set)
                val_x, val_y = xy(val_set)
                n.w = init_w # reset weights to initial values
                n.train( tr_x, tr_y,  c['learning_rate'], a= c['alpha'], l=c['lambda'], max_epochs=max_epochs, tresh=tresh, epoch_f=n.epoch_batch_BP, shuffle_data=False, measure_interval=measure_interval ) # train the network 
                val_error[i] = n.test(val_x, val_y) # test network on this fold and save the resulting error
            val_error = np.mean(val_error) # validation medio sui k fold
            if val_error < best_error_init:
                best_error_init = val_error
        # best_error_init errore minimo che ho trovato con questa configurazione
        if best_error_init < best_error:
            best_error = best_error_init
            best_conf = c
        clear_output(wait=True)
    return best_conf, best_error
        
