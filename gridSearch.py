import pandas as pd
import numpy as np
from itertools import permutations
from itertools import product
from MLP import MLP
from funct import cross_entropy, to_categorical
from scipy.special import softmax
from IPython.display import clear_output
from sklearn.utils import shuffle

"""
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
"""

def get_configurations(params):
    """
    given parameters dictionary, return all possible configurations of parametes
    """
    x = [ dict( zip( params[0], v) ) for v in product( *params[0].values() ) ]
    return x

def read_dataset( filename='monks-1.train', id_col=8 ):
    """
    read dataset by given file name. To be written properly
    """
    data = pd.read_csv( filename, sep=' ', index_col=id_col ) # read dataset
    data = data.drop(columns=data.columns[0]) # drop index column
    data = data.to_numpy() 
    return data

def get_fold(folds,i):
    """
    Given folds of data, returns training set obtained by aggregating all folds but i-th one, that is instead used as and validation set
    """
    # training set is all but i-th fold
    train_set = np.concatenate(folds[:i] + folds[i+1:])

    # i-th fold is validation set
    val_set = folds[i]  
    return train_set, val_set

def k_fold_CV(data, params, k=4, max_epochs=300, tresh=.1, bs=30, measure_interval=10, xy=None):
    """
    Grid search of parameters of the net for given data.
    First divides the data into k folds, then tries all possible congigurations of parameters. For each of them 
    computes validation error on one of the folds after training on remaining data by n_init initializations.
    Best configutation is the one that gave best average validation error over the k folds.
    """

    # important to do this if data is ordered (case of poly regression in test)
    data = shuffle(data)
    #split data into k folds
    folds = np.array_split(data, k)  

    # given the params grid, get all the possible configurations
    configurations = get_configurations(params)

    # variables for keeping the best configuration
    best_error = np.inf # error given by configuraion that gives the best error atm
    best_conf = None #  configuraion that gives the best error atm

    # try all the configurations, for each of them we compute error of n_init MLP 
    # initializations on all of the folds.
    print(f'testing {len(configurations)} configurations \n') # debugging
    for idx_c,c in enumerate(configurations):
        print(f'testing configuration {c}, {idx_c}/{len(configurations)}') # debugging

        # for each fold, train the network and save the validation error on k-th fold
        val_error = [None]*k # we save validation error for each fold
        train_error = [None]*k
        for i in range(k):
            
            # get data form i-th fold
            train_set, val_set = get_fold(folds,i) 

            # split patterns into input and target output
            tr_x, tr_y = xy(train_set)
            val_x, val_y = xy(val_set)
            print(f' {tr_x.shape}, {tr_y.shape}, {val_x.shape}, {val_y.shape} ')

            # new net
            n = MLP(Nodes = c['Nodes'], f = c['f'], f_out=c['f_out'], w_range=c['weights_range']) # inizializzo matrice pesi
            n.train( tr_x, tr_y,  c['learning_rate'], a= c['alpha'], l=c['lambda'], max_epochs=max_epochs, tresh=tresh, bs=bs, shuffle_data=False, measure_interval=measure_interval, verbose=False ) # train the network 

            # compute validation error and save it
            val_error[i] = n.error(val_x, val_y) # test network on this fold and save the resulting error
            train_error[i] = n.error(tr_x,tr_y)

            print(f'fold {i} done, error {val_error[i]}') # debugging
        
        # compute mean validation error on the k folds
        val_error = np.mean(val_error) 
        train_error = np.mean(train_error)
                    
        
        print(f'mean error for this config: validation {val_error} train {train_error} \n')
        if val_error < best_error:
            best_error = val_error
            best_conf = c
        
    print(f'best config {c}: {best_error}')
        # clear_output(wait=True) # debugging

    return best_conf, best_error
        