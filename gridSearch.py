import pandas as pd
import numpy as np
from itertools import permutations
from itertools import product
from MLP import MLP, cross_entropy, to_categorical
from scipy.special import softmax
from IPython.display import clear_output


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

def k_fold_CV(data, params, k=4, n_init=10, max_epochs=300, tresh=.1, measure_interval=10, xy=None):
    """
    Grid search of parameters of the net for given data.
    First divides the data into k folds, then tries all possible congigurations. For each of them 
    computes validation error on one of the folds after training on remaining data by n_init initializations.
    Best configutation is the one that gave best average validation error.
    """
    #split data into 4 folds
    folds = np.array_split(data, k)  

    # given the params grid, get all the possible configurations
    configurations = get_configurations(params)

    # variables for keeping the best configuration
    best_error = np.inf # error given by configuraion that gives the best error atm
    best_conf = None #  configuraion that gives the best error atm

    # try all the configurations, for each of them we compute error of n_init MLP 
    # initializations on all of the folds.
    print(f'testing {len(configurations)} configurations') # debugging
    for idx_c,c in enumerate(configurations):
        best_error_init = np.inf  # best error given by some initialization of an MLP with this configuration
        print(f'testing configuration {c}, {idx_c}/{len(configurations)}') # debugging

        # test the configuration n_init times on every fold
        # this could be done with multithreading 
        # min( list( p.map(do_print,range(0,10)) ) )
        for n in range(n_init):
            print(f'initialization {n}') # debugging

            # initialize a net with the params in this configuration
            n = MLP(Nh = c['hidden_units'], Nu = c['Nu'], Ny= c['Ny'], f = c['activation'], f_out=c['f_out'], w_range=c['weights_range'], w_scale=c['weights_scale'], loss=c['loss'], error=c['error']) # inizializzo matrice pesi
            init_w = np.copy( n.w ) # salvo una copia dei pesi iniziali

            # for each fold, train the network and save the validation error on k-th fold
            # could be turned to multithread but network is same for each fold and that is a problem. maybe a trhread for each init would be better, but then need to consider concurrency... otherwise make k copies of the network.
            # p=mp.Pool(4)
            # p.map(do_print,range(0,10)) 
            # p.close()
            # p.join()

            val_error = [None]*k # we save validation error for each fold
            for i in range(k):
                print(f'fold {i}') # debugging
                
                # get data form i-th fold
                train_set, val_set = get_fold(folds,i) 

                # split patterns into input and target output
                tr_x, tr_y = xy(train_set)
                val_x, val_y = xy(val_set)

                # reset weights to initial values
                n.w = init_w 
                n.train( tr_x, tr_y,  c['learning_rate'], a= c['alpha'], l=c['lambda'], max_epochs=max_epochs, tresh=tresh, mode="batch", shuffle_data=False, measure_interval=measure_interval ) # train the network 

                # compute validation error and save it
                val_error[i] = n.test(val_x, val_y) # test network on this fold and save the resulting error
            
            # compute mean validation error on the k folds, save the one given by the best initialization
            val_error = np.mean(val_error) 
            if val_error < best_error_init:
                best_error_init = val_error
        
        # best_error_init errore minimo che ho trovato con questa configurazione
        # after computing error of best initialization with this configuration, compare it with other configurations and save better one
        if best_error_init < best_error:
            best_error = best_error_init
            best_conf = c

        clear_output(wait=True) # debugging

    return best_conf, best_error
        
