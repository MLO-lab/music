from parafac import models, synthetic, gpu
import pyro
import torch
import pyslurm
import pandas as pd
import seaborn as sns
import torch.multiprocessing as mp
import scanpy as sc
<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
import os
>>>>>>> dev
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score


# Temporary, need to have the metadata in model
def compute_silhouette_score(samples, metadata, metadata_array):
    factors = pd.DataFrame(samples)
    factors = factors.add_prefix('Factor')
<<<<<<< HEAD
    print(factors)
    print(metadata)
    print(metadata_array)
=======
>>>>>>> dev
    factors[metadata] = metadata_array.to_list()
    adata_samples = sc.AnnData(factors.iloc[:, :-1])
    adata_samples.obs = factors[[metadata]]
    cluster_labels = adata_samples.obs[metadata].astype('category').cat.codes  # Mapping categories to integer labels
    data_matrix = adata_samples.X  # Accessing the data matrix
    silhouette_avg = silhouette_score(data_matrix, cluster_labels)
    return(silhouette_avg)

<<<<<<< HEAD
def accuracy_outcome(pred, obs_outcome, val_indices):
    obs_outcome = torch.argmax(obs_outcome, dim=1)
    list_bool = [torch.argmax(pred[n][0], dim=1) == obs_outcome[val_indices] for n in list((range(0,pred.shape[0])))]
    list_int = [t.int() for t in list_bool]
    list_accuracy = [t[t==1].shape[0]/len(val_indices)*100 for t in list_int]
    return(sum(list_accuracy)/len(list_accuracy))
=======
def accuracy_outcome(pred, ground_truth):
    pred = torch.argmax(pred, dim=1)
    ground_truth = torch.argmax(ground_truth, dim=1)
    list_bool = ground_truth == pred
    accuracy = (list_bool[list_bool==True].shape[0]/len(ground_truth))*100
    return(accuracy)
>>>>>>> dev


def posterior_outcome(model,
                        data):
    # Get posterior trace from the validation model
    guide_trace = pyro.poutine.trace(model._guide).get_trace(data)
    # sample observations given latent variables
    blockreplay = pyro.poutine.block(fn = pyro.poutine.replay(model._model, guide_trace), expose=['outcome'])
    posterior_predictive = pyro.sample('pred_obs', blockreplay, data)
    return(posterior_predictive)

def LOO_cross_validation(dataTensor, 
                         obs_outcome,
                         n_features,
                         metadata =  None,
                         metadata_array =  None,
                         R = 15,
                         use_gpu = True,
                         scale_factor = 1,
                         **kwargs):
    
    # Perform leave-one-out cross-validation
    list_accuracy = []
    data_df = pd.DataFrame(columns=['seed', 'scale', 'accuracy', 'fold', 'silhouette'])
    for i in range(1, dataTensor.shape[0]):
        # Exclude the i-th sample for validation
        train_data = torch.cat((dataTensor[:i], dataTensor[i+1:]))
        val_data = dataTensor[i].unsqueeze(0)
        index = list(range(0, dataTensor.shape[0]))
        train_index = index[:i] + index[i+1:]
        accuracy, train, val = train_and_val(i,
                                 train_data, 
                                 val_data,
                                 train_index,
                                 [i],
                                 obs_outcome,
                                 n_features,
                                 metadata,
                                 metadata_array,
                                 R,
                                 use_gpu,
                                 scale_factor,
                                 compute_silhouette = False,
                                 **kwargs)
        list_accuracy.append(accuracy)
        #TODO FILEPATH arg
<<<<<<< HEAD
        torch.save(train, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_accuracy/trained_scale{}_fold{}.pt'.format(scale_factor, i))
        torch.save(val, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_accuracy/val_scale{}_fold{}.pt'.format(scale_factor, i))
=======
        torch.save(train, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/trained_scale{}_fold{}.pt'.format(scale_factor, i))
        torch.save(val, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/val_scale{}_fold{}.pt'.format(scale_factor, i))
>>>>>>> dev
    return(list_accuracy)

def Kfold_cross_validation(dataTensor, 
                     obs_outcome,
                     n_features,
                     metadata =  None,
                     metadata_array =  None,
                     R = 15,
                     use_gpu = True,
                     scale_factor = 1,
                     num_folds = 3,
                     shuffle = True,
                     random_state = 42,
<<<<<<< HEAD
=======
                     parallel = True,
>>>>>>> dev
                     **kwargs):
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)

    list_accuracy = []
<<<<<<< HEAD
    data_df = pd.DataFrame(columns=['seed', 'scale', 'accuracy', 'fold', 'silhouette'])
=======
    data_df = pd.DataFrame(columns=['seed', 'scale', 'train_accuracy', 'val_accuracy', 'fold', 'silhouette'])
>>>>>>> dev
    # Iterate over folds
    for fold, (train_index, val_index) in enumerate(kf.split(dataTensor)):
        # Create training and validation sets for each tensor
        train_data = dataTensor[train_index]
        val_data = dataTensor[val_index]
<<<<<<< HEAD
        accuracy = train_and_val(fold,
=======
        accuracy = train_and_val(dataTensor,
                                 fold,
>>>>>>> dev
                                 train_data, 
                                 val_data,
                                 train_index,
                                 val_index,
                                 obs_outcome,
                                 n_features,
                                 metadata,
                                 metadata_array,
                                 R,
                                 use_gpu,
                                 scale_factor,
                                 **kwargs)
        list_accuracy.append(accuracy)
<<<<<<< HEAD
    return(list_accuracy)


def train_and_val(fold,
=======
        
    return(list_accuracy)


def train_and_val(dataTensor,
                  fold,
>>>>>>> dev
                  train_data, 
                  val_data,
                  train_index,
                  val_index,
                  obs_outcome,
                  n_features,
                  metadata = None,
                  metadata_array = None,
                  R = 15,
                  use_gpu = True,
                  scale_factor = 1,
                  compute_silhouette = True,
                  **kwargs):
    
        trained_model = models.PARAFAC(
            train_data,
            n_features = n_features,
            outcome_obs = obs_outcome[train_index],
            R = R,
            scale_factor = scale_factor,
            use_gpu = use_gpu
        )
    
        loss_history, _ = trained_model.fit(
<<<<<<< HEAD
            n_epochs=500,
            n_particles=1,
            learning_rate=0.001,
            optimizer="clipped",
            verbose=1,
            seed=42
=======
            **kwargs
>>>>>>> dev
        )

        # Get the posterior for the trained samples
        train_samples = trained_model._guide.median()
        A1_sample = train_samples["A1"].detach().cpu()
<<<<<<< HEAD
        torch.save(train_samples, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_accuracy/samples/trained_samples_scale{}_fold{}.pt'.format(scale_factor, fold))

        param_store_trained = pyro.get_param_store().items()
    
=======
        torch.save(trained_model.state_dict(), '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/samples/trained_samples_scale{}_fold{}.pt'.format(scale_factor, fold))

        #param_store_trained = pyro.get_param_store().items()
        pyro.get_param_store().save('/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/saved_params_scale{}_fold{}.save'.format(scale_factor, fold))

        elbo = sns.lineplot(loss_history)
        elbo_fig = elbo.get_figure()
        elbo_fig.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/ELBO/ELBO_scale{}_fold{}.png".format(scale_factor, fold)) 

        '''
>>>>>>> dev
        # Use the trained model to make predictions on the validation set
        validation_model = models.PARAFAC(
            val_data,
            n_features = n_features,
            outcome_obs = None,
            R = R,
            use_gpu = use_gpu,
            scale_factor = scale_factor,
            A2 = train_samples['A2'],
            A3 = train_samples['A3']
        )
    
        loss_history, _ = validation_model.fit(
            n_epochs=500,
            n_particles=1,
            learning_rate=0.001,
            optimizer="clipped",
            verbose=1,
            seed=42
        )
<<<<<<< HEAD

    
        val_samples = validation_model._guide.median()
        A1_sample_val = val_samples["A1"].detach().cpu()
        if(compute_silhouette):
            silhouette = compute_silhouette_score(A1_sample_val, metadata, metadata_array[val_index])
        else:
            silhouette = None

        posterior_predictive = posterior_outcome(validation_model, val_data)
        pred = posterior_predictive['outcome'].to(validation_model.device)
        A1_pred_sample = posterior_predictive["A1"].detach().cpu()
        torch.save(posterior_predictive, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/samples/val_samples_scale{}_fold{}.pt'.format(scale_factor, fold))
        obs_outcome = obs_outcome.to(validation_model.device)
        accuracy = accuracy_outcome(pred, obs_outcome, val_index)
        
        
        A1_sample_df = pd.DataFrame(A1_sample)
        A1_sample_df['NewColumn'] = 'Train'
        A1_sample_val_df = pd.DataFrame(A1_sample_val)
        A1_sample_val_df['NewColumn'] = 'Val'
        # Split on val index so we can have correct labelling of drugs
        #TODO ONLY WORKS FOR LOO, would need sthing else for K fold

        # Split df1 into two parts
        df1_part1 = A1_sample_df.iloc[:val_index[0]]
        df1_part2 = A1_sample_df.iloc[val_index[0]:]

        samples_df = pd.concat([df1_part1, A1_sample_val_df, df1_part2])

        #pair grid
        pair_grid_fig(samples_df, metadata, metadata_array, scale_factor, fold)

        return([42, fold, scale_factor, accuracy, silhouette])
=======
        '''

        train_outcome = trained_model._model.RegressionModel(train_data.to(trained_model.device))
        val_outcome = trained_model._model.RegressionModel(val_data.to(trained_model.device))
        torch.save(val_outcome, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/samples/predicted_outcome_scale{}_fold{}.pt'.format(scale_factor, fold))


        outcome = trained_model._model.RegressionModel(dataTensor.to(trained_model.device))
        samples = pd.DataFrame(trained_model._model.RegressionModel.out2.cpu().detach().numpy())
        # Assign values based on the index lists
        #print(dataTensor.shape)
        #print(samples)
        #print(train_index)
        #print(val_index)
        
        if(compute_silhouette):
            silhouette = compute_silhouette_score(samples, metadata, metadata_array)
        else:
            silhouette = None

        samples['New_Column'] = ''
        samples.loc[train_index, 'New_Column'] = 'Train'
        samples.loc[val_index, 'New_Column'] = 'Val'

        #val_outcome = outcome[val_index]
        #train_outcome = outcome[train_index]
        ground_truth = obs_outcome
        ground_truth = ground_truth.to(trained_model.device)
        train_accuracy = accuracy_outcome(train_outcome, ground_truth[train_index])
        val_accuracy = accuracy_outcome(val_outcome,  ground_truth[val_index])
        
        pair_grid_fig(samples, metadata, metadata_array, scale_factor, fold)

        return([42, fold, scale_factor, train_accuracy, val_accuracy, silhouette])
>>>>>>> dev

        #return([42, fold, scale_factor, accuracy, silhouette], train_samples, posterior_predictive)

#metadata and metadata array to put in training, cause used for silhouette function as well TODO
def pair_grid_fig(samples_factor, metadata, metadata_array, scale, fold):
    samples_factor = samples_factor.add_prefix('Factor')
    samples_factor[metadata] = metadata_array.to_list()
<<<<<<< HEAD
    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[0:5]), hue = metadata)
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
=======
    samples_factor.to_csv("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/out2__scale{}_fold{}.tsv".format(scale, fold))
    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[0:5]), hue = metadata)
    g.map(sns.scatterplot, style=samples_factor["FactorNew_Column"])
>>>>>>> dev
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_05.".format(scale, fold)) 

    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[6:10]), hue = metadata)
<<<<<<< HEAD
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
=======
    g.map(sns.scatterplot, style=samples_factor["FactorNew_Column"])
>>>>>>> dev
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_610.".format(scale, fold)) 

    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[11:15]), hue = metadata)
<<<<<<< HEAD
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
=======
    g.map(sns.scatterplot, style=samples_factor["FactorNew_Column"])
>>>>>>> dev
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_1115.".format(scale, fold)) 

    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[16:20]), hue = metadata)
<<<<<<< HEAD
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
=======
    g.map(sns.scatterplot, style=samples_factor["FactorNew_Column"])
>>>>>>> dev
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_1620.".format(scale, fold)) 


