from parafac import models, synthetic, gpu
import pyro
import torch
import pyslurm
import pandas as pd
import seaborn as sns
import torch.multiprocessing as mp
import scanpy as sc
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score


# Temporary, need to have the metadata in model
def compute_silhouette_score(samples, metadata, metadata_array):
    factors = pd.DataFrame(samples)
    factors = factors.add_prefix('Factor')
    print(factors)
    print(metadata)
    print(metadata_array)
    factors[metadata] = metadata_array.to_list()
    adata_samples = sc.AnnData(factors.iloc[:, :-1])
    adata_samples.obs = factors[[metadata]]
    cluster_labels = adata_samples.obs[metadata].astype('category').cat.codes  # Mapping categories to integer labels
    data_matrix = adata_samples.X  # Accessing the data matrix
    silhouette_avg = silhouette_score(data_matrix, cluster_labels)
    return(silhouette_avg)

def accuracy_outcome(pred, obs_outcome, val_indices):
    obs_outcome = torch.argmax(obs_outcome, dim=1)
    list_bool = [torch.argmax(pred[n][0], dim=1) == obs_outcome[val_indices] for n in list((range(0,pred.shape[0])))]
    list_int = [t.int() for t in list_bool]
    list_accuracy = [t[t==1].shape[0]/len(val_indices)*100 for t in list_int]
    return(sum(list_accuracy)/len(list_accuracy))


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
        torch.save(train, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_accuracy/trained_scale{}_fold{}.pt'.format(scale_factor, i))
        torch.save(val, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_accuracy/val_scale{}_fold{}.pt'.format(scale_factor, i))
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
                     **kwargs):
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)

    list_accuracy = []
    data_df = pd.DataFrame(columns=['seed', 'scale', 'accuracy', 'fold', 'silhouette'])
    # Iterate over folds
    for fold, (train_index, val_index) in enumerate(kf.split(dataTensor)):
        # Create training and validation sets for each tensor
        train_data = dataTensor[train_index]
        val_data = dataTensor[val_index]
        accuracy = train_and_val(fold,
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
    return(list_accuracy)


def train_and_val(fold,
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
            n_epochs=500,
            n_particles=1,
            learning_rate=0.001,
            optimizer="clipped",
            verbose=1,
            seed=42
        )

        # Get the posterior for the trained samples
        train_samples = trained_model._guide.median()
        A1_sample = train_samples["A1"].detach().cpu()
        torch.save(train_samples, '/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_accuracy/samples/trained_samples_scale{}_fold{}.pt'.format(scale_factor, fold))

        param_store_trained = pyro.get_param_store().items()
    
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

        #return([42, fold, scale_factor, accuracy, silhouette], train_samples, posterior_predictive)

#metadata and metadata array to put in training, cause used for silhouette function as well TODO
def pair_grid_fig(samples_factor, metadata, metadata_array, scale, fold):
    samples_factor = samples_factor.add_prefix('Factor')
    samples_factor[metadata] = metadata_array.to_list()
    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[0:5]), hue = metadata)
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_05.".format(scale, fold)) 

    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[6:10]), hue = metadata)
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_610.".format(scale, fold)) 

    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[11:15]), hue = metadata)
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_1115.".format(scale, fold)) 

    g = sns.PairGrid(samples_factor, vars=list(samples_factor.columns[16:20]), hue = metadata)
    g.map(sns.scatterplot, style=samples_factor["FactorNewColumn"])
    g.add_legend()
    g.savefig("/home/kdeazevedo/Codes/github/multi-bayesian-parafac/multi_parafac/output_TCGA/fig/PairGrid_scale{}_fold{}_1620.".format(scale, fold)) 


