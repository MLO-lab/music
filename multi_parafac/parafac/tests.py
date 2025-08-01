from parafac import models, synthetic, gpu, training
import pandas as pd
import pyro
import torch
import itertools
import time
import tqdm
import pyslurm
import torch.multiprocessing as mp
import os
import subprocess
from itertools import repeat

<<<<<<< HEAD
=======
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor

>>>>>>> dev

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def create_element(seeds,
                   n_views = [3],
                   n_drugs = [5],
                   n_samples = [1000],
                   n_features = [12],
                   R = [10],
                   a = [1],
                   b = [0.5],
                   c = [1],
                   d = [0.5],
                   percent = [0]):
    """
    Create element for increase_size function
    seeds : list of seeds
    n_drugs : list of number of drugs, optional
    n_samples : list of number of samples, optional
    n_features : list of number of features, optional
    a : list of concentration for Tau, optional
    b : list of rates for Tau, optional
    c : list of concentration for Lambda, optional
    d : list of rates for Lambda, optional
    """
    return list(itertools.product(seeds, n_views, n_drugs, n_samples, n_features, R, a, b, c, d, percent))

def create_dataset(elem):
    """
    Increase size of samples, drugs or features
    elem : List of seed / n_drugs / n_samples / n_features
    """
    data_list = []
    sim_list = []
    device = f"cuda:{gpu.get_free_gpu_idx()}"
    for i in list(range(0, elem[1])):
        #print("1")
<<<<<<< HEAD
        data = synthetic.DataGenerator(n_drugs = elem[2],
                                      n_features = elem[4],
                                      n_samples = elem[3],
                                      a = elem[6],
                                      b = elem[7],
                                      device = device)
=======
        data = synthetic.DataGenerator(n_samples = elem[2],
                                       n_features = elem[4],
                                       R =  elem[5],
                                       n_drugs = elem[3],
                                       a = elem[6],
                                       b = elem[7],
                                       device = device)
>>>>>>> dev
        #print("2")
        rng = data.generate(seed = elem[0])
        #print("3")
        print(data)
        data.generate_missingness(elem[10])
        #print("4")
        sim_data = data.get_sim_data()
        data_list.append(data)
        sim_list.append(sim_data)
    
    start_time = time.time()
    model = models.PARAFAC([x.Y for x in data_list],
                           [elem[4]]*elem[1],
                           R = elem[5],
                           a = elem[6],
                           b = elem[7],
                           c = elem[8],
                           d = elem[9],
                           device = device)
    # set random seed
    pyro.set_rng_seed(elem[0])
    # clean start
    print("Cleaning parameter store")
    pyro.enable_validation(True)
    pyro.clear_param_store()

    #print("b")
    loss_history, _ = model.fit(
        learning_rate=0.01,
        n_particles=1,
        seed=elem[0]
    )
    end_time = time.time()
<<<<<<< HEAD
=======

    # Use pytorch as backend for tensorly
    tl.set_backend('pytorch')
    tensor = [x.Y for x in data_list]
    weight, factors = parafac(tensor[0], rank=elem[5], init='random')
    
    # Get the decomposed factors
    factor_1, factor_2, factor_3 = factors
    
    # Reconstruct the original tensor
    reconstructed_tensor = cp_to_tensor((weight, factors))
    
    # The inferred tensor is not taken from the posterior, but multiplied from the inferred factor matrices
    inferred_tensor = torch.einsum('ir,jr,kr->ijk', factor_1, factor_2, factor_3)
    #inferred_tensor_with_noise = pyro.sample("Y", dist.Normal(torch.einsum('ir,jr,kr->ijk', factor_1, factor_2, factor_3), 1/tau))
    
    RMSE_Tensorly = RMSELoss(inferred_tensor, tensor[0])
    print("RMSE_Tensorly")
    print(RMSE_Tensorly)
    
    
>>>>>>> dev
    samples = model._guide.median()
    samples['Y'] = torch.einsum('ir,jr,kr->ijk', samples['A1'], samples['A2'], samples['A3'])
    cat_tensor = torch.cat([x['Y_sim'] for x in sim_list], 2)
    new_dict = dict({'Y_sim' : cat_tensor, 'A1_sim' : sim_list[0]['A1_sim'], 'A2_sim' : sim_list[0]['A2_sim'], 'A3_sim' : sim_list[0]['A3_sim']}, **{k: samples[k] for k in ('Y')})
    #TODO: RMSE with NA values
<<<<<<< HEAD
    print(" Y devices")
    print(new_dict['Y'].device)
    print(new_dict['Y_sim'].device)
    data_tuple = (*elem, float(RMSELoss(new_dict['Y'], torch.nan_to_num(new_dict['Y_sim']))), end_time - start_time)
=======
    data_tuple = (*elem, float(RMSELoss(new_dict['Y'], torch.nan_to_num(new_dict['Y_sim']))), float(RMSE_Tensorly.item()), end_time - start_time)
    print("datatuple")
    print(len(data_tuple))
>>>>>>> dev
    return data_tuple

def parallel_create_dataset(procs, **kwargs):
    all_args = create_element(**kwargs)
    
    with Pool(processes=procs) as p:
        results = list(tqdm.tqdm(p.imap(create_dataset, all_args), total=len(all_args)))
    #results = create_dataset(all_args[0])
    
    data_df = pd.DataFrame([results[i][0] for i in range(0, len(results))])
    data_df.columns = ['seed', 'n_views', 'n_drugs', 'n_samples', 'n_features', 
<<<<<<< HEAD
                       'R', 'a', 'b', 'c', 'd', 'percentNA', 'RMSE', 'time']
=======
                       'R', 'a', 'b', 'c', 'd', 'percentNA', 'Model', 'Tensorly', 'time']
>>>>>>> dev
        
    return(data_df)

# Function to create a random tensor and compute its mean
def task_worker(elem, filepath):
    result = create_dataset(elem)
<<<<<<< HEAD
    print(result)
    df = pd.DataFrame([result])
    print(df)
    df.columns = ['seed', 'n_views', 'n_drugs', 'n_samples', 'n_features',
                          'R', 'a', 'b', 'c', 'd', 'percentNA', 'RMSE', 'time']
=======
    print("df")
    print(len(result))
    df = pd.DataFrame([result])
    print(df.shape)
    df.columns = ['seed', 'n_views', 'n_drugs', 'n_samples', 'n_features',
                  'R', 'a', 'b', 'c', 'd', 'percentNA', 'Model',  'Tensorly', 'time']
>>>>>>> dev
    df.to_csv(filepath)

# Function for accuracy and silhouette
def task_worker_accuracy(filepath, dataTensor_filepath, one_hot_encoding_filepath, n_features, n_epochs, scale_factor):
    dataTensor = torch.load(dataTensor_filepath)
    one_hot_encoding = torch.load(one_hot_encoding_filepath)
    long_df_RNA = pd.read_csv("~/Data/RCCL/2_Output_R/RNA_Symbol.tsv", sep = "\t", index_col = 1)
    metadata = pd.read_csv("~/Data/RCCL/2_Output_R/metadata.tsv", sep = "\t", index_col = 0)
    long_df_RNA = long_df_RNA.iloc[: , 1:].T.join(metadata)
    metadata_array = long_df_RNA.index.str.split('_').str[1]
    result = training.LOO_cross_validation(dataTensor, 
                                           one_hot_encoding,
                                           [n_features],
                                           metadata_array = metadata_array,
                                           metadata = "Drug",
                                           R = 20,
                                           use_gpu = True,
                                           scale_factor = scale_factor,
                                           random_state = 42,
                                           n_epochs=n_epochs,
                                           n_particles=1,
<<<<<<< HEAD
                                           learning_rate=0.001,
                                           optimizer="clipped",
                                           verbose=1,
                                           seed=42)
    df = pd.DataFrame(result, columns=['seed', 'fold', 'scale', 'accuracy', 'silhouette'])
    df.to_csv(filepath)

# Function for accuracy and silhouette
def task_worker_accuracy_CV(filepath, dataTensor_filepath, one_hot_encoding_filepath, n_features, n_epochs, scale_factor, compute_silhouette=True):
    dataTensor = torch.load(dataTensor_filepath)
    one_hot_encoding = torch.load(one_hot_encoding_filepath)
    tcga_metadata = pd.read_csv("/home/kdeazevedo/TCGA_metadata.tsv", sep = "\t", index_col = 1)
    metadata_array = tcga_metadata.primary_diagnosis
=======
                                           learning_rate=0.01,
                                           optimizer="clipped",
                                           verbose=1,
                                           seed=42)
    df = pd.DataFrame(result, columns=['seed', 'fold', 'scale', 'train_accuracy', 'val_accuracy', 'silhouette'])
    df.to_csv(filepath)

# Function for accuracy and silhouette
def task_worker_accuracy_CV(filepath, dataTensor_filepath, one_hot_encoding_filepath, n_features, n_epochs, scale_factor, parallel=True, compute_silhouette=True):
    dataTensor = torch.load(dataTensor_filepath)
    one_hot_encoding = torch.load(one_hot_encoding_filepath)
    tcga_metadata = pd.read_csv("/home/kdeazevedo/Codes/github/TCGA_skin_prostate__metadata.tsv", sep = "\t", index_col = 1)
    print(tcga_metadata)
    metadata_array = tcga_metadata.project_id
>>>>>>> dev
    result = training.Kfold_cross_validation(dataTensor, 
                                             one_hot_encoding,
                                             [n_features],
                                             metadata_array = metadata_array,
<<<<<<< HEAD
                                             metadata = "primary_diagnosis",
                                             R = 30,
                                             use_gpu = True,
                                             scale_factor = scale_factor,
=======
                                             metadata = "project_id",
                                             R = 30,
                                             use_gpu = True,
                                             scale_factor = scale_factor,
                                             parallel = parallel,
>>>>>>> dev
                                             random_state = 42,
                                             num_folds = 15,
                                             shuffle = True,
                                             n_epochs=n_epochs,
                                             n_particles=1,
<<<<<<< HEAD
                                             learning_rate=0.001,
                                             optimizer="clipped",
                                             compute_silhouette=compute_silhouette,
                                             verbose=1,
                                             seed=42)
    #print(result)
    df = pd.DataFrame(result, columns=['seed', 'fold', 'scale', 'accuracy', 'silhouette'])
=======
                                             learning_rate=0.1,
                                             optimizer="clipped",
                                             verbose=1,
                                             seed=42,
                                             compute_silhouette=compute_silhouette)
    #print(result)
    df = pd.DataFrame(result, columns=['seed', 'fold', 'scale', 'train_accuracy', 'val_accuracy', 'silhouette'])
>>>>>>> dev
    df.to_csv(filepath)

# Function to submit a Slurm job for a specific task
def submit_slurm_job(task_id, elem, filepath):

    print(task_id)
    print(elem)
    print(filepath)
    # Create a temporary script file with your command
    script_content = f"""#!/bin/bash
    python3 -c 'from parafac.tests import task_worker; task_worker({elem}, "{filepath}")'
    """

    script_file = f"task_{task_id}.out"
    with open(script_file, "w") as f:
        f.write(script_content)
            
    desc = pyslurm.JobSubmitDescription(
        name=f"task_{task_id}",
        nodes=1,
        memory_per_node="4GB",
        partitions="gpu-unlimited",
        time_limit="00:10:00",
        standard_output=f"task_{task_id}.out",
        script=script_file
    )
    job_id = desc.submit()
    print("res of desc submit")
    print(job_id)
    
    job_done = False
    while job_done == False:
        while not job_done:
            job_done = True
            job_status = subprocess.check_output(["scontrol", "show", "job", job_id]).decode('utf-8')
            if "JobState=RUNNING" in job_status or "JobState=PENDING" in job_status:
                job_done = False
                break
            if not job_done:
                time.sleep(10)  # Sleep for some time before checking again

    os.remove(script_file)
    return(data_df)

def parallel_create_dataset_2(procs, **kwargs):
    all_args = create_element(**kwargs)

    data_df = pd.DataFrame(columns=['seed', 'n_views', 'n_drugs', 'n_samples', 'n_features', 
<<<<<<< HEAD
                                    'R', 'a', 'b', 'c', 'd', 'percentNA', 'RMSE', 'time'])
=======
                                    'R', 'a', 'b', 'c', 'd', 'percentNA', 'Model',  'Tensorly', 'time'])
>>>>>>> dev
    data_df.to_csv("test.csv")

    print(zip(range(len(all_args)), all_args, repeat(data_df)))

    print(range(len(all_args)))
    print(all_args)
    print(repeat("test.csv"))
    
    with mp.get_context("spawn").Pool() as pool:
        results = list(tqdm.tqdm(pool.starmap(submit_slurm_job, 
                                              zip(range(len(all_args)), all_args, repeat("test.csv"))
                                             )
                                )
                      ) 

    print(results)
        
    return(results)


# Function to submit a Slurm job for a specific task
def submit_and_get_output(task_id, script_file):
    desc = pyslurm.JobSubmitDescription(
        name=f"task_{task_id}",
        nodes=1,
        memory_per_node="4GB",
        partitions="gpu-unlimited",
        time_limit="00:10:00",
        standard_output=f"task_{task_id}.out",
        script=script_file
    )
    job_id = desc.submit()
    print(job_id)
    return(job_id)

def parallel_create_dataset_3(procs, **kwargs):
    all_args = create_element(**kwargs)
    num_tasks = len(all_args)
    job_info = []

    for i in range(num_tasks):
    # Create a temporary script file with your command
        script_content = f"""#!/bin/bash
        python3 -c 'from parafac.tests import task_worker; task_worker({all_args[i]})'
        """
        
        script_file = f"task_{i}.out"
        with open(script_file, "w") as f:
            f.write(script_content)

        job_id = submit_and_get_output(i, script_file)
        job_info.append(job_id)

        os.remove(script_file)

    print("job_info")
    print(job_info)
    data_df = pd.DataFrame([job_info[i] for i in range(0, len(job_info))])
    data_df.columns = ['seed', 'n_views', 'n_drugs', 'n_samples', 'n_features', 
<<<<<<< HEAD
                       'R', 'a', 'b', 'c', 'd', 'percentNA', 'RMSE', 'time']
=======
                       'R', 'a', 'b', 'c', 'd', 'percentNA', 'Model', 'Tensorly', 'time']
>>>>>>> dev
        
    return(data_df)
