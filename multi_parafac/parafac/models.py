import logging

from parafac.gpu import get_free_gpu_idx

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pandas.api.types import is_string_dtype
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide.guides import AutoNormal, deep_getattr, deep_setattr

from pyro.nn import PyroModule, PyroParam
import pyro.contrib.gp as gp
from pyro.optim import Adam, ClippedAdam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
<<<<<<< HEAD
=======
import gc
>>>>>>> dev

logger = logging.getLogger(__name__)

class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return index, tuple(tensor[index] for tensor in self.tensors)

class PARAFAC(PyroModule):
    def __init__(
        self,
        observations,
        n_features,
        outcome_obs = None,
<<<<<<< HEAD
        p = None,
=======
        p = 1,
>>>>>>> dev
        metadata = None,
        index = None,
        n_samples = 100,
        R = 10,
        a = 1,
        b = 0.5,
        c = 1,
        d = 0.5,
        covariates = None,
        view_names = None,
        device = None,
        use_gpu: bool = True,
        A2 = None,
        A3 = None,
        scale_factor = 1.1,
    ):
        """Parafac module.

        Parameters
        ----------
        observations : MultiView
            Collection of observations as list or dict.
        covariates : SingleView, optional
            Additional observed covariates, by default None
        view_names : List[str], optional
            List of names for each view,
            determines view order as well,
            by default None
        use_gpu : bool, optional
            Whether to train on a GPU, by default True
        """
        super().__init__(name="PARAFAC")

<<<<<<< HEAD
=======
        self.device = device
        torch.set_default_device(self.device)

>>>>>>> dev
        self.covariates = covariates
        self.view_names = view_names
        self.observations = observations
        self.metadata = metadata
        self.index = index
        self.n_samples = n_samples
<<<<<<< HEAD
        print("device arg")
        print(device)
        self.device = device
=======
>>>>>>> dev
        if(device == None):
            self.device = torch.device("cpu")
            if use_gpu and torch.cuda.is_available():
                logger.info("GPU available, running all computations on the GPU.")
                self.device = f"cuda:{get_free_gpu_idx()}"
                #self.device="cuda:0"
<<<<<<< HEAD
        print("self.device")
        print(self.device)
        self.to(self.device)
=======
        #self.to(self.device)
        
>>>>>>> dev

        self._model = None
        self._guide = None
        self._built = False
        self._trained = False
        self._cache = None
        self._informed = None
        
        self.tensor_data = None

        self.outcome_obs = outcome_obs
        self.p = p
        
        self.n_features = n_features
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.R = R
        self.A2 = A2
        self.A3 = A3
        self.scale_factor = scale_factor
        
    def _setup_tensor_data(self):
        if(all(isinstance(x, torch.Tensor) for x in self.observations)):
            if(all(len(x.shape) == 3 for x in self.observations)):
                self.tensor_data = torch.cat(self.observations, 2)

            #if(all(len(x.shape) == 2 for x in list_tensor)):
        
        if(all(isinstance(x, pd.core.frame.DataFrame) for x in self.observations)):
            if(metadata == None):
                logger.info("Metadata required")
            else:
                list_tensor = list()
                index_len = len(set(list(self.metadata[index])))
                drop_col = len(self.metadata.columns)
                for matrix in self.observations:
                    matrix = matrix.join(self.metadata)
                    X = np.array([matrix[matrix[index] == i].iloc[:,:-drop_col].to_numpy() for i in list(range(0,index_len))])
                    list_tensor.append(X)
                self.tensor_data = torch.cat(list_tensor, 2)

<<<<<<< HEAD
        print(self.tensor_data.shape)
        self.tensor_data = self.tensor_data.to(self.device)
                
        return self.tensor_data  
        
    
    def _setup_model_guide(self, batch_size: int):
=======
        self.tensor_data = self.tensor_data.to(self.device)
                
        return self.tensor_data  

    def _custom_guide(self,
                    obs: torch.Tensor = None,
                    obs_outcome: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    A2 = None,
                    A3 = None,
                    scale_factor = 1.1,):
        # Plates to match the model's structure
        rank_plate = self._model.get_plate("rank")
        view_plate = self._model.get_plate("view")
        factor1_plate = self._model.get_plate("factor1")
        factor2_plate = self._model.get_plate("factor2")
        factor3_plate = self._model.get_plate("factor3")
    
        tau_loc = pyro.param("tau_loc", torch.tensor(1.0, device=self.device))
        tau_scale = pyro.param("tau_scale", torch.tensor(0.1, device=self.device), constraint=constraints.positive)
        pyro.sample("tau", dist.Normal(tau_loc, tau_scale))
    
        with view_plate:
            view_shrinkage_loc = pyro.param("view_shrinkage_loc", torch.ones(1, device=self.device))
            view_shrinkage_scale = pyro.param("view_shrinkage_scale", torch.ones(1, device=self.device), constraint=constraints.positive)
            pyro.sample("view_shrinkage", dist.Normal(view_shrinkage_loc, view_shrinkage_scale))
    
        with rank_plate:
            with view_plate:
                rank_scale_loc = pyro.param("rank_scale_loc", torch.ones(1, device=self.device))
                rank_scale_scale = pyro.param("rank_scale_scale", torch.ones(1, device=self.device), constraint=constraints.positive)
                pyro.sample("rank_scale", dist.Normal(rank_scale_loc, rank_scale_scale))
        
            lmbda_loc = pyro.param("lmbda_loc", torch.ones(self.R, device=self.device))
            lmbda_scale = pyro.param("lmbda_scale", torch.ones(self.R, device=self.device), constraint=constraints.positive)
            pyro.sample("lmbda", dist.Normal(lmbda_loc, lmbda_scale))
    
            with factor1_plate:
                A1_loc = pyro.param("A1_loc", torch.randn(self.tensor_data.shape[0], self.R, device=self.device))
                A1_scale = pyro.param("A1_scale", torch.ones(self.tensor_data.shape[0], self.R, device=self.device), constraint=constraints.positive)
                pyro.sample("A1", dist.Normal(A1_loc, A1_scale))
        
            with factor2_plate:
                A2_loc = pyro.param("A2_loc", torch.randn(self.tensor_data.shape[1], self.R, device=self.device))
                A2_scale = pyro.param("A2_scale", torch.ones(self.tensor_data.shape[1], self.R, device=self.device), constraint=constraints.positive)
                pyro.sample("A2", dist.Normal(A2_loc, A2_scale))
        
            with factor3_plate:
                A3_loc = pyro.param("A3_loc", torch.randn(self.tensor_data.shape[2], self.R, device=self.device))
                A3_scale = pyro.param("A3_scale", torch.ones(self.tensor_data.shape[2], self.R, device=self.device), constraint=constraints.positive)
                A3 = pyro.sample("A3", dist.Normal(A3_loc, A3_scale))

        with factor2_plate:
            beta_m_loc = pyro.param("beta_m_loc", torch.zeros(1, 1, device=self.device))
            beta_m_scale = pyro.param("beta_m_scale", torch.ones(1, 1, device=self.device), constraint=constraints.positive)
            beta_m = pyro.sample("beta_m", dist.Normal(beta_m_loc, beta_m_scale))

        
        regression_model = RegressionModel2(self.device, A3, beta_m, self.outcome_obs.shape[1]).to(self.device)
        probs = regression_model(self.tensor_data)
        y_dist = dist.OneHotCategorical(probs=probs)
        with factor1_plate:
            with pyro.poutine.scale(scale=self.scale_factor):
                pyro.sample("outcome", y_dist, infer={"enumerate": "parallel"})

        classification_loss = y_dist.log_prob(self.outcome_obs)
        # Note that the negative sign appears because we're adding this term in the guide
        # and the guide log_prob appears in the ELBO as -log q]
        #TODO: self.alpha
        pyro.factor("classification_loss", -0.01 * classification_loss, has_rsample=False)

        
    
    def _setup_model_guide(self, 
                           obs: torch.Tensor = None,
                            obs_outcome: torch.Tensor = None,
                            mask: torch.Tensor = None,
                            A2 = None,
                            A3 = None,
                            scale_factor = 1.1,):
>>>>>>> dev
        """Setup model and guide.

        Parameters
        ----------
        batch_size : int
            Batch size when subsampling

        Returns
        -------
        bool
            Whether the build was successful
        """
        if not self._built:
            self._model = PARAFACModel(
                self.tensor_data,
                self.n_features,
                self.metadata,
                p = self.p,
                R = self.R,
                a = self.a,
                b = self.b,
                c = self.c,
                d = self.d,
                device=self.device,
            )

<<<<<<< HEAD
=======
            self.tensor_data = self.tensor_data

            def generate_values():
                values = {'tau' : torch.tensor([1]).to(self.device),
                    'lmbda' : torch.tensor([1]*self.R).to(self.device)/torch.tensor([1]*self.R).to(self.device),
                    'A1': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[0], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'A2': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[1], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'A3': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[2], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'Y': self.tensor_data.to(self.device)}
                return pyro.infer.autoguide.initialization.init_to_value(values=values)

>>>>>>> dev
            '''
            init_values={'tau' : torch.tensor([1]).to(self.device),
                    'lmbda' : torch.tensor([1]*self.R).to(self.device)/torch.tensor([1]*self.R).to(self.device),
                    'A1': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[0], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'A2': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[1], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'A3': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[2], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'Y': self.tensor_data.to(self.device)}
            '''
            
<<<<<<< HEAD
            self._guide = AutoNormal(self._model.to(self.device),
                                     #init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(values=init_values), 
                                     init_scale=0.01)
=======
            
            self._guide = AutoNormal(self._model,
                                     #init_loc_fn=pyro.infer.autoguide.initialization.init_to_generated(generate=generate_values), 
                                     init_scale=0.01)
            #print("self._model.device")
            #print(self._model.device)

            #self._guide = self._custom_guide(obs,
            #                                obs_outcome,
            #                                mask,
            #                                A2,
            #                                A3,
            #                                scale_factor)
            print(self._guide)
>>>>>>> dev
            
            self._built = True
        return self._built

    def _setup_optimizer(
        self, batch_size: int, n_epochs: int, learning_rate: float, optimizer: str
    ):
        """Setup SVI optimizer.

        Parameters
        ----------
        batch_size : int
            Batch size
        n_epochs : int
            Number of epochs, needed to schedule learning rate decay
        learning_rate : float
            Learning rate
        optimizer : str
            Optimizer as string, 'adam' or 'clipped'

        Returns
        -------
        pyro.optim.PyroOptim
            pyro or torch optimizer object
        """

<<<<<<< HEAD
=======
        print("optimizer")
>>>>>>> dev
        optim = Adam({"lr": learning_rate, "betas": (0.95, 0.999)})
        if optimizer.lower() == "clipped":
            n_iterations = int(n_epochs * (self.n_samples // batch_size))
            logger.info("Decaying learning rate over %s iterations.", n_iterations)
            gamma = 0.1
            lrd = gamma ** (1 / n_iterations)
            optim = ClippedAdam({"lr": learning_rate, "lrd": lrd})

        self._optimizer = optim
        return self._optimizer

    def _setup_svi(
        self,
        optimizer: pyro.optim.PyroOptim,
        n_particles: int,
        scale: bool = True,
    ):
        """Setup stochastic variational inference.

        Parameters
        ----------
        optimizer : pyro.optim.PyroOptim
            pyro or torch optimizer
        n_particles : int
            Number of particles/samples used to form the ELBO (gradient) estimators
        scale : bool, optional
            Whether to scale ELBO by the number of samples, by default True

        Returns
        -------
        pyro.infer.SVI
            pyro SVI object
        """
        scaler = 1.0
        if scale:
            scaler = 1.0 / self.n_samples

<<<<<<< HEAD
        svi = pyro.infer.SVI(
            model=pyro.poutine.scale(self._model, scale=scaler),
            guide=pyro.poutine.scale(self._guide, scale=scaler),
            optim=optimizer,
            loss=pyro.infer.TraceEnum_ELBO(
=======
        #guide = config_enumerate(self._guide, "parallel", expand=True)
        svi = pyro.infer.SVI(
            model=self._model,
            guide=self._guide,
            optim=optimizer,
            loss=pyro.infer.Trace_ELBO(
>>>>>>> dev
                retain_graph=True,
                num_particles=n_particles,
                vectorize_particles=True,
            ),
        )
        self._svi = svi
        return self._svi

    def _setup_training_data(self):
        """Setup training components.
        Convert observations, covariates and prior scales to torch.Tensor and
        extract mask of missing values to mask SVI updates.

        Returns
        -------
        tuple
            Tuple of (obs, mask, covs, prior_scales)
        """
<<<<<<< HEAD
        train_obs = self.tensor_data.to(self.device)
        mask_obs = ~torch.isnan(train_obs)
=======
        print("train_obs")
        train_obs = self.tensor_data
        train_obs = torch.nan_to_num(train_obs)
        print("mask")
        mask_obs = ~torch.isnan(train_obs)
        print("mask done")
>>>>>>> dev
        #TODO: Find a better way to do this
        if self.outcome_obs is not None:
            outcome_obs = self.outcome_obs.to(self.device) 
        else:
            outcome_obs = self.outcome_obs
        # replace all nans with zeros
        # self.presence mask takes care of gradient updates
<<<<<<< HEAD
        train_obs = torch.nan_to_num(train_obs)
        train_obs = train_obs.to(self.device)
        mask_obs = mask_obs.to(self.device)
=======

        '''
        # Assume train_obs has the shape (D1, D2, D3)
        chunk_size = 1000  # Adjust based on available memory
        
        # Create an empty list to hold processed chunks
        processed_chunks = []

        torch.cuda.empty_cache() 
        # Iterate over the first dimension in chunks
        for i in range(0, train_obs.shape[0], chunk_size):
            print(i)
            #print(i)
            chunk = train_obs[i:i + chunk_size]  # Extract a chunk
            #print("chunk")
            with torch.no_grad(): 
                processed_chunk = torch.nan_to_num(chunk.to(self.device))  # Process the chunk
            #print("chunk nan to num")
            processed_chunks.append(processed_chunk)  # Append the processed chunk
            #print("append")

            # Release the chunk to free up memory
            del chunk, processed_chunk
            gc.collect()
            torch.cuda.empty_cache() 
            torch.cuda.synchronize()  # Ensure all ops are done
            print(f"GPU memory after chunk {i}: {torch.cuda.memory_allocated()} bytes")
        
        # Concatenate the processed chunks back into a single tensor
        train_obs = torch.cat(processed_chunks, dim=0)
        '''

        train_obs = train_obs
        mask_obs = mask_obs
>>>>>>> dev

        #train_covs = None
        #if self.covariates is not None:
        #    train_covs = torch.Tensor(self.covariates)

        #train_prior_scales = None
        #if self._informed:
        #    train_prior_scales = torch.cat(
        #        [torch.Tensor(self.prior_scales[vn]) for vn in self.view_names], 1
        #    )

        return train_obs, outcome_obs, mask_obs

    def fit(
        self,
        batch_size: int = 0,
        n_epochs: int = 1000,
        n_particles: int = None,
        learning_rate: float = 0.005,
        optimizer: str = "clipped",
        callbacks = None,
        verbose: bool = True,
        seed: str = None,
    ):
        """Perform inference.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 0 (all samples)
        n_epochs : int, optional
            Number of iterations over the whole dataset,
            by default 1000
        n_particles : int, optional
            Number of particles/samples used to form the ELBO (gradient) estimators,
            by default 1000 // batch_size
        learning_rate : float, optional
            Learning rate, by default 0.005
        optimizer : str, optional
            Optimizer as string, 'adam' or 'clipped', by default "clipped"
        callbacks : List[Callable], optional
            List of callbacks during training, by default None
        verbose : bool, optional
            Whether to log progress, by default 1
        seed : str, optional
            Training seed, by default None

        Returns
        -------
        tuple
            Tuple of (elbo history, whether training stopped early)
        """
        
        #Setup tensor data
        if not isinstance(self.observations, torch.Tensor):
            self._setup_tensor_data()
        else:
            self.tensor_data = self.observations

<<<<<<< HEAD
=======
        (train_obs, outcome_obs, mask_obs) = self._setup_training_data()

>>>>>>> dev
        # if invalid or out of bounds set to n_samples 
        if batch_size is None or not (0 < batch_size <= self.n_samples):
            batch_size = self.n_samples

        if n_particles is None:
            n_particles = max(1, 1000 // batch_size)
        logger.info("Using %s particles on parallel", n_particles)
        logger.info("Preparing model and guide...")
<<<<<<< HEAD
        self._setup_model_guide(batch_size)
=======
        self._setup_model_guide(train_obs, 
                                outcome_obs, 
                                mask_obs,
                                self.A2, 
                                self.A3, 
                                self.scale_factor)
>>>>>>> dev
        logger.info("Preparing optimizer...")
        optimizer = self._setup_optimizer(
            batch_size, n_epochs, learning_rate, optimizer
        )
        logger.info("Preparing SVI...")
        svi = self._setup_svi(optimizer, n_particles, scale=True)
        logger.info("Preparing training data...")
        #(
            #train_obs,
            #mask_obs,
            #train_covs,
            #train_prior_scales,
        #) = self._setup_training_data()
        
<<<<<<< HEAD
        (train_obs, outcome_obs, mask_obs) = self._setup_training_data()
        
        #train_prior_scales = train_prior_scales.to(self.device)

        if batch_size < self.n_samples:
            logger.info("Using batches of size %s.", batch_size)
            tensors = (train_obs, outcome_obs, mask_obs)
=======
        
        
        #train_prior_scales = train_prior_scales.to(self.device)
        if batch_size < self.n_samples:
            logger.info("Using batches of size %s.", batch_size)
            #outcome obs can't be None, must be empty tensor TODO
            tensors = (train_obs, mask_obs)
>>>>>>> dev
            if self.covariates is not None:
                tensors += (train_covs,)
            data_loader = DataLoader(
                TensorDataset(*tensors),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0, #Changed from 1 to 0
                pin_memory=False, #Changed from str(self.device) != "cpu" to False
                drop_last=False,
            )

            def _step():
                iteration_loss = 0
                for _, (sample_idx, tensors) in enumerate(data_loader):
<<<<<<< HEAD
=======
                    #print(sample_idx.shape)
                    #print(x.shape for x in tensors)
>>>>>>> dev
                    iteration_loss += svi.step(
                        sample_idx.to(self.device),
                        *[tensor.to(self.device) for tensor in tensors],
                        #prior_scales=train_prior_scales,
                    )
                return iteration_loss

        else:
            logger.info("Using complete dataset.")
            train_obs = train_obs
            mask_obs = mask_obs
            outcome_obs = outcome_obs
<<<<<<< HEAD
            #print(train_obs.shape)
            #print(mask_obs.shape)
            #print(outcome_obs.shape)

=======
>>>>>>> dev
            #if train_covs is not None:
            #    train_covs = train_covs.to(self.device)

            def _step():
                return svi.step(
<<<<<<< HEAD
                    train_obs, outcome_obs, mask_obs, self.p,
                    self.A2, self.A3, self.scale_factor
=======
                    #self
                    train_obs.to(self.device), 
                    outcome_obs, 
                    mask_obs.to(self.device),
                    self.A2, 
                    self.A3, 
                    self.scale_factor
>>>>>>> dev
                )

        self.seed = seed
        if seed is not None:
            logger.info("Setting training seed to %s", seed)
            pyro.set_rng_seed(seed)
        # clean start
        logger.info("Cleaning parameter store")
        pyro.enable_validation(True)
        pyro.clear_param_store()

        logger.info("Starting training...")
        stop_early = False
        history = []
        pbar = range(n_epochs)
        if verbose > 0:
            pbar = tqdm(pbar)
            window_size = 5
        for epoch_idx in pbar:
            epoch_loss = _step()
            history.append(epoch_loss)
            if verbose > 0:
                if epoch_idx % window_size == 0 or epoch_idx == n_epochs - 1:
                    pbar.set_postfix({"ELBO": epoch_loss})
            if callbacks is not None:
                # TODO: dont really like this, a bit sloppy
                stop_early = any([callback(history) for callback in callbacks])
                if stop_early:
                    break

        return history, stop_early

class GPModel(PyroModule):
    def __init__(self, input_data, output_data, num_classes):
        
        super(GPModel, self).__init__()
        self.kernel = gp.kernels.RBF(input_dim=input_data.shape[0])
        #print("input data shape")
        #print(input_data)
        #print(input_data.shape)
        #print("output data shape")
        #print(output_data)
        #print(output_data.shape)
       
        self.gp_layer = gp.models.VariationalGP(
            input_data,
            output_data,
            kernel=self.kernel,
            likelihood=gp.likelihoods.MultiClass(num_classes=num_classes),
            whiten=True,
            latent_shape=torch.Size([num_classes]),
        )        

    def train(self):
<<<<<<< HEAD
        print("Train GP")
=======
>>>>>>> dev
        loss = gp.util.train(self.gp_layer, num_steps=1000)
        return loss

    def forward(self, x):
        # GP regression
        #print(x.shape)
        #num_steps = 1000
        mean, var = self.gp_layer(x, full_cov=True)
        #y_hat = self.gp_layer.likelihood(mean, var)
    
        return mean, var

<<<<<<< HEAD
class RegressionModel(torch.nn.Module):
=======
class RegressionModel2(torch.nn.Module):
>>>>>>> dev
    def __init__(self, p, device, weight):
        # p = number of features
        self.device = device
        self.weight = weight.to(self.device)
        super(RegressionModel, self).__init__()
        self.non_linear = torch.nn.Softmax(dim=1).to(self.device)
        self.linear = torch.nn.Linear(weight.shape[1], p, bias=True).to(self.device)
        #loc, scale = torch.zeros(p, R), 10 * torch.ones(p, R)

    def forward(self, x):
        out = self.linear(x)
        return self.non_linear(out)

<<<<<<< HEAD
class RegressionModel2(torch.nn.Module):
    def __init__(self, p, R, device):
        super(RegressionModel, self).__init__()
        self.device = device
        
        # Define logistic regression parameters
        self.beta = torch.nn.Parameter(torch.zeros(p, R).to(self.device))
        print("beta shape")
        print(self.beta.shape)
        
    def forward(self, x, W):
        # Compute activation
        out = self.linear(x * self.beta.T * W.T)
        return torch.exp(out)  # Apply exponential transformation
=======
class RegressionModel(torch.nn.Module):
    def __init__(self, device, W, beta_m, num_class):
        # p = number of features
        self.device = device
        self.W = W.to(self.device)
        self.beta_m = beta_m.to(self.device)
        self.out = None
        self.out3 = None
        self.out2 = None
        super(RegressionModel, self).__init__()
        self = self.to(self.device)
        self.non_linear = torch.nn.Softmax(dim=1).to(self.device)
        self.linear = torch.nn.Linear(W.shape[1], num_class, bias=True)
        #loc, scale = torch.zeros(p, R), 10 * torch.ones(p, R)

    def forward(self, x):
        #out = torch.matmul(self.W, x)
        out = torch.einsum('rp,nmp->rmn', [self.W.T.double(), x.double()]).to(self.device)
        #out2 = torch.matmul(self.beta_m, out)
        torch.cuda.empty_cache()
        out2 = torch.einsum('mo,rmn->rno', [self.beta_m.double(), out]).to(self.device)
        out2 = torch.squeeze(out2, 2).T
        self.out2 = out2
        self.out = out
        out3 = self.linear(out2.float()).to(self.device)
        self.out3 = out3
        res =  self.non_linear(out3).to(self.device)
        return res
>>>>>>> dev

def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(torch.nn.Linear(in_dim, out_dim))
        layers.append(torch.nn.BatchNorm1d(out_dim))
        layers.append(torch.nn.ReLU())
    return nn.Sequential(*layers[:-1])

# Used in parameterizing q(y | z2)
class Classifier(torch.nn.Module):
    def __init__(self, z2_dim, hidden_dims, num_labels):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_labels]
        self.fc = make_fc(dims)

    def forward(self, x):
        logits = self.fc(x)
        return logits


class PARAFACModel(PyroModule):
    def __init__(
        self,
        tensor_data, 
        n_features,
        metadata,
<<<<<<< HEAD
        p = None,
=======
        p = 1,
>>>>>>> dev
        R: int = 10, 
        a: int = 1, 
        b: int = 0.5, 
        c: int = 1, 
        d: int = 0.5,
        device: bool = None
    ):
<<<<<<< HEAD
        """MuVI generative model.
=======
        """PARAFAC generative model.
>>>>>>> dev

        Parameters
        ----------
        tensor_data : 3-d tensor
            Incomplete tensor to infer
        R : int
            Tensor rank
        a : int
            concentration of gamma distribution for Tau
        b : int
            rate of gamma distribution for Tau
        c : int
            concentration of gamma distribution for Lambda
        d : int
            rate of gamma distribution for Lamba
        """
        super().__init__(name="PARAFACModel")
        self.tensor_data = tensor_data
        self.tensor_size = list(tensor_data.shape)
        self.metadata = metadata
        self.n_features = n_features
        self.feature_offsets = [0] + np.cumsum(self.n_features).tolist()
        self.n_views = len(self.n_features)
<<<<<<< HEAD
=======
        self.RegressionModel = None
>>>>>>> dev
        self.R = R
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.p = p
<<<<<<< HEAD
    
        self.device = device
=======
        self.Ws = []
        self.beta_ms = []
        self.A3s = []
        self.output_dict = {}
    
        self.device = device
        self.to(self.device)
>>>>>>> dev

    def get_plate(self, name: str, **kwargs):
        """Get the sampling plate.

        Parameters
        ----------
        name : str
            Name of the plate

        Returns
        -------
        PlateMessenger
            A pyro plate.
        """
        plate_kwargs = {
<<<<<<< HEAD
            "rank": {"name": "rank", "size": self.R, "dim": -1},
            "view": {"name": "view", "size": self.n_views, "dim": -2},
            "factor1": {"name": "sample", "size": self.tensor_size[0]},
            "factor2": {"name": "factor", "size": self.tensor_size[1]},
            "factor3": {"name": "feature", "size": self.tensor_size[2]},
=======
            "rank": {"name": "R", "size": self.R, "dim": -1},
            "view": {"name": "view", "size": self.n_views, "dim": -2},
            "factor1": {"name": "samples", "size": self.tensor_size[0]},
            "factor2": {"name": "slices", "size": self.tensor_size[1]},
            "factor3": {"name": "features", "size": self.tensor_size[2]},
>>>>>>> dev
            "class": {"name": "class", "size": self.p}
        }
        #print("self tensor size")
        #print(self.tensor_size[0])
        return pyro.plate(device= self.device, **{**plate_kwargs[name], **kwargs})

    @config_enumerate
    def forward(
        self,
        obs: torch.Tensor = None,
        obs_outcome: torch.Tensor = None,
        mask: torch.Tensor = None,
<<<<<<< HEAD
        p = None,
=======
>>>>>>> dev
        A2 = None,
        A3 = None,
        scale_factor = 1.1,
    ):
        """Generate samples.

        Parameters
        ----------
        obs : torch.Tensor
            Observations to condition the model on
        obs : torch.Tensor
            Observations to condition the model on
        mask : torch.Tensor
            Binary mask of missing data
            
        Returns
        -------
        dict
            Samples from each sampling site
        """

<<<<<<< HEAD
        output_dict = {}
=======
        #print(obs.shape)
        
>>>>>>> dev
        
        rank_plate = self.get_plate("rank")
        view_plate = self.get_plate("view")
        factor1_plate = self.get_plate("factor1")
        factor2_plate = self.get_plate("factor2")
        factor3_plate = self.get_plate("factor3")
        class_plate = self.get_plate("class")

<<<<<<< HEAD
        tau = pyro.sample("tau", dist.Gamma(self.a, self.b))
        output_dict["tau"] = tau.to(self.device)
=======
        tau = pyro.sample("tau", dist.InverseGamma(self.a, self.b))
        self.output_dict["tau"] = tau.to(self.device)
>>>>>>> dev

        
        # Check memory usage
        #allocated_bytes = torch.cuda.memory_allocated()
        #allocated_mb = allocated_bytes / (1024 * 1024)
        #print("Memory allocated:MB")
        #print(allocated_mb)
        #TODO: If cuda is device
        #torch.cuda.empty_cache()
        
        with view_plate:
            view_shrinkage = pyro.sample("view_shrinkage", dist.HalfCauchy(torch.ones(1, device=self.device)))
<<<<<<< HEAD
            output_dict["view_shrinkage"] = view_shrinkage.to(self.device)
=======
            self.output_dict["view_shrinkage"] = view_shrinkage.to(self.device)
>>>>>>> dev
        
        with rank_plate:
            with view_plate:
                rank_scale = pyro.sample("rank_scale", dist.HalfCauchy(torch.ones(1, device=self.device)))
<<<<<<< HEAD
                output_dict["rank_scale"] = rank_scale.to(self.device)
            lmbda = pyro.sample("lmbda", dist.Gamma(self.c, self.d))
            output_dict["lmbda"] = lmbda.to(self.device)
            #beta = pyro.sample("beta", dist.Normal(torch.zeros(1), torch.ones(1)))
           
            with factor1_plate:
                a = pyro.sample("A1", dist.Normal(torch.zeros(self.R, device=self.device), output_dict["lmbda"]))
                output_dict["A1"] = a.to(self.device)

            with factor2_plate:
                if(A2 != None):
                    output_dict["A2"] = pyro.deterministic("A2", A2).to(self.device)
                else:
                    output_dict["A2"] = pyro.sample("A2", dist.Normal(torch.zeros(self.R, device=self.device), output_dict["lmbda"])).to(self.device)
                
            with factor3_plate:
                if(A3 != None):
                    output_dict["A3"] = pyro.deterministic("A3", A3).to(self.device)
                else:
                    local_scale = pyro.sample("local_scale", dist.HalfCauchy(torch.ones(1, device =  self.device)))
                    output_dict["local_scale"] = local_scale.to(self.device)
                    var = torch.cat(
                        [
                            (
                                output_dict["lmbda"] / 
                                (output_dict["view_shrinkage"][i] 
                                 * output_dict["rank_scale"][i] 
                                 * output_dict["local_scale"][
=======
                self.output_dict["rank_scale"] = rank_scale.to(self.device)
            lmbda = pyro.sample("lmbda", dist.InverseGamma(self.c, self.d))
            #lmbda_old = pyro.sample("lmbda_old", dist.Gamma(self.c, self.d))
            self.output_dict["lmbda"] = lmbda.to(self.device)
            #beta = pyro.sample("beta", dist.Normal(torch.zeros(1), torch.ones(1)))
           
            with factor1_plate:
                a = pyro.sample("A1", dist.Normal(torch.zeros(self.R, device=self.device), torch.ones(1, device=self.device)))
                self.output_dict["A1"] = a.to(self.device)

            with factor2_plate:
                if(A2 != None):
                    self.output_dict["A2"] = pyro.deterministic("A2", A2).to(self.device)
                else:
                    self.output_dict["A2"] = pyro.sample("A2", dist.Normal(torch.zeros(self.R, device=self.device), torch.ones(1, device=self.device))).to(self.device)
                
            with factor3_plate:
                if(A3 != None):
                    self.output_dict["A3"] = pyro.deterministic("A3", A3).to(self.device)
                else:
                    local_scale = pyro.sample("local_scale", dist.HalfCauchy(torch.ones(1, device =  self.device)))
                    self.output_dict["local_scale"] = local_scale.to(self.device)
                    var = torch.cat(
                        [
                            (
                                self.output_dict["lmbda"] / 
                                (self.output_dict["view_shrinkage"][i] 
                                 * self.output_dict["rank_scale"][i] 
                                 * self.output_dict["local_scale"][
>>>>>>> dev
                                    self.feature_offsets[i] : self.feature_offsets[i + 1]
                                 ])
                            )
                            for i in range(self.n_views)
                        ],
                        0,
                    )    
                    #print(var.shape)
<<<<<<< HEAD
                    output_dict["A3"] = pyro.sample("A3", dist.Normal(torch.zeros(self.R, device=self.device), var)).to(self.device)


        #TODO:Class Attribute!!
        p = self.p
        
=======
                    self.output_dict["A3"] = pyro.sample("A3", dist.Normal(torch.zeros(self.R, device=self.device), var)).to(self.device)


        
        #TODO:Class Attribute!!
        p = self.p
        '''
>>>>>>> dev
        with rank_plate:
            with class_plate:
                #self.R
                loc, scale = torch.zeros(p, self.R), 10 * torch.ones(p, self.R)
                weight =  pyro.sample("w_prior", dist.Normal(loc, scale)).to(self.device)


        with factor1_plate:
            with pyro.poutine.scale(scale=scale_factor):
                # Create unit normal priors over the parameters
                regression_model = RegressionModel(p, self.device, weight)
<<<<<<< HEAD
                probs = regression_model(output_dict["A1"])
                output_dict["outcome"] = pyro.sample("outcome",
                                                     dist.OneHotCategorical(probs=probs).to_event(1),
                                                     obs=obs_outcome)
        
        if(mask is None):
            output_dict["Y"] = pyro.sample(
                "Y",
                dist.Normal(
                    torch.einsum("ir,jr,kr->ijk", output_dict["A1"], output_dict["A2"], output_dict["A3"]),
                    1 / torch.sqrt(output_dict["tau"]),
=======
                probs = regression_model(self.output_dict["A1"])
                self.output_dict["outcome"] = pyro.sample("outcome",
                                                     dist.OneHotCategorical(probs=probs).to_event(1),
                                                     obs=obs_outcome)
                                                     
        '''

        
        '''
        with factor2_plate:
            loc, scale = torch.zeros(1, 1, device=self.device), 10 * torch.ones(1, 1, device=self.device)
            self.output_dict["beta_m"] =  pyro.sample("beta_m", dist.Normal(loc, scale)).to(self.device)

        
        with factor1_plate:
            with pyro.poutine.scale(scale=scale_factor):
                # Create unit normal priors over the parameters
                # Maynbe ok with p ?
                regression_model = RegressionModel(self.device, self.output_dict["A3"], self.output_dict["beta_m"], p).to(self.device)
                #regression_model = RegressionModel(self.device, self.output_dict["A3"], self.output_dict["beta_m"], obs_outcome.shape[1]).to(self.device)
                self.RegressionModel = regression_model.to(self.device)
                probs = regression_model(obs)
                y_dist = dist.OneHotCategorical(probs=probs)
                self.output_dict["outcome"] = pyro.sample("outcome",
                                                     y_dist.to_event(1),
                                                     infer={"enumerate": "parallel"},
                                                     obs=obs_outcome).to(self.device)
                self.Ws.append(torch.var_mean(self.RegressionModel.W))
                self.beta_ms.append(self.RegressionModel.beta_m)
                self.A3s.append(self.output_dict["A3"])

        
                classification_loss = y_dist.to_event(1).log_prob(obs_outcome)
                classification_loss = classification_loss.to(self.device)
                # Note that the negative sign appears because we're adding this term in the guide
                # and the guide log_prob appears in the ELBO as -log q]
                #TODO: self.alpha
                pyro.factor("classification_loss", -10 * classification_loss, has_rsample=False)
        '''

    
        
        if(mask is None):
            print("No mask")
            self.output_dict["Y"] = pyro.sample(
                "Y",
                dist.Normal(
                    torch.einsum("ir,jr,kr->ijk", self.output_dict["A1"], self.output_dict["A2"], self.output_dict["A3"]),
                    1 / torch.sqrt(self.output_dict["tau"]),
>>>>>>> dev
                ).to_event(3),
                obs=obs,
                infer={"is_auxiliary": True},
            ).to(self.device)
        else:
<<<<<<< HEAD
            with pyro.poutine.scale(scale=(1/scale_factor)):
                output_dict["Y"] = pyro.sample(
                    "Y",
                    dist.Normal(
                        torch.einsum("ir,jr,kr->ijk", output_dict["A1"], output_dict["A2"], output_dict["A3"]),
                        1 / torch.sqrt(output_dict["tau"]),
                    ).mask(mask).to_event(3),
                    obs=obs,
                    infer={"is_auxiliary": True},
                ).to(self.device)
                
        return output_dict
=======
            mask = mask.int()
            #with pyro.poutine.scale(scale=(1/scale_factor)):
            self.output_dict["Y"] = pyro.sample(
                "Y",
                dist.Normal(
                    torch.einsum("ir,jr,kr->ijk", self.output_dict["A1"], self.output_dict["A2"], self.output_dict["A3"]),
                    1 / torch.sqrt(self.output_dict["tau"]),
                ).mask(mask).to_event(3),
                obs=obs,
                infer={"is_auxiliary": True},
            ).to(self.device)

        self = self.to(self.device)
                
        return self.output_dict
>>>>>>> dev
