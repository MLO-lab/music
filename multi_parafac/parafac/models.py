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
        p = None,
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

        self.covariates = covariates
        self.view_names = view_names
        self.observations = observations
        self.metadata = metadata
        self.index = index
        self.n_samples = n_samples
        print("device arg")
        print(device)
        self.device = device
        if(device == None):
            self.device = torch.device("cpu")
            if use_gpu and torch.cuda.is_available():
                logger.info("GPU available, running all computations on the GPU.")
                self.device = f"cuda:{get_free_gpu_idx()}"
                #self.device="cuda:0"
        print("self.device")
        print(self.device)
        self.to(self.device)

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

        print(self.tensor_data.shape)
        self.tensor_data = self.tensor_data.to(self.device)
                
        return self.tensor_data  
        
    
    def _setup_model_guide(self, batch_size: int):
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

            '''
            init_values={'tau' : torch.tensor([1]).to(self.device),
                    'lmbda' : torch.tensor([1]*self.R).to(self.device)/torch.tensor([1]*self.R).to(self.device),
                    'A1': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[0], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'A2': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[1], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'A3': torch.distributions.MultivariateNormal(torch.zeros(self.tensor_data.shape[2], self.R).to(self.device), torch.eye(self.R).to(self.device)).sample(),
                    'Y': self.tensor_data.to(self.device)}
            '''
            
            self._guide = AutoNormal(self._model.to(self.device),
                                     #init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(values=init_values), 
                                     init_scale=0.01)
            
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

        svi = pyro.infer.SVI(
            model=pyro.poutine.scale(self._model, scale=scaler),
            guide=pyro.poutine.scale(self._guide, scale=scaler),
            optim=optimizer,
            loss=pyro.infer.TraceEnum_ELBO(
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
        train_obs = self.tensor_data.to(self.device)
        mask_obs = ~torch.isnan(train_obs)
        #TODO: Find a better way to do this
        if self.outcome_obs is not None:
            outcome_obs = self.outcome_obs.to(self.device) 
        else:
            outcome_obs = self.outcome_obs
        # replace all nans with zeros
        # self.presence mask takes care of gradient updates
        train_obs = torch.nan_to_num(train_obs)
        train_obs = train_obs.to(self.device)
        mask_obs = mask_obs.to(self.device)

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

        # if invalid or out of bounds set to n_samples 
        if batch_size is None or not (0 < batch_size <= self.n_samples):
            batch_size = self.n_samples

        if n_particles is None:
            n_particles = max(1, 1000 // batch_size)
        logger.info("Using %s particles on parallel", n_particles)
        logger.info("Preparing model and guide...")
        self._setup_model_guide(batch_size)
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
        
        (train_obs, outcome_obs, mask_obs) = self._setup_training_data()
        
        #train_prior_scales = train_prior_scales.to(self.device)

        if batch_size < self.n_samples:
            logger.info("Using batches of size %s.", batch_size)
            tensors = (train_obs, outcome_obs, mask_obs)
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
            #print(train_obs.shape)
            #print(mask_obs.shape)
            #print(outcome_obs.shape)

            #if train_covs is not None:
            #    train_covs = train_covs.to(self.device)

            def _step():
                return svi.step(
                    train_obs, outcome_obs, mask_obs, self.p,
                    self.A2, self.A3, self.scale_factor
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
        print("Train GP")
        loss = gp.util.train(self.gp_layer, num_steps=1000)
        return loss

    def forward(self, x):
        # GP regression
        #print(x.shape)
        #num_steps = 1000
        mean, var = self.gp_layer(x, full_cov=True)
        #y_hat = self.gp_layer.likelihood(mean, var)
    
        return mean, var

class RegressionModel(torch.nn.Module):
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
        p = None,
        R: int = 10, 
        a: int = 1, 
        b: int = 0.5, 
        c: int = 1, 
        d: int = 0.5,
        device: bool = None
    ):
        """MuVI generative model.

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
        self.R = R
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.p = p
    
        self.device = device

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
            "rank": {"name": "rank", "size": self.R, "dim": -1},
            "view": {"name": "view", "size": self.n_views, "dim": -2},
            "factor1": {"name": "sample", "size": self.tensor_size[0]},
            "factor2": {"name": "factor", "size": self.tensor_size[1]},
            "factor3": {"name": "feature", "size": self.tensor_size[2]},
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
        p = None,
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

        output_dict = {}
        
        rank_plate = self.get_plate("rank")
        view_plate = self.get_plate("view")
        factor1_plate = self.get_plate("factor1")
        factor2_plate = self.get_plate("factor2")
        factor3_plate = self.get_plate("factor3")
        class_plate = self.get_plate("class")

        tau = pyro.sample("tau", dist.Gamma(self.a, self.b))
        output_dict["tau"] = tau.to(self.device)

        
        # Check memory usage
        #allocated_bytes = torch.cuda.memory_allocated()
        #allocated_mb = allocated_bytes / (1024 * 1024)
        #print("Memory allocated:MB")
        #print(allocated_mb)
        #TODO: If cuda is device
        #torch.cuda.empty_cache()
        
        with view_plate:
            view_shrinkage = pyro.sample("view_shrinkage", dist.HalfCauchy(torch.ones(1, device=self.device)))
            output_dict["view_shrinkage"] = view_shrinkage.to(self.device)
        
        with rank_plate:
            with view_plate:
                rank_scale = pyro.sample("rank_scale", dist.HalfCauchy(torch.ones(1, device=self.device)))
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
                                    self.feature_offsets[i] : self.feature_offsets[i + 1]
                                 ])
                            )
                            for i in range(self.n_views)
                        ],
                        0,
                    )    
                    #print(var.shape)
                    output_dict["A3"] = pyro.sample("A3", dist.Normal(torch.zeros(self.R, device=self.device), var)).to(self.device)


        #TODO:Class Attribute!!
        p = self.p
        
        with rank_plate:
            with class_plate:
                #self.R
                loc, scale = torch.zeros(p, self.R), 10 * torch.ones(p, self.R)
                weight =  pyro.sample("w_prior", dist.Normal(loc, scale)).to(self.device)


        with factor1_plate:
            with pyro.poutine.scale(scale=scale_factor):
                # Create unit normal priors over the parameters
                regression_model = RegressionModel(p, self.device, weight)
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
                ).to_event(3),
                obs=obs,
                infer={"is_auxiliary": True},
            ).to(self.device)
        else:
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
