"""Data module."""
from parafac.gpu import get_free_gpu_idx

import itertools
import logging
import math
from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np


logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 12,
        n_drugs: int = 5,
        R: int = 10,
        a: float = 1,
        b: float = 0.5,
        use_gpu = True,
        device: str = None
    ) -> None:
        """Generate synthetic data

        Parameters
        ----------
        n_samples : int, optional
            Number of samples, by default 1000
        n_features : int, optional
            Number of features for each view, by default 12
        n_drugs : int, optional
            Number of drugs for each view, by default 5            
        """

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_drugs = n_drugs
        
        self.R = R
        self.a = a
        self.b = b
        
        self.device = device
        if(device == None):
            self.device = torch.device("cpu")
            if use_gpu and torch.cuda.is_available():
                logger.info("GPU available, running all computations on the GPU.")
                self.device = f"cuda:{get_free_gpu_idx()}"
        
        self.A1 = None
        self.A2 = None
        self.A3 = None
        self.Y = None

    def generate(
        self, seed: int = 0, cluster_std: int = 1, center_box: bool = False
    ) -> None:
        
        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)
        
        # We create three sparse matrices that serve as our factor matrices
        cov = torch.diag(torch.tensor([self.a]*self.R)/torch.tensor([self.b]*self.R))
        zeros = torch.zeros([self.R])
<<<<<<< HEAD
        A1 = torch.Tensor(rng.multivariate_normal(zeros, cov, size = self.n_drugs))
        A2 = torch.Tensor(rng.multivariate_normal(zeros, cov, size = self.n_samples))
=======
        A1 = torch.Tensor(rng.multivariate_normal(zeros, cov, size = self.n_samples))
        A2 = torch.Tensor(rng.multivariate_normal(zeros, cov, size = self.n_drugs))
>>>>>>> dev
        A3 = torch.Tensor(rng.multivariate_normal(zeros, cov, size = self.n_features))
                
        # We create the tensor from these matrices, and add noise. The noisy tensor is the one we want to learn.
        new_tensor =  torch.einsum('ir,jr,kr->ijk', A1, A2, A3)
        X_df =  torch.distributions.Normal(new_tensor, 1/torch.sqrt(torch.Tensor([1]))).sample()
        
        self.A1 = A1.to(self.device)
        self.A2 = A2.to(self.device)
        self.A3 = A3.to(self.device)
        self.Y = X_df.to(self.device)
        
        return rng
    
    def get_sim_data(self):
        sim_data = {'A1_sim' : self.A1,
                   'A2_sim' : self.A2,
                   'A3_sim' : self.A3,
                   'Y_sim' : self.Y}
        
        
        return sim_data

 
    def generate_missingness(
        self,
        p: float = 0.1,
        seed=None,
    ):
        
        Y = F.dropout(self.Y, p=p)
        Y[Y == 0] = np.nan
        self.Y = Y