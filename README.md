# Variational Diffusion Autoencoder

Implementation of [Variational Diffusion Auto-encoder: Latent Space Extraction from Pre-trained Diffusion Models (Batzolis++23)](https://arxiv.org/pdf/2304.12141) in `jax` and `equinox`.

The idea here is to remedy the assumption a traditional variational autoencoder (VAE) on the reconstruction likelihood $p(\boldsymbol{x}|\boldsymbol{z})$ assumption by building the likelihood of the data given the latent code out of the sum of the scores of the marginal likelihood of the data $\nabla_{\boldsymbol{x}}p_\theta(\boldsymbol{x})$ (a pre-trained diffusion model) and a variational posterior $\nabla_{\boldsymbol{x}_t}q_{\phi}(\boldsymbol{z}|\boldsymbol{x}_t, t)$. The posterior is modelled by a Gaussian ansatz parameterised with a mean and diagonal covariance, but is a function of the diffusion time.

![alt text](figs/fig.png?raw=true)