# Variational Diffusion Autoencoder

Implementation of [Variational Diffusion Auto-encoder: Latent Space Extraction from Pre-trained Diffusion Models (Batzolis++23)](https://arxiv.org/pdf/2304.12141) in `jax` and `equinox`.

The idea here is to remedy the assumption a traditional variational autoencoder (VAE) on the reconstruction likelihood $p(\boldsymbol{x}|\boldsymbol{z})$ assumption by building the likelihood of the data given a latent encoding with a diffusion model and encoding model.

In practice, this likelihood of the data given a latent code consists of the sum of the scores of the marginal likelihood of the data $\nabla_{\boldsymbol{x}}p_\theta(\boldsymbol{x})$ (a pre-trained diffusion model) and a variational posterior $\nabla_{\boldsymbol{x}(t)}q_{\phi}(\boldsymbol{z}|\boldsymbol{x}(t), t)\approx \nabla_{\boldsymbol{x}(t)} p(\boldsymbol{z}|\boldsymbol{x})$. This approach escapes modelling the intractable evidence of the data. 

The variational posterior is modelled by a Gaussian ansatz parameterised with a mean and diagonal covariance, but is a function of the diffusion time. This approach separates the uses the same VAE objective comprising the reconstruction loss and variational posterior KL, but the gradients only adjust the encoder - this improves the training dynamics of the traditional VAE by forming the VAE objective with a pretrained diffusion model and a varational posterior, only the latter of which is optimised. This also allows for the extraction of a latent space from existing generative models.

The corrector model $c_\psi$ is not implemented as the authors find the variational approximation $q_\phi$ is accurate enough.

![alt text](figs/fig.png?raw=true)