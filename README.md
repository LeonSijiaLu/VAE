# VAE
Variational Autoencoder (VAE)

## Autoencoder
The plain and traditional autoencoder is similar to the file compression algorithm, where we zip and unzip a file.
The difference is that the file compression algorithm would ensure files before and after zipping are the same. However, autoencoder will suffer some losses in the network.

In an autoencoder, we encode some data, then decode as best as we can, as shown below. 
* cat → [1, 2, 3 …]
* dog → [1, 2.1, 3.2 …]

There are some problems:
1.  For example, cat and dog could have similar results. When we decode, we could decode to a dog, when a cat is expected.
2. In other words, we need to capture `Semantic Relationship` between data, we want data to make sense

## Variational Autoencoder
Variational Autoencoder (VAE) is a type of generative model with the goal of learning the latent structure in data and generating similar samples. Given a set of observed variable data x, VAE aims to learn the distribution of a latent variable z, such that we can generate approximate data x from z.
* Encoder: It takes an input `x`, and covert it to latent representation `z`. `z` is a distribution, usually a Gaussian distribution.
* Decoder: It samples a point from the distribution `z`, and attempts to reconstruct the original input as accurately as possible.

The overall concept of VAE is relatively easy to understand, with the two key components being the reparameterization and the loss function. And we will dive into them in details.

### The Reparameterization Trick
Backpropagation allows us to adjust the network’s internal parameters based on the difference between the predicted output and the actual output. However, it struggles when dealing with random sampling, which is inherent in VAEs.

Encoder generates a distribution `z` which the decoder will sample from. Then we can describe `z` as `z = random_sample(μ,σ)`, as you may have noticed, it is a random sample.

#### Why is random sampling non-differentiable?
The main reason lies in the nature of randomness: when we sample from a distribution (such as a Gaussian distribution), the result z is determined by a random process and is not a continuous, deterministic function of the parameters μ and σ. Randomness introduces uncontrollable fluctuations, so it's impossible to compute the partial derivatives through a deterministic mathematical expression.

For example, suppose we sample and get z1, and the next time we might get a different z2. These values are random and not differentiable functions of μ and σ, making it impossible to accurately calculate their partial derivatives with respect to μ or σ. Therefore, during the sampling step, the gradient calculation is 'broken' and cannot be properly propagated to μ and σ."

#### How to solve?
In order to allow backpropagation to work properly during the random sampling step, VAE uses the reparameterization trick. The core idea is to reformulate the sampling step in a differentiable way.

The specific approach is:

z=μ+σ⋅ϵ

where ϵ is a random variable sampled from the standard normal distribution N(0,1). The new formula is also a linear function, meaning it is differentiable. This means we can compute the partial derivatives of the loss function with respect to μ and σ, allowing backpropagation to proceed as usual.