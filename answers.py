r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch

# ==============
# Part 1 answers


def part1_pg_hyperparams():
    # hp = dict(
    #     batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    # )
    hp = dict(batch_size=64, gamma=0.99, beta=0.75, learn_rate=3.3e-3, eps=1e-8, num_workers=0)
    # hp = dict(batch_size=32, gamma=0.985, beta=0.5, learn_rate=2e-3, eps=1e-8, num_workers=2)
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
      gamma=0.963,
      beta=0.72,
      delta=0.9,
      learn_rate=1e-3,
      eps=1e-8,
      num_workers=0,
      )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

When subtracting a baseline, we don't measure a trajectory's worth by it's total reward, but by how much better that total reward is relative to some expected ("baseline") reward value. This helps to reduce the variance of the model by giving a priority to better actions. 
If we dont substract a baseline, the probability of all actions that will achieve positive reward will increase, even if they are "bad". 
With the baseline subtraction, the "bad" actions reward (which is positive but low ) will become negative and their probability will not increase.  <br>
For example, if we have 3 trajectories, $t_1$, $t_2$, $t_3$ with rewards $r_1 = 100$, $r_2 = 49$, $r_3 = 45$. although all rewards are positive, $t_2, t_3$ are "bad" and we will want to increase the probability of $t_1$. If we subtract some baseline (for example $b=50$), $t_1$ probability will be ranked the highest.
"""


part1_q2 = r"""
**Answer**: <br>
As weve seen (in this notebook) 

$$
\begin{align}
v_{\pi}(s) &= \E{g(\tau)|s_0 = s,\pi} \\
q_{\pi}(s,a) &= \E{g(\tau)|s_0 = s,a_0=a,\pi}.
\end{align}
$$

Since the estimated q-values provide an estimation for $$q_{\pi}(a,s)$$, if we sample enough times, we can use them as regression targets to get an approximation for the expectation. """

part1_q3 = r"""
**Answer**: <br>
1. <br> -  *loss_p*: The graphs for cpg and bpg are almost constant and near 0, since in both configurations we subtract the baseline in the policy gradients. all other experiments graphs increase as we go through more episodes, and converges near 0 (or somewhat above). <br> -  *baseline*: both graphs are very similar, we can see that as we go through more episodes the baseline increases. We can also see that the baseline increases as the mean reward increases (not exactly the same but there is some correlation). this makes sense because as the rewards increases, so is the baseline.  <br>  -  *loss_e*: We can see that the cpg and epg are very close to each other while the cpg is better. <br> -  *mean reward*: We can see that the bpg model achieved the best results, and that the cpg results are very close. the other models (epg, vpg) did not perform as well.  <br>

2. The AAC method perform slightly better than the cpg. In the loss_p, we can see that the AAC reaches 0 quite fast, but is not very stable and goes up and down (in small steps), while the CPG is almost constant at 0 from the beginning. <br>
In the loss_e graph we can see that the AAC loss is higher than CPG from the beginning. <br>
In the mean reward graph, the AAC achieved the best rewards, but it is only slightly better than the CPG.
"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=6, h_dim=512, z_dim=16, x_sigma2=5e-4, learn_rate=1e-3, betas=(0.9, 0.999),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r""" 
**Answer**: <br>
$\sigma^2$ is the variance of the parametric likelihood distribution used by the decoder. 
$p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$. <br>
When this parameter has high values, we allow more randomness when going back from the latent space to the instance space. When its low, there will be less randomness and we will get closer to identity mapping. """

part2_q2 = r"""

**answer**: <br>
> 1.  The KL divergence term of the loss can be interpreted as the information gained by using the posterior $q(\bb{Z|X})$ instead of the prior distribution $p(\bb{Z})$. This loss encourages the encoder to distribute all encodings evenly around the center of the latent space, and penalize it when its not distributes evenly. <br> 
The reconstruction loss measures how much the reconstructed image is close to the original output, and penalizes the network for creating outputs different from the input. <br>

>2.  As explained, the KL divergence term can be interpreted as the information gained by using the posterior distribution. when this loss is low, the latent space distribution will become closer to normal distribution. <br> This is a sort of regularization term which will take the distribution returned from the encoder and make it closer to normal distribution, with the information of the encoder. 

>3. Since this is a sort of regularization, it help to prevent over fit. That means that the generated images will be more diverse, instead of generating very similar outputs. <br>

"""

part2_q3 = r"""
**answer**: <br>
The evidence distribution $p(\bb{X})$ represent the likelihood of a data example x under our moder. If we maximize this distribution, it means that it is likely that a data example x was generated from our model.  
"""

part2_q4 = r"""
**answer:** <br>
The values for $\sigma^2$ can be very small and in close range, using log scale will help to numerically stabalize the model and enlarge the value range of sigma (while maintaining the values to be larger than 0, as a variance should be)

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    # hypers = dict(
    #     batch_size=4,
    #     z_dim=128,
    #     data_label=1,
    #     label_noise=0.2,
    #     discriminator_optimizer=dict(
    #         type="Adam",  # Any name in nn.optim like SGD, Adam
    #         lr=3e-3,
    #         betas=(0.5,0.999)
    #         # You an add extra args for the optimizer here
    #     ),
    #     generator_optimizer=dict(
    #         type="Adam",  # Any name in nn.optim like SGD, Adam
    #         lr=3e-3,
    #         betas=(0.5,0.999)
    #         # You an add extra args for the optimizer here
    #     ),
    # )
    hypers = dict(
    batch_size=4,
    z_dim=112,
    data_label=1,
    label_noise=0.2,
    discriminator_optimizer=dict(
        type="Adam",  # Any name in nn.optim like SGD, Adam
        lr=2e-3,
        betas=(0.5,0.999)
        # You an add extra args for the optimizer here
    ),
    generator_optimizer=dict(
        type="Adam",  # Any name in nn.optim like SGD, Adam
        lr=2e-3,
        betas=(0.5,0.999)
        # You an add extra args for the optimizer here
    ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**answer**: <br>
When training a GAN we train its 2 parts seperatly- the discriminator and the generator, in both of the training processes we need to sample outputs.
The first step is training the discriminator, we use sampling from the generator to give the discriminator "fake" examples and train it to discriminate between real and generated data. In that case, we do not need to maintain the gradients when sampling, because we update only the discriminator parameters.  (A further explanation on why we do not want to backpropagate into the generator when training the discriminator can be found in notebook 4 - Unsupervised learning - question 2B (the second question 2)).  <br>
The second step is training the generator, in order to update the generator parameters so it will learn to generate better samples that will be able to "fool" the discriminator, we need to maintain the gradients for this part. 

"""
 

part3_q2 = r"""
When training a GAN to generate images, should we decide to stop training solely based on the fact that the Generator loss is below some threshold? Why or why not?

What does it mean if the discriminator loss remains at a constant value while the generator loss decreases?

**answer**: <br>
1. When training a Gan to generate image we should **Not** decide to stop training solely based on low generator loss. The generator loss depends on the discriminator performance, and can be low when the discriminator perform poorly. i.e, if the discriminator cant discriminate between a real image and some random noise, the generator will achieve a low loss even if the generated image are not at all close to the real data. <br>
We should stop training if the Generator loss is below some threshold **And** the discriminator loss is also below another threshold (regarding the loss we have implemented which we try to minimize). <br>


2. If the discriminator loss remains constant while the generator loss decreases it can mean that both of them are improving. Obviously the generator keeps improving, since its loss is decreasing. If the discriminator loss remains constant, it means that it is "keeping up" with the generator, and also improves in some way. If the discriminator loss would have changes instead of staying constant, it could have meant that it stopped learning (and generator improvement affect the discriminator).
"""

part3_q3 = r"""
**answer**: <br>
First, we can see that the GAN results weve got are much better than the VAE results. 
There are a few main differences between the generated images:
- The VAE images more blurred than the GAN images. <br>
- The GAN  images are more diversed than the VAE images, which are very simmilar to one another. <br>
<br>
The main differences between the model that causing it:
- VAE sample from a learned distribution, while the GAN's generator tries to fool the discriminator into "thinking" that the generated images are real. 
Discriminating between blurred and sharp images is quite simple and a classifier is likely to learn this differnces, so if the generator will generate blurred images, it wont be able to fool the discriminator. On the other hand, when sampling from some distribution, it is likely that there will be some "noise", which will cause the generated images to be blurred. <br>
- As for the diversity, VAE model tries to get an approximation of the distribution from which the input data is taken, when it succeeds, the generated images will be sampled from one distribution and so will be similar to the input images and to each other. The GAN model does not learns the input images directly and just try to create similar images for the discriminator, which results in more diverse outputs.
"""

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    
    x, w, b = ctx.saved_tensors
    dw = 0.5 * grad_output.T @ x
    db = torch.sum(grad_output, dim=0)  # summing over all batch
    dx = grad_output @ w
    return dx, dw, db
    # raise NotImplementedError()
    # ========================
