import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        modules = []
        print('in size: ', in_size)
        channels = [in_size[0],in_size[1],in_size[1]*2,in_size[1]*4,in_size[1]*8]
        convs = [(4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1)]
        for i, (kernel, stride, padding) in zip(range(len(channels)-1), convs):
            modules.append(torch.nn.Conv2d(in_channels=channels[i],
                                out_channels=channels[i+1],
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                bias=False)
            )
            modules.append(torch.nn.BatchNorm2d(channels[i+1], momentum=0.1))
            modules.append(torch.nn.LeakyReLU(negative_slope=0.25, inplace=True))
            # modules.append(torch.nn.Dropout2d(0.2))
        modules.append(torch.nn.Conv2d(in_channels=512,
                                out_channels=1,
                                kernel_size=4,
                                stride=1,
                                padding=0,
                                bias=False))
        self.discriminator = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        N = x.shape[0]
        y = self.discriminator(x).reshape((N, -1))
        # raise NotImplementedError()
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        dim = z_dim*featuremap_size*2
        # self.pre_process = nn.Sequential(nn.Linear(in_features=z_dim,
        #                       out_features=64*8,
        #                       bias=False),nn.LeakyReLU(0.15))
        channels = [z_dim,dim,dim//2,dim//4,dim//8,out_channels]
        convs = [(1, 0), (2, 1), (2, 1), (2, 1),(2, 1)]
        # channels = list(reversed([32, 64, 128, 256])) + [out_channels]
        # print(channels)
        # convs = list(reversed([(4, 0), (4, 4),(1, 1), (1, 1)]))
        for i, (stride, padding) in zip(range(len(channels)-1), convs):
            modules.append(torch.nn.ConvTranspose2d(in_channels=channels[i],
                                out_channels=channels[i+1],
                                kernel_size=featuremap_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
            )
            if i< (len(channels)-2):
                modules.append(torch.nn.BatchNorm2d(channels[i+1], momentum=0.1))
                modules.append(torch.nn.ReLU(inplace=True))
                # modules.append(torch.nn.Dropout2d(0.2))
        modules.append(torch.nn.Tanh())
        self.cnn = nn.Sequential(*modules)
        # raise NotImplementedError()
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        self.train(with_grad)
        space = torch.randn(n, self.z_dim, requires_grad=with_grad, device=device)
        if not with_grad:
            with torch.no_grad():
                samples = self.forward(space)
        else:
            samples = self.forward(space)
        self.train()
        # raise NotImplementedError()
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        # x=self.pre_process(z)
        x=z.view(-1, self.z_dim, 1, 1)
        x = self.cnn(x)
        # raise NotImplementedError()
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    
    lower, upper = data_label-0.5*label_noise, data_label+0.5*label_noise
    N = y_data.shape[0]
    m = torch.distributions.uniform.Uniform(torch.tensor([lower]), torch.tensor([upper]))
    true_data_labels = m.sample([N])
    
    lower, upper = 1-data_label-0.5*label_noise, 1-data_label+0.5*label_noise
    N = y_generated.shape[0]
    m = torch.distributions.uniform.Uniform(torch.tensor([lower]), torch.tensor([upper]))
    gen_data_labels = m.sample([N])
    
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_data = criterion(y_data, true_data_labels.squeeze().to(y_data.device))
    loss_generated = criterion(y_generated, gen_data_labels.squeeze().to(y_generated.device))
    
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    labels = torch.ones_like(y_generated)*(data_label)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_generated, labels.to(y_generated.device))
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    N = x_data.shape[0]
    generated_data = gen_model.sample(N)
    disc_real_scores = dsc_model(x_data).squeeze()
    disc_gen_scores = dsc_model(generated_data).squeeze()
    dsc_optimizer.zero_grad()
    dsc_loss = dsc_loss_fn(disc_real_scores, disc_gen_scores)
    dsc_loss.backward()
    dsc_optimizer.step()
    # raise NotImplementedError()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    N = x_data.shape[0]
    generated_data = gen_model.sample(N, with_grad=True)
    disc_gen_scores = dsc_model(generated_data)
    gen_optimizer.zero_grad()
    gen_loss = gen_loss_fn(disc_gen_scores)
    gen_loss.backward()
    gen_optimizer.step()
    # raise NotImplementedError()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if torch.tensor(dsc_losses[-1]) + torch.tensor(gen_losses[-1]) / 4 == torch.min(torch.tensor(dsc_losses) + torch.tensor(gen_losses) / 4):
        torch.save(gen_model, checkpoint_file)
        saved = True
    # raise NotImplementedError()
    # ========================
    return saved
