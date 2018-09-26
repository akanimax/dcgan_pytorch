""" script for checking the conditional generation of a trained dcgan generator """

import torch as th
import matplotlib.pyplot as plt

from attn_gan_pytorch.Networks import ConditionalGenerator
from attn_gan_pytorch.Utils import get_layer
from attn_gan_pytorch.ConfigManagement import get_config
from train import OneHotCIFAR10

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

label = 9
num_samples = 12
latent_size = 128
num_classes = 10
config_file = "configs/dcgan-cifar-10-conditional/gen.conf"
model_file = "models/dcgan-cifar-10-conditional/GAN_GEN_12.pth"

gen_conf = get_config(config_file)
gen_conf = list(map(get_layer, gen_conf.architecture))
generator = ConditionalGenerator(gen_conf, 128).to(device)
generator.load_state_dict(th.load(model_file))

# generate one-hot samples:
labels = th.LongTensor([label for _ in range(num_samples)])
one_hot_labels = OneHotCIFAR10.one_hot_embedding(labels, num_classes)

# perform inference here:
expanded_labels = th.unsqueeze(th.unsqueeze(one_hot_labels, -1), -1)
latent_input = th.randn(
    num_samples,
    latent_size - num_classes,
    1, 1
)
gan_input = th.cat((latent_input, expanded_labels), dim=1).to(device)

generated_images = generator(gan_input).detach()

for generated_image in generated_images:
    plt.figure(figsize=(2, 2))
    plt.imshow(generated_image.permute(1, 2, 0) / 2 + 0.5)
    plt.show()
