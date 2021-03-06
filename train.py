""" script for training a DCGAN on for given dataset """

import torch as th
import argparse
import numpy as np

from torch.backends import cudnn
from torchvision.datasets import CIFAR10

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# enable fast training
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_config", action="store", type=str,
                        default="configs/dcgan-cifar-10-conditional/gen.conf",
                        help="default configuration for generator network")

    parser.add_argument("--discriminator_config", action="store", type=str,
                        default="configs/dcgan-cifar-10-conditional/dis.conf",
                        help="default configuration for discriminator network")

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--discriminator_file", action="store", type=str,
                        default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="data/cifar-10_with_labels",
                        help="path for the images directory")

    parser.add_argument("--sample_dir", action="store", type=str,
                        default="samples/dcgan-cifar-10-conditional",
                        help="path for the generated samples directory")

    parser.add_argument("--model_dir", action="store", type=str,
                        default="models/dcgan-cifar-10-conditional",
                        help="path for saved models directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="standard-gan",
                        help="loss function to be used: 'hinge', " +
                             "'relativistic-hinge', 'standard-gan'")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=128,
                        help="latent size for the generator")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=64,
                        help="batch_size for training")

    parser.add_argument("--start", action="store", type=int,
                        default=1,
                        help="starting epoch number")

    parser.add_argument("--num_epochs", action="store", type=int,
                        default=12,
                        help="number of epochs for training")

    parser.add_argument("--feedback_factor", action="store", type=int,
                        default=62,
                        help="number of logs to generate per epoch")

    parser.add_argument("--checkpoint_factor", action="store", type=int,
                        default=1,
                        help="save model per n epochs")

    parser.add_argument("--g_lr", action="store", type=float,
                        default=0.0002,
                        help="learning rate for generator")

    parser.add_argument("--d_lr", action="store", type=float,
                        default=0.0002,
                        help="learning rate for discriminator")

    parser.add_argument("--data_percentage", action="store", type=float,
                        default=100,
                        help="percentage of data to use")

    parser.add_argument("--num_workers", action="store", type=int,
                        default=3,
                        help="number of parallel workers for reading files")

    args = parser.parse_args()

    return args


# create the one-hot encoded cifar-10 dataset
class OneHotCIFAR10(CIFAR10):
    """
    extends the CIFAR10 and one-hot encodes the labels
    """

    @staticmethod
    def one_hot_embedding(labels, num_classes):
        """
        Embedding labels to one-hot form
        :param labels: (LongTensor) class labels, sized [N,]
        :param num_classes: (int) number of classes
        :return:(tensor) encoded labels, sized [N, #classes]
        """
        y = th.eye(num_classes)
        return y[labels]

    def __getitem__(self, index):
        """
        one-hot encodes the labels before outputting
        :param index: integer index
        :return: images, labels: tensor[B x 3 x 32 x 32], tensor[B x 10]
        """
        images, labels = super().__getitem__(index)
        labels = self.one_hot_embedding(labels, 10)

        return images, labels


def randomizer(correct_labels):
    """
    randomly change the correct_labels for mismatch
    :param correct_labels: input correct labels
    :return: mismatched labels
    """
    int_labels = th.argmax(correct_labels, dim=-1)
    ranges = [list(range(correct_labels.shape[-1]))
              for _ in range(correct_labels.shape[0])]
    for i in range(len(int_labels)):
        ranges[i].remove(int_labels[i])

    mismatched_int_labels = list(map(lambda x: np.random.choice(x), ranges))
    mismatched_labels = OneHotCIFAR10.one_hot_embedding(
        th.LongTensor(mismatched_int_labels),
        correct_labels.shape[-1]
    ).to(device)

    # return the so calculated mismatched labels:
    return mismatched_labels


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from attn_gan_pytorch.Utils import get_layer
    from attn_gan_pytorch.ConfigManagement import get_config
    from attn_gan_pytorch.Networks import ConditionalGenerator, \
        ConditionalDiscriminator, ConditionalGAN
    from data_processing.DataLoader import get_transform, get_data_loader
    from attn_gan_pytorch.Losses import HingeGAN, RelativisticAverageHingeGAN
    from attn_gan_pytorch.Losses import StandardGAN

    # create a data source:
    cifar_10_dataset = OneHotCIFAR10(args.images_dir, transform=get_transform((32, 32)))
    data = get_data_loader(cifar_10_dataset, args.batch_size, args.num_workers)

    # create generator object:
    gen_conf = get_config(args.generator_config)
    gen_conf = list(map(get_layer, gen_conf.architecture))
    generator = ConditionalGenerator(gen_conf, args.latent_size)

    if args.generator_file is not None:
        # load the weights into generator
        generator.load_state_dict(th.load(args.generator_file))

    print("Generator Configuration: ")
    print(generator)

    # create discriminator object:
    dis_conf = get_config(args.discriminator_config)
    dis_conf = list(map(get_layer, dis_conf.architecture))
    discriminator = ConditionalDiscriminator(dis_conf[:-1], dis_conf[-1])

    if args.discriminator_file is not None:
        # load the weights into discriminator
        discriminator.load_state_dict(th.load(args.discriminator_file))

    print("Discriminator Configuration: ")
    print(discriminator)

    # create a gan from these
    conditional_dcgan = ConditionalGAN(generator, discriminator, device=device)

    # create optimizer for generator:
    gen_optim = th.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()),
                              args.g_lr, [0.5, 0.999])

    dis_optim = th.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                              args.d_lr, [0.5, 0.999])

    loss_name = args.loss_function.lower()

    if loss_name == "hinge":
        loss = HingeGAN
    elif loss_name == "relativistic-hinge":
        loss = RelativisticAverageHingeGAN
    elif loss_name == "standard-gan":
        loss = StandardGAN
    else:
        raise Exception("Unknown loss function requested")

    # train the GAN
    conditional_dcgan.train(
        data,
        gen_optim,
        dis_optim,
        loss_fn=loss(device, discriminator),
        num_epochs=args.num_epochs,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        feedback_factor=args.feedback_factor,
        num_samples=64,
        matching_aware=True,
        mismatcher=randomizer,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir,
        log_dir=args.model_dir,
        start=args.start
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
