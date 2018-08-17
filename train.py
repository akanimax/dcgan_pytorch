""" script for training a Self Attention GAN on celeba images """

import torch as th
import argparse

from torch.backends import cudnn
from attn_gan_pytorch.Losses import GANLoss

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
                        default="configs/dcgan/gen.conf",
                        help="default configuration for generator network")

    parser.add_argument("--discriminator_config", action="store", type=str,
                        default="configs/dcgan/dis.conf",
                        help="default configuration for discriminator network")

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--discriminator_file", action="store", type=str,
                        default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="data/celeba",
                        help="path for the images directory")

    parser.add_argument("--sample_dir", action="store", type=str,
                        default="samples/dcgan",
                        help="path for the generated samples directory")

    parser.add_argument("--model_dir", action="store", type=str,
                        default="models/dcgan",
                        help="path for saved models directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="standard-gan",
                        help="loss function to be used: 'hinge', 'relativistic-hinge'")

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


# define the StandardGAN loss function
class StandardGAN(GANLoss):

    def __init__(self, dev, dis):
        from torch.nn import BCELoss

        super().__init__(dev, dis)

        # define the criterion object
        self.criterion = BCELoss()

    def dis_loss(self, real_samps, fake_samps):
        # calculate the real loss:
        real_loss = self.criterion(th.squeeze(self.dis(real_samps)),
                                   th.ones(real_samps.shape[0]).to(self.device))
        # calculate the fake loss:
        fake_loss = self.criterion(th.squeeze(self.dis(fake_samps)),
                                   th.zeros(fake_samps.shape[0]).to(self.device))

        # return final loss as average of the two:
        return (real_loss + fake_loss) / 2


    def gen_loss(self, _, fake_samps):
        return self.criterion(th.squeeze(self.dis(fake_samps)),
                              th.ones(fake_samps.shape[0]).to(self.device))


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from attn_gan_pytorch.Utils import get_layer
    from attn_gan_pytorch.ConfigManagement import get_config
    from attn_gan_pytorch.Networks import Generator, Discriminator, GAN
    from data_processing.DataLoader import FlatDirectoryImageDataset, \
        get_transform, get_data_loader
    from attn_gan_pytorch.Losses import HingeGAN, RelativisticAverageHingeGAN

    # create a data source:
    celeba_dataset = FlatDirectoryImageDataset(args.images_dir,
                                               transform=get_transform((64, 64)))
    data = get_data_loader(celeba_dataset, args.batch_size, args.num_workers)

    # create generator object:
    gen_conf = get_config(args.generator_config)
    gen_conf = list(map(get_layer, gen_conf.architecture))
    generator = Generator(gen_conf, args.latent_size)

    if args.generator_file is not None:
        # load the weights into generator
        generator.load_state_dict(th.load(args.generator_file))

    print("Generator Configuration: ")
    print(generator)

    # create discriminator object:
    dis_conf = get_config(args.discriminator_config)
    dis_conf = list(map(get_layer, dis_conf.architecture))
    discriminator = Discriminator(dis_conf)

    if args.discriminator_file is not None:
        # load the weights into discriminator
        discriminator.load_state_dict(th.load(args.discriminator_file))

    print("Discriminator Configuration: ")
    print(discriminator)

    # create a gan from these
    dcgan = GAN(generator, discriminator, device=device)

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
    dcgan.train(
        data,
        gen_optim,
        dis_optim,
        loss_fn=loss(device, discriminator),
        num_epochs=args.num_epochs,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        feedback_factor=args.feedback_factor,
        num_samples=64,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir,
        log_dir=args.model_dir,
        start=args.start
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
