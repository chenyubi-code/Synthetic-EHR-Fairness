import argparse
import os
import numpy as np
import time
import random
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()

experimentName = os.path.splitext(os.path.basename(__file__))[0]

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument("--DATASETPATH", type=str,
                    required=True,
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--n_epochs_pretrain", type=int, default=100,
                    help="number of epochs of pretraining the autoencoder")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument("--cuda", type=str2bool, default=False,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=str2bool, default=False,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between samples")
parser.add_argument("--epoch_time_show", type=str2bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=100, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=str2bool, default=False, help="Minibatch averaging")

parser.add_argument("--training", type=str2bool, default=False, help="Training status")
parser.add_argument("--generate", type=str2bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--expPATH", type=str, required=True,
                    help="Training status")
parser.add_argument("--manual_seed", type=int, default=202601, help="Repeat seed")
parser.add_argument("--n_generate", type=int, default=30000, help="Number of synthetic rows to generate")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.expPATH, exist_ok=True)

opt.manualSeed = int(opt.manual_seed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
np.random.seed(opt.manualSeed)
cudnn.benchmark = False
cudnn.deterministic = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

device = torch.device("cuda:0" if opt.cuda else "cpu")

data = np.load(os.path.expanduser(opt.DATASETPATH), allow_pickle=False)
if data.ndim != 2 or data.shape[0] < opt.batch_size:
    raise ValueError("The training matrix must have at least batch_size rows; reduce --batch_size for a small trial.")
if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
    raise ValueError("Expected a complete binary feature matrix.")

sampleSize = data.shape[0]
featureSize = data.shape[1]

# Use every input row for model fitting and diagnostic sampling.
trainData = data.astype(np.float32)
testData = data.astype(np.float32)


class Dataset:
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
           pass

        return torch.from_numpy(sample)


dataset_train_object = Dataset(data=trainData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=False, num_workers=0, drop_last=True, sampler=samplerRandom)

dataset_test_object = Dataset(data=testData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=False, num_workers=0, drop_last=True, sampler=samplerRandom)

random_samples = next(iter(dataloader_test))
feature_size = random_samples.size()[1]

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        n_channels_base = 4

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0,
                      dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16 * n_channels_base, out_channels=32 * n_channels_base, kernel_size=2, stride=1,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=1, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=4, padding=0,
                               dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=7, stride=4,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=7, stride=3,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=7, stride=2,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=3, stride=2,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        x = self.decoder(x)
        return torch.squeeze(x)

    def decode(self, x):
        x = self.decoder(x)
        return torch.squeeze(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.genDim = 128
        self.linear1 = nn.Linear(opt.latent_dim, self.genDim)
        self.bn1 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(opt.latent_dim, self.genDim)
        self.bn2 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()

    def forward(self, x):
        residual = x
        temp = self.activation1(self.bn1(self.linear1(x)))
        out1 = temp + residual

        residual = out1
        temp = self.activation2(self.bn2(self.linear2(out1)))
        out2 = temp + residual
        return out2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disDim = 256

        ma_coef = 1
        if opt.minibatch_averaging:
            ma_coef = ma_coef * 2

        self.model = nn.Sequential(
            nn.Linear(ma_coef * dataset_train_object.featureSize, self.disDim),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(int(self.disDim), 1)
        )

    def forward(self, x):

        if opt.minibatch_averaging:
            x_mean = torch.mean(x, 0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), 1)

        output = self.model(x)
        return output


def generator_loss(y_fake, y_true):
    """Return the non-saturating generator loss."""
    epsilon = 1e-12
    return -0.5 * torch.mean(torch.log(y_fake + epsilon))


def autoencoder_loss(x_output, y_target):
    """Sum binary cross-entropy over features, then average across patients."""
    epsilon = 1e-12
    min_dim = min(x_output.shape[1], y_target.shape[1])
    
    x_output = x_output[:, :min_dim]
    y_target = y_target[:, :min_dim]

    term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
    loss = torch.mean(-torch.sum(term, 1), 0)
    return loss


def discriminator_loss(outputs):
    """Return the Wasserstein discriminator loss."""
    loss = torch.mean((1 - labels) * outputs) - torch.mean(labels * outputs)
    return loss


def discriminator_accuracy(predicted, y_true):
    """Compute discriminator classification accuracy."""
    total = y_true.size(0)
    correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def sample_transform(sample):
    """Threshold generated probabilities at 0.5."""
    sample[sample >= 0.5] = 1
    sample[sample < 0.5] = 0
    return sample

def match_feature_size_tensor(x, target_dim):

    if x.dim() == 1:
        x = x.view(1, -1)

    current_dim = x.shape[1]

    if current_dim < target_dim:

        pad_size = target_dim - current_dim

        padding = torch.zeros(
            x.shape[0],
            pad_size,
            device=x.device,
            dtype=x.dtype
        )

        x = torch.cat([x, padding], dim=1)

    elif current_dim > target_dim:

        x = x[:, :target_dim]

    return x

def weights_init(m):
    """Initialize convolution, normalization, and linear layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


generatorModel = Generator()
discriminatorModel = Discriminator()
autoencoderModel = Autoencoder()
autoencoderDecoder = autoencoderModel.decoder

Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1


if torch.cuda.device_count() > 1 and opt.multiplegpu:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  generatorModel = nn.DataParallel(generatorModel, list(range(opt.num_gpu)))
  discriminatorModel = nn.DataParallel(discriminatorModel, list(range(opt.num_gpu)))
  autoencoderModel = nn.DataParallel(autoencoderModel, list(range(opt.num_gpu)))
  autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(opt.num_gpu)))

if opt.cuda:
    generatorModel.cuda()
    discriminatorModel.cuda()
    autoencoderModel.cuda()
    autoencoderDecoder.cuda()
    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)
autoencoderModel.apply(weights_init)

g_params = [{'params': generatorModel.parameters()},
            {'params': autoencoderDecoder.parameters(), 'lr': 1e-4}]
optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_A = torch.optim.Adam(autoencoderModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

if opt.training:

    for epoch_pre in range(opt.n_epochs_pretrain):
        for i, samples in enumerate(dataloader_train):

            real_samples = Variable(samples.type(Tensor))

            recons_samples = autoencoderModel(real_samples)

            a_loss = autoencoder_loss(recons_samples, real_samples)

            optimizer_A.zero_grad()

            a_loss.backward()
            optimizer_A.step()

            batches_done = epoch_pre * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                    % (epoch_pre + 1, opt.n_epochs_pretrain, i, len(dataloader_train), a_loss.item())
                    , flush=True)

    gen_iterations = 0
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()
        for i, samples in enumerate(dataloader_train):

            valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)

            real_samples = Variable(samples.type(Tensor))

            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            for p in discriminatorModel.parameters():
                p.requires_grad = True

            # Train the critic more often at the beginning and every 500 updates.
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                n_iter_D = 100
            else:
                n_iter_D = opt.n_iter_D
            j = 0
            while j < n_iter_D:
                j += 1

                # Weight clipping enforces the critic's Lipschitz constraint.
                for p in discriminatorModel.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                optimizer_D.zero_grad()

                errD_real = torch.mean(discriminatorModel(real_samples), dim=0)
                errD_real.backward(one)

                z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

                fake_samples = generatorModel(z)

                fake_samples = torch.squeeze(autoencoderDecoder(fake_samples.unsqueeze(dim=2)))

                fake_samples = match_feature_size_tensor(
                    fake_samples,
                    dataset_train_object.featureSize
                )
                
                errD_fake = torch.mean(discriminatorModel(fake_samples.detach()),dim=0)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake

                optimizer_D.step()

            for p in discriminatorModel.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()

            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            fake_samples = generatorModel(z)

            fake_samples = torch.squeeze(autoencoderDecoder(fake_samples.unsqueeze(dim=2)))

            fake_samples = match_feature_size_tensor(
                fake_samples,
                dataset_train_object.featureSize
            )

            errG = torch.mean(discriminatorModel(fake_samples), dim=0)
            errG.backward(one)

            optimizer_G.step()
            gen_iterations += 1

        with torch.no_grad():

            real_samples_test = next(iter(dataloader_test))
            real_samples_test = Variable(real_samples_test.type(Tensor))
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            fake_samples_test_temp = generatorModel(z)
            fake_samples_test = torch.squeeze(autoencoderDecoder(fake_samples_test_temp.unsqueeze(dim=2)))

            fake_samples_test = match_feature_size_tensor(
                fake_samples_test,
                dataset_train_object.featureSize
            )

            out_real_test = discriminatorModel(real_samples_test).view(-1)
            accuracy_real_test = discriminator_accuracy(F.sigmoid(out_real_test), valid)

            out_fake_test = discriminatorModel(fake_samples_test.detach()).view(-1)
            accuracy_fake_test = discriminator_accuracy(F.sigmoid(out_fake_test), fake)

            reconst_samples_test = autoencoderModel(real_samples_test)
            a_loss_test = autoencoder_loss(reconst_samples_test, real_samples_test)

        print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.3f Loss_G: %.3f Loss_D_real: %.3f Loss_D_fake %.3f'
              % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
                 errD.item(), errG.item(), errD_real.item(), errD_fake.item()), flush=True)

        print(
            "TEST: [Epoch %d/%d] [Batch %d/%d] [A loss: %.2f] [real accuracy: %.2f] [fake accuracy: %.2f]"
            % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
               a_loss_test.item(), accuracy_real_test,
               accuracy_fake_test)
            , flush=True)

        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0 or epoch + 1 == opt.n_epochs:
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'Autoencoder_state_dict': autoencoderModel.state_dict(),
                'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_A_state_dict': optimizer_A.state_dict(),
            }, os.path.join(opt.expPATH, "model_epoch_%d.pth" % (epoch + 1)))


if opt.generate:

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    checkpoint = torch.load(
        os.path.join(opt.expPATH, "model_epoch_%d.pth" % opt.n_epochs),
        map_location=device
    )

    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
    autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

    generatorModel.eval()
    autoencoderModel.eval()
    autoencoderDecoder.eval()

    num_fake_samples = int(opt.n_generate)

    gen_batches = []
    n_batches = int(np.ceil(num_fake_samples / opt.batch_size))

    print("Generating synthetic data...")
    print("Target N:", num_fake_samples)
    print("Batch size:", opt.batch_size)
    print("Number of batches:", n_batches)
    print("Feature size:", dataset_train_object.featureSize)

    with torch.no_grad():
        for i in range(n_batches):

            current_batch_size = min(
                opt.batch_size,
                num_fake_samples - i * opt.batch_size
            )

            z = torch.randn(
                current_batch_size,
                opt.latent_dim,
                device=device
            )

            gen_samples_tensor = generatorModel(z)

            gen_samples_decoded = torch.squeeze(
                autoencoderDecoder(gen_samples_tensor.unsqueeze(dim=2))
            )

            gen_samples_decoded = match_feature_size_tensor(
                gen_samples_decoded,
                dataset_train_object.featureSize
            )

            gen_samples_decoded = gen_samples_decoded.detach().cpu().numpy()

            assert np.isnan(gen_samples_decoded).sum() == 0

            gen_batches.append(gen_samples_decoded)

    gen_probs = np.vstack(gen_batches)
    gen_probs = gen_probs[:num_fake_samples, :].astype(np.float32)

    prob_path = os.path.join(opt.expPATH, "synthetic_prob.npy")
    np.save(prob_path, gen_probs, allow_pickle=False)
