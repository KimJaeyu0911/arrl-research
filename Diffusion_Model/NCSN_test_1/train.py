import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

from ncsn_pytorch.models import ScoreNet
from ncsn_pytorch.utils import annealed_langevin_dynamics, get_sigmas, sample_noise


def train(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if args.dataset == 'mnist':
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        ds = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        in_channels = 1
    elif args.dataset == 'image':
        # Load images from a local folder. Images will be resized and converted to grayscale.
        transform = T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
        if not os.path.isdir(args.image_dir):
            raise ValueError(f"Image directory not found: {args.image_dir}")
        # Use ImageFolder if there are class subfolders, otherwise load all images in the folder.
        entries = [os.path.join(args.image_dir, e) for e in os.listdir(args.image_dir)]
        has_subdirs = any(os.path.isdir(p) for p in entries)
        if has_subdirs:
            ds = datasets.ImageFolder(root=args.image_dir, transform=transform)
        else:
            from PIL import Image
            import glob

            class FlatImageDataset(torch.utils.data.Dataset):
                def __init__(self, root, transform=None):
                    self.files = sorted(glob.glob(os.path.join(root, '*.*')))
                    self.transform = transform

                def __len__(self):
                    return len(self.files)

                def __getitem__(self, idx):
                    path = self.files[idx]
                    img = Image.open(path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    return img, 0

            ds = FlatImageDataset(args.image_dir, transform=transform)
        in_channels = 1
    else:
        raise ValueError("Unsupported dataset: use 'mnist' or 'image'")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ScoreNet(in_channels=in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    sigmas = get_sigmas(n_scales=args.n_scales)

    for epoch in range(args.epochs):
        pbar = tqdm(loader)
        for x, _ in pbar:
            x = x.to(device)
            noise, sigma_chosen = sample_noise(x, sigmas, device=device)
            x_noisy = x + noise
            pred = model(x_noisy, sigma_chosen)
            # Denoising score matching target: score = - (x_noisy - x) / sigma^2 = -noise / sigma^2
            sigma_view = sigma_chosen.view(-1, 1, 1, 1)
            target = -noise / (sigma_view ** 2)
            loss = mse(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {epoch} loss {loss.item():.4f}")

        torch.save({'model': model.state_dict()}, f'checkpoints/model_epoch{epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='image', choices=['mnist', 'image'],
                        help="Choose 'mnist' or 'image' (local folder)")
    parser.add_argument('--image-dir', default='/home/arrl/jaeyu/Diffusion_Model/NCSN_test_1/image',
                        help='Path to local images (ImageFolder structure)')
    parser.add_argument('--img-size', type=int, default=28, help='Resize images to this size (square)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default=None, help='cuda or cpu (optional)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n-scales', type=int, default=10)
    args = parser.parse_args()
    train(args)
