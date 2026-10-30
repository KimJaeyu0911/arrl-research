import argparse
import torch
import math
import matplotlib.pyplot as plt
from ncsn_pytorch.models import ScoreNet
from ncsn_pytorch.utils import get_sigmas, annealed_langevin_dynamics


def sample(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = ScoreNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    sigmas = get_sigmas(n_scales=args.n_scales)
    img_size = args.img_size
    samples = annealed_langevin_dynamics(model, (args.num_samples, 1, img_size, img_size), sigmas,
                                         n_steps=args.steps, step_size=args.step_size, device=device)
    samples = samples.clamp(-1, 1)

    # show grid
    # arrange samples in a square-ish grid
    n_cols = int(math.ceil(math.sqrt(args.num_samples)))
    n_rows = int(math.ceil(args.num_samples / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    for idx in range(n_rows * n_cols):
        ax = axs[idx]
        if idx < args.num_samples:
            ax.imshow(samples[idx, 0].cpu().numpy() * 0.5 + 0.5, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.out)
    print('Saved', args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-samples', type=int, default=16)
    parser.add_argument('--n-scales', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--img-size', type=int, default=28, help='Image size for sampling')
    parser.add_argument('--out', default='samples.png')
    parser.add_argument('--step-size', type=float, default=0.0002, help='Langevin step size')
    parser.add_argument('--device', default=None, help='cuda or cpu (optional)')
    args = parser.parse_args()
    sample(args)
