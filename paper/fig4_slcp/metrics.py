from os.path import join
import argparse
import numpy as np
import torch
from sbibm.metrics.c2st import c2st
from sbibm.metrics.mmd import mmd


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run inference for toy data.")
    parser.add_argument('--obs', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument('--seq', type=int)
    parser.add_argument("--inf", type=str)
    args = parser.parse_args()

    seq = (args.seq == 1)
    cfgdir = './configs'
    wdir = './slcp'

    print(f"Configuration: {args}")

    # METRICS
    out_dir = f'./slcp/res/{args.inf}_obs{args.obs}_N{args.N}'
    print(f"Output directory: {out_dir}")

    ref = np.load(join(wdir, 'ref.npy'))
    samps = np.load(join(out_dir, 'single_samples.npy'))

    ref = ref[args.obs-1]
    ref = torch.Tensor(ref)
    samps = torch.Tensor(samps)

    results = {}
    print('Running c2st...')
    results['c2st'] = c2st(ref, samps).item()
    print('Running mmd...')
    results['mmd'] = mmd(ref, samps).item()

    print(results)
    np.save(join(out_dir, 'metrics.npy'), results)

    print("Done.")
