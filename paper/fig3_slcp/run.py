import sys
import os
from os.path import join
import argparse
import numpy as np
from ili.inference import InferenceRunner
from ili.validation.runner import ValidationRunner
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run inference for toy data.")
    parser.add_argument('--obs', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument("--inf", type=str)
    args = parser.parse_args()

    seq = (args.inf[0] == 's') or (args.inf[0] == 'm')
    cfgdir = './configs'
    cfgname = args.inf
    if args.inf[0] == 's':
        cfgname = cfgname[1:]

    print(f"Configuration: {args}")

    # DATA
    if seq:
        from ili.dataloaders import SBISimulator
        import sbibm
        task = sbibm.get_task("slcp")
        simulator = task.get_simulator()
        loader = SBISimulator.from_config(
            join(cfgdir, 'data', 'seq.yaml'),
            xobs_file=f'obs_{args.obs}.npy',
            num_simulations=args.N//10,
        )
        loader.set_simulator(simulator)
    else:
        from ili.dataloaders import StaticNumpyLoader
        loader = StaticNumpyLoader.from_config(
            join(cfgdir, 'data', 'static.yaml'),
            xobs_file=f'obs_{args.obs}.npy',
            x_file=f'{args.N}_x.npy',
            theta_file=f'{args.N}_theta.npy'
        )

    # INFERENCE
    out_dir = f'./slcp/res/{args.inf}_obs{args.obs}_N{args.N}'
    print(f"Output directory: {out_dir}")

    if args.inf == 'mcabc':
        from ili.inference import ABCRunner
        train_args = dict(num_simulations=args.N, quantile=100/args.N)
        runner = ABCRunner.from_config(
            join(cfgdir, 'infer', f'{cfgname}.yaml'),
            out_dir=out_dir,
            train_args=train_args,
        )
        kde = runner(loader=loader)
        os.remove(join(out_dir, 'samples.pkl'))
        samples = kde.sample(10000).numpy()
        np.save(join(out_dir, 'single_samples.npy'), samples)
        sys.exit(0)

    kwargs = dict(
        out_dir=out_dir
    )
    if seq:
        kwargs['model'] = dict(engine=args.inf.upper())
    runner = InferenceRunner.from_config(
        join(cfgdir, 'infer', f'{cfgname}.yaml'), **kwargs)
    runner(loader=loader)

    # VALIDATION
    kwargs = dict(
        out_dir=out_dir
    )
    val_runner = ValidationRunner.from_config(
        join(cfgdir, 'val', f'{cfgname}.yaml'), **kwargs)
    val_runner(loader=loader)

    sys.exit(0)
