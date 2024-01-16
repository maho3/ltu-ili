from os.path import join
import argparse
from ili.dataloaders import SBISimulator, StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation.runner import ValidationRunner


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

    print(f"Configuration: {args}")

    # DATA
    kwargs = dict(
        xobs_file=f'obs_{args.obs}.npy',
        x_file=f'{args.N}_x.npy',
        theta_file=f'{args.N}_theta.npy',
    )
    if seq:
        raise NotImplementedError
        loader = SBISimulator.from_config(
            join(cfgdir, 'data', 'seq.yaml'), **kwargs)
        loader.set_simulator(simulator)
    else:
        loader = StaticNumpyLoader.from_config(
            join(cfgdir, 'data', 'static.yaml'), **kwargs)

    # INFERENCE
    out_dir = f'./slcp/res/{args.inf}_obs{args.obs}_N{args.N}'
    print(f"Output directory: {out_dir}")

    kwargs = dict(
        out_dir=out_dir
    )
    runner = InferenceRunner.from_config(
        join(cfgdir, 'infer', f'{args.inf}.yaml'), **kwargs)
    runner(loader=loader)

    # VALIDATION
    kwargs = dict(
        out_dir=out_dir
    )
    val_runner = ValidationRunner.from_config(
        join(cfgdir, 'val', f'{args.inf}.yaml'), **kwargs)
    val_runner(loader=loader)
