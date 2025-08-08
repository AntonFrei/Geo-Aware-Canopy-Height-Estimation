import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import wandb

from runner import Runner
from utilities import GeneralUtility

warnings.filterwarnings('ignore')

#GPU Selection - Currently not necessary
'''
import torch

#print("Total GPUs available:", torch.cuda.device_count())

#for i in range(torch.cuda.device_count()):
#    print(f"GPU {i} - Name: {torch.cuda.get_device_name(i)}")

#print("Selected GPU before:", torch.cuda.current_device())

torch.cuda.set_device(1)

# Check which device is being used
print("Selected GPU after:", torch.cuda.current_device())
#print("Device name:", torch.cuda.get_device_name())
'''

debug = "--debug" in sys.argv

defaults = dict(
    # System
    seed=1,

    # Data
    dataset='icml_2024_global_rh100',
    batch_size=10, #Optimum per paper:32 - default 5 - memory/speed tradeoff, batchsize_15 => 1 Epoch ~ 600 Iterations

    # Architecture
    arch='unet',  # Defaults to unet
    backbone='resnet50',  # Defaults to resnet50
    use_pretrained_model=False,

    # Optimization
    ## In runner.py get_dataloaders():
    #cut_off = 1000  # Reduce from 3000 to 1000
    optim='AdamW',  # Defaults to AdamW
    loss_name='shift_huber',  # Defaults to shift_l1
    n_iterations=24000, #6000 / 18000 for satclip
    log_freq=500,
    initial_lr=5e-4, #Default 1e-3 / without geo 1e-4 for more input
    weight_decay=1e-3, #default 1e-3 / 1e-1
    use_standardization=True,
    use_augmentation=False,
    use_label_rescaling=False,

    #Coordinates
    use_coord_encoding=False, #!!!!!!
    coord_encoder="fourier",  # Options: raw, wrap, wrap_lon_only,fourier, satclip
    coord_injection_mode="input",  # NEW: Options: "input", "feature_maps"


    # Efficiency
    fp16=False,
    use_memmap=False,
    num_workers_per_gpu=8,   # Defaults to 8

    # Other
    use_weighted_sampler='g10',#'g10' is the default
    use_weighting_quantile=10,
    use_swa=False,
    use_mixup=False,
    use_grad_clipping=True,
    use_input_clipping=False,   # Must be in [False, None, 1, 2, 5]
    n_lr_cycles=0,
    cyclic_mode='triangular2',
    )

#if not debug:
    # Set everything to None recursively
#    defaults = GeneralUtility.fill_dict_with_none(defaults)
#defaults = GeneralUtility.fill_dict_with_none(defaults)


# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)


@contextmanager
def tempdir():
    username = getpass.getuser()
    tmp_root = '/scratch/local/' + username
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir('/scratch/local/') and not os.path.isdir(tmp_root):
        os.mkdir(tmp_root)
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        assert 'htc-' not in os.uname().nodename, "Not allowed to write to /tmp on htc- machines."
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # Check if we are running on the GCP cluster, if so, mark as potentially preempted
    is_htc = 'htc-' in os.uname().nodename
    is_gcp = 'gpu' in os.uname().nodename and not is_htc
    if is_gcp:
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting()  # Note: This potentially overwrites the config when a run is resumed -> problems with tmp_dir

    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
