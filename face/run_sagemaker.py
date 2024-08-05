import argparse

import sagemaker
from sagemaker.pytorch import PyTorch

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--num-workers', type=int)
parser.add_argument('--save-interval', type=int, default=100,
                    help='Number of batches between checkpoints')
parser.add_argument('--src-checkpoint', type=str, required=False)
parser.add_argument('--dst-checkpoint', type=str, required=True)
args = parser.parse_args()

hyperparameters = {
    "epochs": args.epochs,
    'batch-size': args.batch_size,
    'learning-rate': args.learning_rate,
    'num-workers': args.num_workers,
    'data-file': 'fer2013.csv',
    'dst-checkpoint': args.dst_checkpoint
}

if args.src_checkpoint is not None:
    hyperparameters['src-checkpoint'] = args.src_checkpoint

session = sagemaker.Session()

s3_input_data = 's3://etunes-bucket/data/fer2013.csv'
s3_checkpoint_dir = 's3://etunes-bucket/checkpoints/'

pytorch_estimator = PyTorch(
    git_config={
        'repo': 'git@github.com:canivit/etunes.git',
    },
    source_dir='face',
    role='etunes',
    entry_point='train_model.py',
    framework_version='1.13',  # PyTorch version
    py_version='py39',  # Python version
    instance_count=1,
    # instance_type='ml.c5.4xlarge',
    instance_type='ml.p3.8xlarge',
    hyperparameters=hyperparameters,
    checkpoint_s3_uri=s3_checkpoint_dir,
)

train_input = sagemaker.inputs.TrainingInput(s3_input_data)
pytorch_estimator.fit({'train': train_input})
