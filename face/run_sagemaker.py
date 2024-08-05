import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
training_jobs = session.list_hubs()
print(training_jobs)

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
    hyperparameters={
        'epochs': 400,
        'batch-size': 64,
        'learning-rate': 0.001,
        'num-workers': 16,
        'data-file': 'fer2013.csv',
        'src-checkpoint': 'checkpoint_3.pt',
        'dst-checkpoint': 'checkpoint_4.pt',
    },
    checkpoint_s3_uri=s3_checkpoint_dir,
)

train_input = sagemaker.inputs.TrainingInput(s3_input_data)
pytorch_estimator.fit({'train': train_input})
