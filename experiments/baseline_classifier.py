import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from voicemap.datasets import OmniglotDataset
from voicemap.models import get_omniglot_classifier
from voicemap.train import fit
from voicemap.callbacks import *
from config import PATH

assert torch.cuda.is_available()
device = torch.device('cuda')


##############
# Parameters #
##############
batchsize = 64
test_fraction = 0.1
num_tasks = 100
k_way = 5
n_shot = 1


###################
# Create datasets #
###################
background = OmniglotDataset('background')
evaluation = OmniglotDataset('evaluation')


indices = range(len(background))
class_ids = background.df['class_id'].values
train_indices, test_indices, _, _ = train_test_split(indices, class_ids, test_size=test_fraction,
                                                     stratify=class_ids)
train = torch.utils.data.Subset(background, train_indices)
test = torch.utils.data.Subset(background, test_indices)

# Model
model = get_omniglot_classifier(background.num_classes())
model.to(device, dtype=torch.double)


############
# Training #
############
train_loader = DataLoader(train, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss().cuda()


def prepare_batch(batch):
    # Move to GPU and convert targets to int
    x, y = batch
    return x.cuda(), y.long().cuda()


def prepare_n_shot_batch(query, support):
    query = torch.from_numpy(query[0]).to(device, dtype=torch.double)
    support = torch.from_numpy(support[0]).to(device, dtype=torch.double)
    return query, support


callbacks = [
    EvaluateMetrics(test_loader),
    # Evaluate n-shot on tasks on unseen classes
    NShotTaskEvaluation(num_tasks=num_tasks, n_shot=n_shot, k_way=k_way, dataset=evaluation,
                        prepare_batch=prepare_n_shot_batch, prefix='test_'),
    ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=5, verbose=True),
    ModelCheckpoint(filepath=PATH + '/models/omniglot_classifier.torch',
                    monitor='val_categorical_accuracy'),
    CSVLogger(PATH + '/logs/omniglot_classifier.csv'),
]


torch.backends.cudnn.benchmark = True
fit(
    model,
    opt,
    loss_fn,
    epochs=30,
    dataloader=train_loader,
    prepare_batch=prepare_batch,
    callbacks=callbacks,
    metrics=['categorical_accuracy']
)
