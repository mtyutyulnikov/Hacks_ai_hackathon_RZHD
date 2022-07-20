import os
os.environ['PYTORCH_JIT'] = '0'

import copy
import json
import time
import torch
import click
import numpy as np
# import transformers

from skimage import io
from pathlib import Path
from collections import defaultdict
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split

import nn_tools.utils.config as cfg

from nn_tools.models import UnetSm
from nn_tools.utils import init_determenistic
from nn_tools.metrics.segmentation import jaccard_score
from nn_tools.process.pre import ( apply_image_augmentations, apply_only_image_augmentations )
from nn_tools.losses.segmentation import ( cross_entropy_with_logits_loss, 
                                           focal_with_logits_loss,
                                           dice_with_logits_loss )
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import os
from torch import nn


config = dict()

debugging_info = { 'epoch_loss_trace': list(),
                   'epoch_score_trace': list(),
                   'epoch_additional_score_trace': list(),
                   'epoch_times': list(),
                   'max_memory_consumption': 0.,
                 }

def init_precode():
#     from precode.augmentations.unet3d_sm import train_aug, infer_aug, pre_aug

    import albumentations as A
    from albumentations.pytorch import ToTensor


    config['TRAIN_AUG'] = A.Compose([
        A.Resize( height=336, 
                    width=597, 
                    p=1. ),

        #Rost augs
        A.OneOf([
            A.RandomFog( alpha_coef=0.04,
                         fog_coef_lower=0.1,
                         fog_coef_upper=0.4,
                         p=0.15 )
        ], p=1.),
        A.CLAHE(p=0.5),
        A.ToGray(p=0.3),
        A.VerticalFlip(p=0.5),
        A.Cutout(),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        A.GridDistortion(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.ChannelShuffle(p=1.), 
    ])

    config['EVAL_AUG'] = A.Compose([
        A.Resize( height=336, 
                    width=597, 
                    p=1. ),
    ])

def init_global_config(**kwargs):
    cfg.init_timestamp(config)
    cfg.init_run_command(config)
    cfg.init_kwargs(config, kwargs)
    cfg.init_logging(config, __name__, config['LOGGER_TYPE'], filename=config['PREFIX']+'logger_name.txt')
    cfg.init_device(config)
    cfg.init_verboser(config, logger=config['LOGGER'])
    cfg.init_options(config)


class RJDDataset(Dataset):
    def __init__( self, imgs, masks = None, aug = None):
        self.imgs = imgs
        self.masks = masks
        self.aug = aug

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> dict:
        imgpath = self.imgs[idx]
        img = io.imread(imgpath)

        if self.masks is not None:
            mask = io.imread(self.masks[idx])

            mask = mask[:, :, 0]
            mask[mask == 6] = 1
            mask[mask == 7] = 2
            mask[mask == 10] = 3

        if self.aug is not None:
            auged = self.aug(image=img, mask=mask)
            img, mask = auged['image'], auged['mask']

        img = np.moveaxis(img, -1, 0)

        return img, mask

def collate_fn(batch):
    imgs, masks = zip(*batch)

    imgs = torch_float(imgs, torch.device('cpu'))
    masks = torch_long(masks, torch.device('cpu'))

    return imgs, masks

def load_data_with_labels(data):
    # datapath = Path(os.getenv('DATAPATH'))
    datapath = Path('../../timur/rjd/data')
    print(datapath)


    ALL_IMAGES = sorted((datapath / 'train' / 'images').glob('*.png'))
    ALL_MASKS = sorted((datapath / 'train' / 'mask').glob('*.png'))

    indices = np.arange(len(ALL_IMAGES))

    train_indices, valid_indices = train_test_split(
      indices, test_size=0.15, random_state=1996, shuffle=True
    )

    data['train'] = RJDDataset(
        [ ALL_IMAGES[idx] for idx in train_indices ],
        [ ALL_MASKS[idx] for idx in train_indices ],
        config['TRAIN_AUG']
    )

    data['val'] = RJDDataset(
        [ ALL_IMAGES[idx] for idx in valid_indices ],
        [ ALL_MASKS[idx] for idx in valid_indices ],
        config['EVAL_AUG']
    )

def load_data():
    data = { }

    load_data_with_labels(data)

    return data

def torch_long(data, device):
    return Variable(torch.LongTensor(data)).to(device)

def torch_float(data, device):
    return Variable(torch.FloatTensor(data), requires_grad=True).to(device)

def create_model(num_classes):
    # model = UnetSm( in_channels=3,
    #                 out_channels=num_classes,
    #                 encoder_name=config['BACKBONE'],
    #                 encoder_weights='ssl',
    #                 decoder_attention_type='scse' )
    # model = UNETR(
    #         in_channels=3, 
    #         out_channels=num_classes,
    #         img_size=224,
    #         spatial_dims=2
    #     )

    class SegFormer4model(nn.Module):
        def __init__(self):
            super(SegFormer4model, self).__init__()
            
            self.base_model = SegformerForSemanticSegmentation().from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.linear = nn.Linear(150, num_classes) 

            self.scaler = nn.Upsample(scale_factor=4, mode='nearest')

            
        def forward(self, inputs):
            segformer_output = self.base_model(inputs).logits   
            segformer_output = segformer_output.transpose(1, 3).transpose(1, 2)

            outputs = self.linear(segformer_output)
            outputs = outputs.transpose(1, 2).transpose(1, 3)
            outputs = self.scaler(outputs)
            return outputs

    model = SegFormer4model()
    # state = torch.load('./10-07-22:17-15_model_segformer.pth', map_location=torch.device('cpu'))
    # model.load_state_dict(state)
    return model

def inner_supervised(model, imgs_batch, masks_batch):
    imgs_batch = imgs_batch.to(config['DEVICE'])
    masks_batch = masks_batch.to(config['DEVICE'])

    logits_batch = model(imgs_batch)
    # print(logits_batch.shape, masks_batch.shape)

    loss = dice_with_logits_loss(masks_batch, logits_batch, average='macro', averaged_classes=np.arange(4)) + \
           cross_entropy_with_logits_loss(masks_batch, logits_batch)

    return loss

def inner_train_loop(model, opt, dataset):
    model.train()

    batch_losses = list()

    dataloader = DataLoader( dataset,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=True,
                            #  sampler=RandomSampler(dataset),
                             collate_fn=collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=2 )

    for step_idx, (imgs_batch, masks_batch) in config['VERBOSER'](enumerate(dataloader), total=len(dataloader)):
        loss_sup = inner_supervised(model, imgs_batch, masks_batch)

        loss = loss_sup

        loss.backward()

        if (step_idx + 1) % config['ACCUMULATION_STEP'] == 0:
            opt.step()
            model.zero_grad()

        batch_losses.append(loss.item())

    return np.mean(batch_losses)

def inner_val_loop(model, dataset):
    model.eval()

    metrics = defaultdict(list)
    
    dataloader = DataLoader( dataset,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=False,
                            #  sampler=RandomSampler(dataset),
                             collate_fn=collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=2 )

    for imgs_batch, masks_batch in config['VERBOSER'](dataloader):
        with torch.no_grad():
            imgs_batch = imgs_batch.to(config['DEVICE'])
            masks_batch = masks_batch.to(config['DEVICE'])

            logits_batch = model(imgs_batch)

            pred_masks_batch = logits_batch.argmax(axis=1)

            masks_batch = masks_batch.cpu().data.numpy().astype(np.uint8)
            pred_masks_batch = pred_masks_batch.cpu().data.numpy().astype(np.uint8)

            for mask, pred_mask in zip(masks_batch, pred_masks_batch):
                for metric, average in [ ('jaccard_score', 'none'),
                                         ('jaccard_score', 'macro') ]:
                    score = globals()[metric](mask, pred_mask, average=average)

                    if average == 'none':
                        for idx, scorei in zip(np.unique(mask), score):
                            metrics[metric + '_' + average + '_' + str(idx)].append(scorei)
                    else:
                        metrics[metric + '_' + average].append(score)

    additional_scores = {}

    for key in metrics:
        mean = float(np.mean(metrics[key]))
        additional_scores[f'{key}'] = mean

    score = additional_scores['jaccard_score_macro']

    return score, additional_scores

def fit(model, data):
    train_losses = list()
    val_scores = list()

    model.to(config['DEVICE'])

    opt = torch.optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], eps=1e-8)

    epochs_without_going_up = 0
    best_score = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(config['EPOCHS']):
        start_time = time.perf_counter()

        loss = inner_train_loop( model,
                                 opt,
                                 data['train'] )

        config['LOGGER'].info(f'epoch - {epoch+1} loss - {loss:.6f}')
        train_losses.append(loss)

        score, additional = inner_val_loop( model,
                                            data['val'] )

        val_scores.append(score)
        config['LOGGER'].info(f'epoch - {epoch+1} score - {100 * score:.2f}%')

        for key in additional:
            config['LOGGER'].info(f'epoch - {epoch+1} {key} - {100 * additional[key]:.2f}%')

        if best_score < score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_going_up = 0

            store(model)
        else:
            epochs_without_going_up += 1

        if epochs_without_going_up == config['STOP_EPOCHS']:
            break

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        config['LOGGER'].info(f'elapsed time {elapsed_time:.2f} s')
        config['LOGGER'].info(f'epoch without improve {epochs_without_going_up}')

        if config['DEBUG']:
            debugging_info['epoch_loss_trace'].append(round(loss, 3))
            debugging_info['epoch_score_trace'].append(round(100*score, 3))
            debugging_info['epoch_times'].append(round(elapsed_time, 3))

            for key in additional:
                additional[key] = round(100*additional[key], 3)

            debugging_info['epoch_additional_score_trace'].append(additional)

    model.load_state_dict(best_state)

def store(model):
    state = model.state_dict()
    path = config['PREFIX'] + config['MODELNAME']
    torch.save(state, path)

def store_debug():
    if not config['DEBUG']:
        return

    if torch.cuda.is_available():
        debugging_info['max_memory_consumption'] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    else:
        pass

    with open(config['PREFIX'] + 'debug_name.json', 'w') as f:
        json.dump(debugging_info, f)

@click.command()
@click.option('--learning_rate', '-lr', type=float, default=1e-3)
@click.option('--batch_size', '-bs', type=int, default=12)
@click.option('--epochs', '-e', type=int, default=30, help='The number of epoch per train loop')
@click.option('--accumulation_step', '-as', type=int, default=1, help='The number of iteration to accumulate gradients')
@click.option('--stop_epochs', '-se', type=int, default=5)
@click.option('--modelname', '-mn', type=str, default='model_segformer_rectangle_size.pth')
# @click.option('--backbone', '-bone', type=str, default='resnext50_32x4d')
@click.option('--njobs', type=int, default=1, help='The number of jobs to run in parallel.')
@click.option('--logger_type', '-lt', type=click.Choice(['stream', 'file'], case_sensitive=False), default='stream')
@click.option('--verbose', is_flag=True, help='Whether progress bars are showed')
@click.option('--debug', is_flag=True, help='Whether debug info is stored')
@click.option('--debugname', '-dn', type=str, default='debug_name.json')


def main(**kwargs):
    init_determenistic()
    init_global_config(**kwargs)
    init_precode()

    config['N_CLASSES'] = 4  

    for key in config:
        if key != 'LOGGER':
            config['LOGGER'].info(f'{key} {config[key]}')
            debugging_info[key.lower()] = str(config[key])

    data = load_data()

    config['LOGGER'].info(f'create model')
    model = create_model(num_classes=config['N_CLASSES'])

    config['LOGGER'].info(f'fit model')
    fit(model, data)

    config['LOGGER'].info(f'store model')
    store(model)

    store_debug()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
