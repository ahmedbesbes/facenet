import os
import shutil
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.modules.distance import PairwiseDistance
import torchvision
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from eval_metrics import evaluate, plot_roc
from utils import TripletLoss
from models.facenet_resnet34 import FaceNetModel
from models.facenet_resnet_inception import InceptionResnetV1
from data_loader import TripletFaceDataset, get_dataloader


parser = argparse.ArgumentParser(
    description='Face Recognition using Triplet Loss')

parser.add_argument('--start-epoch', default=0, type=int, metavar='SE',
                    help='start epoch (default: 0   )')
parser.add_argument('--num-epochs', default=200, type=int, metavar='NE',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--num-classes', default=10000, type=int, metavar='NC',
                    help='number of clases (default: 10000)')
parser.add_argument('--test-size', default=0.2, type=float,
                    help='ratio of validation set')
parser.add_argument('--num-train-triplets', default=10000, type=int, metavar='NTT',
                    help='number of triplets for training (default: 10000)')
parser.add_argument('--num-valid-triplets', default=10000, type=int, metavar='NVT',
                    help='number of triplets for vaidation (default: 10000)')
parser.add_argument('--embedding-size', default=128, type=int, metavar='ES',
                    help='embedding size (default: 128)')
parser.add_argument('--batch-size', default=64, type=int, metavar='BS',
                    help='batch size (default: 128)')
parser.add_argument('--num-workers', default=8, type=int, metavar='NW',
                    help='number of workers (default: 8)')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate (sou: 0.001)')
parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
                    help='margin (default: 0.5)')
parser.add_argument('--identity-csv-name', default='/data_science/computer_vision/data/celeba/identities.csv',
                    type=str, help='files with image id and identity id')
parser.add_argument('--root-dir', default='/data_science/computer_vision/data/celeba/', type=str,
                    help='root directory for data')
parser.add_argument('--save-model', default=1, choices=[0, 1], type=int)
parser.add_argument('--epochs-save', default=5, type=int,
                    help='number of epochs to save model')
parser.add_argument('--log-path', default='./logs/',
                    type=str, help="path of tensorboard logs")
parser.add_argument('--flush-history', default=0, choices=[
                    0, 1], type=int, help="flag to whether or not remove tensorboard logs")
parser.add_argument('--pretrained', default=0,
                    choices=[0, 1], type=int, help="flag to whether or not use pretrained weights")
parser.add_argument('--model', type=str,
                    choices=['resnet34', 'inception'], default='resnet34', help='backbone model')
parser.add_argument('--model', type=str, default=None,
                    help="path to model checkpoint")

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)


def main():
    if args.flush_history == 1:
        objects = os.listdir(args.log_path)
        for f in objects:
            if os.path.isdir(os.path.join(args.log_path, f)):
                shutil.rmtree(os.path.join(args.log_path, f))

    now = datetime.now()
    date_id = now.strftime("%Y%m%d-%H%M%S")

    logdir = os.path.join(args.log_path, date_id)
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    identities = pd.read_csv(args.identity_csv_name)
    identities = identities[identities['rejected'] != 1]
    stats_identities = pd.DataFrame(identities['class'].value_counts())
    stats_identities.reset_index(inplace=True)
    stats_identities.columns = ['class', 'n_images']

    train_ids, valid_ids = train_test_split(stats_identities['class'].values,
                                            stratify=stats_identities['n_images'].values,
                                            test_size=args.test_size)
    train_identities = identities[identities['class'].isin(train_ids)]
    valid_identities = identities[identities['class'].isin(valid_ids)]

    if args.model == 'resnet34':
        model = FaceNetModel(embedding_size=args.embedding_size,
                             num_classes=args.num_classes,
                             pretrained=bool(args.pretrained)).to(device)
    elif args.model == 'inception':
        model = InceptionResnetV1(num_classes=args.num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    best_loss = np.inf

    for epoch in range(args.start_epoch, args.num_epochs + args.start_epoch):

        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch,
                                     args.num_epochs + args.start_epoch - 1))

        data_loaders, data_size = get_dataloader(args.root_dir,
                                                 train_identities,
                                                 valid_identities,
                                                 args.num_train_triplets,
                                                 args.num_valid_triplets,
                                                 args.batch_size,
                                                 args.num_workers)

        best_loss = train_valid(model,
                                optimizer,
                                scheduler,
                                epoch,
                                data_loaders,
                                data_size,
                                writer,
                                best_loss,
                                date_id)

    print(80 * '=')


def train_valid(model, optimizer, scheduler, epoch, dataloaders, data_size, writer, best_loss, date_id):

    for phase in ['train', 'valid']:

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()

        for i, batch_sample in tqdm(enumerate(dataloaders[phase]),
                                    leave=False,
                                    total=len(dataloaders[phase])):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(
                    anc_img), model(pos_img), model(neg_img)

                # choose the hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist <
                       args.margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets].to(device)
                pos_hard_embed = pos_embed[hard_triplets].to(device)
                neg_hard_embed = neg_embed[hard_triplets].to(device)

                triplet_loss = TripletLoss(args.margin).forward(
                    anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                # dists = l2_dist.forward(anc_embed, pos_embed)
                # distances.append(dists.data.cpu().numpy())
                # labels.append(np.ones(dists.size(0)))

                # dists = l2_dist.forward(anc_embed, neg_embed)
                # distances.append(dists.data.cpu().numpy())
                # labels.append(np.zeros(dists.size(0)))

                triplet_loss_sum += triplet_loss.item()

                writer.add_scalar(
                    f'{phase}/loss',
                    triplet_loss_sum / (i+1),
                    epoch * len(dataloaders[phase]) + i
                )

        avg_triplet_loss = triplet_loss_sum / len(dataloaders[phase])
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))

        writer.add_scalar(
            f'{phase}/loss/epoch',
            avg_triplet_loss,
            epoch
        )

        if bool(args.save_model):
            if phase == 'valid':
                if avg_triplet_loss < best_loss:
                    best_loss = avg_triplet_loss
                    torch.save({'epoch': epoch,
                                'best_loss': best_loss,
                                'state_dict': model.state_dict()},
                               './log/checkpoint_epoch_{}.pth'.format(date_id))

    return best_loss


if __name__ == '__main__':

    main()
