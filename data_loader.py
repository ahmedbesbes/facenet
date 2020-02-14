import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, identities, num_triplets, transform=None):

        self.root_dir = root_dir
        self.df = identities
        self.num_triplets = num_triplets
        self.transform = transform
        self.training_triplets = self.generate_triplets(
            self.df, self.num_triplets)

    @staticmethod
    def generate_triplets(df, num_triplets):

        triplets = []
        grouped_by_class = (df.groupby('class')
                            .apply(lambda row: list(row['name'])))

        face_classes = grouped_by_class.to_dict()

        grouped_by_class = pd.DataFrame(grouped_by_class)
        grouped_by_class.reset_index(inplace=True)
        grouped_by_class.columns = ['class', 'images']
        grouped_by_class['n_images'] = grouped_by_class['images'].map(len)

        for _ in tqdm(range(num_triplets), leave=False):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''
            pos_class = (grouped_by_class[grouped_by_class['n_images'] > 2]['class']
                         .sample(1)
                         .values[0])
            neg_class = (grouped_by_class[grouped_by_class['class'] != pos_class]['class']
                         .sample(1)
                         .values[0])

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc, ipos = np.random.choice(range(len(face_classes[pos_class])),
                                              2,
                                              replace=False)
            ineg = np.random.randint(0, len(face_classes[neg_class]))
            triplets.append([face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                             pos_class, neg_class, pos_name, neg_name])

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[
            idx]

        anc_img = os.path.join(
            self.root_dir, 'cropped', anc_id)
        pos_img = os.path.join(
            self.root_dir, 'cropped', pos_id)
        neg_img = os.path.join(
            self.root_dir, 'cropped', neg_id)

        anc_img = io.imread(anc_img)
        pos_img = io.imread(pos_img)
        neg_img = io.imread(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img,
                  'pos_img': pos_img,
                  'neg_img': neg_img,
                  'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):

        return len(self.training_triplets)


def get_dataloader(root_dir,
                   train_identities,     valid_identities,
                   num_train_triplets, num_valid_triplets,
                   batch_size,         num_workers):

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}

    face_dataset = {
        'train': TripletFaceDataset(root_dir=root_dir,
                                    identities=train_identities,
                                    num_triplets=num_train_triplets,
                                    transform=data_transforms['train']),
        'valid': TripletFaceDataset(root_dir=root_dir,
                                    identities=valid_identities,
                                    num_triplets=num_valid_triplets,
                                    transform=data_transforms['valid'])}

    dataloaders = {
        x: torch.utils.data.DataLoader(
            face_dataset[x], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x in ['train', 'valid']}

    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size
