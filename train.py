import cv2
import gc
import os
import csv
import glob
import time
import datetime

# Basics
import pandas as pd
import numpy as np

# SKlearn
from sklearn import model_selection
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data Augmentation for Image Preprocessing
from albumentations import (VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue,
                            RandomResizedCrop, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2, ToTensor
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, resnet50, vgg16, vgg16_bn, vgg19, vgg19_bn, inception_v3, squeezenet1_0, densenet161
import warnings
warnings.filterwarnings("ignore")

def get_data():
    # My Train: with imputed missing values + OHE
    my_train = pd.read_csv('../input/siim-melanoma-prep-data/train_clean.csv')

    # Drop path columns and Diagnosis (it won't be available during TEST)
    # We'll rewrite them once the data is concatenated
    to_drop = ['path_dicom', 'path_jpeg', 'diagnosis']
    for drop in to_drop:
        if drop in my_train.columns:
            my_train.drop([drop], axis=1, inplace=True)

    # Roman's Train: with added data for Malignant category
    roman_train = pd.read_csv('../input/../input/melanoma-external-malignant-256/train_concat.csv')

    # --- Before concatenatenating both together, let's preprocess roman_train ---
    # Replace NAN with 0 for patient_id
    roman_train['patient_id'] = roman_train['patient_id'].fillna(0)

    # OHE
    to_encode = ['sex', 'anatom_site_general_challenge']
    encoded_all = []

    roman_train[to_encode[0]] = roman_train[to_encode[0]].astype(str)
    roman_train[to_encode[1]] = roman_train[to_encode[1]].astype(str)

    label_encoder = LabelEncoder()

    for column in to_encode:
        encoded = label_encoder.fit_transform(roman_train[column])
        encoded_all.append(encoded)

    roman_train[to_encode[0]] = encoded_all[0]
    roman_train[to_encode[1]] = encoded_all[1]

    # Give all columns the same name
    roman_train.columns = my_train.columns

    # --- Concatenate info which is not available in my_train ---
    common_images = my_train['dcm_name'].unique()
    new_data = roman_train[~roman_train['dcm_name'].isin(common_images)]

    # Merge all together
    train_df = pd.concat([my_train, new_data], axis=0)


    # Create path column to image folder for both Train and Test
    path_train = '../input/melanoma-external-malignant-256/train/train/'
    train_df['path_jpg'] = path_train + train_df['dcm_name'] + '.jpg'


    # --- Last final thing: NORMALIZE! ---
    train_df['age'] = train_df['age'].fillna(-1)

    normalized_train = preprocessing.normalize(train_df[['sex', 'age', 'anatomy']])

    train_df['sex'] = normalized_train[:, 0]
    train_df['age'] = normalized_train[:, 1]
    train_df['anatomy'] = normalized_train[:, 2]

    train_df, test_df = model_selection.train_test_split(
        train_df,
        train_size=0.8,
        test_size=0.2,
        random_state=8  # make sure train and test are always the same
    )

    print('Train: {:,}'.format(len(train_df)), '\n' +
          'Test: {:,}'.format(len(test_df)))

    return train_df, test_df

class MelanomaDataset(Dataset):

    def __init__(self, dataframe, vertical_flip, horizontal_flip,
                 is_train=True):
        self.dataframe, self.is_train = dataframe, is_train
        self.vertical_flip, self.horizontal_flip = vertical_flip, horizontal_flip

        # Data Augmentation (custom for each dataset type)
        if is_train:
            self.transform = Compose([RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
                                      ShiftScaleRotate(rotate_limit=90, scale_limit = [0.7, 1]),
                                      HorizontalFlip(p = self.horizontal_flip),
                                      VerticalFlip(p = self.vertical_flip),
                                      HueSaturationValue(sat_shift_limit=[0.7, 1.3],
                                                         hue_shift_limit=[-0.1, 0.1]),
                                      RandomBrightnessContrast(brightness_limit=[0.01, 0.1],
                                                               contrast_limit= [0.01, 0.1]),
                                      #Normalize(),
                                      ToTensor()])

        else:
            self.transform = Compose([  # Normalize(),
            Resize(height=224, width=224),
            ToTensor()])


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, index):
        # Select path and read image
        image_path = self.dataframe['path_jpg'][index]
        image = cv2.imread(image_path)

        # For this image also import .csv information (sex, age, anatomy)
        csv_data = np.array(self.dataframe.iloc[index][['sex', 'age', 'anatomy']].values,
                            dtype=np.float32)

        # Apply transforms
        image = self.transform(image=image)

        # Extract image from dictionary
        image = image['image']

        # If train/valid: image + class | If test: only image
        return (image, csv_data), self.dataframe['target'][index]


class ResNet50Network(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size

        # Define Feature part (IMAGE)
        self.features = resnet50(pretrained=True)  # 1000 neurons out
        # (CSV data)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))

        # Define Classification part
        self.classification = nn.Linear(1000 + 250, output_size)

    def forward(self, image, csv_data, prints=False):

        if prints: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)

        # Concatenate layers from image with layers from csv_data
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)

        return out


class EfficientNetwork(nn.Module):
    def __init__(self, output_size, no_columns, b4=False, b2=False):
        super().__init__()
        self.b4, self.b2, self.no_columns = b4, b2, no_columns

        # Define Feature part (IMAGE)
        if b4:
            self.features = EfficientNet.from_pretrained('efficientnet-b4')
        elif b2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b7')

        # (CSV)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))

        # Define Classification part
        if b4:
            self.classification = nn.Sequential(nn.Linear(1792 + 250, output_size))
        elif b2:
            self.classification = nn.Sequential(nn.Linear(1408 + 250, output_size))
        else:
            self.classification = nn.Sequential(nn.Linear(2560 + 250, output_size))

    def forward(self, image, csv_data, prints=False):

        if prints: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # IMAGE CNN
        image = self.features.extract_features(image)
        if prints: print('Features Image shape:', image.shape)

        if self.b4:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1792)
        elif self.b2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 2560)
        if prints: print('Image Reshaped shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)

        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)

        return out


class DenseNetNetwork(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size
        # Define Feature part (IMAGE)
        self.features = densenet161(pretrained=True)
        # (CSV data)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        # Define Classification part
        self.classification = nn.Linear(1000 + 250, output_size)

    def forward(self, image, csv_data, prints=False):

        if prints: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)

        # Concatenate layers from image with layers from csv_data
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)

        return out


class SqueezeNet(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size
        # Define Feature part (IMAGE)
        self.features = squeezenet1_0(pretrained=True)
        # (CSV data)
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        # Define Classification part
        self.classification = nn.Linear(1000 + 250, output_size)

    def forward(self, image, csv_data, prints=False):

        if prints: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)

        # Concatenate layers from image with layers from csv_data
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)

        return out


class custom_vgg16(nn.Module):

    def __init__(self, output_size, no_columns, v16=True, v16_bn=False, v19=False):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size
        self.v16, self.v16_bn, self.v19 = v16, v16_bn, v19

        # Define Feature part (IMAGE)
        if v16:
            self.features = vgg16(pretrained=True)
        elif v16_bn:
            self.features = vgg16_bn(pretrained=True)
        elif v19:
            self.features = vgg19(pretrained=True)
        else:
            self.features = vgg19_bn(pretrained=True)

        # (CSV)
        # keep this the same for all models you try
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))

        # Define Classification part
        # you'll need to change 1792 to whatever size the model outputs in self.features.
        if v16:
            self.classification = nn.Sequential(nn.Linear(1000 + 250, output_size))
        elif v16_bn:
            self.features = nn.Sequential(nn.Linear(1000 + 250, output_size))
        elif v19:
            self.features = nn.Sequential(nn.Linear(1000 + 250, output_size))
        else:
            self.features = nn.Sequential(nn.Linear(1000 + 250, output_size))

    def forward(self, image, csv_data, prints=False):

        if prints: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # IMAGE CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)

        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)

        return out


class CustomNetwork(nn.Module):
    def __init__(self, output_size, no_columns, device, set_params=None):
        super().__init__()
        self.no_columns = no_columns
        self.n_layers = set_params['n_layers']
        self.conv_filters = set_params['conv_filters']
        self.kernel_size = set_params['kernel_size']
        self.pool_size = set_params['pool_size']
        self.dropout_rate = set_params['dropout_rate']

        self.cnn_layers = []

        # CONV LAYERS
        for i in range(self.n_layers):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=3 if i == 0 else self.conv_filters,
                          out_channels=self.conv_filters,
                          kernel_size=self.kernel_size),
                # nn.BatchNorm2d(self.conv_filters),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)
            ).to(device))

        self.max_pool = nn.MaxPool2d(kernel_size=self.pool_size).to(device)

        # CSV data
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2)).to(device)

        # Classification part

        # calculating the right size for the linear layer
        # i = input_size
        # o = output
        # p = padding
        # k = kernel_size
        # s = stride
        # d = dilation
        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        # in our case, s=1, p=0, d=1, so for each layer o=i-(k-1)
        self.output_dimensions = ((224 - self.n_layers * (
                    self.kernel_size - 1)) // self.pool_size) ** 2 * self.conv_filters
        self.classification = nn.Linear(self.output_dimensions + 250, output_size).to(device)

    def forward(self, image, csv_data, prints=False):

        if prints: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # IMAGE CNN
        for i in range(self.n_layers):
            image = self.cnn_layers[i](image)
            # print(image.shape)

        # one maxpool
        image = self.max_pool(image)
        if prints: print('Features Image shape:', image.shape)

        # not sure I like this average pool thing. Let's just Flatten() instead
        # image = F.avg_pool2d(image, image.size()[2:]).reshape(self.batch_size, -1)
        image = image.reshape(-1, self.output_dimensions)
        if prints: print('Image Reshaped shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints: print('CSV Data:', csv_data.shape)

        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if prints: print('Out shape:', out.shape)

        return out


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    prc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, roc, prc, f1


def train_folds(model, train_df, test_df, version='v1', set_params=None, debug=False):

    if debug:
        train_df = train_df[:100]
        test_df = test_df[:100]
    train_len = len(train_df)

    # Create Object
    group_fold = GroupKFold(n_splits=set_params['K'])

    # Generate indices to split data into training and test set.
    folds = group_fold.split(X=np.zeros(train_len),
                             y=train_df['target'],
                             groups=train_df['ID'].tolist())

    for fold, (train_index, valid_index) in enumerate(folds):
        # Append to .txt
        with open(f"logs_{version}_fold{fold}.txt", 'a+') as f:
            print('-' * 10, 'Fold:', fold + 1, '-' * 10, file=f)
        print('-' * 10, 'Fold:', fold + 1, '-' * 10)

        # --- Create Instances ---
        # Best ROC score in this fold
        best_roc = None
        # Reset patience before every fold
        patience_f = set_params['patience']

        # Initiate the model
        training_model = model

        optimizer = torch.optim.Adam(training_model.parameters(),
                                     lr=set_params['learning_rate'],  # learning_rate,
                                     weight_decay=set_params['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max',
                                      patience=set_params['lr_patience'], verbose=True, factor=set_params['lr_factor'])
        criterion = nn.BCEWithLogitsLoss()

        # --- Read in Data ---
        train_data = train_df.iloc[train_index].reset_index(drop=True)
        valid_data = train_df.iloc[valid_index].reset_index(drop=True)

        # Create Data instances
        train = MelanomaDataset(train_data, vertical_flip=set_params['vertical_flip'], horizontal_flip=set_params['horizontal_flip'],
                                is_train=True)
        valid = MelanomaDataset(valid_data, vertical_flip=set_params['vertical_flip'], horizontal_flip=set_params['horizontal_flip'],
                                is_train=False)


        # Dataloaders
        train_loader = DataLoader(train, batch_size=set_params['train_batch_size'], shuffle=True, num_workers=set_params['num_workers'])
        # shuffle=False! Otherwise function won't work!!!
                # how do I know? ^^
        valid_loader = DataLoader(valid, batch_size=set_params['val_test_batch_size'], shuffle=False, num_workers=set_params['num_workers'])

        # === EPOCHS ===
        print('training...')
        for epoch in range(set_params['epochs']):
            start_time = time.time()
            correct = 0
            train_losses = 0

            # === TRAIN ===
            # Sets the module in training mode.
            training_model.train()

            for (images, csv_data), labels in train_loader:
                # Save them to device
                images = torch.tensor(images, device=set_params['device'], dtype=torch.float32)
                csv_data = torch.tensor(csv_data, device=set_params['device'], dtype=torch.float32)
                labels = torch.tensor(labels, device=set_params['device'], dtype=torch.float32)

                # Clear gradients first; very important, usually done BEFORE prediction
                optimizer.zero_grad()

                # Log Probabilities & Backpropagation
                out = training_model(images, csv_data)

                loss = criterion(out, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                # --- Save information after this batch ---
                # Save loss
                train_losses += loss.item()
                # From log probabilities to actual probabilities
                train_preds = torch.round(torch.sigmoid(out)) # 0 and 1
                # Number of correct predictions
                correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()

            # Compute Train Accuracy
            train_acc = correct / len(train_index)


            # === EVAL ===
            # Sets the model in evaluation mode
            training_model.eval()

            # Create matrix to store evaluation predictions (for accuracy)
            valid_preds = torch.zeros(size = (len(valid_index), 1), device=set_params['device'], dtype=torch.float32)

            # Disables gradients (we need to be sure no optimization happens)
            with torch.no_grad():
                for k, ((images, csv_data), labels) in enumerate(valid_loader):
                    images = torch.tensor(images, device=set_params['device'], dtype=torch.float32)
                    csv_data = torch.tensor(csv_data, device=set_params['device'], dtype=torch.float32)

                    out = training_model(images, csv_data)
                    pred = torch.sigmoid(out)
                    valid_preds[k*images.shape[0] : k*images.shape[0] + images.shape[0]] = pred

                labels = valid_data['target'].values

                valid_acc, valid_roc, valid_prc, valid_f1 = compute_metrics(labels, torch.round(valid_preds.cpu()))

                # Compute time on Train + Eval
                duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]

                # PRINT INFO
                # Append to .txt file
                with open(f"logs_{version}_fold{fold}.txt", 'a+') as f:
                    print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'.\
                     format(duration, epoch+1, set_params['epochs'], train_losses, train_acc, valid_acc, valid_roc), file=f)
                # Print to console
                print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'.\
                     format(duration, epoch+1, set_params['epochs'], train_losses, train_acc, valid_acc, valid_roc))


                # === SAVE MODEL ===
                # Update scheduler (for learning_rate)
                scheduler.step(valid_roc)

                # Update best_roc
                if not best_roc: # If best_roc = None
                    best_roc = valid_roc
                    torch.save(training_model.state_dict(),
                               f"{version}_Fold{fold+1}_Epoch{epoch+1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}.pth")
                    continue

                if valid_roc > best_roc:
                    best_roc = valid_roc
                    # Reset patience (because we have improvement)
                    patience_f = set_params['patience']
                    for filename in glob.glob(f"{version}_Fold*"):  # remove all prev checkpoints for this model
                        os.remove(filename)
                    torch.save(training_model.state_dict(),
                               f"{version}_Fold{fold+1}_Epoch{epoch+1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}.pth")

                else:
                    # Decrease patience (no improvement in ROC)
                    patience_f = patience_f - 1
                    if patience_f == 0:
                        with open(f"logs_{version}_fold{fold}.txt", 'a+') as f:
                            print('Early stopping (no improvement since 3 models) | Best ROC: {}'.\
                                  format(best_roc), file=f)
                        print('Early stopping (no improvement since 3 models) | Best ROC: {}'.\
                              format(best_roc))
                        break


        # === INFERENCE ===
        print('testing...')
        training_model.eval()
        test_df = test_df.reset_index(drop=True)
        test = MelanomaDataset(test_df, vertical_flip=set_params['vertical_flip'], horizontal_flip=set_params['horizontal_flip'],
                               is_train=False)
        test_loader = DataLoader(test, batch_size=set_params['val_test_batch_size'], shuffle=False, num_workers=set_params['num_workers'])
        test_preds = torch.zeros(size=(len(test_df), 1), device=set_params['device'], dtype=torch.float32)

        with torch.no_grad():
            for k, ((images, csv_data), labels) in enumerate(test_loader):
                images = torch.tensor(images, device=set_params['device'], dtype=torch.float32)
                csv_data = torch.tensor(csv_data, device=set_params['device'], dtype=torch.float32)

                out = training_model(images, csv_data)
                pred = torch.sigmoid(out)
                test_preds[k * images.shape[0]: k * images.shape[0] + images.shape[0]] = pred

            # Compute accuracy
            labels = test_df['target'].values
            test_acc, test_roc, test_prc, test_f1 = compute_metrics(labels, torch.round(test_preds.cpu()))
            with open(f"results_{version}.csv", 'a+') as h: # append
                writer = csv.writer(h)
                writer.writerow([test_acc, test_roc, test_prc, test_f1])
            with open(f"preds_{version}_fold{fold}.txt", 'w+') as g:
                labels = np.expand_dims(labels, 1).astype(float)
                arr = test_preds.data.cpu().numpy()
                np.savetxt(g, np.concatenate([labels, arr], axis=1))

        # === CLEANING ===
        # Clear memory
        del train, valid, train_loader, valid_loader, images, labels
        # Garbage collector
        gc.collect()


def main(model_type=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device available now:', device)

    set_params = {
        'device': device,
        'epochs': 15,
        'K': 10,
        'patience': 3,
        'TTA': 3,
        'num_workers': 8,
        'learning_rate': 0.0005,
        'weight_decay': 0.0,
        'lr_patience': 1,
        'lr_factor': 0.4,
        'train_batch_size': 64,
        'val_test_batch_size': 64,
        'vertical_flip': 0.5,
        'horizontal_flip': 0.5,
        'csv_columns': ['sex', 'age', 'anatomy']
    }

    if model_type == 'resnet':
        model = ResNet50Network(output_size=1, no_columns=3).to(device)
    elif model_type == 'vgg':
        model = custom_vgg16(output_size=1, no_columns=3, v16=True).to(device)
    elif model_type == 'efficient':
        model = EfficientNetwork(output_size=1, no_columns=3, b4=False, b2=True).to(device)
    elif model_type == 'dense':
        model = DenseNetNetwork(output_size=1, no_columns=3).to(device)
    elif model_type == 'custom':
        set_params['learning_rate'] = 0.00977
        set_params['n_layers'] = 5
        set_params['batch_size'] = 8
        set_params['conv_filters'] = 11
        set_params['kernel_size'] = 4
        set_params['pool_size'] = 3
        set_params['dropout_rate'] = 0.4
        model = CustomNetwork(output_size=1, no_columns=3, device=device, set_params=set_params).to(device)
    train_df, test_df = get_data()
    train_folds(model, train_df, test_df, version=model_type, set_params=set_params, debug=True)

main('resnet')