import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', type=str, help='Path al log de dades dentrenament')
args = parser.parse_args()


def read_train_valid_loss(root_dir):
    train_logs = list(pathlib.Path(root_dir).glob('train.csv'))
    valid_logs = list(pathlib.Path(root_dir).glob('valid.csv'))
    train_loss, valid_loss, train_epochs, valid_epochs = [],[],[],[]
    for log in train_logs:
        with open(log, 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            i = 0
            for line in reader:
                x = line[0].split(",")
                if i != 0:
                    train_epochs.append(int(x[0]))
                    train_loss.append(round(float(x[1]),3))
                i = i+1
    for log in valid_logs:
        with open(log, 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            i = 0
            for line in reader:
                x = line[0].split(",")
                if i != 0:
                    valid_epochs.append(int(x[0]))
                    valid_loss.append(round(float(x[1]),3))
                i = i+1
    
    return train_loss, valid_loss, train_epochs, valid_epochs

def main():
    args = parser.parse_args()
    if not (args.root_dir):
        parser.error('No root directory found  --root-dir')
    else:
        train_loss, valid_loss, train_epochs, valid_epochs = read_train_valid_loss(str(args.root_dir))
        f = plt.figure(figsize=(15,5))
        plt.title('Triplet Loss (train / validation)')
        plt.plot(train_epochs, train_loss, color = '#2c3e50', label = 'Train')
        plt.xlabel('Epoch', fontsize = 15)
        plt.ylabel('Loss', fontsize = 15)
        plt.ylim(0, max(train_loss)+0.1*max(train_loss))
        plt.plot(valid_epochs, valid_loss, color = '#16a085', label = 'Validation')
        plt.legend(frameon=True,loc = 'upper right')
        plt.xticks(np.arange(min(train_epochs), max(train_epochs)+1, len(train_epochs)//15))
        plt.savefig('log/loss.jpg', dpi=f.dpi)
        plt.show();
        print('Min train loss achived '+str(min(train_loss)))
        print('Min validation loss achived '+str(min(valid_loss)))


if __name__ == '__main__':
    main()
