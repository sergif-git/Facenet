import argparse
import torch
from torchvision import transforms
import torchvision
from PIL import Image
from models import FaceNetModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from random import sample, choice
import json
import numpy
import torchvision
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from models import model_920
from eval_metrics import evaluate, plot_roc
import pandas as pd
import string
torch.cuda.empty_cache()
import gc
gc.collect()
from mtcnn.mtcnn import MTCNN
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--test-root-dir', type=str, help='Path al directori dimatges de test')
parser.add_argument('--test-csv-name', type=str, help='Path al csv de les imatges de test')
parser.add_argument('--pretrain', action='store_true', help='Utilitza el model preentrenat')
parser.add_argument('--load-last', type=str, help='Path al checkpoint de lentrenament')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = args.test_root_dir
df = pd.read_csv(args.test_csv_name)
class_row_counts = df['class'].value_counts()
images_above1_index = class_row_counts[class_row_counts > 2].index.tolist()
images_above1 = df[df['class'].isin(images_above1_index)]

pretrain = args.pretrain
model = FaceNetModel(pretrained=pretrain)
model.to(device)
if args.load_last:
    checkpoint = args.load_last
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
model = torch.nn.DataParallel(model)
    
trfrm = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
topil = transforms.ToPILImage()
totensor = transforms.Compose(trfrm.transforms[:-1])


def get_all_pairs():
    pairs = []
    i = 0;
    total = len(df['id'])
    noms = df['id']
    while i < total:
        j = i
        while j != total:
            pairs.append((images_above1.iloc[i], images_above1.iloc[j]))
            j = j + 1
        i = i + 1
    return pairs


def get_path(root, item):
    return os.path.join(root, str(item['name']), str(item['id']) + "."+str(item['ext']))


def main():
    all_pairs = get_all_pairs()
    model.eval()
    mida = len(df['class'])
    matrixDistances = numpy.zeros((mida, mida))
    with torch.no_grad():
        i, j, diag = 0, 0, 0
        for num, item in enumerate(all_pairs, 1):
            a,b = map(lambda i: trfrm(Image.open(get_path(root, i))).unsqueeze(0).to(device), item)
            embed1, embed2 = model(a), model(b)
            euclidean_distance = F.pairwise_distance(embed1, embed2)
            euclidean_distance = round(euclidean_distance.item(),3)
            name1, name2 = item[0]['id'], "prova" #item[1]['id']        
            matrixDistances[i][j] = euclidean_distance
            matrixDistances[j][i] = euclidean_distance
            if i == mida-1:
                diag = diag +1
                i = diag
                j = j + 1
            else:
                i = i + 1
    plt.matshow(matrixDistances)
    plt.colorbar(orientation="vertical")
    labels = df['id']
    a = numpy.arange(mida)
    plt.xlabel('Embedding', fontsize=10)
    plt.ylabel('Embedding', fontsize=10)
    plt.xticks(a,labels, rotation='vertical')
    plt.yticks(a,labels)
    plt.tick_params(axis='x', which='both',bottom=True,top=False,
        labelbottom=True,labeltop=False,rotation=60,labelsize=3)
    plt.tick_params(axis='y',labelsize=3)
    plt.title("Model 3 - Distancies embeddings test", fontsize=10)
    plt.savefig('pretrainDistClasses.png', dpi=1500)

    plt.show()


if __name__ == '__main__':
    main()