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
from collections import defaultdict
import pandas as pd
from models import model_920
from eval_metrics import evaluate, plot_roc
import pandas as pd
import string
import math
from torch.nn.modules.distance import PairwiseDistance
torch.cuda.empty_cache()
import gc
gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument('--test-root-dir', type=str, help='Path al directori dimatges de test')
parser.add_argument('--test-csv-name', type=str, help='Path al csv de les imatges de test')
parser.add_argument('--num-classif', default=60, type=int, metavar='NE', help='Utilitza n parells aleatoris del dataset de test (default: 60)')
parser.add_argument('--all-pairs', action='store_true', help='Utilitza tots els parells del dataset de test')
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


def get_random_pairs(size):
    pairs = []
    while len(pairs) < size:
        must_same = choice([True, False])
        if must_same:
            class_ = images_above1.sample().iloc[0]['class']
            same_pairs = images_above1[images_above1['class'] == class_].sample(2)
            pairs.append((same_pairs.iloc[0], same_pairs.iloc[1]))
        else:
            pair1 = images_above1.sample()
            pair2 = images_above1[images_above1['class'] != pair1.iloc[0]['class']].sample()
            pairs.append((pair1.iloc[0], pair2.iloc[0]))
    return pairs


def get_path(root, item):
    return os.path.join(root, str(item['name']), str(item['id']) + "."+str(item['ext']))


def cosine_similarity(embedding1, embedding2):
    dot = numpy.sum(numpy.multiply(embedding1.cpu().numpy(), embedding2.cpu().numpy()))
    norm = numpy.linalg.norm(embedding1.cpu().numpy(), axis=1) * numpy.linalg.norm(embedding2.cpu().numpy(), axis=1)
    similarity = dot / norm
    dist = numpy.arccos(similarity) / math.pi
    return dist


def main():
    l2_dist = PairwiseDistance(2)
    max_correct = 0
    best = 0
    if args.all_pairs:
        pairs = get_all_pairs()
    else:
        pairs = get_random_pairs(args.num_classif)
    result = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for threshold in numpy.arange(6, 13, 0.5):
            print(threshold)
            correct = 0
            for num, item in enumerate(pairs, 1):
                a,b = map(lambda i: trfrm(Image.open(get_path(root, i))).unsqueeze(0).to(device), item)
                embed1, embed2 = model.forward(a), model.forward(b)
                euclidean_distance = l2_dist.forward(embed1, embed2)
                not_same = euclidean_distance > threshold
                diff_class = int(item[0]['class'] != item[1]['class'])
                result[threshold].append((diff_class == not_same).item())
                if diff_class == not_same:
                    correct = correct + 1
            if max_correct < correct:
                max_correct = correct
                best= threshold

    x = list(result.keys())
    y = [sum(i) for i in result.values()]
    print('Major nombre classificacions correctes : '+str(max_correct)+' amb llindar de valor '+str(round(best,3)))
    plt.xlabel('Threshold', fontsize=15)
    plt.ylabel('Classificacions correctes', fontsize=15)
    plt.plot(x,y)
    plt.title("Model 3 - Test distancia euclidiana")
    plt.savefig('pretrainThreshTest.png', dpi=1500)
    plt.show()


if __name__ == '__main__':
    main()