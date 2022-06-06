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
from numpy import dot
from numpy.linalg import norm

parser = argparse.ArgumentParser()
parser.add_argument('--test-root-dir', type=str, help='Path al directori dimatges de test')
parser.add_argument('--test-csv-name', type=str, help='Path al csv de les imatges de test')
parser.add_argument('--pretrain', action='store_true', help='Utilitza el model preentrenat')
parser.add_argument('--load-last', type=str, help='Path al checkpoint de lentrenament')
parser.add_argument('--use-euclidian', action='store_true', help='Utilitza la millor representació de cada classe')
parser.add_argument('--threshold', default=6.0, type=float, metavar='MG',
                    help='Distancia per considerar un parell de embeddings iguals (default: 6.0)')

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
topil = transforms.ToPILImage()
totensor = transforms.Compose(trfrm.transforms[:-1])


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


def imshow(img,text=None,should_save=False):
    npimg = torchvision.utils.make_grid(img).numpy()
    plt.axis("off")
    if text:
        plt.text(220, 243, text, fontweight='bold', horizontalalignment='center',fontsize=3,
            bbox=dict(facecolor='white', alpha=0.8))
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))


def get_path(root, item):
    return os.path.join(root, str(item['name']), str(item['id']) + "."+str(item['ext']))


def cosinesimilarity(emb1, emb2):
    emb1 = emb1.squeeze(0).cpu()
    emb2 = emb2.squeeze(0).cpu()
    emb1 = emb1.numpy()
    emb2 = emb2.numpy()
    cosine_similarity = dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return cosine_similarity


def main():
    random_pairs = get_random_pairs(20)
    threshold = args.threshold
    model.eval()
    mida = len(df['class'])
    nCorrect = 0
    nTotal = 0
    with torch.no_grad():
        i, j, diag = 0, 0, 0
        for num, item in enumerate(random_pairs, 1):
            a,b = map(lambda i: trfrm(Image.open(get_path(root, i))).unsqueeze(0).to(device), item)
            x0, x1 = map(lambda i: totensor(Image.open(get_path(root, i))).unsqueeze(0), item)
            embed1, embed2 = model(a), model(b)
            concatenated = torch.cat((x0,x1),0)
            
            if args.use_euclidian:
                euclidean_distance = F.pairwise_distance(embed1, embed2)
                not_same = euclidean_distance > threshold
                pred = ("Iguals", "Diferents")[not_same]
                diff_class = int(item[0]['class'] != item[1]['class'])
                actual = ("Incorrecte", "Correcte")[diff_class == not_same]
                text = f"Distancia: {euclidean_distance.item():.2f}\nPredicció: {pred} \n({actual})"
            else:
                similarity = cosinesimilarity(embed1, embed2)
                not_same = similarity <= threshold
                pred = ("Iguals", "Diferents")[not_same]
                diff_class = int(item[0]['class'] != item[1]['class'])
                actual = ("Incorrecte", "Correcte")[diff_class == not_same]
                text = f"Similitud: {similarity.item():.2f}\nPredicció: {pred} \n({actual})"
            name1, name2 = item[0]['id'], item[1]['id'] 
            if actual == 'Correcte':
                nCorrect = nCorrect +1
            plt.rcParams["font.size"] = "4"
            plt.subplot(5,4,num)
            imshow(concatenated, text)
            nTotal = nTotal + 1
    
    if args.pretrain:
        plt.title("Model 3 - Classificaccions")
        plt.savefig('pretrainRandomClassif.png', dpi=1500)
    else:
        plt.title("Model original - Classificaccions")
        plt.savefig('originalRandomClassifs.png', dpi=1500)

    plt.show()
    print('Total correct predictions: '+str(nCorrect)+'/'+str(nTotal)+' ('+str(round(nCorrect/nTotal,3))+')')


if __name__ == '__main__':
    main()