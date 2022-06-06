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
import glob
import os
import math
from sklearn.decomposition import PCA
import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='Utilitza el model preentrenat')
parser.add_argument('--pcaplot', action='store_true', help='Utilitza el model preentrenat')
parser.add_argument('--path-dataset', type=str, help='Path a la classe a plotar')
parser.add_argument('--load-last', type=str, help='Path al checkpoint de lentrenament')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def getImages(path):
    image_list = []
    for filename in glob.glob(str(path)+'/*.png'):
        im=Image.open(filename)
        image_list.append(totensor(im))
    return image_list


def main():
    model.eval()
    directory = args.path_dataset
    files = 3
    idxPlot = 1
    fig = plt.figure(figsize=(5.,5.), tight_layout=True)
    plt.title("Model original - Embeddings per classe", fontsize=10)  
    embedListAveragePerClass = []
    valorsListAveragePerClass = []
    llegenda = []
    for subdirectory in os.listdir(directory):   
        image_list = getImages(os.path.join(directory, subdirectory)) 
        llegenda.append(str(os.path.basename(subdirectory)))
        ax = plt.subplot(files,3,idxPlot)
        ax.set_title(("Embeddings de "+str(os.path.basename(subdirectory))), fontsize=6)
        ax.set_ylabel('Valor dins idx', fontsize=4)
        ax.set_xlabel('Index dins embedding', fontsize=4)
        valors = numpy.arange(0,128,1)  
        embedList = [] 
        for image in image_list:     
            embed1 = model.forward(image.unsqueeze(0))
            embed1 = embed1.detach().cpu().numpy()
            embedList.append(embed1)
            ax.plot(valors, embed1[0], linewidth=1)
        embedListAveragePerClass.append((numpy.average(embedList, axis=0)[0]))
        valorsListAveragePerClass.append(valors)
        idxPlot = idxPlot + 1    
    plt.savefig('embeddingsPerClass.jpg', dpi=3000)
    plt.show()
    
    for i in range(len(embedListAveragePerClass)):
        plt.plot(valorsListAveragePerClass[i],embedListAveragePerClass[i], linewidth=1)
    plt.title("Model original - Embeddings per classe", fontsize=10)    
    plt.xlabel('Idx dins embedding', fontsize=10)
    plt.ylabel('Valor dins idx', fontsize=10)
    plt.legend(fontsize=15) # using a size in points
    plt.legend(llegenda)
    plt.savefig('embeddingsMeanPerClass.jpg', dpi=1500)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Define Data
    x0 = embedListAveragePerClass[0]
    x1 = embedListAveragePerClass[1]
    x2 = embedListAveragePerClass[2]
    x3 = embedListAveragePerClass[3]
    x4 = embedListAveragePerClass[4]
    x5 = embedListAveragePerClass[5]
    x6 = embedListAveragePerClass[6]
    x7 = embedListAveragePerClass[7]
    x8 = embedListAveragePerClass[8]
    # Plot
    ax.scatter(x0,x1,x2,x3,x4,x5,x6,x7,x8, color='red')
    # Display
    plt.show()


if __name__ == '__main__':
    main()
