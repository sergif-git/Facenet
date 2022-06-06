from scipy import misc
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
import matplotlib.ticker as mticker
import pandas as pd
from models import model_920
from eval_metrics import evaluate, plot_roc
import pandas as pd
import string
import glob
import os
import math
from mtcnn.mtcnn import MTCNN
from torchvision import datasets
import argparse
import sys
import time
import cv2
import datetime
from sklearn.metrics import confusion_matrix
torch.cuda.empty_cache()
import gc
import math
from numpy import dot
from numpy.linalg import norm
from sklearn import svm
gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='Utilitza el model preentrenat')
parser.add_argument('--load-last', type=str, help='Path al checkpoint de lentrenament')
parser.add_argument('--path-dataset', type=str, help='Path al dataset per generar els embeddings a comparar')
parser.add_argument('--path-dataset-testing', type=str, help='Path al dataset a testejar')
parser.add_argument('--load-embeddings', type=str, help='Path als embeddings dusuaris')
parser.add_argument('--use-min', action='store_true', help='Utilitza la millor representació de cada classe')
parser.add_argument('--use-euclidian', action='store_true', help='Utilitza la millor representació de cada classe')
parser.add_argument('--euclidianThresh', default=6.0, type=float, metavar='MG', help='Distancia euclidiana maxima per no considerar reconeixament')
parser.add_argument('--cosineThresh', default=0.8, type=float, metavar='MG', help='Similitud cosinosiudal minima per considerar reconeixament')
parser.add_argument('--use-svm', action='store_true', help='Utilitza el SVM com a classificador')


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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
topil = transforms.ToPILImage()
totensor = transforms.Compose(trfrm.transforms[:-1])


def cosinesimilarity(emb1, emb2):
    emb1 = emb1.squeeze(0).cpu()
    emb2 = emb2.squeeze(0).cpu()
    emb1 = emb1.numpy()
    emb2 = emb2.numpy()
    cosine_similarity = dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return cosine_similarity


def getImages(path):
    image_list = []
    for filename in glob.glob(str(path)+'/*.png'):
        image_list.append(filename)
    return image_list


def createEmbeddings(path,i):    
    if i ==0:
        embedListMinimDistClass = []
        noms = []
        embedList = [] 
        for subdirectory in os.listdir(path):  
            image_list = getImages(os.path.join(path, subdirectory)) 
            embedList = []
            # Tots els embeddings de la classe
            for image in image_list:      
                a = trfrm(Image.open(image)).unsqueeze(0).to(device)
                embed1 = model(a)
                embedList.append(embed1)
            distances = []
            # Per cada embedding de la classe
            for embeding in embedList:
                # Distancia a tots
                dist = 0
                for emb in embedList:
                    dist = dist + F.pairwise_distance(embeding, emb)
                distances.append(dist)
            idx_min = distances.index(min(distances))
            embedListMinimDistClass.append(embedList[idx_min])
            noms.append(str(os.path.basename(subdirectory)))
        data = [embedListMinimDistClass, noms]
        torch.save(data, 'minimPerClasse.pt') # saving data.pt file
    elif i == 1:
        embedListAveragePerClass = []
        noms = []
        embedList = [] 
        for subdirectory in os.listdir(path):   
            image_list = getImages(os.path.join(path, subdirectory)) 
            embedList = []
            for image in image_list:      
                a = trfrm(Image.open(image)).unsqueeze(0).to(device)
                embed1 = model(a)
                embedList.append(embed1)
            embedListAveragePerClass.append(torch.mean(torch.stack(embedList),dim=0))            
            noms.append(str(os.path.basename(subdirectory)))
        data = [embedListAveragePerClass, noms]
        torch.save(data, 'mitjaPerClasse.pt') # saving data.pt file
    else:
        noms = []
        embedList = [] 
        for subdirectory in os.listdir(path):   
            image_list = getImages(os.path.join(path, subdirectory)) 
            for image in image_list:      
                a = trfrm(Image.open(image)).unsqueeze(0).to(device)
                embedList.append(model(a))
                noms.append(str(os.path.basename(subdirectory)))
        data = [embedList, noms]
        torch.save(data, 'totsPerClasse.pt') # saving data.pt file


def testAccuracy(saved_data):
    embedding_list = saved_data[0]
    name_list = saved_data[1]
    if not args.use_svm:
        name_list.append('unknown')
        mida = len(name_list)
        matrixConf = numpy.zeros((mida,mida))
    else:
        mida = len(name_list)//25
        matrixConf = numpy.zeros((mida,mida))
    pathTest = args.path_dataset_testing
    
    clf = svm.SVC()
    if args.use_svm:
        embedding_list2 = []
        name_list2 = []
        for e in embedding_list:
            embedding_list2.append(e.cpu().numpy()[0].reshape(1,128)[0]) 
        embedding_list=embedding_list2
        for n in name_list:
            name_list2.append(n)
        name_list=name_list2
        clf.fit(embedding_list,name_list)
    for subdirectory in os.listdir(pathTest):
        nameTest = subdirectory
        image_list = getImages(os.path.join(pathTest, subdirectory)) 
        for imageTest in image_list:
            a = trfrm(Image.open(imageTest)).unsqueeze(0).to(device)
            embedTest = model(a)
            dist_list = []
            if args.use_svm:
                predict = clf.predict([embedTest.cpu().numpy()[0]])
                namePredicted=predict[0]
                matrixConf[name_list.index(nameTest)//25][name_list.index(namePredicted)//25] = \
                    matrixConf[name_list.index(nameTest)//25][name_list.index(namePredicted)//25] + 1
            else:
                for idx, embTrain in enumerate(embedding_list):                
                    if args.use_euclidian:
                        euclidean_distance = F.pairwise_distance(embedTest, embTrain)
                        dist_list.append(euclidean_distance[0])
                    else:
                        similarity = cosinesimilarity(embedTest, embTrain)
                        dist_list.append(similarity)
                    namePredicted = 'unknown'
                    if args.use_euclidian:
                        idx = dist_list.index(min(dist_list))
                        distance = dist_list[idx]
                        if distance < args.euclidianThresh:
                            namePredicted = name_list[idx]                
                    else:
                        idx = numpy.argmax(dist_list)
                        distance = dist_list[idx]
                        if distance > args.cosineThresh:
                            namePredicted = name_list[idx]
                matrixConf[name_list.index(nameTest)][name_list.index(namePredicted)] = \
                    matrixConf[name_list.index(nameTest)][name_list.index(namePredicted)] + 1
    print("Actual (x) vs predictions (y):")
    print(matrixConf)

    if not args.use_svm:
        mida=mida-1
    actualClass = 0
    tpTotal, tnTotal, fpTotal, fnTotal = 0,0,0,0
    for clas in range(0, mida):
        tp, tn, fp, fn = 0,0,0,0
        for i in range(0, mida):
            for j in range(0, mida):
                if ((i == actualClass) and (j == actualClass)):
                    tp = matrixConf[i][j]
                if ((i == actualClass) and (j != actualClass)):
                    fn = fn + matrixConf[i][j]
                if ((j == actualClass) and (i != actualClass)):
                    fp = fp + matrixConf[i][j]
                if ((i != actualClass) and (j != actualClass)):
                    tn = tn + matrixConf[i][j]
        miniConfMatrix = numpy.array([[tp, fn],[fp, tn]])
        if not args.use_svm:
            print("Individual confusion matrix for class :", str(name_list[actualClass]))
        else:
            print("Individual confusion matrix for class :", str(name_list[((actualClass+1)*25)-1]))
        print(miniConfMatrix)
        tpTotal = tpTotal + tp
        tnTotal = tnTotal + tn
        fpTotal = fpTotal + fp
        fnTotal = fnTotal + fn
        actualClass = actualClass + 1

    print("Total true positives: "+ str(tpTotal))
    print("Total true negatives: "+ str(tnTotal))
    print("Total false positives: "+ str(fpTotal))
    print("Total false negatives: "+ str(fnTotal))
    print("Final confusion matrix:")
    conMatrix = numpy.array([[tpTotal,fnTotal],[fpTotal,tnTotal]])
    print(conMatrix)

    accuracy = round(((tpTotal)/(tpTotal+fnTotal)),2)
    print("Model accuracy: "+str(accuracy))


'''
        # Accuracy -> Total de correctes contra total
        # Cada quant el classificador es correcte
        accuracy = round(((tp+tn)/(tp+tn+fp+fn)),2)
        print("Accuracy (Ratio good classifications): "+str(accuracy))
        
        # Misclassification -> Total de incorrectes contra totat
        # Cada quant el classificador es incorrecte
        misclassification = round(((fp+fn)/(tp+tn+fp+fn)),2)
        print("Misclassification (Ratio not good classifications): "+str(misclassification))
        
        # True positive rate
        # Sensitivity o Recall -> Verdaders positius contra positius actuals, objectiu minimitzar falsos negatius
        # Quan es positiu quant diu que es positiu (encerta)
        recall = round((tp/(tp+fn)),2)
        print("Sensitivity, recall o true positive rate (Actual positive - predicted positive): "+str(recall))

        # False positive rate
        # Quan es negatiu quantes vegades diu que es positiu (sequivoca)
        if tnTotal==0 and fpTotal==0:
            fpRate=0
        else:
            fpRate = round((fp/(tn+fp)),2)
        print("False positive rate (Actual negative - predicted positive): "+str(fpRate))

        # True negative rate
        # Specify -> Negatius contra negatius actuals, objectiu minimitzar falsos positius
        # Quan es negatiu quantes vegades diu que es negatiu (encerta)
        if tnTotal==0 and fpTotal==0:
            specify=0
        else:
            specify = round((tn/(tn+fp)),2)
        print("Specify, true negative rate (Actual negative - predicted negative): "+str(specify))

        # False negative rate
        # Quan es positiu quantes vegades diu que es negatiu (sequivoca)
        fnRate = round((fn/(tp+fn)),2)
        print("False negative rate (Actual positive - predicted negative): "+str(fnRate))

        # Precision -> Verdaders positius contra positius predits, objectiu minimitzar falsos positius
        # Quan es positiu quantes vegades prediu correctament
        precision = round((tp/(tp+fp)),2)
        print("Precision (Correct predicted positives): "+str(precision))

        # F-Measure -> Objectiu minimitzar falsos positius i falsos negatius
        f_measure = round(((2*precision*recall)/(precision+recall)),2)
        print("F-Measure (Precision and recall combination): "+str(f_measure))
'''



def main():
    model.eval()
    with torch.no_grad():
        if args.load_embeddings:
            saved_data = torch.load(args.load_embeddings) # loading data.pt file
            testAccuracy(saved_data)
        else:
            if args.path_dataset:
                if args.use_min:
                    createEmbeddings(args.path_dataset,0)
                    saved_data = torch.load('minimPerClasse.pt') # loading data.pt file
                elif args.use_svm:
                    createEmbeddings(args.path_dataset,2)
                    saved_data = torch.load('totsPerClasse.pt') # loading data.pt file
                else:
                    createEmbeddings(args.path_dataset,1)
                    saved_data = torch.load('mitjaPerClasse.pt') # loading data.pt file
                testAccuracy(saved_data)
            else:
                print("No embeddings to compare")




if __name__ == '__main__':
    main()
