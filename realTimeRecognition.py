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
import tensorflow as tf
import matplotlib.pyplot as plt
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
torch.cuda.empty_cache()
import detect_face
import gc
import math
from numpy import dot
from numpy.linalg import norm
from sklearn import svm
gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='Utilitza el model preentrenat')
parser.add_argument('--load-last', type=str, help='Path al checkpoint de lentrenament')
parser.add_argument('--path-dataset', type=str, help='Path a la classe a generar els embeddings')
parser.add_argument('--load-embeddings', type=str, help='Path als embeddings dusuaris ja creats')
parser.add_argument('--use-svm', action='store_true', help='Utilitza el SVM com a classificador')
parser.add_argument('--use-min', action='store_true', help='Utilitza la millor representació de cada classe')
parser.add_argument('--use-euclidian', action='store_true', help='Utilitza la millor representació de cada classe')
parser.add_argument('--euclidianThresh', default=6.0, type=float, metavar='MG', help='Distancia euclidiana maxima per no considerar reconeixament')
parser.add_argument('--cosineThresh', default=0.8, type=float, metavar='MG', help='Similitud cosinosiudal minima per considerar reconeixament')


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


def realTimeRecognition(saved_data): 
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    if args.use_svm:
        embedding_list2 = []
        name_list2 = []
        for e in embedding_list:
            embedding_list2.append(e.cpu().numpy()[0].reshape(1,128)[0]) 
        embedding_list=embedding_list2
        for n in name_list:
            name_list2.append(n)
        name_list=name_list2
        clf = svm.SVC()
        clf.fit(embedding_list,name_list)
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 608)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    face_detector = MTCNN()
    while True:
        ret, frame = video_capture.read()
        if ret == 0:
            print("Error: check if webcam is connected")
            return        
        
        data=face_detector.detect_faces(frame)  
        img_size = numpy.asarray(frame.shape)[0:2] 
        for res in data:
            if res['confidence'] < 0.99:
                continue
            box=res['box'] 
            box[0] = numpy.maximum(box[0]-44/2, 0)
            box[1] = numpy.maximum(box[1]-44/2, 0)
            box[2] = numpy.minimum(box[2]+88/2, img_size[1])
            box[3] = numpy.minimum(box[3]+88/2, img_size[0])
            pt_1 = (int(box[0]),int(box[1]))
            pt_2 = (int(box[0]+box[2]),int(box[1]+box[3]))
            img=frame[int(box[1]):int(box[1])+int(box[3]),int(box[0]):int(box[0])+ int(box[2])]
            img =  Image.fromarray(img).resize(size=(224, 224))            
            img = cv2.cvtColor(numpy.array(img), cv2.COLOR_BGR2RGB)            
            #imgframe=img
            img = trfrm(img).unsqueeze(0).to(device) 
            a = model(img)            
            name = 'Unknown'
            distance = 0
            if args.use_svm:
                predict = clf.predict([a.cpu().numpy()[0]])
                name=predict[0]
                '''predict = clf.predict_proba([a.cpu().numpy()[0]])
                predict=predict[0]
                max_val = numpy.max(predict)
                name=name_list[numpy.where(predict == max_val)[0][0]]'''
            else:
                dist_list = [] # list of matched distances    
                for idx, emb_db in enumerate(embedding_list):
                    if args.use_euclidian:
                        euclidean_distance = F.pairwise_distance(a, emb_db)
                        dist_list.append(euclidean_distance[0])  
                    else:
                        similarity = cosinesimilarity(a, emb_db)
                        dist_list.append(similarity)    
                if args.use_euclidian:
                    idx_min = dist_list.index(min(dist_list))  
                    distance = dist_list[idx_min]
                    print(str(name_list[idx_min])+" - "+str(distance))
                    if distance < args.euclidianThresh:
                        name = name_list[idx_min]
                else:
                    idx_max = dist_list.index(max(dist_list))
                    distance = dist_list[idx_max]
                    print(str(name_list[idx_max])+" - "+str(distance))
                    if distance > args.cosineThresh:                
                        name = name_list[idx_max]
            if name == 'Unknown':
                cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                if args.use_svm:
                    cv2.putText(frame, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
                else:
                    cv2.putText(frame, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
        cv2.imshow('Real time recognition', frame)

        keyPressed = cv2.waitKey(1) & 0xFF
        if keyPressed == 27: # ESC key
            break
        elif keyPressed == 13: # ENTER key
            cv2.imwrite("captura" + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg", frame);
            print('Screenshot saved!')
    video_capture.release()
    cv2.destroyAllWindows()
    #plt.imshow(imgframe)
    plt.show()



def main():
    model.eval()
    with torch.no_grad():
        if args.load_embeddings:
            saved_data = torch.load(args.load_embeddings) # loading data.pt file
            realTimeRecognition(saved_data)
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
                realTimeRecognition(saved_data)
            else:
                print("No embeddings to compare for real time recognition")


if __name__ == '__main__':
    main()
