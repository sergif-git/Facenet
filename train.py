import argparse
import datetime
import time
import math
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from torch.nn.modules.distance import PairwiseDistance
from torch.optim import lr_scheduler
from data_loader import get_dataloader
from write_csv_for_making_dataset import write_csv
from eval_metrics import evaluate, plot_roc
from loss import TripletLoss
from models import FaceNetModel, model_920
from utils import ModelSaver, init_log_just_created
torch.cuda.empty_cache()
import gc
gc.collect()

# Parametres generals per entrenar
parser = argparse.ArgumentParser()
parser.add_argument('--num-epochs', default=1000, type=int, metavar='NE',
                    help='Nombre de epochs per entrenament (default: 200)')
parser.add_argument('--num-train-triplets', default=2000, type=int, metavar='NTT',
                    help='Nombre de triplets per entrenament (default: 300)')
parser.add_argument('--num-valid-triplets', default=1000, type=int, metavar='NVT',
                    help='Nombre de triplets per validació (default: 270)')
parser.add_argument('--batch-size', default=9, type=int, metavar='BS',
                    help='Mida del batch (default: 9)')
parser.add_argument('--num-workers', default=4, type=int, metavar='NW',
                    help='Nombre de processos en paral·lel (default: 4)')
parser.add_argument('--learning-rate', default=0.0001, type=float, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--step-size', default=50, type=int, metavar='SZ',
                    help='Passos per la reducció del learning learning rate (default: 50)')
parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
                    help='Marge per als parells negatius de la funció triplet loss (default: 0.5)')
parser.add_argument('--train-root-dir', type=str, help='Path a les dades dentrenament')
parser.add_argument('--valid-root-dir', type=str, help='Path a les dades de validació')
parser.add_argument('--train-csv-name', type=str, help='Path al csv de les imatges dentrenament')
parser.add_argument('--valid-csv-name', type=str, help='Path al csv de les imatges de validació')
parser.add_argument('--pretrain', action='store_true')


args = parser.parse_args()
# Crida a la utilització de la gpu si aquesta esta disponible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Distancia quatratica entre dos vectors
l2_dist = PairwiseDistance(2)
# Instancia la classe implementada a 'utils.py' per guardar el model en qualsevol moment
modelsaver = ModelSaver()


# Guarda el model si ha obtingut millor precisió
def save_if_best(state, acc):
    modelsaver.save_if_best(acc, state)


def main():
    # Crea si no existeixen els csv corresponents al entrenament del model preentrenat
    init_log_just_created("log/valid.csv")
    init_log_just_created("log/train.csv")
    print('Number of epochs to train: '+str(args.num_epochs))
    print('Batch size: '+str(args.batch_size))
    print('Number of training triplets: '+str(args.num_train_triplets))
    print('Number of validation triplets: '+str(args.num_valid_triplets))
    print('Initial learning rate to train: '+str(args.learning_rate))
    print('Learning decayed every '+str(args.step_size)+' epochs')
    print('Triplet loss margin to train: '+str(args.margin))
    
    # Carrega del model facenet definit per la classe 'FaceNetModel' dins l'arxiu 'utils.py'
    pretrain = args.pretrain
    model = FaceNetModel(pretrained=pretrain)
    model.to(device) # Carrega del model a memoria gpu/cpu
    #El classificador no sera entrenat
    model.freeze_only(['fc', 'classifier'])
        
    # Carrega de la triplet loss definida per la classe 'TripletLoss' dins l'arxiu 'loss.py' a memoria gpu/cpu
    triplet_loss = TripletLoss(args.margin).to(device)

    # Optimitzador adam
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    # Redueix el learning rate en valor igual a gamma cada cada step_size vegades
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    model = torch.nn.DataParallel(model) # Paral·lelitza el model en memòria


# Inici entrenament
    # Per cada epoch
    start_epoch = 0
    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, args.num_epochs + start_epoch - 1))

        time0 = time.time()
        # Obtenció de l'iterable data_loader per al dataset donat
        data_loaders, data_size = get_dataloader(args.train_root_dir, args.valid_root_dir,
                                                 args.train_csv_name, args.valid_csv_name,
                                                 args.num_train_triplets, args.num_valid_triplets,
                                                 args.batch_size, args.num_workers)
        # Crida per entrenar el model
        train_valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size)
        print(f'  Execution time                 = {time.time() - time0}')
    print(80 * '=')
    print('Triplet loss training finished')


def save_last_checkpoint(state):
    torch.save(state, 'log/last_checkpoint.pth')


def train_valid(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size):
    for phase in ['train', 'valid']:

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]

        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))

        write_csv(f'log/{phase}.csv', [epoch, avg_triplet_loss])

        if phase == 'valid':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.module.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'loss': avg_triplet_loss
                                  })
        



def train_valid1(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size,last_tri_loss_train, last_tri_loss_val, last_acc_train, last_acc_val):
    cas = 0
    for phase in ['train', 'valid']:
        labels, distances = [], []
        triplet_loss_sum = 0.0
        if phase == 'train':
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
            model.train()
        else:
            model.eval()
        for batch_idx, batch_sample in enumerate(dataloaders[phase]):
            if cas == 0:
                anc_img = batch_sample['anc_img'].to(device)
                pos_img = batch_sample['pos_img'].to(device)
                neg_img = batch_sample['neg_img'].to(device)

                with torch.set_grad_enabled(phase == 'train'):

                    # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                    anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                    # choose the semi hard negatives only for "training"
                    pos_dist = l2_dist.forward(anc_embed, pos_embed)
                    neg_dist = l2_dist.forward(anc_embed, neg_embed)

                    all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                    if phase == 'train':
                        hard_triplets = np.where(all == 1)
                        if len(hard_triplets[0]) == 0:
                            continue
                    else:
                        hard_triplets = np.where(all >= 0)

                    anc_hard_embed = anc_embed[hard_triplets]
                    pos_hard_embed = pos_embed[hard_triplets]
                    neg_hard_embed = neg_embed[hard_triplets]

                    triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                    if phase == 'train':
                        optimizer.zero_grad()
                        triplet_loss.backward()
                        optimizer.step()

                    distances.append(pos_dist.data.cpu().numpy())
                    labels.append(np.ones(pos_dist.size(0)))

                    distances.append(neg_dist.data.cpu().numpy())
                    labels.append(np.zeros(neg_dist.size(0)))

                    triplet_loss_sum += triplet_loss.item()

        # Si per aquest batch no tinc cap semi hard triplet
        # o be no en tinc prou pel calcular accuracy, no computo res, em quedo amb els resultats anteriors
        if len(distances)==0 or len(distances)<10 or cas==1:
            cas=1
            print('Semi hard embeddings not founded in this batch')
            if phase=='train':
                avg_triplet_loss=last_tri_loss_train
                accuracy=last_acc_train
            else:
                avg_triplet_loss=last_tri_loss_val
                accuracy=last_acc_val
        else:
            avg_triplet_loss = triplet_loss_sum / data_size[phase]
            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for dist in distances for subdist in dist])
            accuracy = evaluate(distances, labels)

        if phase=='train':
            last_tri_loss_train=avg_triplet_loss
            last_acc_train=accuracy
        else:
            last_tri_loss_val=avg_triplet_loss
            last_acc_val=accuracy

        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_csv(f'log/{phase}.csv', [epoch, np.mean(accuracy), avg_triplet_loss])

        if phase == 'valid':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.module.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy': np.mean(accuracy),
                                  'loss': avg_triplet_loss
                                  })
            save_if_best({'epoch': epoch,
                          'state_dict': model.module.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy': np.mean(accuracy),
                          'loss': avg_triplet_loss
                          }, np.mean(accuracy))
    return last_tri_loss_train, last_tri_loss_val, last_acc_train, last_acc_val


if __name__ == '__main__':
    main()
