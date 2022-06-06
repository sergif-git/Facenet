# PFG
Facial recognition with machine learning - Degree Final Project

These projecte uses machine learning to solve facial recognition in real time.
Models are trained using the FaceNet CNN architecture and the triplet loss function.
Input images needs to be 224 x 244 pixels size and the output resulting embeddings are 128.
To extract and align faces from original images MTCNN is used.
A SVC is used to classify embeddings for each class.

Multiple programs are served to train, validate, and use the trained model, note that all these programs are optimized to validate a trained model on 9 users


ALIGN IMAGES
for N in {1..4}; do python align_dataset_mtcnn.py /home/Desktop/project/datasets/wild  /home/Desktop/project/datasets/wildaligned --image_size 224 --margin 44 --gpu_memory_fraction 0.25 & done

CSV GENERATION (used to load faces)
python write_csv_for_making_dataset.py --root-dir= /home/Desktop/project/datasets/wildalignedtrain --final-file= /home/Desktop/project/datasets/wildaligned/oficinaaligneddatasettrain.csv

TRIPLET LOSS TRAINING
python train.py --train-root-dir /home/Desktop/project/datasets/wildalignedtrain --train-csv-name /home/Desktop/project/datasets/wildalignedtrain.csv --valid-root-dir /home/Desktop/project/datasets/wildalignedval --valid-csv-name /home/Desktop/project/datasets/wildalignedval.csv --num-epochs 400 --pretrain --margin 0.5

TRAINING PLOT
python plottrainresults.py --root-dir /home/Desktop/project/log/modeloriginal

EMBEDDINGS DISTANCES PLOT
python distancies.py --test-root-dir /home/Desktop/project/datasets/wildalignedtrain --test-csv-name /home/Desktop/project/datasets/wildalignedtest.csv --pretrain
![pretrainDistClasses](https://user-images.githubusercontent.com/27964097/172219458-79437028-4233-4da7-8915-77a9bb0322e2.png)

EMBEDDINGS VALUES PLOT
python embeddingsPlot.py --pretrain --path-dataset /home/Desktop/project/datasets/wildalignedtest
![embeddingsMeanPerClass](https://user-images.githubusercontent.com/27964097/172219403-277e715e-a027-43cf-8adb-369e4641ae77.jpg)
![embeddingsPerClass](https://user-images.githubusercontent.com/27964097/172219407-4cdacb15-ac4e-4154-9d35-013355a5646f.jpg)

EUCLIDIAN THRESHOLD TEST
python thresholdtest.py --test-root-dir /home/Desktop/project/datasets/wildalignedtest --test-csv-name /home/Desktop/project/datasets/wildalignedtest.csv --pretrain --num-classif 1000
![pretrainThreshTest](https://user-images.githubusercontent.com/27964097/172219430-7978ed31-8e2d-4b38-90b8-0dacdcd45593.png)

COSINE SIMILARITY TEST
python cosinetest.py --test-root-dir /home/Desktop/project/datasets/wildalignedtest --test-csv-name /home/Desktop/project/datasets/wildalignedtest.csv --pretrain --num-classif 1000
![pretrainCosineTest](https://user-images.githubusercontent.com/27964097/172219441-5596789b-591c-4ff9-9fa8-8ed5f640d163.png)

CLASSIFICATION PLOT
python classify.py --test-root-dir /home/Desktop/project/datasets/wildalignedtest --test-csv-name /home/Desktop/project/datasets/wildalignedtest.csv --use-euclidian --threshold 9.0
![pretrainRandomClassif](https://user-images.githubusercontent.com/27964097/172219479-806c7285-def8-4e50-bbad-de61f765d8be.png)

CONFUSION MATRIX (usine minium embeddings, mean embedding, or SVC)
python modelAccuracy.py --path-dataset /home/Desktop/project/datasets/wildalignedtrain --path-dataset-testing /home/Desktop/project/datasets/wildalignedtest --use-euclidian --euclidianThresh 9.0

REAL TIME RECOGNITION
python realTimeRecognition.py --pretrain --load-last './log/modelx/last_checkpoint.pth' --path-dataset /home/Desktop/project/datasets/wildalignedtrain --use-euclidian --euclidianThresh 8.0
