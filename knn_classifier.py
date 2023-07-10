from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import os,sys
from pre_processing import preProcessing

dataset_after_processing_path = './hhd_AP/'
def getDataByName(name,mode=1,dirNum=0):
    count=0
    images=[]
    labels=[]
    path_string = f'{dataset_after_processing_path}/{name}'
    if mode: #training or validation
        for dir in os.listdir(path_string):
            for img in os.listdir(f'{path_string}/{dir}'):
                image = cv2.imread(f'{path_string}/{dir}/{img}')    
                images.append(image.flatten())
                labels.append(dir)

    else:
        for img in os.listdir(f'{path_string}/{dirNum}'):
            image = cv2.imread(f'{path_string}/{dirNum}/{img}')    
            images.append(image.flatten())
            labels.append(str(dirNum))

    return np.array(images),np.array(labels)

#choosing the best k value
def chooseKValue(TrainImgs,TrainLbls,ValidImgs,ValidLbls):
    k_dict = {}
    for k in range(1,16,2):
        model = KNeighborsClassifier(n_neighbors = k, metric='euclidean')
        model.fit(TrainImgs,TrainLbls)
        acc = model.score(ValidImgs,ValidLbls)
        k_dict[k] = acc
    return max(k_dict,key=k_dict.get)

#training the model with the best k value, and 80% of the photos
def modelTraining():
    TrainImgs,TrainLbls= getDataByName('training')
    ValidImgs,ValidLbls= getDataByName('validation')
    K = chooseKValue(TrainImgs,TrainLbls,ValidImgs,ValidLbls)
    model = KNeighborsClassifier(n_neighbors = K, metric='euclidean')
    model.fit(TrainImgs,TrainLbls)
    return model,K

#Testing phase
def modelTesting(model,K):
    acc_total = 0
    #for confusion matrix
    predicts = np.array([])
    realLabels = np.array([])

    #accuracy results with precentage
    f = open('results.txt','w')
    f.write(f'K = {K}\nLetter    Accuracy\n\n')

    for i in range(0,27):
        images,labels= getDataByName('testing',0,i)
        acc = model.score(images,labels)
        acc_total+=acc
        predicts = np.concatenate((predicts,model.predict(images)))
        realLabels = np.concatenate((realLabels,labels))
        #write to results file the percentage of each letter
        f.write(f'{i:<10} {acc*100:.2f}%\n')
    #write to results file the avg percentage 
    acc_avg = acc_total/27
    f.write(f'\nAverage accuracy: {acc_avg*100:.2f}%\n')

    #confusion matrix
    cm=confusion_matrix(realLabels, predicts, labels=[str(i) for i in range(0,27)])
    f = open('confusion_matrix.txt','w')
    f.write(f'{"":<5}')
    [f.write(f'{str(i):<5}') for i in range(0,27)]
    f.write('\n_________________________________________________________________________________________________________________________________________')
    for i in range(0,27):
        f.write(f'\n{str(i):<2}{"|":<3}')
        for x in cm[i]:
            f.write(f'{str(x):<5}')


def main():
    path = sys.argv[1]
    preProcessing(path)
    model,K = modelTraining()
    modelTesting(model,K)

if __name__ == "__main__":
    main()