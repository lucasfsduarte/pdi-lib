import sys
import cv2 as cv
import numpy as np
import pandas as pd
import mahotas, csv
import matplotlib.pyplot as plt
import time
import timeit

from matplotlib import pyplot as plt
from math import copysign, log10
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def cannyFilter(img):
    """
    Aplica o filtro de Canny e gera uma nova imagem.

    Args:
        img: imagem a ser utilzada.

    Returns:
        img_canny: imagem com o filtro aplicado.
    """
    img_canny = cv.Canny(img, 50, 100)
    return img_canny

def momentoZernikeWebCam(image):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #cv2.imshow("saida",gray)
    #cv2.waitKey(100)

    #aplicando filtro de media
    filtroMedia = cv.GaussianBlur(gray,(5,5),0)

    ret3,segmentada = cv.threshold(filtroMedia,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    segmentada = cv.erode(segmentada, None, iterations=1)
    segmentada = cv.resize(segmentada, (400,400))

    cnts, hierarchy  = cv.findContours(segmentada.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # a função findCountours devolve vários contornos da imagem original
    # existentes devidos a pequenos objetos que ainda aparecem na imagem.
    # este codigo armazena em maior_index o indice do maior contorno obtido
    # e em segunda_index, o segundo maior. O maior contorno tem que ser a mao...
    maior_area = 0; segunda_area = 0; maior_index = 0; segunda_index = 0

    for i, c in enumerate(cnts):
        area = cv.contourArea(c)
        if (area > maior_area):
            if (area > segunda_area):
                segunda_area = maior_area
                maior_area = area
                maior_index = i
        elif (area > segunda_area):
            segunda_area = area
            segunda_index = i

    mask = np.zeros(image.shape[:2], dtype="uint8")
    #desenha o maior contorno
    cv.drawContours(image, cnts, maior_index, (0, 255, 0), 2)
    image2 = cv.resize(image, (400,400))

    #define o menor retangulo circunscrito no objeto detectado como parametro para recortar o objeto e
    # obter uma imagem destacada do objeto
    (x,y,w,h) = cv.boundingRect(cnts[maior_index])
    destacada = mask[y:y+h, x:x+w]
    #calcula os monetos e converte para uma array, devolvendo este valor
    caracteristicas = mahotas.features.zernike_moments(segmentada, cv.minEnclosingCircle(c)[1], degree=8)
    caracteristicasVetorizadas = np.asarray(caracteristicas)

    caracteristicasVetorizadas = str(caracteristicasVetorizadas).replace('\n', '')
    caracteristicasVetorizadas = (caracteristicasVetorizadas[1:]).replace(']', '')
    caracteristicasVetorizadas = caracteristicasVetorizadas.replace('  ', ' ')

    return caracteristicasVetorizadas

def momentoHuWebCam(img):
    # Threshold image
    _,img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

    # Calculate Moments
    moment = cv.moments(img)

    # Calculate Hu Moments
    huMoments = cv.HuMoments(moment)

    # file = open(nomeDir + ".csv", "a")
    resultadoHu = ""
    maxIteracoes = 7
    for i in range(0,maxIteracoes):

        valor = "{:5f}".format(-1*copysign(1.0,huMoments[i])*log10(abs(huMoments[i])))
        #file.write(valor + ", ")
        resultadoHu += str(valor) + " "

    # file.close()
    return resultadoHu

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
          # xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
