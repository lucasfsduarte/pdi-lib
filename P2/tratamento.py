from lib.lib import *
from math import copysign, log10
import mahotas, csv

def trataImagens():
    """
    Trata as imagens com a utilização de filtros.

    Args:
        nomePasta: nome da pasta que contém as imagens.

    Returns:
        None.
    """
    for i in range(19): # 19 representa a quantidade de imagens que tem no repositorio "contornosExtraidos"


        nomeImg = "contornos" + str("_%s.png" % i)
        print(nomeImg)
        nomeDiretorio = "contornosExtraidos/"
        img = cv.imread(nomeDiretorio + nomeImg)
        img = cv.resize(img, (400,400))

        # Zernike
        momentoZernike("CSV/" + "dados", img)

        criaLabels("CSV/", str(i))


def momentoZernike(nomeDir, image):

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

    file = open(nomeDir + ".csv", "a")

    caracteristicasVetorizadas = str(caracteristicasVetorizadas).replace('\n', '')
    caracteristicasVetorizadas = (caracteristicasVetorizadas[1:]).replace(']', '')
    caracteristicasVetorizadas = caracteristicasVetorizadas.replace('  ', ' ')
    caracteristicasVetorizadas = caracteristicasVetorizadas.replace(' ', ',')
    linha_resultante = caracteristicasVetorizadas.replace(',,', ',')
    linha_resultante = linha_resultante.replace('  ', ' ')
    linha_resultante = linha_resultante.split(',')

    # Remover item vazio '' inserido incorretamente
    while(True):
        try:
            linha_resultante.remove('')
        except:
            break

    linha_resultante = ','.join(linha_resultante)
    linha_resultante = linha_resultante.replace(',', ', ')

    file.write(linha_resultante + "\n") # SE FOR USAR O HU, CONCATENAR A STRHU AQUI NO COMEÇO
    file.close()

def criaLabels(nomeDir, letra):
    file = open(nomeDir + "labels" + ".csv", "a")
    file.write(letra + "\n")
    file.close()

trataImagens()
