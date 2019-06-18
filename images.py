from general import *

def imgRead(path, grey=0):
    if grey == 1: return cv.imread(path, 0)
    else: return cv.imread(path)


def imgWrite(path, img):
    cv.imwrite(path, img)


def imgShow(img, desc='image'):
    cv.imshow(desc, img)


def imgSplit(img):
    return cv.split(img)


def imgResize(img, xy):
    return cv.resize(img, xy)


def imgBinaryThresh(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(grey, 127, 255, cv.THRESH_BINARY)
    return th


def imgCrop(img, max, min):
    # cv.rectangle(img, max, min, (0, 0, 255), 0)
    return img[min[0]:max[0], min[1]:max[1]]


def imgOtsuThresh(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(grey, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th


def imgCmpColors(color, colors):
    """
    Módulo responsável por classificar uma cor X em termos de semelhança com
    outras cores.

    Args:
        color: cor a ser classificada (em bgr);
        colors: lista de cores de referência (centroides, em bgr).

    Returns:
        bestColor: a cor entre a lista de cores mais próxima à cor X.
    """
    bestIndex = 766
    bestColor = [0, 0, 0]
    for i in range(0, len(colors)):
        b = abs(int(color[0]) - int(colors[i][0]))
        g = abs(int(color[1]) - int(colors[i][1]))
        r = abs(int(color[2]) - int(colors[i][2]))
        total = b + g + r
        if total < bestIndex:
            bestIndex = total
            bestColor = colors[i]
    return bestColor

def imgKmeans(img, k):
    """
        Módulo de execução do algoritmo K-means sobre uma imagem.

        Args:
            img: imagem a ser utilizada;
            k: int - parâmetro do algoritmo (número de centroides).

        Returns:
            center: lista de centroides gerados (bgr).
    """
    kImg = img
    z = kImg.reshape((-1,3))
    z = np.float32(z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((kImg.shape))
    # cv.imshow('res2',res2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return res2, center

def imgGenHistogram(img, description):
    """
        Módulo responsável por aplicar o algoritmo de Otsu e gerar o histograma.

        Args:
            img: imagem base;
            description: nome do png gerado.

        Returns:
            None.
    """
    hImg = img
    ret, imgf = cv.threshold(hImg, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.subplot(3,1,1), plt.imshow(hImg,cmap = 'gray')
    plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,2), plt.hist(hImg.ravel(), 256)
    plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
    plt.title('Histogram'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,1,3), plt.imshow(imgf,cmap = 'gray')
    plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])
    plt.savefig('histogramas/img_%s.png' % description)
    plt.close()

def imgConvert(img, conv):
    if conv == 'hsv': return cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elif conv == 'lab': return cv.cvtColor(img, cv.COLOR_BGR2LAB)
    elif conv == 'hsi': return cv.cvtColor(img, cv.COLOR_BGR2HSI)
    elif conv == 'gray': return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else: return None

def imgGaussian(img):
    return cv.GaussianBlur(img, (3, 3), 0)

def imgWait():
    cv.waitKey(0)
    cv.destroyAllWindows()
