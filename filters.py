from general import *
from images import imgGaussian

# Gera o histograma de uma imagem;
def filGenHistogram(img):

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    return hist.cumsum()


# Equaliza uma imagem a partir de seu histograma;
def filEqualizateImage(img):

    hist = filGenHistogram(img)
    cdf_m = np.ma.masked_equal(hist, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]


# Aplica o filtro de média e gera uma nova imagem;
def filMean(img, anchor):

    # anchor: tamanho da matriz da máscara do filtro (3, 5, 7);
    mean = cv.blur(img, (anchor, anchor))
    return mean


# Aplica o filtro de mediana e gera uma nova imagem;
def filMedian(img, k):

    # k: tamanho linear de abertura (3, 5, 7);
    median = cv.medianBlur(img, k)
    return median


# Aplica o filtro de canny e gera uma nova imagem;
def filCanny(img):

    canny = cv.Canny(img, 100, 200)
    return canny


# Aplica o filtro de sobel e gera uma nova imagem;
def filSobel(img):

    gaussian = imgGaussian(img);
    sobelx = cv.Sobel(gaussian, cv.CV_8U, 1, 0, ksize = 5)
    sobely = cv.Sobel(gaussian, cv.CV_8U, 0, 1, ksize = 5)
    return sobelx + sobely


# Aplica o filtro de prewitt e gera uma nova imagem;
def filPrewitt(img):

    gaussian = imgGaussian(img);
    kernelx = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
    kernely = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
    prewittx = cv.filter2D(gaussian, -1, kernelx)
    prewitty = cv.filter2D(gaussian, -1, kernely)
    return prewittx + prewitty


# Aplica o filtro de roberts e gera uma nova imagem;
def filRoberts(img):

    edges = cv.Canny(img, 100, 200)
    return edges
