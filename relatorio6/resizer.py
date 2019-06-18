import cv2

def resizeImg(imgPath, letter, index, width=120, height=120):
    img = cv2.imread(imgPath)
    newimg = cv2.resize(img, (width, height))

    cv2.imwrite('alphabet/{}{}.jpg'.format(letter, index), newimg)

def getImgPath(imgName, format=None):
    if (format is not None):
        return 'images/{}.{}'.format(imgName, format)
    return 'images/{}'.format(imgName)

alphabet = {
    'a': ['images/a_{}.jpg'.format(i) for i in range(1, 11)],
    'b': ['images/b_{}.jpg'.format(i) for i in range(1, 11)],
    'c': ['images/c_{}.jpg'.format(i) for i in range(1, 11)],
    'd': ['images/d_{}.jpg'.format(i) for i in range(1, 11)],
    'e': ['images/e{}.jpg'.format(i) for i in range(1, 11)],
    'f': ['images/f{}.jpg'.format(i) for i in range(1, 11)],
    'g': ['images/g{}.jpg'.format(i) for i in range(1, 11)],
    'h': ['images/h{}.jpg'.format(i) for i in range(1, 11)],
    'i': ['images/I{}.jpg'.format(i) for i in range(1, 11)],
    'j': ['images/J{}.jpg'.format(i) for i in range(1, 11)],
    'k': ['images/K{}.jpg'.format(i) for i in range(1, 11)],
    'l': ['images/L{}.jpg'.format(i) for i in range(1, 11)],
    'm': ['images/m{}.jpg'.format(i) for i in range(0, 10)],
    'n': ['images/n{}.jpg'.format(i) for i in range(0, 10)],
    'o': ['images/o{}.jpg'.format(i) for i in range(1, 11)],
    'p': ['images/p{}.jpg'.format(i) for i in range(1, 11)],
    'q': ['images/q{}.jpg'.format(i) for i in range(1, 11)],
    'r': ['images/r{}.jpg'.format(i) for i in range(1, 11)],
    's': ['images/s{}.jpg'.format(i) for i in range(1, 11)],
    't': ['images/t{}.jpg'.format(i) for i in range(1, 11)],
    'u': ['images/u{}.jpg'.format(i) for i in range(0, 10)],
    'v': ['images/v_{}.jpg'.format(i) for i in range(1, 11)],
    'w': ['images/{}.jpg'.format(i) for i in range(1, 11)],
    'x': ['images/x{}.jpg'.format(i) for i in range(1, 11)],
    'y': ['images/y{}.jpg'.format(i) for i in range(1, 11)],
    'z': ['images/z{}.jpg'.format(i) for i in range(1, 11)]
}


for letter in alphabet:
    for i in range(len(alphabet[letter])):
        img = alphabet[letter][i]
        resizeImg(img, letter, i, width=240, height=240)
