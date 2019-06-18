import cv2
import numpy as np
import random as rng
import random as rng
rng.seed(12345)
def contornoConvexo(imagem):
    threshold = 128
    # Detect edges using Canny
    canny_output = cv2.Canny(imagem, threshold, threshold * 2)
    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        drawing2 = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(drawing2, contours, i, color)

        # Primeira execucao para gerar imagens base
        cv2.imwrite("contornosExtraidos/contornos" + "_%s.png" % i, drawing2)
        # Segunda execucao para gerar imagens teste
        # cv2.imwrite("contornosTeste/contornos" + "_%s.png" % i, drawing2)
        cv2.drawContours(drawing2, hull_list, i, color)

    # Show in a window
    # cv2.imshow('Contours', drawing)

# Let's load a simple image with 3 black squares

# Imagem base
image = cv2.imread('imgs/shapes.jpg')
# Imagem para classificar
# image = cv2.imread('imgs/shapes.jpg')

cv2.waitKey(0)
imagem2=image

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contorno, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contorno)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contorno, -1, (0, 255, 0), 3)
# detectando o centro de cada objeto
i = 1
imagem2=image
for c in contorno:
    M = cv2.moments(c)
    centrox = int(M['m10'] / M['m00'])
    centroy = int(M['m01'] / M['m00'])
    cv2.circle(image, (centrox, centroy), 5, (255, 255, 255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(i), (centrox, centroy), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # imprimindo area
    print("Area do objeto " + str(i) + ": ", cv2.contourArea(c), "\tPerímetro " + str(i) + ": ",
          round(cv2.arcLength(c, True), 2))
    x,y,w,h=cv2.boundingRect(c)
    imagem2 = cv2.rectangle(imagem2,(x,y),(x+w,y+h),(0,0,255),3)
    (x,y),raio=cv2.minEnclosingCircle(c)
    centro=(int(x),int(y))
    raio=int(raio)
    imagem2=cv2.circle(imagem2,centro,raio,(0,255,0),2)
    # cv2.imwrite("teste2" + "_%s.png" % i, imagem2)
    i = i + 1

i=1
for c in contorno:
    x,y,w,h=cv2.boundingRect(c)
    area=cv2.contourArea(c)
    area_retangulo=w*h
    extensao=round(float(area/area_retangulo),2)
    aspecto= round(float(w/h),2)
    # major Axis, minor axis -> Ma,ma
    (x,y),(Ma,ma),angulo=cv2.fitEllipse(c)
    angulo=round(angulo,2)
    print("razao de aspecto da imagem ", i, aspecto, "\tExtensao: ", extensao,"\tangulo: ",angulo )
    i = i + 1

contornoConvexo(gray)
# cv2.imshow('contorno', image)
# cv2.imshow('Bounding Box', imagem2)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
FONT_HERSHEY_SIMPLEX
normal size sans-serif font

FONT_HERSHEY_PLAIN
small size sans-serif font

FONT_HERSHEY_DUPLEX
normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)

FONT_HERSHEY_COMPLEX
normal size serif font

FONT_HERSHEY_TRIPLEX
normal size serif font (more complex than FONT_HERSHEY_COMPLEX)

FONT_HERSHEY_COMPLEX_SMALL
smaller version of FONT_HERSHEY_COMPLEX

FONT_HERSHEY_SCRIPT_SIMPLEX
hand-writing style font

FONT_HERSHEY_SCRIPT_COMPLEX
more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX

FONT_ITALIC
flag for italic font
'''
