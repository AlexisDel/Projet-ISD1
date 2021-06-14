import numpy as np
import cv2
import os.path


''' renvoie le centre de gravité des pixels de <img> différents de <background> '''
def center(img, background=[0,0,0]):

    #dimensions de l'image
    width, height, channel = img.shape

    # somme des positions des pixels en x et en y
    mx, my = 0, 0

    # nombre de pixels
    n = 0

    # on fait la somme des position des pixels détectés
    for x in range(width):
        for y in range(height):
            if not np.array_equal(np.array(img[x,y]), np.array(background)):
                n = n + 1
                mx = mx + x
                my = my + y

    # si aucun pixel n'a été détecté, on renvoie le centre de l'image
    if n == 0:
        return width//2, height//2
        
    # moyenne des positions des pixels
    return int(mx/n), int(my/n)



''' renvoie le crop de <img> de centre (<cx>;<cy>) et de taille <size> '''
def crop(img, cx, cy, size=32):
    width, height, channel = img.shape
    if cx-size//2 < 0:
        cx = size//2
    if cy-size//2 < 0:
        cy = size//2
    if cx+size//2 >= width:
        cx = width-size//2
    if cy+size//2 >= height:
        cy = height-size//2
    return img[cx-size//2:cx+size//2, cy-size//2:cy+size//2]



''' renvoie une image identique à <img>, mais les pixels du background sont noirs '''
def foreground(img):

    # on rend l'image floue
    blur = cv2.GaussianBlur(img, (5,5), 10, 10)

    # on utilise l'espace de couleurs HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # filtre vert-jaune
    lower1 = np.array([22, 60, 128])
    upper1 = np.array([55, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # filtre rouge 1
    lower2 = np.array([0, 80, 128])
    upper2 = np.array([10, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # filtre rouge 2
    lower3 = np.array([170, 80, 128])
    upper3 = np.array([180, 255, 255])
    mask3 = cv2.inRange(hsv, lower3, upper3)

    # superposition des filtres
    maskf = cv2.bitwise_or(mask1, mask2)
    maskf = cv2.bitwise_or(maskf, mask3)

    # application du filtre final
    return cv2.bitwise_and(img, img, mask=maskf)


''' afficher l'image dans une fenêtre '''
def show_img(img, w=360, h=360):
    cwindow = cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', w, h)
    cv2.imshow('img', img)
    cv2.waitKey(0)
