#Importation des bibliothèques
import os
import cv2
import numpy as np
from sys import path
from utilities import *
from fonctions_projet import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score as sklearn_metric


''' main '''

#Création des répertoires
os.system("mkdir fruit")
os.system("mkdir crop_fruit")
os.system("mkdir crop")

#Définition de l'emplacement des répertoires
raw_data_path = "raw_data/"		#répertoires du dataset initial
fruit_path = "fruit/"
crop_fruit_path = "crop_fruit/"
crop_path = "crop/"

#Nombre d'images de pommes et de bananes
nimg_a = 333
nimg_b = 158

#Crop des pommes
for n in range(1, nimg_a+1):
	img_name = "/a"

	if n < 10:
		img_name = img_name + "0"

	img_name = img_name + str(n) + ".png"

	img = cv2.imread(raw_data_path + img_name)

	fruit = foreground(img)

	cx, cy = center(fruit)

	crop_img = crop(img, cx, cy)

	crop_fruit = crop(fruit, cx, cy)

	cv2.imwrite(crop_path + img_name, crop_img)
	cv2.imwrite(fruit_path + img_name, fruit)
	cv2.imwrite(crop_fruit_path + img_name, crop_fruit)


#Crop des bananes
for n in range(1, nimg_b+1):
	img_name = "/b"

	if n < 10:
		img_name = img_name + "0"

	img_name = img_name + str(n) + ".png"

	img = cv2.imread(raw_data_path + img_name)

	fruit = foreground(img)

	cx, cy = center(fruit)

	crop_img = crop(img, cx, cy)

	crop_fruit = crop(fruit, cx, cy)

	cv2.imwrite(crop_path + img_name, crop_img)
	cv2.imwrite(fruit_path + img_name, fruit)
	cv2.imwrite(crop_fruit_path + img_name, crop_fruit)


#Récupération des images crop
data_dir = 'crop_fruit/'
a_files = get_files(data_dir, 'a')
b_files = get_files(data_dir, 'b')

#Preprocessing du dataset
X, Y = preprocess_data(a_files, b_files, extract_cropped_image)

#Sauvegarde du dataset
my_data_dir = './my_dataset/'
file_name = os.path.join(my_data_dir, 'CROP_data.csv') # Cropped data (pixels)
data_to_csv(X, Y, file_name=file_name)
df = pd.read_csv(os.path.join(my_data_dir, 'CROP_data.csv'))
df.head()

#Application de l'algorithme des k (k=3) plus proches voisins au dataset
sklearn_model = KNeighborsClassifier(n_neighbors=3)

#Sauvegarde des résultats
p_tr, s_tr, p_te, s_te = df_cross_validate(df, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()

#Affichage des résultats
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
