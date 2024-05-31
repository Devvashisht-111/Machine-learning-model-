!pip install bing-image-downloader
!mkdir Dataset_image
from bing_image_downloader import downloader
downloader.download("rose",limit=30,output_dir="Dataset_image",adult_filter_off=True)
downloader.download("apple fruit",limit=30,output_dir="Dataset_image",adult_filter_off=True)
downloader.download("icecream cone",limit=30,output_dir="Dataset_image",adult_filter_off=True)
#The code is use for preprocessing
#Data is in array format so, the data will be flatten and resize
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
Final=[]
Images=[]
Flatten_data=[]
DATADIR = '/content/Dataset_image'
CATEGORIES = ['rose','apple fruit','icecream cone']
for i in CATEGORIES:
  Class_index = CATEGORIES.index(i)#label encoding the values
  path = os.path.join(DATADIR,i)
  for j in os.listdir(path):
    image_array = imread(os.path.join(path,j))
    resized_array = resize(image_array,(150,150,3))
    Flatten_data.append(resized_array.flatten())
    Images.append(resized_array)
    Final.append(Class_index)
Flatten_data = np.array(Flatten_data)
Final = np.array(Final)
Images= np.array(Images)
unique,count = np.unique(Final,return_counts=True)
plt.bar(CATEGORIES,count)
#split data into two set train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Flatten_data,Final,test_size=0.3,random_state=109)
from sklearn.model_selection import GridSearchCV
from sklearn import svm

param_grid = [{'C':[1,10,100,1000],'kernel':['linear']},
              {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}]
svc = svm.SVC(probability=True)
clf = GridSearchCV(svc,param_grid)
clf.fit(x_train , y_train)
#checking the predicted output
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)
#saving model using pickle library
import pickle
pickle.dump(clf,open('image_model.p','wb'))
model= pickle.load(open('image_model.p','rb'))
#testing a new iamge
Flatten_data = []
url = input('Enter the URL')
img= imread(url)
resized_array = resize(img,(150,150,3))
Flatten_data.append(resized_array.flatten())
Flatten_data = np.array(Flatten_data)
print(img.shape)
plt.imshow(resized_array)
y_out = model.predict(Flatten_data)
y_out = CATEGORIES[y_out[0]]
print(f' PREDICTED OUTPUT: {y_out}')
