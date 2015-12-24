import cv2
from matplotlib import pyplot as plt
import sklearn
import numpy as np
import pickle as pk
from os import listdir

plt.style.use('ggplot')

NR_WORDS = 1000

from os import listdir

def load_images(imgs_paths, gray=False):
    for path in imgs_paths:
        img = cv2.imread(path)
        
        if gray:
            yield cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            yield cv2.imread(path)
            
# SIFT features detector and extractor
sift = cv2.xfeatures2d.SIFT_create()

# FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

def train_bow(detector, matcher, extractor=None):
    if extractor == None:
        extractor = detector
    
    bow_extractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
    
    vocabulary = pk.load(open('vocabulary_1000w.p', 'rb'))
    
    bow_extractor.setVocabulary(vocabulary)
    
    return bow_extractor
    
detector = sift
extractor = sift

sift_bow_extractor = train_bow(detector, flann, extractor=extractor)

train_folder = 'data/train/'

imgs_paths = [train_folder + filepath for filepath in listdir(train_folder)]

best_clf = pk.load(open('svm_classifier.p', 'rb'))

def save_labels_csv(labels):

    pk.dump(labels, open('labels.p', 'wb'))

    indexes = np.asmatrix(range(1, len(labels)+1)).transpose()
    labels = np.asmatrix(labels).transpose()
    
    indexed_labels = np.concatenate((indexes, labels), axis=1)
    
    np.savetxt('results_1000.csv', 
               indexed_labels,
               fmt='%d',
               delimiter=',',
               header='id,label',
               comments='')
               
test_folder = 'data/test1/'

test_imgs_paths = [test_folder + filepath for filepath in listdir(test_folder)]

pred = []

test_imgs = load_images(test_imgs_paths, gray=True)

for i, img in enumerate(test_imgs):
    if( i % 100 == 0 ):
        print(i)

    kp = detector.detect(img)
    img_features = sift_bow_extractor.compute(img, kp)

    try:
      p = best_clf.predict(img_features)
    except:
      p = np.array(1)

    pred.append(p)
        
save_labels_csv(pred)
