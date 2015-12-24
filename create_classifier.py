import cv2
from matplotlib import pyplot as plt
import sklearn
import numpy as np
import pickle as pk
from os import listdir
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import clone as skl_clone
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import sys

plt.style.use('ggplot')

NR_WORDS = 1000

train_folder = 'data/train/'
imgs_paths = [train_folder + filepath for filepath in listdir(train_folder)]
labels = [1 if "dog" in path else 0 for path in imgs_paths]


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

features = np.empty((0, NR_WORDS))
imgs = load_images(imgs_paths, gray=True)

for img in imgs:
    kp = detector.detect(img)

    img_features = sift_bow_extractor.compute(img, kp)

    features = np.concatenate((features, img_features), axis=0)

features.shape

pk.dump(features, open('features_1000w.p', 'wb'))

sys.exit(0)

labels = np.asarray(labels)
labels.shape

pk.dump(labels, open('labels_1000w.p', 'wb'))

train_folder = 'data/train/'

imgs_paths = [train_folder + filepath for filepath in listdir(train_folder)]


def k_fold_model_select(features, labels, raw_classifiers, n_folds=10, weigh_samples_fn=None):
    # weigh_samples_fn is explained below
    # assumes that the raw_classifier output is in probability

    # split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.3,
                                                        stratify=labels,
                                                        random_state=0)


    # use stratified k-fold cross validation to select the model
    skf = StratifiedKFold(y_train, n_folds=n_folds)

    best_classifier = None
    best_score = float('-inf')

    for train_index, validation_index in skf:
        for raw_classifier in raw_classifiers:
            classifier = skl_clone(raw_classifier)
            classifier = classifier.fit(X_train[train_index], y_train[train_index])

            if weigh_samples_fn != None:
                y_pred = classifier.predict(X_train[validation_index])
                sample_weight = weigh_samples_fn(y_train[validation_index], y_pred)
            else:
                sample_weight = None

            score = accuracy_score(classifier.predict(X_train[validation_index]), y_train[validation_index],
                                     sample_weight=sample_weight)

            if score > best_score:
                best_classifier = classifier
                best_score = score

    # compute the confusion matrix
    y_pred = best_classifier.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    # now compute the score for the test data of the best found classifier
    if weigh_samples_fn != None:
        sample_weight = weigh_samples_fn(y_test, y_pred)
    else:
        sample_weight = None
    test_score = accuracy_score(best_classifier.predict(X_test), y_test, sample_weight=sample_weight)

    # obtain the classification report
    report = classification_report(y_test, y_pred, target_names=['cat', 'dog'], sample_weight=sample_weight)

    # obtain ROC curve
    y_test_bin = label_binarize(y_test, classes=[0, 1])
    y_prob = best_classifier.predict_proba(X_test)

    #fpr, tpr, _ = roc_curve(y_test_bin[:, 1], y_prob[:, 1])
    fpr, tpr, _ = roc_curve(y_test_bin, y_prob[:, 1])
    roc_info = (best_classifier.__class__.__name__, (fpr, tpr))

    return (test_score, report, conf_mat, roc_info, best_classifier)


knn = KNeighborsClassifier(weights='distance', algorithm='auto')
knn_score, knn_rep, knn_cm, knn_roc, knn_clf = k_fold_model_select(features, labels, [knn])

print("Nearest Neighbors")
print("Score:", knn_score)
print("Confusion matrix:", knn_cm, sep='\n')
print("Classification report:", knn_rep, sep='\n')
knn_clf = knn_clf.fit(features, labels)
pk.dump(knn_clf, open('knn_clf.p', 'wb'))
