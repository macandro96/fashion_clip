# read json file and create target labels
import json
import pickle

path = '../data/validation.json'
with open(path) as f:
    label = json.load(f)

image_target = {}
anns = label['annotations']

#'labelId': ['62', '17', '66', '214', '105', '137', '85'], 'imageId': '45'
for ann in anns:
    img = ann['imageId']
    labels = ann['labelId']
    labels = [int(i)-1 for i in labels]
    #one hot encoding of size 228
    one_hot = [0]*228
    for i in labels:
        one_hot[i] = 1
    image_target[img] = one_hot

one_hot_path = '../data/target_labels.json'
with open(one_hot_path, 'w') as f:
    json.dump(image_target, f)

path = '../data/validation_scores_zero_shot.pkl'
with open(path, 'rb') as f:
    scores = pickle.load(f)