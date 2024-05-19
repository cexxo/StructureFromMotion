import cv2
from torchvision import transforms
import numpy as np
import sys
from SuperGluePretrainedNetwork.models.matching import Matching

def normalize_coordinates(row_i, col_j, img):
    num_rows, num_cols = img.shape[:2]
    x = (col_j-num_cols/2)/(num_cols - 1.)
    y = (row_i-num_rows/2)/(num_rows - 1.)
    return x, y

path_to_image=sys.argv[1]
path_to_image2 = sys.argv[2]
image_index_i = sys.argv[3]
image_index_j = sys.argv[4]


image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(path_to_image2, cv2.IMREAD_GRAYSCALE)

image_color = cv2.imread(path_to_image)
image2_color = cv2.imread(path_to_image2)

transform = transforms.ToTensor()
image_t = transform(image) 
image_t2 = transform(image2) 
config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }
matching = Matching(config).eval().to('cpu')
data = {}
data['image0'] = image_t.unsqueeze(0)
data['image1'] = image_t2.unsqueeze(0)
m1 = matching.forward(data)
cosedasalvare = []
temp = []
count = 0
count2 = 0
try:
    with open('output_py.txt') as f:
        for line in f:
            pass
        last_line = line.split()
    offset = int(last_line[1]) + 1
except:
    offset = 0
id = 0
for i in m1['matches0'][0]:
    if int(i) == -1:
        count+=1
        continue
    temp = [
                int(image_index_i), 
                id+offset
            ]
    x, y = normalize_coordinates(int(m1['keypoints0'][0][id][0]), int(m1['keypoints0'][0][id][1]), image)
    temp2 = [x,y]
    cosedasalvare.append(temp+temp2)
    temp3 = [
                int(image_index_j),
                id+offset
            ]
    x2, y2 = normalize_coordinates(int(m1['keypoints1'][0][int(i)][0]),int(m1['keypoints1'][0][int(i)][1]),image)
    temp4 = [
                x2,y2
            ]
    cosedasalvare.append(temp3+temp4)
    id +=1

print("matches found in image {}: {}, matches discarded: {}".format(image_index_i,len(m1['matches0'][0]),count))
print("matches found in image {}: {}, matches discarded: {}".format(image_index_j,len(m1['matches1'][0]),count2))
print("total number of written points: {}".format(len(cosedasalvare)))
f=open('output_py.txt','a')
np.savetxt(f,cosedasalvare,fmt='%g')
number_to_be_defined = 0
colori_da_salvare = []
for i in m1['matches0'][0]:
    if int(i) == -1:
        count+=1
        continue
    rgb1 = np.array(image_color[ int(m1['keypoints0'][0][number_to_be_defined][1]),int(m1['keypoints0'][0][number_to_be_defined][0])])
    rgb2 = np.array(image2_color[int(m1['keypoints1'][0][int(i)][1]),int(m1['keypoints1'][0][int(i)][0])])
    colori_da_salvare.append(rgb1)
    colori_da_salvare.append(rgb2)
    number_to_be_defined+=1
z=open('color_py.txt','a')
np.savetxt(z,colori_da_salvare,fmt='%g')