# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
import cv2
import numpy as np
from skimage.morphology import label

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

img = cv2.imread(r'I:\Learning\Machine Learning\Kaggle\Data Science Bowl 2018\Data\stage1_train\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\images\0fe691c27c3dcf767bc22539e10c840f894270c82fc60c8f0d81ee9e7c5b9509.png',cv2.IMREAD_GRAYSCALE)

rle = list(prob_to_rles(img))
print (rle)