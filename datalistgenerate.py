import copy
import os
import numpy as np
import nibabel as nib
import cv2
from skimage import segmentation



def run(image):

    seg_map = segmentation.slic(image,  n_segments=512, compactness=10.,)
    seg_map2 = np.zeros(seg_map.shape)
    for i in np.unique(seg_map):
        pos = np.where(seg_map == i)
        seg_map2[pos] = np.mean(image[pos])
    seg_map2 = np.array(seg_map2/10,dtype='int')*10

    return seg_map2


def select_slices(segmentation):
    indeces = np.sum(np.sum(segmentation, axis=1), axis=1)
    zmin, zmax = np.min(np.where(indeces > 0)[0]), np.max(np.where(indeces > 0)[0])
    zmin, zmax = int(zmin + (zmax - zmin) * 0.10), int(zmax - (zmax - zmin) * 0.10)
    # indeces = np.where(indeces > 60)[0]
    # gap = (np.max(indeces) - np.min(indeces)) // 4
    gap = (zmax - zmin) // 4
    # if gap < num_slices:
    #     num_slices = gap
    slices = []
    for i in range(4):
        slices.append(zmin + i * gap)


    return slices
def save_one_montage(fpath,file,save_path):

    fn = os.path.join(fpath,file)
    img = nib.load(fn)
    imgArr = img.get_fdata()
    subID = file.split('.')[0]

    ## CT thresholding
    imgArr[np.where(imgArr <= -1500)] = -1500
    imgArr[np.where(imgArr >= 100)] = 100
    imgArr = (imgArr + 1500) / 1600 * 255

    ## get corresponding segmentation masks
    seg = nib.load(fn.replace('ct','lung'))
    segArr = seg.get_fdata()
    segMask = copy.deepcopy(segArr)
    segMask[np.where(segArr != 0)] = 1

    ## get z-axis indeces for the montage
    z_lung = select_slices(segMask)


    imgArr_selected = copy.deepcopy(imgArr[:, :, z_lung])
    seg_selected = copy.deepcopy(segArr[:, :, z_lung])


    imgArrMonUp = np.vstack((imgArr_selected[:, :, 0], imgArr_selected[:, :, 1]))
    imgArrMonDown = np.vstack((imgArr_selected[:, :, 2], imgArr_selected[:, :, 3]))
    imgArrMon = np.hstack((imgArrMonUp, imgArrMonDown))

    imgArrMonUp = np.vstack((seg_selected[:, :, 0], seg_selected[:, :, 1]))
    imgArrMonDown = np.vstack((seg_selected[:, :, 2], seg_selected[:, :, 3]))
    segArrMon = np.hstack((imgArrMonUp, imgArrMonDown))

    save_fn = '%s_%i_%i_%i_%i.png'%(subID,z_lung[0],z_lung[1],z_lung[2],z_lung[3])
    cv2.imwrite(os.path.join(save_path,'Segmentation',save_fn), segArrMon)
    cv2.imwrite(os.path.join(save_path,'CTMontage',save_fn), segArrMon)

    unsupervised_imgArrMon = \
        np.array(np.expand_dims(imgArrMon, 2).repeat(3, axis=2),dtype=np.uint8)
    unsupervised_segArr = run(unsupervised_imgArrMon)

    cv2.imwrite(os.path.join(save_path,'USeg',save_fn), unsupervised_segArr)





if __name__ == '__main__':

    fpath = './NII_path'
    save_path = './montage_path'
    imgs = os.listdir(fpath)
    for file in imgs:
        save_one_montage(fpath,file,save_path)

