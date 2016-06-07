"""
This script is for calculating similarities based on a 2D alignment model
that finds the affine transformation that best aligns two images and
calculates pixel distance after the transformation is applied.

Created on Apr 8, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

import cPickle as pkl

import numpy as np
import scipy.ndimage as spimg
import pandas as pd
import pandasql as psql


def center_and_scale_image(img, fig_width=200, fig_height=200, img_width=250, img_height=250):
    # find bounding box
    posx, posy = np.nonzero(img > 0.0)
    bb_x1, bb_x2 = np.min(posx), np.max(posx)
    bb_y1, bb_y2 = np.min(posy), np.max(posy)

    fig = img[bb_x1:(bb_x2+1), bb_y1:(bb_y2+1)]
    figw, figh = fig.shape

    fig = spimg.zoom(fig, (fig_width / float(figw), fig_height / float(figh)), order=1)

    # calculate center
    posx, posy = np.nonzero(fig > 0.0)
    centerx, centery = np.mean(posx), np.mean(posy)

    # place figure into the center of image
    fig_x1 = np.round((img_width / 2.0) - centerx)
    fig_x2 = fig_x1 + fig_width
    fig_y1 = np.round((img_height / 2.0) - centery)
    fig_y2 = fig_y1 + fig_height
    if fig_x1 < 0 or fig_y1 < 0 or fig_x2 > img_width or fig_y2 > img_height:
        raise RuntimeError("Figure cannot be placed into the image.")

    img = np.zeros((img_width, img_height))
    img[fig_x1:fig_x2, fig_y1:fig_y2] = fig

    return img


def calculate_affine_feature_distance(feature_list1, feature_list2):
    """
    Calculate the affine distance between two shapes (represented as feature lists) based solely on features.
    Affine distance is defined as the distance between two shapes after they are aligned as well as possible using an
    affine transform.
    We use the formula from (Eqn. 6 in paper)
        Werman, M., & Weinshall, D. (1995). Similarity and affine invariant distances between 2D point sets.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 17(8), 810-814.

    Parameters:
        feature_list1 (ndarray): 2xn array of feature positions for shape 1
        feature_list2 (ndarray): 2xn array of feature positions for shape 2

    Returns:
        float: affine distance between shapes defined by input feature lists
    """
    # center both feature lists
    c1x = np.mean(feature_list1[0])
    c1y = np.mean(feature_list1[1])
    c2x = np.mean(feature_list2[0])
    c2y = np.mean(feature_list2[1])

    feature_list1[0] = feature_list1[0] - c1x
    feature_list1[1] = feature_list1[1] - c1y
    feature_list2[0] = feature_list2[0] - c2x
    feature_list2[1] = feature_list2[1] - c2y

    # use the formula to calculate distance
    t1 = np.dot(np.linalg.pinv(feature_list1), feature_list1)
    t2 = np.dot(np.linalg.pinv(feature_list2), feature_list2)

    dist = 2 - np.trace(np.dot(t1, t2))
    return dist


def calculate_affine_image_distance(feature_list1, image1, feature_list2, image2):
    """
    Calculate the affine distance between two shapes (represented as feature lists) based on images.
    Affine distance is defined as the distance between two shapes after they are aligned as well as possible using an
    affine transform. Feature lists are used to find the affine transform that best aligns shape1 with shape2; then,
    this transform is applied to first image and pixel distance between this transformed image and second image is
    calculated.

    Parameters:
        feature_list1 (ndarray): 2xn array of feature positions for shape 1
        image1 (ndarray): image of shape 1
        feature_list2 (ndarray): 2xn array of feature positions for shape 2
        image2 (ndarray): image of shape 2

    Returns:
        float: affine distance between shapes defined by input feature lists
    """
    # center both feature lists
    c1x = np.mean(feature_list1[0])
    c1y = np.mean(feature_list1[1])
    c2x = np.mean(feature_list2[0])
    c2y = np.mean(feature_list2[1])

    feature_list1[0] = feature_list1[0] - c1x
    feature_list1[1] = feature_list1[1] - c1y
    feature_list2[0] = feature_list2[0] - c2x
    feature_list2[1] = feature_list2[1] - c2y

    # find the best linear transform
    a = np.dot(feature_list1, np.linalg.pinv(feature_list2))

    # calculate offset, i.e., translation amount
    old_center = np.array([c1x, c1y])
    new_center = np.array([c2x, c2y])
    offset = old_center - np.dot(a, new_center)

    # affine transform first image
    transformed_image = spimg.affine_transform(image1, a, offset, order=1)

    dist = np.mean(np.square((transformed_image - image2) / np.max(image2)))
    return dist


if __name__ == "__main__":
    stimuli_folder = "../../stimuli/stimuli20150624_144833/single_view"

    # load feature lists precalculated for alignment (see calculate_alignment_feature_lists script)
    feature_lists = pkl.load(open('AlignmentFeatureLists.pkl'))

    df = pd.DataFrame(index=range(0, 80), columns=['Target', 'Comparison', 'FeatureDistance', 'AffineImageDistance',
                                                   'AffineFeatureDistance', 'CenteredScaledDistance'])

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']

    viewpoints = eval(open("{0:s}/viewpoints.txt".format(stimuli_folder)).read())
    i = 0
    for obj in objects:
        print(obj)
        obj_img = spimg.imread("{0:s}/{1:s}.png".format(stimuli_folder, obj), mode='F')
        obj_normalized_img = center_and_scale_image(obj_img)

        for transformation in transformations:
            print("\t{0:s}".format(transformation))
            comparison = "{0:s}_{1:s}".format(obj, transformation)
            comp_img = spimg.imread("{0:s}/{1:s}.png".format(stimuli_folder, comparison), mode='F')
            comp_normalized_img = center_and_scale_image(comp_img)

            centered_scaled_dist = np.mean(np.square((obj_normalized_img - comp_normalized_img) /
                                                     np.max(comp_normalized_img)))

            affine_img_dist = calculate_affine_image_distance(feature_lists[obj], obj_img, feature_lists[comparison],
                                                              comp_img)

            affine_feat_dist = calculate_affine_feature_distance(feature_lists[obj], feature_lists[comparison])

            no_alignment_feature_dist = np.mean(np.square(feature_lists[obj] - feature_lists[comparison]))

            df.loc[i] = [obj, comparison, no_alignment_feature_dist, float(affine_img_dist), affine_feat_dist, centered_scaled_dist]
            i += 1

    # calculate model predictions
    predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "d1.FeatureDistance<d2.FeatureDistance as Align_NoAlignment_Prediction, "
                             "d1.AffineImageDistance<d2.AffineImageDistance as Align_AffineImage_Prediction, "
                             "d1.AffineFeatureDistance<d2.AffineFeatureDistance as Align_AffineFeature_Prediction, "
                             "d1.CenteredScaledDistance<d2.CenteredScaledDistance as Align_CenteredScaled_Prediction "
                             "from df as d1, df as d2 "
                             "where d1.Target=d2.Target and d1.Comparison<d2.Comparison", env=locals())

    # write to disk
    open('Alignment_ModelPredictions.txt', 'w').write(predictions.to_string())
    # open('../../../R/BDAoOSS_Synthetic/VP_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('Alignment_ModelDistances.txt', 'w').write(df.to_string())
