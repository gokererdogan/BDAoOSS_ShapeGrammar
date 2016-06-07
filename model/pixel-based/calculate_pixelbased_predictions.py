# coding=utf-8
"""
Big data analysis of object shape representations
Calculates the predictions of the pixel-based model that calculates the Euclidean distance between images.

Created on Jun 1, 2016

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
"""

import numpy as np
import scipy.misc as misc

import pandas as pd
import pandasql as psql


if __name__ == "__main__":
    stimuli_folder = '../../stimuli/stimuli20150624_144833/single_view'

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']
    variations = [o + '_' + t for t in transformations for o in objects]

    df = pd.DataFrame(index=np.arange(0, len(objects) * len(transformations)),
                      columns=['Target', 'Comparison', 'Distance', 'DistanceInverted'], dtype=float)

    i = 0
    for obj in objects:
        print obj
        target_file = "{0:s}/{1:s}.png".format(stimuli_folder, obj)
        target_img = misc.imread(target_file)
        target_img = np.asarray(target_img, dtype=np.float64) / 255.0
        for transformation in transformations:
            comparison_file = "{0:s}/{1:s}_{2:s}.png".format(stimuli_folder, obj, transformation)
            comparison_img = misc.imread(comparison_file)
            # beware overflow! imread returns uint images, convert to float!
            comparison_img = np.asarray(comparison_img, dtype=np.float64) / 255.0
            dist = np.sum(np.square(target_img - comparison_img))

            inverted_comparison_file = "{0:s}_inverted/{1:s}_{2:s}.png".format(stimuli_folder, obj, transformation)
            inverted_comparison_img = misc.imread(inverted_comparison_file)
            # beware overflow! imread returns uint images, convert to float!
            inverted_comparison_img = np.asarray(inverted_comparison_img, dtype=np.float64) / 255.0
            dist_inv = np.sum(np.square(target_img - inverted_comparison_img))

            df.loc[i] = [obj, obj+'_'+transformation, dist, dist_inv]
            i += 1

    predictions = psql.sqldf("select d1.Target as Target, d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "d1.Distance<d2.Distance as PixelBased_Prediction "
                             "from df as d1, df as d2 "
                             "where d1.Target = d2.Target and d1.Comparison<d2.Comparison",
                             env=locals())

    open("PixelBased_Distances.txt", "w").write(df.to_string())
    open("PixelBased_Predictions.txt", "w").write(predictions.to_string())



