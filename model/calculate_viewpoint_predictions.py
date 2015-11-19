"""
This script is for calculating similarities based on a simple model 
that simply looks at the difference in viewpoint, i.e., the angle of
rotation of the camera around z axis. This is meant to be a simple
control model.

Created on Nov 19, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import pandas as pd
import pandasql as psql
import cPickle as pkl

if __name__ == "__main__":

    stimuli_folder = "../stimuli/stimuli20150624_144833"
    
    df = pd.DataFrame(index=range(0, 8), columns=['Target', 'Comparison', 'ViewpointDifference'])

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']

    viewpoints = eval(open("{0:s}/viewpoints.txt".format(stimuli_folder)).read())
    i = 0
    for obj in objects:
        print(obj)
        obj_vp = viewpoints[obj]

        for transformation in transformations:
            print("\t{0:s}".format(transformation))
            comparison = "{0:s}_{1:s}".format(obj, transformation)
            comp_vp = viewpoints[comparison]
			
            distance = min((obj_vp - comp_vp) % 360, (comp_vp - obj_vp) % 360)
            df.loc[i] = [obj, comparison, distance]
            i += 1

    # calculate model predictions
    predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "d1.ViewpointDifference<d2.ViewpointDifference as VP_Prediction "  
                             "from df as d1, df as d2 "
                             "where d1.Target = d2.Target and d1.Comparison<d2.Comparison", env=locals())

    # write to disk
    open('VP_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('../../../R/BDAoOSS_Synthetic/VP_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('VP_ModelDistances.txt', 'w').write(df.to_string())

