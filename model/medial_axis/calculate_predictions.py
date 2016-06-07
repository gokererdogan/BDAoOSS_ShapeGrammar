"""
Big data analysis of object shape representations

Calculate the predictions of the medial_axis model from similarities between shapes.

Created on May 5, 2016

Goker Erdogan
https://github.com/gokererdogan
"""

import pandas as pd
import pandasql as psql

similarities = pd.read_csv('MA_ModelSimilarities.csv')

# calculate logp(Shape_comparison|Shape_target) and logp(Shape_target|Shape_comparison)

similarities['p_comp_target'] = similarities['LogP(S_c|SK_t)'] + similarities['LogP(SK_t|S_t)']
similarities['p_target_comp'] = similarities['LogP(S_t|SK_c)'] + similarities['LogP(SK_c|S_c)']
similarities['p_avg'] = similarities['LogP(S_t|SK_c)'] + similarities['LogP(SK_c|S_c)']

df = similarities[['Target', 'Comparison', 'p_comp_target', 'p_target_comp', 'p_avg']]

predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                         "d1.p_comp_target > d2.p_comp_target as MA_pcomp_Prediction, "
                         "d1.p_target_comp > d2.p_target_comp as MA_ptarget_Prediction, "
                         "d1.p_avg > d2.p_avg as MA_pavg_Prediction "
                         "from df as d1, df as d2 where d1.Target = d2.Target "
                         "and d1.Comparison < d2.Comparison",
                         env=locals())

open('MA_ModelPredictions.txt', 'w').write(predictions.to_string())
