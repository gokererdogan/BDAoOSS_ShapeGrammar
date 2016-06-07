"""
Big data analysis of object shape representations

Calculate the predictions of HMAX model from distances between shapes.

Created on May 9, 2016

Goker Erdogan
https://github.com/gokererdogan
"""

import pandas as pd
import pandasql as psql

similarities = pd.read_csv('HMAX_ModelSimilarities.csv')

predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                         "d1.c1_1 < d2.c1_1 as HMAX_C1_1_Prediction, "
                         "d1.c1_2 < d2.c1_2 as HMAX_C1_2_Prediction, "
                         "d1.c1_3 < d2.c1_3 as HMAX_C1_3_Prediction, "
                         "d1.c1_4 < d2.c1_4 as HMAX_C1_4_Prediction, "
                         "d1.c1_5 < d2.c1_5 as HMAX_C1_5_Prediction, "
                         "d1.c1_6 < d2.c1_6 as HMAX_C1_6_Prediction, "
                         "d1.c1_7 < d2.c1_7 as HMAX_C1_7_Prediction, "
                         "d1.c1_8 < d2.c1_8 as HMAX_C1_8_Prediction, "
                         "d1.c1 < d2.c1 as HMAX_C1_Prediction, "
                         "d1.c2_1 < d2.c2_1 as HMAX_C2_1_Prediction, "
                         "d1.c2_2 < d2.c2_2 as HMAX_C2_2_Prediction, "
                         "d1.c2_3 < d2.c2_3 as HMAX_C2_3_Prediction, "
                         "d1.c2_4 < d2.c2_4 as HMAX_C2_4_Prediction, "
                         "d1.c2_5 < d2.c2_5 as HMAX_C2_5_Prediction, "
                         "d1.c2_6 < d2.c2_6 as HMAX_C2_6_Prediction, "
                         "d1.c2_7 < d2.c2_7 as HMAX_C2_7_Prediction, "
                         "d1.c2_8 < d2.c2_8 as HMAX_C2_8_Prediction, "
                         "d1.c2 < d2.c2 as HMAX_C2_Prediction "
                         "from similarities as d1, similarities as d2 where d1.Target = d2.Target "
                         "and d1.Comparison < d2.Comparison",
                         env=locals())

open('HMAX_ModelPredictions.txt', 'w').write(predictions.to_string())
