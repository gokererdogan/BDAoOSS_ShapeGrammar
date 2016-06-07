# coding=utf-8
"""
Big data analysis of object shape representations
Calculates the predictions of AlexNet for the experiment with inverted images.

Created on Jun 1, 2016

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
"""

import numpy as np
import pandas as pd

LAYER_COUNT = 14


if __name__ == "__main__":
    # we ran the inverted experiment with only high confidence trials from our original experiment. hence we need to
    # get a list of these high confidence trials from the results for our original experiment.
    experiment_results = pd.read_table('../HumanPredictions_20151118.txt', sep=" ", header=0)
    high_confidence = experiment_results.loc[np.logical_or(experiment_results['HumanPred'] < .2,
                                                           experiment_results['HumanPred'] > .8), :]
    trial_count = high_confidence.shape[0]

    # load the calculated distances from AlexNet
    alexnet_distances = pd.read_table('AlexNet_ModelDistances.txt', header=0, sep=" ", index_col=0, skipinitialspace=True)

    predictions = pd.DataFrame(columns=['Comparison1', 'Comparison2', 'Layer', 'Prediction'])

    for i in range(trial_count):
        comparison1 = high_confidence.iloc[i, 0]
        comparison2 = high_confidence.iloc[i, 1]
        # in the experiment with inverted images, we only presented the comparison that was picked by the majority of
        # subjects in the original experiment in the inverted orientation
        subjects_chose_1 = high_confidence.iloc[i, 2] > .5

        if subjects_chose_1:
            dist1 = alexnet_distances.loc[alexnet_distances['Comparison'] == comparison1, ['Layer', 'DistanceInverted']]
            dist1.rename(columns={'DistanceInverted': 'Distance'}, inplace=True)
            dist2 = alexnet_distances.loc[alexnet_distances['Comparison'] == comparison2, ['Layer', 'Distance']]
        else:
            dist1 = alexnet_distances.loc[alexnet_distances['Comparison'] == comparison1, ['Layer', 'Distance']]
            dist2 = alexnet_distances.loc[alexnet_distances['Comparison'] == comparison2, ['Layer', 'DistanceInverted']]
            dist2.rename(columns={'DistanceInverted': 'Distance'}, inplace=True)

        dists = pd.merge(dist1, dist2, on=['Layer'], how='inner', suffixes=['1', '2'])
        dists['Prediction'] = np.asarray(dists['Distance1'] < dists['Distance2'], dtype=np.int)
        trial_predictions = pd.DataFrame(data={'Comparison1': comparison1, 'Comparison2': comparison2,
                                               'Layer': dists['Layer'], 'Prediction': dists['Prediction']})

        predictions = predictions.append(trial_predictions)

    # Put prediction from each layer to a separate column
    predictions = pd.pivot_table(predictions, index=['Comparison1', 'Comparison2'], columns='Layer',
                                 values='Prediction')
    # convert MultiIndex to columns
    predictions.reset_index(inplace=True)
    del predictions.columns.name  # get rid of name

    open("AlexNet_Predictions_Inverted.txt", "w").write(predictions.to_string())



