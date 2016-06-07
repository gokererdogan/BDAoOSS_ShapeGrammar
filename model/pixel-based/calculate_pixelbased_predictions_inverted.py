# coding=utf-8
"""
Big data analysis of object shape representations
Calculates the predictions of the pixel-based model for the experiment with inverted images.

Created on Jun 1, 2016

@author: goker erdogan
gokererdogan@gmail.com
https://github.com/gokererdogan/
"""

import numpy as np

import pandas as pd


if __name__ == "__main__":
    # we ran the inverted experiment with only high confidence trials from our original experiment. hence we need to
    # get a list of these high confidence trials from the results for our original experiment.
    experiment_results = pd.read_table('../HumanPredictions_20151118.txt', sep=" ", header=0)
    high_confidence = experiment_results.loc[np.logical_or(experiment_results['HumanPred'] < .2,
                                                           experiment_results['HumanPred'] > .8), :]
    trial_count = high_confidence.shape[0]

    # load the calculated pixel based distances
    pb_distances = pd.read_table('PixelBased_Distances.txt', header=0, sep=" ", index_col=0, skipinitialspace=True)

    predictions = pd.DataFrame(index=range(trial_count), columns=['Comparison1', 'Comparison2', 'PB_Prediction'])

    for i in range(trial_count):
        comparison1 = high_confidence.iloc[i, 0]
        comparison2 = high_confidence.iloc[i, 1]
        # in the experiment with inverted images, we only presented the comparison that was picked by the majority of
        # subjects in the original experiment in the inverted orientation
        subjects_chose_1 = high_confidence.iloc[i, 2] > .5

        if subjects_chose_1:
            dist1 = pb_distances.loc[pb_distances['Comparison'] == comparison1, 'DistanceInverted'].iloc[0]
            dist2 = pb_distances.loc[pb_distances['Comparison'] == comparison2, 'Distance'].iloc[0]
        else:
            dist1 = pb_distances.loc[pb_distances['Comparison'] == comparison1, 'Distance'].iloc[0]
            dist2 = pb_distances.loc[pb_distances['Comparison'] == comparison2, 'DistanceInverted'].iloc[0]

        predictions.loc[i] = [comparison1, comparison2, 1 if dist1 < dist2 else 0]

    open("PixelBased_Predictions_Inverted.txt", "w").write(predictions.to_string())



