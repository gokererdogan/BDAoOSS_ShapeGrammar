import itertools

import numpy as np
import pandas as pd

import sklearn.svm as svm
import sklearn.preprocessing as pp

import gmllib.experiment as exp


def fit_svm(model_name, subject_data_file, threshold, C, nfolds, pca):
    subject_data = pd.read_table(subject_data_file, sep=" ")
    net, layer = model_name
    row_labels = eval(open('outputs/{0:s}_row_labels.txt'.format(net)).read())

    if not pca:
        stimuli_outputs = np.load('outputs/{0:s}_{1:s}.npy'.format(net, layer))
    else:
        stimuli_outputs = np.load('outputs/{0:s}_{1:s}_pca.npy'.format(net, layer))

    x, y, relations = model_outputs_to_svm_data(stimuli_outputs, subject_data, row_labels, threshold=threshold)

    normalizer = pp.StandardScaler()

    accuracies = []
    for n in range(nfolds):
        train_ix, test_ix = split_data(relations, row_labels, objects_in_train=7)
        train_x = x[train_ix,:]
        test_x = x[test_ix,:]
        train_y = y[train_ix]
        test_y = y[test_ix]

        train_x = normalizer.fit_transform(train_x)
        test_x = normalizer.transform(test_x)

        svm_model = svm.SVC(C=C, kernel='linear')
        svm_model.fit(train_x, train_y)
        svm_pred = svm_model.predict(test_x)
        accuracies.append(np.mean(test_y == svm_pred))

    print("{0:s}_{1:s} ({2:.2f}): {3:f}+-{4:f}".format(net, layer, threshold, np.mean(accuracies), np.std(accuracies) * 2))

    results = {'Accuracy': np.mean(accuracies), 'AccuracySD': np.std(accuracies)}
    return results


def model_outputs_to_svm_data(object_x, human_predictions, labels, threshold=0.5):
    dim_object_x = object_x.shape[1]
    n_x = np.sum(np.logical_or(human_predictions.HumanPred <= (1 - threshold), human_predictions.HumanPred > threshold))
    x = np.zeros((n_x, 2 * dim_object_x))
    y = np.zeros(n_x)
    relations = []

    i = 0
    for _, row in human_predictions.iterrows():
        object_name = row['Comparison1'].split('_')[0]
        object_id = labels.index(object_name)
        comparison1_id = labels.index(row['Comparison1'])
        comparison2_id = labels.index(row['Comparison2'])

        if row['HumanPred'] > threshold or row['HumanPred'] <= (1 - threshold):
            relations.append((row['Comparison1'], row['Comparison2']))
            x[i, 0:dim_object_x] = np.square(object_x[comparison1_id] - object_x[object_id])
            x[i, dim_object_x:] = np.square(object_x[comparison2_id] - object_x[object_id])
            y[i] = row['HumanPred'] < threshold
            i += 1
        else:
            continue

    return x, y, relations


def split_data(rels, labels, objects_in_train=7):
    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    train_objects = np.random.choice(objects, size=objects_in_train, replace=False)
    train_ix = []
    test_ix = []
    for i, rel in enumerate(rels):
        obj_name = rel[0].split('_')[0]
        if obj_name in train_objects:
            train_ix.append(i)
        else:
            test_ix.append(i)
    np.random.shuffle(train_ix)
    np.random.shuffle(test_ix)
    return train_ix, test_ix


def test():
    subject_data_file = 'HumanPredictions_20151118.txt'
    outputs = np.load('outputs/alexnet_prob.npy')
    subject_data = pd.read_table(subject_data_file, sep=" ")
    row_labels = eval(open('outputs/alexnet_row_labels.txt').read())
    x, y, relations = model_outputs_to_svm_data(outputs, subject_data, row_labels, threshold=0.5)

    normalizer = pp.StandardScaler()

    accuracies = []
    for n in range(50):
        train_ix, test_ix = split_data(relations, row_labels, objects_in_train=7)
        train_x = x[train_ix,:]
        test_x = x[test_ix,:]
        train_y = y[train_ix]
        test_y = y[test_ix]

        train_x = normalizer.fit_transform(train_x)
        test_x = normalizer.transform(test_x)

        svm_model = svm.SVC(C=1.0, kernel='linear')
        svm_model.fit(train_x, train_y)
        svm_pred = svm_model.predict(test_x)
        accuracies.append(np.mean(test_y == svm_pred))

    print("\n{0:f}+-{1:f}".format(np.mean(accuracies), np.std(accuracies) * 2))


if __name__ == "__main__":
    subject_data_file = 'HumanPredictions_20151118.txt'

    models = {'alexnet': ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5',
                          'fc6', 'fc7', 'fc8', 'prob'],
              'googlenet': ['pool1_norm1', 'conv2_norm2', 'inception_3a_output', 'inception_3b_output', 'pool3_3x3_s2',
                            'inception_4a_output', 'inception_4b_output', 'inception_4c_output', 'inception_4d_output',
                            'inception_4e_output', 'pool4_3x3_s2', 'inception_5a_output', 'inception_5b_output',
                            'pool5_7x7_s1', 'loss3_classifier', 'prob'],
              'pb': ['']}

    model_names = []
    for net, layers in models.iteritems():
        for layer in layers:
            model_names.append((net, layer))

    experiment = exp.Experiment(name='fit_svm_object_split', experiment_method=fit_svm, model_name=model_names,
                                subject_data_file=subject_data_file, threshold=[0.5, 0.8],
                                C=0.1, nfolds=50, pca=True)

    experiment.run(parallel=True, num_processes=10)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/FitSVMSplitObject.csv')

