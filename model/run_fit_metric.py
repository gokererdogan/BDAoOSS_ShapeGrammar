import numpy as np
import pandas as pd

import sklearn.preprocessing as pp

import gmllib.metric_learning as metric
import gmllib.experiment as exp


def fit_metric(model_name, subject_data_file, threshold, C, nfolds, normalize, pca, method, rank):
    subject_data = pd.read_table(subject_data_file, sep=" ")
    net, layer = model_name
    row_labels = eval(open('outputs/{0:s}_row_labels.txt'.format(net)).read())
    if not pca:
        x = np.load('outputs/{0:s}_{1:s}.npy'.format(net, layer))
    else:
        x = np.load('outputs/{0:s}_{1:s}_pca.npy'.format(net, layer))

    scaler = pp.StandardScaler()

    accuracies = []
    for n in range(nfolds):
        train_x, train_relations, test_x, test_relations = split_data(subject_data, x, row_labels, objects_in_train=7,
                                                                      threshold=threshold)

        if normalize:
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        if method == 'low_rank':
            A, objective, accuracy, converged = metric.learn_low_rank_metric(train_x, train_relations, cost=C, rank=rank,
                                                                             method='SLSQP', tol=1e-6, verbose=False)
        elif method == 'diag':
            A, objective, accuracy, converged = metric.learn_diagonal_metric(train_x, train_relations, cost=C,
                                                                             method='L-BFGS-B', tol=1e-6, verbose=False)

        test_accuracy = metric.calculate_accuracy(test_x, A, test_relations)
        accuracies.append(test_accuracy)

    print("{0:35s} ({1:4.2f}): {2:.4f}+-{3:.4f}".format(net+'_'+layer, threshold, np.mean(accuracies),
                                                        np.std(accuracies) * 2))

    results = {'Accuracy': np.mean(accuracies), 'AccuracySD': np.std(accuracies)}
    return results


def subject_data_to_relative_comparisons(data, labels, threshold=0.5):
    relations = []
    for i, row in data.iterrows():
        if row['HumanPred'] > threshold:
            similar = row['Comparison1']
            dissimilar = row['Comparison2']
        elif row['HumanPred'] <= (1 - threshold):
            similar = row['Comparison2']
            dissimilar = row['Comparison1']
        else:
            continue

        object_name = similar.split('_')[0]
        if object_name in labels:
            object_id = labels.index(object_name)
            similar_id = labels.index(similar)
            dissimilar_id = labels.index(dissimilar)

            relations.append((object_id, similar_id, dissimilar_id))

    return relations


def split_data(subject_data, model_output, labels, objects_in_train=7, threshold=0.5):
    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    train_objs = np.random.choice(objects, size=objects_in_train, replace=False)
    train_rows = []
    test_rows = []
    train_labels = []
    test_labels = []
    for i, label in enumerate(labels):
        obj_name = label.split('_')[0]
        if obj_name in train_objs:
            train_rows.append(i)
            train_labels.append(label)
        else:
            test_rows.append(i)
            test_labels.append(label)

    train_x = model_output[train_rows, :]
    test_x = model_output[test_rows, :]
    train_relations = subject_data_to_relative_comparisons(subject_data, train_labels, threshold=threshold)
    test_relations = subject_data_to_relative_comparisons(subject_data, test_labels, threshold=threshold)

    return train_x, train_relations, test_x, test_relations


def predict_from_train(train_rels, test_rels):
    correct = 0.0
    for rel in test_rels:
        o = rel[0]
        x = rel[1]
        y = rel[2]

        xa = set([r[2] for r in train_rels if r[0] == o and r[1] == x])
        ay = set([r[1] for r in train_rels if r[0] == o and r[2] == y])

        ax = set([r[1] for r in train_rels if r[0] == o and r[2] == x])
        ya = set([r[2] for r in train_rels if r[0] == o and r[1] == y])

        if xa.intersection(ay) and not ax.intersection(ya):
            # if intersect(xa, ay) not empty and intersect(ya, ax) is empty
            correct += 1.0
        elif not xa.intersection(ay) and not ax.intersection(ya):  # if both intersections are empty
            correct += 0.5
        elif xa.intersection(ay) and ax.intersection(ya):  # if both not empty
            correct += 0.5

    return correct / len(test_rels)


def calculate_random_split_accuracy(test_size=0.25, threshold=0.5, nfolds=50):
    subject_data_file = 'HumanPredictions_20151118.txt'
    subject_data = pd.read_table(subject_data_file, sep=" ")
    row_labels = eval(open('outputs/pb_row_labels.txt').read())
    relations = subject_data_to_relative_comparisons(subject_data, row_labels, threshold=threshold)
    N = len(relations)
    train_N = int(test_size * N)
    accuracies = []
    for n in range(nfolds):
        np.random.shuffle(relations)
        train = relations[0:train_N]
        test = relations[train_N:]
        accuracies.append(predict_from_train(train, test))

    return np.mean(accuracies), np.std(accuracies)


def test():
    subject_data_file = 'HumanPredictions_20151118.txt'
    outputs = np.load('outputs/alexnet_prob_pca.npy')
    subject_data = pd.read_table(subject_data_file, sep=" ")
    row_labels = eval(open('outputs/alexnet_row_labels.txt').read())

    C = 1.0
    normalize = False
    scaler = pp.StandardScaler()

    train_accuracies = []
    test_accuracies = []
    for n in range(20):
        print('.'),
        train_x, train_rels, test_x, test_rels = split_data(subject_data, outputs, row_labels, objects_in_train=7,
                                                            threshold=0.8)

        if normalize:
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

        """
        A, objective, accuracy, converged = metric.learn_diagonal_metric(train_x, train_rels, cost=C, method='L-BFGS-B',
                                                                         tol=1e-6, verbose=False)

        """

        A, objective, accuracy, converged = metric.learn_low_rank_metric(train_x, train_rels, cost=C, method='SLSQP',
                                                                         rank=5, tol=1e-6, verbose=False)

        test_accuracy = metric.calculate_accuracy(test_x, A, test_rels)

        train_accuracies.append(accuracy)
        test_accuracies.append(test_accuracy)

    print("\nTrain: {0:f}+-{1:f}".format(np.mean(train_accuracies), np.std(train_accuracies) * 2))
    print("Test: {0:f}+-{1:f}".format(np.mean(test_accuracies), np.std(test_accuracies) * 2))


def reduce_dims_with_pca():
    import sklearn.decomposition as d
    models = {'alexnet': ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5',
                          'fc6', 'fc7', 'fc8', 'prob'],
              'googlenet': ['pool1_norm1', 'conv2_norm2', 'inception_3a_output', 'inception_3b_output', 'pool3_3x3_s2',
                            'inception_4a_output', 'inception_4b_output', 'inception_4c_output', 'inception_4d_output',
                            'inception_4e_output', 'pool4_3x3_s2', 'inception_5a_output', 'inception_5b_output',
                            'pool5_7x7_s1', 'loss3_classifier', 'prob'],
              'pb': ['']}

    pca = d.PCA(n_components=0.95)

    for net, layers in models.iteritems():
        for layer in layers:
            print(net+'_'+layer)
            x = np.load('outputs/{0:s}_{1:s}.npy'.format(net, layer))
            x = pca.fit_transform(x)
            np.save('outputs/{0:s}_{1:s}_pca.npy'.format(net, layer), x)

if __name__ == "__main__":
    subject_data_file = 'HumanPredictions_20151118.txt'
    models = {'alexnet': ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5',
                          'fc6', 'fc7', 'fc8', 'prob'],
              'googlenet': ['pool1_norm1', 'conv2_norm2', 'inception_3a_output', 'inception_3b_output', 'pool3_3x3_s2',
                            'inception_4a_output', 'inception_4b_output', 'inception_4c_output', 'inception_4d_output',
                            'inception_4e_output', 'pool4_3x3_s2', 'inception_5a_output', 'inception_5b_output',
                            'pool5_7x7_s1', 'loss3_classifier', 'prob'],
              'hmax': ['c1_0', 'c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5', 'c1_6', 'c1_7', 'c1',
                       'c2_0', 'c2_1', 'c2_2', 'c2_3', 'c2_4', 'c2_5', 'c2_6', 'c2_7', 'c2', ]}

    models = {'hmax': ['c1_0', 'c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5', 'c1_6', 'c1_7', 'c1',
                       'c2_0', 'c2_1', 'c2_2', 'c2_3', 'c2_4', 'c2_5', 'c2_6', 'c2_7', 'c2', ]}

    models = {'fb': ['']}

    # 'pb': ['']}

    # models = {'random': ['1000', '10000']}
    # models = {'pb': ['']}

    model_names = []
    for net, layers in models.iteritems():
        for layer in layers:
            model_names.append((net, layer))

    experiment = exp.Experiment(name='fit_metric_object_split', experiment_method=fit_metric, model_name=model_names,
                                subject_data_file=subject_data_file,
                                threshold=[0.5, 0.8],
                                C=1.0,
                                nfolds=50,
                                normalize=[False, True],
                                pca=False,
                                method='low_rank',
                                # method='diag',
                                rank=[5, 10, 20])
                                # rank=5)

    experiment.run(parallel=True, num_processes=10)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/FitMetricSplitObject.csv')

