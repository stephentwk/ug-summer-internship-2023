import torch
import pandas as pd
from Bio import SeqIO
from collections import Counter
from sklearn.cluster import k_means_
from sklearn.cluster import KMeans
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import sklearn.metrics as metrics
import numpy as np

import warnings

warnings.filterwarnings('ignore')


def init_seed(seed):
    torch.cuda.cudnn_enabled = False    # also disable cudnn to maximize reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class EpochSampler(object):
    """
    EpochSampler: yield permuted indexes at each epoch.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, indices):
        """
        Initialize the EpochSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - iterations: number of epochs
        """
        super(EpochSampler, self).__init__()
        self.indices = indices

    def __iter__(self):
        """
        yield a batch of indexes
        """

        while (True):
            shuffled_idx = self.indices[torch.randperm(len(self.indices))]

            yield shuffled_idx

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class FullNet(nn.Module):
    def __init__(self, x_dim, hid_dim=64, z_dim=64, p_drop=0.2):
        super(FullNet, self).__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            full_block(x_dim, hid_dim, p_drop),
            full_block(hid_dim, z_dim, p_drop),
        )

        self.decoder = nn.Sequential(
            full_block(z_dim, hid_dim, p_drop),
            full_block(hid_dim, x_dim, p_drop),
        )

    def forward(self, x):
        # sort the virus sequences by length in descending order
        sorted_lengths, sorted_indices = torch.sort(torch.LongTensor([len(seq) for seq in x]), descending=True)
        sorted_sequences = [x[i] for i in sorted_indices]

        # pad the virus sequences to the maximum length
        padded_sequences = rnn_utils.pad_sequence(sorted_sequences, batch_first=True, padding_value=0)

        encoded = self.encoder(padded_sequences)
        decoded = self.decoder(encoded)

        # revert the virus sequences to their original order
        _, unsorted_indices = torch.sort(sorted_indices)
        encoded = encoded[unsorted_indices]
        decoded = decoded[unsorted_indices]

        return encoded, decoded


def euclidean_dist(x, y):
    """
    compute the Euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def init_step(dataset, model, device, n_clusters):
    n_examples = len(dataset)

    X = torch.stack([dataset[i] for i in range(n_examples)])

    # run kmeans clustering
    X = X.to(device)
    encoded, _ = model(X)
    kmeans = KMeans(n_clusters, random_state=0).fit(encoded.data.cpu().numpy())
    landmark_encoded = torch.tensor(kmeans.cluster_centers_, device=device)

    return landmark_encoded


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def compute_landmarks_train(embeddings, target, prev_landmarks=None, tau=0.2):
    """
    Computing landmarks of each class in the labeled meta-dataset. Landmark is a closed form solution of
    minimizing distance to the mean and maximizing distance to other landmarks. If tau=0, landmarks are
    just mean of data points.
    embeddings: embeddings of the labeled dataset
    target: labels in the labeled dataset
    prev_landmarks: landmarks from previous iteration
    tau: regularizer for inter- and intra-cluster distance
    """

    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))

    landmarks_mean = torch.stack([embeddings[idx_class].mean(0) for idx_class in class_idxs]).squeeze()

    if prev_landmarks is None or tau == 0:
        return landmarks_mean

    suma = prev_landmarks.sum(0)
    n_landmark = prev_landmarks.shape[0]
    landmark_dist_part = (tau / (n_landmark - 1)) * torch.stack([suma - p for p in prev_landmarks])
    landmarks = 1 / (1 - tau) * (landmarks_mean - landmark_dist_part)

    return landmarks


def loss_test_basic(encoded, prototypes):
    dists = euclidean_dist(encoded, prototypes)
    min_dist = torch.min(dists, 1)

    y_hat = min_dist[1]
    args_uniq = torch.unique(y_hat, sorted=True)
    args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])

    # get_distances
    min_dist = min_dist[0]

    loss_val = torch.stack([min_dist[y_hat == idx_class].mean(0) for idx_class in args_uniq]).mean()

    return loss_val, args_count


def loss_task(encoded, prototypes, target, criterion='dist'):
    """
    Calculate loss.
    criterion: NNLoss - assign to closest prototype and calculate NNLoss
         dist - loss is distance to prototype that example needs to be assigned to
                and -distance to prototypes from other class
    """

    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))

    # prepare targets so they start from 0,1
    for idx, v in enumerate(uniq):
        target[target == v] = idx

    dists = euclidean_dist(encoded, prototypes)

    if criterion == 'NNLoss':

        loss = torch.nn.NLLLoss()
        log_p_y = nn.functional.log_softmax(-dists, dim=1)

        loss_val = loss(log_p_y, target)
        _, y_hat = log_p_y.max(1)

    elif criterion == 'dist':

        loss_val = torch.stack(
            [dists[idx_example, idx_proto].mean(0) for idx_proto, idx_example in enumerate(class_idxs)]).mean()
        y_hat = torch.max(-dists, 1)[1]

    acc_val = y_hat.eq(target.squeeze()).float().mean()

    return loss_val, acc_val


def loss_test(encoded, prototypes, tau):
    loss_val_test, args_count = loss_test_basic(encoded, prototypes)

    if tau > 0:
        dists = euclidean_dist(prototypes, prototypes)
        n_proto = prototypes.shape[0]
        loss_val2 = - torch.sum(dists) / (n_proto * n_proto - n_proto)

        loss_val_test += tau * loss_val2

    return loss_val_test, args_count


def adjust_range(y):
    """Assures that the range of indices if from 0 to n-1."""
    y = np.array(y, dtype=np.int64)
    val_set = set(y)
    mapping = {val: i for i, val in enumerate(val_set)}
    y = np.array([mapping[val] for val in y], dtype=np.int64)
    return y


def hungarian_match(y_true, y_pred):
    """Matches predicted labels to the original using hungarian algorithm."""

    y_true = adjust_range(y_true)
    y_pred = adjust_range(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(-w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    d = {i: j for i, j in ind}
    y_pred = np.array([d[v] for v in y_pred])

    return y_true, y_pred


def set_scores(scores, y_true, y_pred, scoring):
    labels = list(set(y_true))

    for metric in scoring:
        if metric == 'accuracy':
            scores[metric] = metrics.accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            scores[metric] = metrics.precision_score(y_true, y_pred, labels, average='macro')
        elif metric == 'recall':
            scores[metric] = metrics.recall_score(y_true, y_pred, labels, average='macro')
        elif metric == 'f1_score':
            scores[metric] = metrics.f1_score(y_true, y_pred, labels, average='macro')
        elif metric == 'nmi':
            scores[metric] = metrics.normalized_mutual_info_score(y_true, y_pred)
        elif metric == 'adj_mi':
            scores[metric] = metrics.adjusted_mutual_info_score(y_true, y_pred)
        elif metric == 'adj_rand':
            scores[metric] = metrics.adjusted_rand_score(y_true, y_pred)


def compute_scores(y_true, y_pred, scoring={'accuracy', 'precision', 'recall', 'nmi', 'adj_rand', 'f1_score',
                                            'adj_mi'}):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    scores = {}
    y_true, y_pred = hungarian_match(y_true, y_pred)
    set_scores(scores, y_true, y_pred, scoring)

    return scores


def name_virus_families(self, dataset, landmark_all, id_to_numeric_class, top_match=5, umap_reduce_dim=True, ndim=10):
    """
    For each test cluster, estimate sigma and mean. Fit Gaussian distribution with that mean and sigma and calculate the
    probability of each of the train landmarks to be the neighbor to the mean data point.
    Normalization is performed in regard to all other landmarks in train.
    landmarks: virus family landmarks also returned by function train
    id_to_numeric_class: dictionary with virus family name of previously seen virus families as key, and their cluster
    idx as value

    return: interp_names: dictionary with novel virus family cluster index as key and probabilities to all previously
    seen virus families as value

    WORKING IN PROGRESS...
    """
    return


def main():
    """
    1. Initialize seed
    """
    seed = 2
    init_seed(seed)

    """
    2. Preprocess dataset
    """
    # read .tsv metadata of IMG/VR
    metadata = pd.read_csv("./IMGVR/IMGVR_meta.tsv", sep="\t")
    metadata = metadata.loc[:, ['UVIG', 'Ecosystem classification', 'Topology', 'geNomad score', 'Length',
                                'Gene content (total genes;cds;tRNA;geNomad marker)', 'Taxonomic classification',
                                'Host taxonomy prediction']]

    # filter out the outliers
    metadata = metadata[metadata['Length'] < 61610]

    # read .fasta files for virus sequences
    # fasta_files = ["./IMGVR/group_1.fasta"]
    fasta_files = ["./IMGVR/group_1.fasta", "./IMGVR/group_2.fasta", "./IMGVR/group_3.fasta", "./IMGVR/group_4.fasta",
                   "./IMGVR/group_5.fasta", "./IMGVR/group_6.fasta", "./IMGVR/group_7.fasta", "./IMGVR/group_8.fasta",
                   "./IMGVR/group_9.fasta", "./IMGVR/group_10.fasta", "./IMGVR/group_11.fasta",
                   "./IMGVR/group_12.fasta", "./IMGVR/group_13.fasta", "./IMGVR/group_14.fasta",
                   "./IMGVR/group_15.fasta", "./IMGVR/group_16.fasta", "./IMGVR/group_17.fasta",
                   "./IMGVR/group_18.fasta", "./IMGVR/group_19.fasta", "./IMGVR/group_20.fasta",
                   "./IMGVR/group_21.fasta", "./IMGVR/group_22.fasta", "./IMGVR/group_23.fasta",
                   "./IMGVR/group_24.fasta", "./IMGVR/group_25.fasta", "./IMGVR/group_26.fasta",
                   "./IMGVR/group_27.fasta", "./IMGVR/group_28.fasta", "./IMGVR/group_29.fasta"]
    sequences = {}
    for fasta_file in fasta_files:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id.split('|')[0]] = str(record.seq)

    # add a column of sequences only if the ID of the sequence is present in the metadata
    temp_series = metadata['UVIG'].map(sequences)
    metadata['Sequence'] = temp_series.fillna('Not given')

    # filter only those with sequence present
    metadata = metadata[metadata["Sequence"] != 'Not given']

    # extract only the viruses that have been classified on a family level
    extract_family = lambda s: s.split(';')[5] if len(s.split(';')) > 1 else ''
    metadata['Taxonomic classification'] = metadata['Taxonomic classification'].apply(extract_family)
    metadata = metadata[metadata['Taxonomic classification'] != '']

    # extract only the virus families that have a frequency of at least 5
    metadata = metadata.groupby('Taxonomic classification').filter(lambda x: len(x) >= 5)

    # add a column that stores the numeric class of the viruses
    unique_families = metadata['Taxonomic classification'].unique()
    family_to_numeric = {family: i for i, family in enumerate(unique_families)}
    metadata['Numeric class'] = metadata['Taxonomic classification'].map(family_to_numeric)

    """
    3. Obtain train_data, test_data and ID_to_numeric_class
    """
    groups = metadata.groupby('Taxonomic classification')
    train_data = []
    test_data = []
    for _, group in groups:
        group = group.sample(frac=1, random_state=seed)

        # split the group in train and test sets with a 5:1 ratio
        train_size = int(len(group) * 5 / 6)
        train_set = group[:train_size]
        test_set = group[train_size:]
        train_data.append(train_set)
        test_data.append(test_set)

    # create the train and test datasets in two separate dataframes
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    pd.set_option('display.max_columns', None)

    # # print the frequencies of each of the 150 classes
    # train_counts = train_data['Taxonomic classification'].value_counts()
    # train_freq_list = train_counts.tolist()
    # print(train_freq_list)
    # print(len(train_data['Taxonomic classification'].unique()), len(train_data['Numeric class'].unique()))
    # test_counts = test_data['Taxonomic classification'].value_counts()
    # test_freq_list = test_counts.tolist()
    # print(test_freq_list)
    # print(len(test_data['Taxonomic classification'].unique()), len(test_data['Numeric class'].unique()))

    # create a dictionary storing the respective numeric family number for each virus ID
    id_to_numeric_class = dict(zip(metadata['UVIG'], metadata['Numeric class']))

    """
    4. Initialize parameters
    """
    family_count = Counter(test_data['Taxonomic classification'])
    avg_score_direct = np.zeros((len(family_count), 5))
    n_clusters = len(metadata['Taxonomic classification'].unique())
    hid_dim_1 = 1000
    hid_dim_2 = 100
    p_drop = 0.2
    epochs = 30
    learning_rate = 0.001
    lr_scheduler_step = 20
    lr_scheduler_gamma = 0.5  # StepLR learning rate scheduler gamma
    tau = 0.2  # regularizer for inter-cluster distance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for idx, test_data in enumerate(test_data):     # run for every unique families in the test/unlabeled data
        """
        5. Initialize model, loss function and optimizer
        """
        sequence_max_len = metadata['Length'].max()
        mars = FullNet(sequence_max_len, hid_dim_1, hid_dim_2, p_drop).to(device)

        """
        6. Initialize 2 optimizers, one for embedding purposes and one for testing evaluation,
           also initialize landmarks for testing and training, landmark refers to a closed form solution of minimizing
           distance to the mean and maximizing distance to other landmarks, largely related to clusters
        """
        train_frequencies = train_data['Taxonomic classification'].value_counts()
        train_freq_list = train_frequencies.tolist()  # stores the count of data items of each class/cluster
        # print(train_freq_list)
        family_count = Counter(train_data['Taxonomic classification'])
        sorted_family_count = sorted(family_count.items(), key=lambda x: x[1], reverse=True)

        # create a big list of lists storing sequences of each family in descending order of frequency
        test_family_sequences = test_data['Sequence'].values.tolist()
        sorted_train_family_sequences = []
        for family_name, count in sorted_family_count:
            sequences = [sequence for family, sequence in zip(train_data['Taxonomic classification'],
                                                              train_data['Sequence']) if family == family_name]
            sorted_train_family_sequences.append(sequences)

        # initialize landmarks for optimizers
        landmark_test = [torch.zeros(size=(1, hid_dim_2), requires_grad=True, device=device) for _ in range(n_clusters)]
        landmark_train = [torch.zeros(size=(frequency, hid_dim_2), requires_grad=True, device=device) for frequency in
                          train_freq_list]
        kmeans_init_test = init_step(test_family_sequences, mars, device, n_clusters=n_clusters)
        kmeans_init_train = [init_step(sequence_list, mars, device, n_clusters=n_clusters) for sequence_list in
                             sorted_train_family_sequences]
        with torch.no_grad():
            [landmark.copy_(kmeans_init_train[idx]) for idx, landmark in enumerate(landmark_train)]
            [landmark_test[i].copy_(kmeans_init_test[i, :]) for i in range(kmeans_init_test.shape[0])]

        optim = torch.optim.Adam(params=list(mars.encoder.parameters()), lr=learning_rate)
        optim_landmark_test = torch.optim.Adam(params=landmark_test, lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                       gamma=lr_scheduler_gamma,
                                                       step_size=lr_scheduler_step)

        """
        7. training epochs/rounds
        """
        train_iter = [iter(family) for family in sorted_train_family_sequences]
        test_iter = iter(test_family_sequences)
        # also initialize the validation set
        val_split = 0.8
        target = torch.tensor(test_family_sequences)
        unique_targets = torch.unique(target, sorted=True)
        class_idxs = list(map(lambda c: target.eq(c).nonzero(), unique_targets))
        class_idxs = [idx[torch.randperm(len(idx))] for idx in class_idxs]
        val_idx = torch.cat([idx[int(val_split * len(idx)):] for idx in class_idxs])
        val_loader = DataLoader(test_data,
                                batch_sampler=EpochSampler(val_idx),
                                pin_memory=True)
        val_iter = [iter(data) for data in val_loader]

        best_acc = 0
        for epoch in range(1, epochs + 1):
            """
            7.1 - main training/testing part
            """
            mars.train()
            set_requires_grad(mars, False)
            for landmark in landmark_test:
                landmark.requires_grad = False
            optim_landmark_test.zero_grad()

            # update centroids
            task_idx = torch.randperm(len(train_iter))
            for task in task_idx:
                task = int(task)
                x, y, _ = next(train_iter[task])
                x, y = x.to(device), y.to(device)
                encoded, _ = mars(x)
                current_landmark_train = compute_landmarks_train(encoded, y, landmark_train[task], tau=tau)
                landmark_train[task] = current_landmark_train.data

            for landmark in landmark_test:
                landmark.requires_grad = True

            x, y_test, _ = next(test_iter)
            x = x.to(device)
            encoded, _ = mars(x)
            loss, argument_count = loss_test(encoded, torch.stack(landmark_test).squeeze(), tau)

            loss.backward()
            optim_landmark_test.step()

            # update embedding & calculate accuracies
            set_requires_grad(True)
            for landmark in landmark_test:
                landmark.requires_grad = False

            optim.zero_grad()

            total_accuracy = 0
            total_loss = 0
            n_tasks = 0
            mean_accuracy = 0  # also stores the accuracy of the entire training round/epoch

            task_idx = torch.randperm(len(train_iter))
            for task in task_idx:
                task = int(task)
                x, y, _ = next(train_iter[task])
                x, y = x.to(device), y.to(device)
                encoded, _ = mars(x)
                loss, acc = loss_task(encoded, landmark_train[task], y, criterion='dist')
                total_loss += loss
                total_accuracy += acc.item()
                n_tasks += 1

            if n_tasks > 0:
                mean_accuracy = total_accuracy / n_tasks

            # testing part
            x, _, _ = next(test_iter)
            x = x.to(device)
            encoded, _ = mars(x)
            loss, _ = loss_test(encoded, torch.stack(landmark_test).squeeze(), tau)
            total_loss += loss
            n_tasks += 1

            mean_loss = total_loss / n_tasks

            mean_loss.backward()
            optim.step()

            if epoch == epochs:
                print(f'\n=== Epoch: {epoch} ===')
                print(f'Training accuracy: {mean_accuracy}')

            '''
            7.2 - validation & evaluation process
            '''
            if val_loader is None:
                continue
            mars.eval()

            with torch.no_grad():
                # one epoch/round of validation
                n_tasks_val = len(val_iter)
                task_idx_val = torch.randperm(n_tasks_val)

                total_loss_val = 0
                total_accuracy_val = 0

                for task in task_idx_val:
                    x, y, _ = next(val_iter[task])
                    x, y = x.to(device), y.to(device)
                    encoded = mars(x)
                    # essentially comparing with the previous landmarks
                    loss, acc = loss_task(encoded, landmark_train[task], y, criterion='dist')
                    total_loss_val += loss
                    total_accuracy_val += acc.item()
                mean_accuracy_val = total_accuracy_val / n_tasks_val
                mean_loss_val = total_loss_val / n_tasks_val

                if mean_accuracy_val > best_acc:
                    print('Saving model...')
                    best_acc = mean_accuracy_val
                    best_state = mars.state_dict()
                postfix = ' (Best)' if mean_accuracy_val >= best_acc else f' (Best: {best_acc})'
                print(f'Validation loss: {mean_loss_val}, accuracy: {mean_accuracy_val}{postfix}')
            lr_scheduler.step()

        """
        8. final touches after training, testing and evaluation, note that adata saving is not needed as file/function
           calling parts are avoided and the dataset does not involve the usage of adata (which differs from the original
           MARS model)
        """
        if val_loader is None:
            best_state = mars.state_dict()  # note that the best state is saved last
        landmark_all = landmark_train + [torch.stack(landmark_test).squeeze()]

        # assign labels for the viruses
        torch.no_grad()
        mars.eval()
        test_iter = iter(test_family_sequences)
        x_test, y_true, viruses = next(test_iter)
        x_test = x_test.to(device)
        encoded_test, _ = mars(x_test)

        dists = euclidean_dist(encoded_test, landmark_test)
        y_pred = torch.min(dists, 1)[1]

        # saving back to the test_data after assigning cluster labels to the unlabeled meta-dataset
        test_data['Landmarks'] = y_pred
        test_data['Predicted family number'] = list(encoded_test)

        # compute the scores
        eval_scores = compute_scores(y_true, y_pred)

        """
        9. name virus families (WIP)
        """
        name_virus_families(test_data, landmark_all, id_to_numeric_class)

        """
        10. final score calculation and output
        """
        avg_score_direct[idx, 0] = eval_scores['accuracy']
        avg_score_direct[idx, 1] = eval_scores['f1_score']
        avg_score_direct[idx, 2] = eval_scores['nmi']
        avg_score_direct[idx, 3] = eval_scores['adj_rand']
        avg_score_direct[idx, 4] = eval_scores['adj_mi']

        print(f"For {test_data.metadata}, Accuracy: {eval_scores['accuracy']}, F1_score: {eval_scores['f1_score']}, "
              f"NMI: {eval_scores['nmi']}, Adj_Rand: {eval_scores['adj_rand']}, Adj_MI: {eval_scores['adj_mi']}")

    avg_score_direct = np.mean(avg_score_direct, axis=0)
    print(f'\nFor the entire testing dataset, Accuracy: {avg_score_direct[0]}, F1_score: {avg_score_direct[1]}, '
          f'NMI: {avg_score_direct[2]}, Adj_Rand: {avg_score_direct[3]}, Adj_MI: {avg_score_direct[4]}')


if __name__ == '__main__':
    main()
