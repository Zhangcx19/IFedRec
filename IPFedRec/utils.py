"""
    Some handy functions for pytroch model training ...
"""
import torch
import logging
import numpy as np
import scipy.sparse as sp
import pandas as pd
import copy


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = sp.spdiags(idf, 0, col, col)
    return tf * idf


def load_data(data_path):
    """load data, including train interaction, validation interaction, test interaction and item raw features"""
    item_content_file = data_path + '/item_features.npy'
    item_content = np.load(item_content_file)

    train_file = data_path + '/train.csv'
    train = pd.read_csv(train_file, dtype=np.int32)
    user_ids = list(set(train['uid'].values))
    train_item_ids = list(set(train['iid'].values))
    train_item_content = item_content[train_item_ids]
    train_item_ids_map = {iid: i for i, iid in enumerate(train_item_ids)}
    for i in train_item_ids_map.keys():
        train['iid'].replace(i, train_item_ids_map[i], inplace=True)

    test_file = data_path + '/test.csv'
    test = pd.read_csv(test_file, dtype=np.int32)
    test_item_ids = list(set(test['iid'].values))
    test_item_content = item_content[test_item_ids]
    test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}

    vali_file = data_path + '/vali.csv'
    vali = pd.read_csv(vali_file, dtype=np.int32)
    vali_item_ids = list(set(vali['iid'].values))
    vali_item_content = item_content[vali_item_ids]
    vali_item_ids_map = {iid: i for i, iid in enumerate(vali_item_ids)}

    data_dict = {'train': train, 'train_item_content': train_item_content, 'user_ids': user_ids,
                 'vali': vali, 'vali_item_content': vali_item_content, 'vali_item_ids_map': vali_item_ids_map,
                 'test': test, 'test_item_content': test_item_content, 'test_item_ids_map': test_item_ids_map,
                 }
    return data_dict


def negative_sampling(train_data, num_negatives):
    """sample negative instances for training, refer to Heater."""
    # warm items in training set.
    item_warm = np.unique(train_data['iid'].values)
    # arrange the training data with form {user_1: [[user_1], [user_1_item], [user_1_rating]],...}.
    train_dict = {}
    single_user, user_item, user_rating = [], [], []
    grouped_train_data = train_data.groupby('uid')
    for userId, user_train_data in grouped_train_data:
        temp = copy.deepcopy(item_warm)
        for row in user_train_data.itertuples():
            single_user.append(int(row.uid))
            user_item.append(int(row.iid))
            user_rating.append(float(1))
            temp = np.delete(temp, np.where(temp == row.iid))
            for i in range(num_negatives):
                single_user.append(int(row.uid))
                negative_item = np.random.choice(temp)
                user_item.append(int(negative_item))
                user_rating.append(float(0))
                temp = np.delete(temp, np.where(temp == negative_item))
        train_dict[userId] = [single_user, user_item, user_rating]
        single_user = []
        user_item = []
        user_rating = []
    return train_dict


def compute_metrics(evaluate_data, user_item_preds, item_ids_map, recall_k):
    """compute evaluation metrics for cold-start items."""
    """input:
    evaluate_data: (uid, iid) dataframe.
    user_item_preds: cold-start item prediction for each user.
    item_ids_map: {ori_id: reindex_id} dict.
    recall_k: top_k metrics.
       output:
    recall, precision, ndcg
    """
    pred = []
    target_rows, target_columns = [], []
    temp = 0
    for uid in user_item_preds.keys():
        # predicted location for each user.
        user_pred = user_item_preds[uid]
        _, user_pred_all = user_pred.topk(k=recall_k[-1])
        user_pred_all = user_pred_all.cpu()
        pred.append(user_pred_all.tolist())

        # cold-start items real location for each user.
        user_cs_items = list(evaluate_data[evaluate_data['uid']==uid]['iid'].unique())
        # record sparse target matrix indexes.
        for item in user_cs_items:
            target_rows.append(temp)
            target_columns.append(item_ids_map[item])
        temp += 1
    pred = np.array(pred)
    target = sp.coo_matrix(
        (np.ones(len(evaluate_data)),
         (target_rows, target_columns)),
        shape=[len(pred), len(item_ids_map)]
    )
    recall, precision, ndcg = [], [], []
    idcg_array = np.arange(recall_k[-1]) + 1
    idcg_array = 1 / np.log2(idcg_array + 1)
    idcg_table = np.zeros(recall_k[-1])
    for i in range(recall_k[-1]):
        idcg_table[i] = np.sum(idcg_array[:(i + 1)])
    for at_k in recall_k:
        preds_k = pred[:, :at_k]
        x = sp.lil_matrix(target.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = np.multiply(target.todense(), x.todense())
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(target, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x_coo = sp.coo_matrix(x.todense())
        rows = x_coo.row
        cols = x_coo.col
        target_csr = target.tocsr()
        dcg_array = target_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(target, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k - 1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    return recall, precision, ndcg


def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss