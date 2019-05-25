import argparse
import pickle as pkl
import sys
import csv
import zipfile

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm


def load_zip(fname):
    """
    This function load the zip file
    :param zipfile: file path of a zipfile
    :return: reader of the zipped file
    """
    train = zipfile.ZipFile(fname, 'r')
    try:
        path = train.filelist[0]
        print(f"File found at {path}")
    except IndexError:
        print("csv not found in the zip file")
        sys.exit(0)
    return train.open(path)


def feeder(data, batch_size=10000):
    """
    Read csv line by line, return a batch of size batch_size
    :param data: opened file
    :param batch_size: size of the batch
    :return:
    """
    while True:
        xs = []
        ys = []
        for _ in range(batch_size):
            line = next(data).rstrip().decode().split(',')
            # a bit fof a hack but i'm short of time
            if len(line) > 7:
                y = int(line[-2:][-1])
                x = line[:-2]
            else:
                y = 0
                x = line[1:]
            x[-1] = pd.to_datetime(x[-1]).value // 10 ** 9
            x = np.array(x, dtype=np.int32)
            xs.append(x)
            ys.append(y)

        yield np.array(xs), np.array(ys)


def train(data, batch_size=10000, test_every=10, max_steps=int(1e6), n_epochs=1, log_file=None, model_path=None):
    """
    Peform training
    :param data: take an opened zipfile
    :param batch_size: size of the bach
    :param test_every: number of test over time. This also define the train test split
    :param max_steps: number of total line to read from the zip file
    :param n_epochs:  number of epochs over a single minibatch
    :param log_file: log file for training
    :return: column name
    """
    columns = next(data)

    feed = feeder(data, batch_size)
    sampler = SMOTEENN()
    if model_path is not None:
        with open(model_path,'rb') as f:
            clf = pkl.load(f)
    else:
        clf = SGDClassifier()

    for global_step in tqdm(range(max_steps)):
        try:
            x_tr, y_tr = next(feed)
            x_tr = minmax_scale(x_tr)
        except StopIteration:
            feed = feeder(data, batch_size)
            continue
        for _ in range(n_epochs):
            try:
                x_tr, y_tr = sampler.fit_resample(x_tr, y_tr)
            except ValueError as e:
                tqdm.write(str(e))
                continue
            clf.fit(x_tr, y_tr)

        if global_step % test_every == 0:
            y_hat = clf.predict(x_tr)
            tqdm.write(classification_report_imbalanced(y_tr, y_hat), file=log_file)
            fname = f"model/clf_{global_step}.pkl"
            with open(fname, 'wb') as f:
                pkl.dump(clf, f)
            tqdm.write(f"File saved as {fname}")
    return columns


def test(data, model_path, logs_path='logs', batch_size=1000, dest_path='logs'):
    """
    Perform test and prediction
    :param data: take an opened zipfile
    :param model_path: pkl file of the classifier
    :param logs_path: path of the log file
    :param batch_size: batch_size
    :param dest_path: path of the prediction file
    :return:
    """
    columns = next(data)

    feed = feeder(data, batch_size)
    try:
        with open(model_path, 'rb') as f:
            clf = pkl.load(f)
    except IOError:
        print("Model not found. Train first")
        sys.exit(0)
    print("reading total lines in the file....")
    # n_lines = len(data.readlines())
    n_lines = 1000
    print(n_lines)
    idx = 0
    for step in tqdm(range(n_lines // batch_size)):
        try:
            x_tr, y_tr = next(feed)
            x_tr = minmax_scale(x_tr)
        except StopIteration:
            break
        y_hat = clf.predict(x_tr)
        with open(logs_path + '/report.txt', 'a+') as f:
            tqdm.write(classification_report_imbalanced(y_tr, y_hat), file=f)
        with open(dest_path + '/preds.csv', 'a+') as f:
            writer = csv.writer(f)
            for r in y_hat:
                writer.writerow([idx, r])
                idx += 1


def predict(csv_path, model_path, dest_path='logs/predictions.csv'):
    df = pd.read_csv(csv_path)
    try:
        with open(model_path, 'rb') as f:
            clf = pkl.load(f)
    except IOError:
        print("Model not found. Train first")
        sys.exit(0)

    y_hat = clf.predict(df)
    y_hat.to_csv(dest_path)
    print(f"File saved at {dest_path}")


def main(args):
    if args.mode == "train":
        tr_set = load_zip(args.training_set)
        train(tr_set, batch_size=args.batch_size,model_path=args.model)
    elif args.mode == "test":
        test_set = load_zip(args.test_set)
        test(test_set, batch_size=args.batch_size, model_path=args.model, logs_path=args.log_path)
    elif args.mode == "predict":
        predict(args.prediction_set, model_path=args.model_path, dest_path=args.dest_path)
    else:
        print('Mode not found. Exiting.')
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10000)
    parser.add_argument('--training_set', default='train.csv.zip')
    parser.add_argument('--test_set', default='test.csv.zip')
    parser.add_argument('--prediction_set', default='train_sample.csv')
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--mode', default='test')
    parser.add_argument('--model', default='model/clf_0.pkl')
    parser.add_argument('--log_path', default='logs/')
    parser.add_argument('--dest_path', default='log')
    args = parser.parse_args()
    main(args)
