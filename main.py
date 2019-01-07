import os
import pandas as pd
import numpy as np
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_model import StockDataSet
from model_rnn import LstmRNN
from matplotlib import pyplot as plt

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 1, "Input size [1]")
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

if not os.path.exists("logs"):
    os.mkdir("logs")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def load_sp500(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05):
    """not documented, from original fork, only for training"""
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)
        ]

    # Load metadata of s & p 500 stocks
    info = pd.read_csv("data/constituents-financials.csv")
    info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
    info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
    print info['file_exists'].value_counts().to_dict()

    info = info[info['file_exists'] == True].reset_index(drop=True)
    info = info.sort('market_cap', ascending=False).reset_index(drop=True)

    if k is not None:
        info = info.head(k)

    print "Head of S&P 500 info:\n", info.head()

    # Generate embedding meta file
    info[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)

    return [
        StockDataSet(row['symbol'],
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=0.05,
                     read_from_twosigma=False)
        for _, row in info.iterrows()]

def model_predict(sess, dataset_list, target_stock, label, visualize=False):
    """
    method for predicting new data with rnn model
    Args:
        sess: (tensorflow.python.client.session.Session)
        dataset_list: [(data_model.StockDataSet)] *NOTE
        target_stock: (str)
        label: (int)
        visualize: (bool) plots predictions
    returns:
        test_predictions, test_loss

    NOTE:
        Arg dataset_list is generated as follows
    dataset_list = [StockDataSet(
        FILE_NAME,
        input_size=FLAGS.input_size,
        num_steps=FLAGS.num_steps,
        test_ratio=1)]

    for more info: help(data_model.StockDataSet.__init__)
    """
    print("[main.py] START PREDICTION STAGE")
    print("[main.[y] target stock: "+target_stock)

    # Merged test data of different stocks.
    merged_test_X = []
    merged_test_y = []
    merged_test_labels = []

    for label_, d_ in enumerate(dataset_list):
        merged_test_X += list(d_.test_X)
        merged_test_y += list(d_.test_y)
        merged_test_labels += [[label]] * len(d_.test_X)

    test_data_feed = {
        sess.graph.get_tensor_by_name('learning_rate:0'): 0.0,
        sess.graph.get_tensor_by_name('keep_prob:0'): 1.0,
        sess.graph.get_tensor_by_name('inputs:0'): merged_test_X,
        sess.graph.get_tensor_by_name('targets:0'): merged_test_y,
        sess.graph.get_tensor_by_name('stock_labels:0'): merged_test_labels,
    }
    prediction = sess.graph.get_tensor_by_name('add:0')
    loss = sess.graph.get_tensor_by_name('loss_mse_test:0')
    test_prediction, test_loss = sess.run([prediction, loss], test_data_feed)

    #test_prediction are normalized prices (not returns)

    print("[main.py] GOT PREDICTIONS OF SHAPE")
    print(test_prediction.shape)

    if visualize:
        i = 9
        print("printing labels[{}]".format(i))
        pred = np.transpose(test_prediction)[i] * 5
        real = np.transpose(merged_test_y)[i]
        plt.plot(pred, label='pred')
        plt.plot(real, label='real')
        plt.legend()
        plt.show()

    return test_prediction, test_loss

def binary_score(target_data, test_prediction, target_stock):
    """
    evaluates and store a simple binary score 
    for model predictions

    Args:
        target_data: (data_model.StockDataSet)
        test_predictions: (numpy.ndarray)
           shape (len_prediction, FLAGS.input_size)
        target_stock: (str) for name printing
    return:
        binary_score: (float)
    """
    binary_score = 0
    for i in range(len(target_data.test_X)):
        last_close = target_data.test_X[i][-1][-1]
        #  _X[i][-1][-1], first [-1] is for the last 10 days, second [-1] is for the last day of last 10 days
        real_next10 = target_data.test_y[i][-1]
        pred_next10 = test_prediction[i][-1]

        binary = int(((pred_next10 - last_close) * (real_next10 - last_close)) >= 0)
        # are pred and real in the same direction?
        binary_score += binary
    from operator import truediv
    score1 = truediv(binary_score, len(target_data.test_X))
    print("[main] PREDICTIONS: binary score on "+target_stock)
    print(score1)
    score_file = "./logs/stock_rnn_lstm128_step30_input10_embed3/scores/train_data"
    print("[main] writing scores on "+score_file)
    with open(score_file,'a') as sf:
        sf.write(target_stock+" "+str(score1)+"\n")


def main(_):
    """procedure with two user cases

    1) train model:
    python main.py --stock_count=400 --embed_size=3 --input_size=10 --max_epoch=50

    2) if already trained predict for hardcoded target_stock
     and computes binary metric:
    python main.py --stock_count=400 --embed_size=3 --input_size=10 --max_epoch=50 --train=False
    """
    pp.pprint(flags.FLAGS.__flags)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:

        show_all_variables()

        if FLAGS.train:
            # TRIGGER TRAIN ROUTINE
            # stock_data_list for training

            # loads train data from two_sigma
            DATA_FOLDER = "~/Desktop/Coding/AI/two-sigma-kaggle/kernels/data"
            mixed_test_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_test_df.csv"))
            # mixed_train_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_train_df.csv"))
            train_dataset=mixed_test_df


            from time import time, ctime
            start_time = time()
            print("[main.py] start training data processing from two-sigma dataset, "+ctime())
            stock_data_list = []
            assetCodes = mixed_test_df['assetCode'].unique().tolist()
            for process, assetCode in enumerate(assetCodes):
                if process % 500 == 0: print("process = "+str(process))

                asset_train_dataset = train_dataset[train_dataset['assetCode'] == assetCode]
                asset_train_dataset = pd.DataFrame({'Close':asset_train_dataset['close']}).reset_index(drop=True)

                # if asset has too few datapoint don't insert or will break data process
                if  FLAGS.num_steps * FLAGS.input_size + 1 < len(asset_train_dataset):
                    stock_data_list.append(
                        StockDataSet(assetCode,
                                     input_size=FLAGS.input_size,
                                     num_steps=FLAGS.num_steps,
                                     test_ratio=0.05,
                                     read_from_twosigma=True,
                                     train_dataset = asset_train_dataset
                                     )
                        )

            print("[main.py] data processing done TIME:",str(time() - start_time))

            print("[main.py] initialize rnn_model with stock_count: "+str(len(stock_data_list)))
            FLAGS.stock_count = len(stock_data_list)
            rnn_model = LstmRNN(
                sess,
                stock_count =FLAGS.stock_count,
                lstm_size=FLAGS.lstm_size,
                num_layers=FLAGS.num_layers,
                num_steps=FLAGS.num_steps,
                input_size=FLAGS.input_size,
                embed_size=FLAGS.embed_size,
            )

            rnn_model.train(stock_data_list, FLAGS)

        else:
            # TRIGGER PREDICTION ROUTINE
            if not rnn_model.load()[0]:
                raise Exception("[!] Train a model first, then run test mode")

            target_stock = "AAPL.O"
            label = 8
            dataset_list = [StockDataSet(
                target_stock,
                input_size=FLAGS.input_size,
                num_steps=FLAGS.num_steps,
                test_ratio=1)]

            test_prediction, test_loss = model_predict(sess, dataset_list, target_stock, label)

            target_data = dataset_list[0]
            binary_score(target_data, test_prediction, target_stock)


if __name__ == '__main__':
    tf.app.run()
