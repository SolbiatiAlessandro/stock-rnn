"""
This is a template for the APIs of models to be used into the stacking framework.
run with Python 3.x
"""
from time import time, ctime
import pandas as pd
import numpy as np
import pickle as pk
from matplotlib import pyplot as plt
from datetime import datetime, date
import sys
import os
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_model import StockDataSet
from model_rnn import LstmRNN

pp = pprint.PrettyPrinter()

class model():
    """
    model description
    
    python main.py --embed_size=3 --input_size=10 --max_epoch=50

    FEATURES:

    ISSUES:

    this class is for a model (that can also be
    a combination of bagged models)
    The commonality of the bagged models is that
    they share the feature generation
    """

    def __init__(self, name, num_steps=3, input_size=10, embed_size=3, max_epoch=5):
        self.name             = name
        self.type             = None
        self.model = None
        self.training_results = None
        self.assetCode_mapping = None
        print("\ninit model {}".format(self.name))
        sys.path.insert(0, '../') # this is for imports from /kernels
        if not os.path.exists("logs"):
            os.mkdir("logs")

        # stock-rnn FLAGS
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
        flags.DEFINE_integer("max_epoch", 5, "Total training epoches. [50]")
        flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
        flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
        flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
        flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

        FLAGS = flags.FLAGS
        self.FLAGS = FLAGS

        FLAGS.num_steps = num_steps
        FLAGS.input_size = input_size
        FLAGS.embed_size=embed_size
        FLAGS.max_epoch=max_epoch


        self.run_config = tf.ConfigProto()
        self.run_config.gpu_options.allow_growth = True

    def _preprocess(self, market_data):
        """optional data preprocessing
        NOTE: use of this method is DEPRECATED and is only kept
        for backward compatibility
        """
        try:
            market_data = market_data.loc[market_data['time']>=date(2010, 1, 1)]
        except TypeError: # if 'time' is a string value
            print("[_generate_features] 'time' is of type str and not datetime")
            if not market_data.loc[market_data['time']>="2010"].empty:
                # if dates are before 2010 means dataset is for testing
                market_data = market_data.loc[market_data['time']>="2010"]
        assert market_data.empty == False
        return market_data
        

    def _generate_features(self, market_data, news_data):
        """
        GENERAL:
        given the original market_data and news_data
        generate new features, doesn't change original data.
        NOTE: data cleaning and preprocessing is not here,
        here is only feats engineering
        
        MODEL SPECIFIC:

        Args:
            market_train_df: pandas.DataFrame
            news_train_df: pandas.DataFrame

        Returns:
            complete_features: pandas.DataFrame
        """
        start_time = time()
        if verbose: print("Starting features generation for model {}, {}".format(self.name, ctime()))

        complete_features = market_data.copy()

        if 'returnsOpenNextMktres10' in complete_features.columns:
            complete_features.drop(['returnsOpenNextMktres10'],axis=1,inplace=True)


        # generate features here..


        if verbose: print("Finished features generation for model {}, TIME {}".format(self.name, time()-start_time))
        return complete_features

    def _generate_target(self, Y):
        """
        given Y generate binary labels
        returns:
            up, r : (binary labels), (returns)
        """
        binary_labels = Y >= 0
        return binary_labels.astype(int).values, Y.values

    def train(self, X, Y, verbose=False, load=True):
        """
        GENERAL:
        basic method to train a model with given data
        model will be inside self.model after training
        
        MODEL SPECIFIC:
        
        
        Args:
            X: [market_train_df, news_train_df]
            Y: [target]
            verbose: (bool)
            load: load model if possible instead of training
        Returns:
            (optional) training_results
        """

        start_time = time()
        if verbose: print("Starting training for model {}, {}\n".format(self.name, ctime()))
            
        time_reference = X[0]['time'] #time is dropped in preprocessing, but is needed later for metrics eval
        universe_reference = X[0]['universe']


        ####################
        ## stock-rnn code ##
        ####################

        self.sess = tf.Session(config=self.run_config)
        self.sess.__enter__()
        sess = self.sess
        import pickle as pk
        try:
            self.assetCode_mapping = pk.load(open("competition_mapping.pkl","rb"))
            print("[train] MAPPING FOUND, skip data processing and training")
        except:
            print("[train] NO MAPPING FOUND, starting data processing")
            train_dataset = X[0]

            stock_data_list = []
            assetCodes = train_dataset['assetCode'].unique().tolist()
            for process, assetCode in enumerate(assetCodes):
                if process % 500 == 0: print("[train] processing single asset data = "+str(process)+"/"+str(len(assetCodes)))

                asset_train_dataset = train_dataset[train_dataset['assetCode'] == assetCode]
                asset_train_dataset = pd.DataFrame({'Close':asset_train_dataset['close']}).reset_index(drop=True)

                # if asset has too few datapoint don't insert or will break data process
                if  self.FLAGS.num_steps * self.FLAGS.input_size + 1 < len(asset_train_dataset):
                    stock_data_list.append(
                        StockDataSet(assetCode,
                                     input_size=self.FLAGS.input_size,
                                     num_steps=self.FLAGS.num_steps,
                                     test_ratio=0.05,
                                     read_from_twosigma=True,
                                     train_dataset = asset_train_dataset
                                     )
                        )

            print("[train] data processing done TIME:",str(time() - start_time))

            mapping = {}
            for label, stockDataSet in enumerate(stock_data_list):
                mapping[stockDataSet.stock_sym] = label
            pk.dump(mapping, open("temp_mapping","wb"))
            # save map of asset codes
            self.assetCode_mapping = mapping

        self.FLAGS.stock_count = len(self.assetCode_mapping)

        print("[train] initialize rnn_model with stock_count: "+str(self.FLAGS.stock_count))
        self.model = LstmRNN(
            sess,
            stock_count =self.FLAGS.stock_count,
            lstm_size=self.FLAGS.lstm_size,
            num_layers=self.FLAGS.num_layers,
            num_steps=self.FLAGS.num_steps,
            input_size=self.FLAGS.input_size,
            embed_size=self.FLAGS.embed_size,
        )

        if load:
            try:
                self._load()
                print("#"*30+"\n[WARNING] TRAINING SKIPPED, MODEL LOADED FROM MEMORY\n"+"[INFO] if you want to avoid skipping training, change model name\n"+"#"*30)
                if verbose: print("\nFinished training for model {}, TIME {}".format(self.name, time()-start_time))
                return
            except:
                print("Tried to load model but didn't find any")
                pass

        assert len(stock_data_list) > 0
        self.model.train(stock_data_list, self.FLAGS)


        del X, train_dataset

        if verbose: print("\nFinished training for model {}, TIME {}".format(self.name, time()-start_time))

        return None #training results


    def rnn_predict(self, sess, dataset_list, target_stock, label, visualize=False):
        """
        method for SINGLE STOCK predicting new data with rnn model
        from /stock-rnn/main.py
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
        print("[main.y] target stock: "+target_stock)

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

    def single_asset_predict(self, test_df, tail_df, target_stock, verbose=True):
        """
        predict values for single asset
        Args:
            test_df: mixed, used for prediction
            tail_dF: mixed, used for lagged values
            target_stock: (str) asset code to predict
        Return:
            test_prediction

        self.model should already be trained or loaded  this poin
        self.sess should be an already initilaized tf.session with called .__enter__()
        """
        start_time = time()
        if verbose: print("Starting single_asset_predict for model {}, {}".format(self.name, ctime()))

        sess = self.sess
        if not self.model:
            raise Exception("[single_asset_pred] no model found! first run self.train")
        if not self.assetCode_mapping:
            import pickle as pk
            self.assetCode_mapping = pk.load(open("competition_mapping.pkl","rb"))

        label = self.assetCode_mapping[target_stock]
        print("[single_asset_pred] calling StockDataSet on {} {}".format(target_stock, label))
        dataset_list = [StockDataSet(
            target_stock,
            input_size=self.FLAGS.input_size,
            num_steps=self.FLAGS.num_steps,
            test_ratio=1,
            train_dataset = tail_df,
            test_dataset = test_df
            )]

        test_prediction, test_loss = self.rnn_predict(sess, dataset_list, target_stock, label)

        if verbose: print("Finished single_asset_predict for model {}, TIME {}".format(self.name, time()-start_time))
        return test_prediction
    
    def predict(self, X, verbose=False):
        """
        given a block of X features gives prediction for everyrow

        Args:
            X: [market_train_df, news_train_df]
        Returns:
            y: pandas.Series
        """
        start_time = time()
        if verbose: print("Starting prediction for model {}, {}".format(self.name, ctime()))

        X_test = self._generate_features(X[0], X[1], verbose=verbose)
        if verbose: print("X_test shape {}".format(X_test.shape))

        # predict code here..
        y_test = []

        if verbose: print("Finished prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test

    def predict_rolling(self, historical_df, market_obs_df, verbose=False):
        """
        predict features from X, uses historical for (lagged) feature generation
        to be used with rolling prediciton structure from competition

        Args:
            historical_df: [market_train_df, news_train_df]
            market_obs_df: from rolling prediction generator
        """
        start_time = time()
        if verbose: print("Starting rolled prediction for model {}, {}".format(self.name, ctime()))

        X_test = self._generate_features(historical_df[0], historical_df[1], verbose=verbose, normalize=normalize, normalize_vals=normalize_vals, output_len=len(market_obs_df))
        X_test.reset_index(drop=True,inplace=True)
        if verbose: print("X_test shape {}".format(X_test.shape))

        # prediction code here..


        y_test = None

        if verbose: print("Finished rolled prediction for model {}, TIME {}".format(self.name, time()-start_time))
        return y_test


    def inspect(self, X):
        """
        visualize and examine the training of the model
        ONLY FOR GRADIENT BOOSTED TREES
        Args:
            X: for the shap values

        MODEL SPECIFIC:
        plots training results and feature importance
        """
        if not self.training_results:
            print("Error: No training results available")
        else:
            print("printing training results..")
            for _label, key in self.training_results.items():
                for label, result in key.items():
                    plt.plot(result,label=_label+" "+label)
            plt.title("Training results")
            plt.legend()
            plt.show()

        if not self.model1:
            print("Error: No model available")
        else:
            print("printing feature importance..")
            f=lgb.plot_importance(self.model1)
            f.figure.set_size_inches(10, 30) 
            plt.show()

    def _postprocess(self, predictions, normalize=True):
        """
        post processing of predictions

        Args:
            predictions: list(np.array) might be from
                different models
        Return:
            predictions: np.array

        MODEL SPECIFIC:
        the postprocessing is needed to ensemble bagged
        models and to map prediction interval from [0, 1] 
        to [-1, 1]
        """
        y_test = sum(predictions)/len(predictions)
        if normalize:
            y_test = (y_test-y_test.min())/(y_test.max()-y_test.min())
        y_test = y_test * 2 - 1
        return y_test

    def _clean_data(self, data):
        """
        originally from function mis_impute in
        https://www.kaggle.com/guowenrui/sigma-eda-versionnew

        Args:
            data: pd.DataFrame
        returns:
            cleaned data (not in place)
        """
        for i in data.columns:
            if data[i].dtype == "object":
                    data[i] = data[i].fillna("other")
            elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
                    data[i] = data[i].fillna(data[i].mean())
                    # I am just filling the mean of all stocks together?
                    # should fill with the mean of the singular stock
            else:
                    pass
        return data

    def _save(self):
        """
        save models to memory into pickle/self.name

        RaiseException: if can't save
        """
        to_save = self.model
        
        if not to_save:
            print("[_save] Error: not all models are trained")
            print(to_save)
        else:
            if not os.path.exists("pickle"):
                os.mkdir("pickle")
            save_name = os.path.join("pickle",self.name+"_.pkl")
            with open(save_name,"wb") as f:
                pk.dump(to_save, f)
                print("[_save] saved models to "+save_name)

    def _load(self):
        """
        load models to memory from pickle/self.name
        cant use pickle here

        RaiseExcpetion: can't find model
        """
        if not self.model:
            raise Exception("[!] Can't find model")
        if not self.model.load()[0]:
            raise Exception("[!] Can't load model")
        print("[_load] models loaded succesfully")

