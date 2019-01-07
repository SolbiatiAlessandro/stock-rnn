import numpy as np
import os
import pandas as pd
import random
import time

random.seed(time.time())


class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 close_price_only=True,
                 read_from_twosigma=True):
        """
        Args:
            stock_sym: (str) filename
            input_size, num_steps: (int) used for data processing
            test_ratio: 1 for testing
            read_from_twosigma: (bool) toggle integration

        NOTE:

        data are read from DATA_FOLDER/stock_sym, and they
        must be in the format:

        raw_df: pd.DataFrame, shape(data_len, columns) 
        only constrain is that:
        assert 'Close' in columns == True

        """
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized
        self.read_from_two_sigma = read_from_twosigma
        # Read csv file
        if not read_from_twosigma:
            DATA_FOLDER = "data"
            raw_df = pd.read_csv(os.path.join(DATA_FOLDER, "%s.csv" % stock_sym))
        else:

            # NOTE: requirements for data size
            # if you want to predict starting from t1, you
            # need to have data starting from t0 where
            # t0 = t1 - self.num_steps * self.input_size
            # for the single asset

            DATA_FOLDER = "~/Desktop/Coding/AI/two-sigma-kaggle/kernels/data"
            print("[dataLoader] loading two-sigma data")
            mixed_test_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_test_df.csv"))
            mixed_train_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_train_df.csv"))
            tail_size = self.input_size * self.num_steps
            tail_time = mixed_train_df['time'].unique()[-tail_size-1]
            mixed_tail_df = mixed_train_df[mixed_train_df['time'] > tail_time]
            del mixed_train_df

            test_df = pd.DataFrame()

            # try different suffixes for mapping assetCode name
            # from stock-rnn to two-sigma convention
            if test_df.empty:
                tail_df = mixed_tail_df[mixed_tail_df['assetCode'] == self.stock_sym+".O"]['close']
                test_df = mixed_test_df[mixed_test_df['assetCode'] == self.stock_sym+".O"]['close']
            if test_df.empty:
                tail_df = mixed_tail_df[mixed_tail_df['assetCode'] == self.stock_sym+".A"]['close']
                test_df = mixed_test_df[mixed_test_df['assetCode'] == self.stock_sym+".A"]['close']
            if test_df.empty:
                tail_df = mixed_tail_df[mixed_tail_df['assetCode'] == self.stock_sym+".N"]['close']
                test_df = mixed_test_df[mixed_test_df['assetCode'] == self.stock_sym+".N"]['close']

            # test_df is the df to make predictions
            test_df = pd.DataFrame({'Close':test_df}).reset_index(drop=True)
            # tail_df is the df to calculate lagged values
            tail_df = pd.DataFrame({'Close':tail_df}).reset_index(drop=True)

        if test_df.empty: raise ValueError(": can't find {} in two-sigma dataset".format(self.stock_sym))


        try: assert 'Close' in list(test_df.columns)
        except: exit("[DataFormattingError] Close columns not present in "+os.path.join(DATA_FOLDER, "%s.csv" % stock_sym))

        # Merge into one sequence
        if close_price_only:
            self.tail_seq = tail_df['Close'].tolist()
            self.test_seq = test_df['Close'].tolist()
        else:
            raise NotImplemented(" implemented only close prices")

        self.tail_seq = np.array(self.tail_seq)
        self.test_seq = np.array(self.test_seq)
        if read_from_twosigma:
            self.test_X, self.test_y = self._prepare_prediction_data(self.test_seq, self.tail_seq)
        else:
            self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.test_seq)

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_prediction_data(self, test_seq, tail_seq): 
        """
        Args:
            test_seq: numpy.ndarray(float), last arbitrary N values
                of asset close to predict
            tail_seq: numpy.ndarray(float), last tail_size values
                of asset close to compute lagged feats
            
        Returns:
            X, y
        """
        # split into items of input_size

        # predictions will start from this element of the concat df
        prediction_start = len(tail_seq)
        concat_seq = np.append(tail_seq, test_seq)
        assert concat_seq[prediction_start] == test_seq[0]

        # ORIGINAL CODE: it was split range(len(seq) // input_size)
        # in order to have non overlapping X. Of course since I need day to day
        # predictions this is not ok.

        # concat_seq is a np.array of float
        # seq is list of np.array of len input_size
        seq = [np.array(concat_seq[i : i + self.input_size]) for i in range(len(concat_seq) - self.input_size)]

        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i + 1][-1] - 1.0 for i, curr in enumerate(seq[1:])]
        # TODO: issue? was this bug also in original branch? should check 
        # seq[i + 1] was seq[i] and element were normalized on 8th and not on 9th

        range_val = len(seq) - self.num_steps 
        # reason and condition on tail_seq is that
        # len(tail_seq) + len(test_seq) > self.num_steps * self.input_size
        assert range_val > 0

        # ORIGINAL CODE
        # y = np.array([seq[i + self.num_steps] for i in range(range_val)])
        # X = np.array([seq[i: i + self.num_steps] for i in range(range_val)])


        # there are N = len(test_seq) - self.input_size, len(y) = N
        y = np.array(seq[prediction_start :])
        assert y.shape == (len(test_seq) - self.input_size,
                self.input_size)
                

        # for every y[i] point add the last num_steps chunks
        # of length input_size to X[i]
        X = []
        for k in range(len(y)):
            X.append([
                seq[prediction_start + k - \
                (self.num_steps - i) * (self.input_size)] \
                for i in range(self.num_steps) \
                ])
        X = np.array(X)

        assert X.shape == (y.shape[0], self.num_steps, self.input_size)
        # the following are all assertions on data consistency
        # they work only if self.normalized = False
        # check in case of refactoring
        """
        assert X[0][-1][-1] == concat_seq[prediction_start - 1]
        assert X[0][-2][-1] == concat_seq[prediction_start - 11]
        assert X[1][-1][-1] == concat_seq[prediction_start]
        assert X[1][-2][-1] == concat_seq[prediction_start - 10]
        assert X[2][-1][-1] == concat_seq[prediction_start + 1]
        assert X[2][-2][-1] == concat_seq[prediction_start - 9]
        """

        print("[_prepare_data_prediction] prepare X with shape {}".format(X.shape))
        print("[_prepare_data_prediction] prepare y with shape {}".format(y.shape))
        return X, y



    def _prepare_data(self, seq): #TODO: reformat _prepare_data_prediction
        # and keep this with originial code
        """
        Args:
            seq: raw_df['Close'].tolist()
        Returns:
            train_X, train_y, test_X, test_y
        """
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
              for i in range(len(seq) // self.input_size)]


        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
