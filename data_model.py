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
            DATA_FOLDER = "~/Desktop/Coding/AI/two-sigma-kaggle/kernels/data/market_test_df.csv"
            mixed_df = pd.read_csv(DATA_FOLDER)
            raw_df = pd.DataFrame()
            # try different suffixes for mapping assetCode name
            # from stock-rnn to two-sigma convention
            if raw_df.empty:
                raw_df = mixed_df[mixed_df['assetCode'] == self.stock_sym+".O"]['close']
            if raw_df.empty:
                raw_df = mixed_df[mixed_df['assetCode'] == self.stock_sym+".A"]['close']
            if raw_df.empty:
                raw_df = mixed_df[mixed_df['assetCode'] == self.stock_sym+".N"]['close']
            raw_df = pd.DataFrame({'Close':raw_df})

        if raw_df.empty: raise ValueError(": can't find {} in two-sigma dataset".format(self.stock_sym))


        try: assert 'Close' in list(raw_df.columns)
        except: exit("[DataFormattingError] Close columns not present in "+os.path.join(DATA_FOLDER, "%s.csv" % stock_sym))

        # Merge into one sequence
        if close_price_only:
            self.raw_seq = raw_df['Close'].tolist()
        else:
            self.raw_seq = [price for tup in raw_df[['Open', 'Close']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq)
        if read_from_twosigma:
            try:
                self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_prediction_data(self.raw_seq)
            except Exception as e:
                print("[_prepare_data_prediction] Exception: "+str(e.message))
                import pdb;pdb.set_trace() 
        else:
            try:
                self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)
            except Exception as e:
                print("[_prepare_data] Exception: "+str(e.message))
                import pdb;pdb.set_trace() 

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_prediction_data(self, seq): 
        """
        Args:
            seq: raw_df['Close'].tolist()
        Returns:
            train_X, train_y, test_X, test_y
        """
        # split into items of input_size

        # ORIGINAL CODE: it was split range(len(seq) // input_size)
        # in order to have non overlapping X. Of course since I need day to day
        # predictions this is not ok.
        #
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size]) for i in range(len(seq))]

        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:]) if len(curr) > 0]

        # split into groups of num_steps

        # so my sequences are too short,the original code breaks
        # because X is empty, this is the original code:

        # the original code would split num_steps, where every step is long size_input
        # my size input is fixed to 10, so I would put self.num_step to 1 or 2?
        # or I can try to concatenate previous dataset

        # I will modify and raise a warning, for now I use dull values

        import pdb;pdb.set_trace() 
        range_val = len(seq) - self.num_steps 
        if range_val <= 0:
            print("[_prepare_data] WARNING: using dull values for part of training")
            range_val = len(seq)

        X = np.array([seq[i: i + self.num_steps] for i in range(range_val)])
        y = np.array([seq[i + self.num_steps] for i in range(range_val)])

        import pdb;pdb.set_trace() 
        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

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
