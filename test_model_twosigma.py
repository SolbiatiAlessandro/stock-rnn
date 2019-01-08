import unittest
import model_twosigma
import pandas as pd
import numpy as np
import os

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

class testcase(unittest.TestCase):

    def setUp(self):
        DATA_FOLDER = "~/Desktop/Coding/AI/two-sigma-kaggle/kernels/data"
        import os
        self.news_train_df = None
        print("\n[test_model_twosigma.py/setUp] loading data..")
        self.market_train_df_head = pd.read_csv(os.path.join(DATA_FOLDER, "market_train_df_head.csv")).drop('Unnamed: 0', axis=1)
        self.market_train_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_train_df.csv")).drop('Unnamed: 0', axis=1)
        self.market_test_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_test_df.csv")).drop('Unnamed: 0', axis=1)
        self.market_train_df['time'] = pd.to_datetime(self.market_train_df['time'])
        
        self.market_cols = list(self.market_train_df.columns)

        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop(['returnsOpenNextMktres10'], axis=1)
        self.market_train_df['time'] = pd.to_datetime(self.market_train_df['time'])

    @unittest.skip("wait")
    def test_generate_features(self):
        """
        this is one of the most important tests,
        the idea is that it needs to  make sure that all the
        generated features are exactly as imagined.
        !!NOT ONLY DIMENTIONAL AND SANITY CHECK!!
        you actually need to validate your hypotesis on the feats

        example:
        in two-sigma-kaggle I was generating lagged features but
        I forgot to add groupby('asset') and so all the features
        were basically crap. I got low score and I had no idea why.
        I was only check that the feature were generated! 
        """
        m = model_twosigma.model('example')
        complete_features = m._generate_features(self.market_train_df, self.news_train_df, verbose=True, normalize=False)

        # _generate_features must not change the given dataset in place
        self.assertListEqual(list(self.market_train_df.columns), self.market_cols)
        self.assertListEqual(list(self.news_train_df.columns), self.news_cols)

        self.assertFalse(complete_features.empty)

    @unittest.skip("for later")
    def test_train(self):
        """ OK on commit e6e63c6 """
        m = model_twosigma.model('COMPETITION', num_steps=4, embed_size=4, max_epoch=50)
        self.assertTrue(m.model is None)
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True, load=True)
        try:import model_rnn
        except:pass
        self.assertEqual(type(m.model), model_rnn.LstmRNN)
        print("train test OK")

    @unittest.skip("for later")
    def test_single_asset_predict(self):
        """working benchmark on b63e5298"""
        m = model_twosigma.model('COMPETITION', num_steps=4, embed_size=4, max_epoch=50)
        self.assertTrue(m.model is None)
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True, load=True)
        try:import model_rnn
        except:pass
        self.assertEqual(type(m.model), model_rnn.LstmRNN)

        DATA_FOLDER = "~/Desktop/Coding/AI/two-sigma-kaggle/kernels/data"
        print("[test_single_asset_predict] laoding data for testing")
        mixed_test_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_test_df.csv"))
        mixed_train_df = self.market_train_df
        sym = "MSFT.O"
        got = m.single_asset_predict(mixed_test_df, mixed_train_df, sym)
        assert len(got) > 0
        assert got.shape[1]==10

        from sigma_score import sigma_score
        single = mixed_test_df[mixed_test_df['assetCode'] == sym]
        target = single['returnsOpenNextMktres10']
        predictions = [batch[-1] for batch in got]

        assert len(target) == len(predictions) + 10

        x_t = target[:len(predictions)] * predictions
        score = x_t.mean() / x_t.std()

        print("SIGMA SCORE: "+str(score))

    @unittest.skip("for later")
    def test_predict(self):
        """FROM OLD test_model_lgbm_71"""
        X_test  = [self.market_train_df.iloc[-20:], self.news_train_df[-20:]]
        y_test = self.target[-20:]
        
        m = model_twosigma.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)

        got = m.predict(X_test, verbose=True)

        #sanity check on prediction sizes
        self.assertTrue(len(got) > 0)
        self.assertEqual(X_test[0].shape[0], len(got))
        self.assertEqual(len(y_test), len(got))
        print("predictions test OK")

    #@unittest.skip("for later")
    def test_predict_rolling(self):
        import pickle as pk
        try:
            days = pk.load(open("pickle/days.pkl","rb"))
            print("[test_predict_rolling] loaded days from memory")
        except:
            days = []
            market_test_df = self.market_test_df
            print("[test_predict_rolling] generating days")
            for date in market_test_df['time'].unique():
                market_obs_df = market_test_df[market_test_df['time'] == date].drop(['returnsOpenNextMktres10','universe'],axis=1)
                predictions_template_df = pd.DataFrame({'assetCode':market_test_df[market_test_df['time'] == date]['assetCode'],
                                                                                    'confidenceValue':0.0})
                days.append([market_obs_df,None,predictions_template_df])
            pk.dump(days, open("pickle/days.pkl","wb"))

        model = model_twosigma.model('COMPETITION', num_steps=4, embed_size=4, max_epoch=50)
        self.assertTrue(model.model is None)
        model.train([self.market_train_df, self.news_train_df], self.target, verbose=True, load=True)
        try:import model_rnn
        except:pass
        self.assertEqual(type(model.model), model_rnn.LstmRNN)


        # the following is simulation code from submission kernel

        print("[test_predict_rolling] starting rolling simulation")
        import time
        PREDICTIONS = []
        n_days = 0
        prep_time = 0
        prediction_time = 0
        n_lag=[40] # num_steps * input_size = 40
        packaging_time = 0
        total_market_obs_df = []
        for (market_obs_df, news_obs_df, predictions_template_df) in days[:2]:
            n_days +=1
            if (n_days%50==0):
                pass
                #print(n_days,end=' ')
            t = time.time()
            #market_obs_df['time'] = market_obs_df['time'].dt.date

            total_market_obs_df.append(market_obs_df)
            if len(total_market_obs_df)==1:
                history_df = total_market_obs_df[0]
            else:
                history_df = pd.concat(total_market_obs_df[-(np.max(n_lag)+1):])
            
            confidence = model.predict_rolling([history_df, None], market_obs_df, verbose=True)      
               
            preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
            predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
            PREDICTIONS.append(predictions_template_df)

        for i, ref in enumerate(PREDICTIONS):
            df = pd.DataFrame({'assetCode':PREDICTIONS[i]['assetCode'],'ref':PREDICTIONS[i]['confidenceValue'],'compare':PREDICTIONS[i]['confidenceValue']})
            try:
                self.assertTrue(all(df.iloc[:,1] == df.iloc[:,2]))
            except:
                print("AssertionError: rolling predictions not correct")
                import pdb;pdb.set_trace()
                pass

    @unittest.skip("do not print")
    def test_inspect(self):
        m = model_twosigma.model('example')
        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        m.inspect(self.market_train_df)

    @unittest.skip("this is computationally heavy")
    def test_train_with_fulldataset(self):
        m = model_twosigma.model('example')
        self.assertTrue(m.model is None)

        print("loading full dataset ..")
        self.market_train_df = pd.read_csv("../data/market_train_df.csv").drop('Unnamed: 0', axis=1)
        self.news_train_df = None
        
        self.market_cols = list(self.market_train_df.columns)

        self.target = self.market_train_df['returnsOpenNextMktres10']
        self.market_train_df.drop('returnsOpenNextMktres10',axis=1)

        m.train([self.market_train_df, self.news_train_df], self.target, verbose=True)
        self.assertEqual(type(m.model), m.type)
        print("train test OK")

        m.inspect(self.market_train_df) #looks healthy

        got = m.predict([self.market_train_df[-100:], None], verbose=True, do_shap=True)

        print(got.describe())

    @unittest.skip("wait")
    def test_prediction_postprocessing(self):
        m = model_twosigma.model('example')
        model1_predictions = np.full(100, 0.4)
        model2_predictions = np.full(100, 0.6)
        y_test = m._postprocess([model1_predictions, model2_predictions])
        # test bagging 
        self.assertEqual(y_test.shape, (100, ))
        # test mapping
        self.assertTrue(all(np.full(100, 0) == y_test))
        print("test_prediction_postprocessing OK")

    @unittest.skip("wait")
    def test_clean_data(self):
        m = model_twosigma.model('example')
        dirty_array = np.full(10,5,dtype=float)
        dirty_array[4] = np.nan # generate artificial nans

        m._clean_data(pd.DataFrame(dirty_array))
        self.assertEqual(dirty_array[4], 5.0)

    @unittest.skip("wait")
    def test_save_load(self):
        m = model_twosigma.model('example')
        m.name = "save_test"
        m.model1 = 7
        m.model2 = 1
        m.model3 = 2
        m.model4 = 3
        m.model5 = 4
        m.model6 = 5
        m._save()

        n = model_twosigma.model('example')
        n.name = "save_test"
        n._load()
        self.assertEqual(n.model1,  7)
        self.assertEqual(n.model2,  1)
        self.assertEqual(n.model3,  2)
        self.assertEqual(n.model4,  3)
        self.assertEqual(n.model5,  4)
        self.assertEqual(n.model6,  5)
        

if __name__=="__main__":
    unittest.main()
