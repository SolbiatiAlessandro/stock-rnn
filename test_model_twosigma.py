import unittest
import model_twosigma
import pandas as pd
import numpy as np
import os


class testcase(unittest.TestCase):

    def setUp(self):
        DATA_FOLDER = "~/Desktop/Coding/AI/two-sigma-kaggle/kernels/data"
        import os
        self.news_train_df = None
        print("\n[test_model_twosigma.py/setUp] loading data..")
        self.market_train_df_head = pd.read_csv(os.path.join(DATA_FOLDER, "market_train_df_head.csv")).drop('Unnamed: 0', axis=1)
        self.market_train_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_train_df.csv")).drop('Unnamed: 0', axis=1)
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

    #@unittest.skip("for later")
    def test_single_asset_predict(self):
        m = model_twosigma.model('COMPETITION', num_steps=4, embed_size=4, max_epoch=50)
        self.assertTrue(m.model is None)
        m.train([self.market_train_df_head, self.news_train_df], self.target, verbose=True, load=True)
        try:import model_rnn
        except:pass
        self.assertEqual(type(m.model), model_rnn.LstmRNN)

        DATA_FOLDER = "~/Desktop/Coding/AI/two-sigma-kaggle/kernels/data"
        print("[test_single_asset_predict] laoding data for testing")
        mixed_test_df = pd.read_csv(os.path.join(DATA_FOLDER, "market_test_df.csv"))
        mixed_train_df = self.market_train_df
        got = m.single_asset_predict(mixed_test_df, mixed_train_df, "AAPL.O")
        assert len(got) > 0

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

    @unittest.skip("for later")
    def test_predict_rolling(self):
        """FROM OLD test_model_lgbm_71"""
        import pickle as pk
        with open("pickle/rolling_predictions_dataset.pkl","rb") as f:
            days = pk.load(f)
        mins,maxs,rng = pk.load(open("pickle/normalizing.pkl","rb"))
        model = model_twosigma.model('DecisionTree.model_twosigma')
        model._load()

        PREDICTIONS = pk.load(open("pickle/_ref_rolling_predictions.pkl","rb"))

        # the following is simulation code from submission kernel

        import time
        _COMPARE_PREDICTIONS = []
        n_days = 0
        prep_time = 0
        prediction_time = 0
        n_lag=[3,7,14]
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
                
            
            confidence = model.predict_rolling([history_df, None], market_obs_df, verbose=True, normalize=True, normalize_vals = [maxs,mins])      
               
            preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
            predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
            _COMPARE_PREDICTIONS.append(predictions_template_df)

        for i, ref in enumerate(_COMPARE_PREDICTIONS):
            df = pd.DataFrame({'assetCode':PREDICTIONS[i]['assetCode'],'ref':PREDICTIONS[i]['confidenceValue'],'compare':_COMPARE_PREDICTIONS[i]['confidenceValue']})
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
