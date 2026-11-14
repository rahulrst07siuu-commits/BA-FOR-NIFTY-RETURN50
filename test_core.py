import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from modules.features import build_daily_features, build_monthly_research, daily_model_matrix
from modules.data_io import data_audit, load_workbook
from modules.validation import assert_chronological
from modules.models import chronological_regression
from modules.sip_research import sip_regression

class DataTests(unittest.TestCase):
    def test_daily_feature_target_shift(self):
        dates=pd.bdate_range('2024-01-01',periods=70); close=np.linspace(100,140,70)+np.sin(np.arange(70))
        df=pd.DataFrame({'Date':dates,'Nifty_Open':close-.5,'Nifty_High':close+1,'Nifty_Low':close-1,'Nifty_Close':close,'IndiaVIX_Close':15})
        f=build_daily_features(df)
        self.assertAlmostEqual(f.loc[0,'target_next_close'],close[1])
        self.assertTrue(pd.isna(f.iloc[-1]['target_next_return_pct']))
        self.assertIn('rsi_14',f.columns); self.assertIn('sma_50',f.columns)
        X,y=daily_model_matrix(f)
        self.assertNotIn('Target_Next_Session_Return_Pct',X.columns)
        self.assertNotIn('Target_Next_Session_Close',X.columns)
        self.assertNotIn('target_next_return_pct',X.columns)
    def test_monthly_separate_merge(self):
        m=pd.DataFrame({'Month':pd.date_range('2020-01-01',periods=4,freq='MS'),'SIP Inflow (cr)':[1,2,3,4],'FII Net Flow (cr)':[-1,1,-2,2],'FII Selling (0/1)':[1,0,1,0]})
        r=build_monthly_research(m)
        self.assertEqual(len(r),4); self.assertEqual(r['SIP x FII Selling'].tolist(),[1,0,3,0])
    def test_audit_missing_and_duplicates(self):
        d=pd.DataFrame({'Date':['2024-01-01','2024-01-01'],'x':[1,np.nan]})
        a=data_audit(d,'Date'); self.assertEqual(a['duplicate_dates'],1); self.assertEqual(a['missing_cells'],1)
    def test_chronological_guard(self):
        self.assertTrue(assert_chronological(pd.date_range('2020-01-01',periods=4),pd.date_range('2020-01-05',periods=2)))
        with self.assertRaises(ValueError): assert_chronological(pd.date_range('2020-01-01',periods=5),pd.date_range('2020-01-05',periods=2))
    def test_small_sample_blocks_training(self):
        X=pd.DataFrame({'x':np.arange(22)}); y=pd.Series(np.arange(22),name='y')
        r=chronological_regression(X,y,min_rows=80); self.assertEqual(r['status'],'insufficient_data')
    def test_sip_regression_requires_real_monthly_outcome(self):
        m=pd.DataFrame({'SIP Inflow (cr)':np.arange(50),'FII Net Flow (cr)':-np.arange(50),'FII Selling (0/1)':np.ones(50)})
        r=sip_regression(m); self.assertEqual(r['status'],'missing_dependent_variable')

if __name__=='__main__': unittest.main()
