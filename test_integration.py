import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from modules.models import chronological_regression, chronological_classification
from modules.validation import walk_forward_rmse
from modules.models import _pipe
from sklearn.linear_model import Ridge

class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12); n=150
        x=rng.normal(size=n); y=.4*x+rng.normal(scale=.3,size=n)
        self.X=pd.DataFrame({'x':x,'noise':rng.normal(size=n)})
        self.y=pd.Series(y,name='target')
    def test_chronological_regression_trains(self):
        r=chronological_regression(self.X,self.y,min_rows=80)
        self.assertEqual(r['status'],'ok'); self.assertGreater(r['train_rows'],r['test_rows']); self.assertIn('Training-mean baseline',r['results']['Model'].tolist())
    def test_classification_pipeline(self):
        y=(self.y>0).astype(int)
        r=chronological_classification(self.X,y,min_rows=80)
        self.assertEqual(r['status'],'ok'); self.assertTrue({'Accuracy','Precision','Recall','F1'}.issubset(r['results'].columns))
    def test_walk_forward(self):
        r=walk_forward_rmse(_pipe(Ridge(),True),self.X,self.y,n_splits=5)
        self.assertEqual(r['status'],'ok'); self.assertEqual(len(r['folds']),5)

if __name__=='__main__': unittest.main()
