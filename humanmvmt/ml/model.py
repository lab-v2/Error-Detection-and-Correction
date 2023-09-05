import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

    
class Model:
    def __init__(self, model = None, df = None, map_file = None, hp_dict = None, cv = None, rfe_cv = None, test_size = None):
        self.model = model
        self.map_file = map_file
        self.hp_dict = hp_dict
        self.cv = cv
        self.rfe_cv = rfe_cv
        self.test_size = test_size
        if df is not None:
            self.df = df
            if map_file is not None:
                self.df['label'] = self.df['label'].map(self.map_file)                                                                                                                                    
            self.X = self.df.drop(['label', 'segment'], axis=1)
            self.Y = self.df['label']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = self.test_size, 
                                                                                    random_state = 1999, stratify = self.Y)

    def fit(self, save_model_path = None):
        print('Searching best parameters..')
        clf = GridSearchCV(self.model(), self.hp_dict, scoring='roc_auc', cv=self.cv)
        clf.fit(self.X_train, self.y_train)
        print('Best parameters extracted..')

        # Fitting model with best params
        print('Fitting model with best parameters..')
        best_model = self.model(**clf.best_params_)
        best_model.fit(self.X_train, self.y_train)
        print('Model with best parameters completed..')

        # Recursive feature elimination based feature selection
        print('Eleminating features using RFECV..')
        selector = RFECV(best_model, step=1, cv=self.rfe_cv, scoring = 'roc_auc_ovr_weighted')
        selector = selector.fit(self.X_train, self.y_train)
        print('Feature selection completed..')

        self.dropped_features = set(self.X.columns).difference(set(set(selector.get_feature_names_out())))
        self.best_features = selector.get_feature_names_out()

        if save_model_path is not None:
            with open(save_model_path, 'wb') as f:
                pickle.dump(selector, f)

    def predict(self, saved_model_path = None):
        if not self.model:
            with open(saved_model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        # Get predictions
        y_tr_pred = self.model.predict(self.X_train)
        y_te_pred = self.model.predict(self.X_test)
        return self.y_train, y_tr_pred, self.y_test, y_te_pred
