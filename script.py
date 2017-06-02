import warnings
warnings.filterwarnings("ignore")

import math

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

regressor = {'boosting': GradientBoostingRegressor(),
             'rf': RandomForestRegressor(),
             'lasso': Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False,
                            max_iter=1000, tol=0.0001, selection='cyclic'),
             'linear': LinearRegression(),
             'elastic': ElasticNet(),
             'xgboosting': XGBRegressor()}

parameters = {'boosting': {'loss': ['huber'],
                           'learning_rate': [0.01, 0.05, 0.1],
                           'n_estimators': [100, 1000, 3000, 5000],
                           'max_depth': [2,3,6,9],
                           'min_samples_split': [5, 10, 15],
                           'min_samples_leaf': [10, 15, 20],
                           'max_features': ['sqrt', 'log2']},
              'rf': {'n_estimators': [10, 100, 1000],
                     'max_features': ['sqrt', 'log2'],
                     'max_depth': [None, 10, 20, 40, 80],
                     'min_samples_split': [2, 8, 16],
                     'min_samples_leaf': [1, 5, 10]},
              'lasso': {},
              'linear': {},
              'elastic': {},
              'xgboosting': {'max_depth': [2,3,4,5,6],
                             'learning_rate': [0.01, 0.05, 0.1],
                             'n_estimators': [1000, 3000, 5000, 7000],
                             'colsample_bytree': [0.2],
                             'gamma': [0.0],
                             'min_child_weight': [1.5],
                             'reg_alpha': [0.9],
                             'reg_lambda': [0.6],
                             'subsample': [0.2]}}

class Dataset():

    def __init__(self, filename):
        self.filename = filename
        self._import()

    def _import(self):
        """ Import training data """
        self.data = pd.read_csv(self.filename)

    def delete_rows(self):
        """ Eliminate rows with too many missing values """
        self.data = self.data

    def substitute_missing_values(self):
        """ Substitute missing values with appropriate replacements """
        self.data['LotFrontage'].fillna(0, inplace=True)
        self.data['Alley'].fillna('NoAlley', inplace=True)
        self.data['MasVnrType'].fillna('None', inplace=True)
        self.data['MasVnrArea'].fillna(0, inplace=True)
        self.data['BsmtQual'].fillna('None', inplace=True)
        self.data['BsmtCond'].fillna('None', inplace=True)
        self.data['BsmtExposure'].fillna('None', inplace=True)
        self.data['BsmtFinType1'].fillna('None', inplace=True)
        self.data['BsmtFinType2'].fillna('None', inplace=True)
        self.data['Electrical'].fillna('Unkown', inplace=True)
        self.data['FireplaceQu'].fillna('None', inplace=True)
        self.data['GarageType'].fillna('None', inplace=True)
        self.data['GarageYrBlt'].fillna(np.mean(self.data['GarageYrBlt']), inplace=True)
        self.data['GarageFinish'].fillna('None', inplace=True)
        self.data['GarageQual'].fillna('None', inplace=True)
        self.data['GarageCond'].fillna('None', inplace=True)
        self.data['PoolQC'].fillna('None', inplace=True)
        self.data['Fence'].fillna('None', inplace=True)
        self.data['GarageArea'].fillna(0, inplace=True)

        """ Missing values in submission_dataset """
        self.data['Exterior1st'].fillna('None', inplace=True)
        self.data['MSZoning'].fillna('None', inplace=True)
        self.data['KitchenQual'].fillna('None', inplace=True)
        self.data['SaleType'].fillna('None', inplace=True)
        self.data['Exterior2nd'].fillna('None', inplace=True)
        self.data['BsmtFinSF1'].fillna(0, inplace=True)
        self.data['BsmtFinSF2'].fillna(0, inplace=True)
        self.data['BsmtUnfSF'].fillna(0, inplace=True)
        self.data['TotalBsmtSF'].fillna(0, inplace=True)
        self.data['BsmtFullBath'].fillna(0, inplace=True)
        self.data['BsmtHalfBath'].fillna(0, inplace=True)
        self.data['GarageCars'].fillna(0, inplace=True)

    def extract_new_features(self):
        """ Derive new features, based on original ones """
        self.data['LogLotArea'] = np.log(self.data['LotArea'] + 1)
        self.data['HasMasonry'] = self.data['MasVnrArea'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasType2'] = self.data['BsmtFinSF2'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasUnfinishedBasement'] = self.data['BsmtUnfSF'].apply(lambda x: 0 if x == 0 else 1)
        self.data['LogTotalBsmtSF'] = np.log(self.data['TotalBsmtSF'] + 1)
        self.data['Log1stFlrSF'] = np.log(self.data['1stFlrSF'] + 1)
        self.data['Log2ndFlrSF'] = np.log(self.data['2ndFlrSF'] + 1)
        self.data['HasSecondFloor'] = self.data['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)
        self.data['DuplexArea'] = self.data['1stFlrSF'] + self.data['2ndFlrSF']
        self.data['LogDuplexArea'] = np.log(self.data['DuplexArea'] + 1)
        self.data['TriplexArea'] = self.data['DuplexArea'] + self.data['TotalBsmtSF']
        self.data['LogTriplexArea'] = np.log(self.data['TriplexArea'] + 1)
        self.data['FourplexArea'] = self.data['TriplexArea'] + self.data['GarageArea']
        self.data['LogFourplexArea'] = np.log(self.data['FourplexArea'] + 1)
        self.data['TotalArea'] = self.data['FourplexArea'] + self.data['WoodDeckSF'] + self.data['OpenPorchSF'] + self.data['EnclosedPorch'] + self.data['3SsnPorch'] + self.data['ScreenPorch']
        self.data['LogTotalArea'] = np.log(self.data['TotalArea'] + 1)
        self.data['TotalLivingArea'] = self.data['TotalArea'] - self.data['GarageArea']
        self.data['LogTotalLivingArea'] = np.log(self.data['TotalLivingArea'] + 1)
        self.data['HasLowQualityFinishing'] = self.data['LowQualFinSF'].apply(lambda x: 0 if x == 0 else 1)
        self.data['LogGrLivArea'] = np.log(self.data['GrLivArea'] + 1)
        self.data['HasHalfBathAboveGrade'] = self.data['HalfBath'].apply(lambda x: 0 if x == 0 else 1)
        self.data['TotalBathAboveGrade'] = self.data['FullBath'] + self.data['HalfBath']
        self.data['HasBathAboveGrade'] = self.data['TotalBathAboveGrade'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasFireplace'] = self.data['Fireplaces'].apply(lambda x: 0 if x == 0 else 1)
        self.data['LogGarageArea'] = np.log(self.data['GarageArea'] + 1)
        self.data['HasGarage'] = self.data['GarageArea'].apply(lambda x: 0 if x == 0 else 1)
        self.data['StillAvailableArea'] = self.data['LotArea'] - self.data['1stFlrSF'] - self.data['GarageArea']
        self.data['HasAvailableArea'] = self.data['StillAvailableArea'].apply(lambda x: 0 if x < 10000 else 1)
        self.data['HasWoodDeck'] = self.data['WoodDeckSF'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasOpenPorch'] = self.data['OpenPorchSF'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasEnclosedPorch'] = self.data['EnclosedPorch'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasThreeSeasonPorch'] = self.data['3SsnPorch'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasScreenPorch'] = self.data['ScreenPorch'].apply(lambda x: 0 if x == 0 else 1)
        self.data['TotalPorchArea'] = self.data['OpenPorchSF'] + self.data['EnclosedPorch'] + self.data['3SsnPorch'] + self.data['ScreenPorch']
        self.data['HasPorch'] = self.data['TotalPorchArea'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HasPool'] = self.data['PoolArea'].apply(lambda x: 0 if x == 0 else 1)
        self.data['HouseAge'] = self.data['YrSold'] - self.data['YearBuilt']
        self.data['NewHouse'] = self.data['HouseAge'].apply(lambda x: 1 if x == 0 else 0)
        self.data['HouseRemodelAge'] = self.data['YrSold'] - self.data['YearRemodAdd']
        self.data['RecentlyRemodeled'] = self.data['HouseRemodelAge'].apply(lambda x: 1 if x == 0 else 0)
        self.data['GarageAge'] = self.data['YrSold'] - self.data['GarageYrBlt']
        self.data['NewGarage'] = self.data['GarageAge'].apply(lambda x: 1 if x == 0 else 0)

    def delete_features(self):
        """ Delete features found to be useless during EDA """
        self.data = self.data.drop('Utilities', axis=1)
        self.data = self.data.drop('Condition2', axis=1)
        self.data = self.data.drop('RoofMatl', axis=1)
        self.data = self.data.drop('LowQualFinSF', axis=1)
        self.data = self.data.drop('MiscFeature', axis=1)
        self.data = self.data.drop('GarageCond', axis=1)

        """ I'll delete this feature for now. It's messing up the one-hot encoding. """
        self.data = self.data.drop('PoolQC', axis=1)

    def apply_label_encoding_to(self, column_name, label_encoding):
        self.data[column_name] = self.data[column_name].apply(lambda x: label_encoding[column_name][x])

    def apply_one_hot_encoding_to(self, column_name):
        self.data = pd.concat([self.data, pd.get_dummies(self.data[column_name]).add_prefix(column_name+' = ')], axis=1)
        self.data = self.data.drop(column_name, axis=1)

    def transform_features(self):
        """ Turn categorical features into numerical ones """

        """ Label encoding """
        label_encoding = {'MSSubClass': {150: 0,
                                         180: 1,
                                         30: 2,
                                         45: 3,
                                         190: 4,
                                         50: 5,
                                         90: 6,
                                         85: 7,
                                         40: 8,
                                         160: 9,
                                         70: 10,
                                         20: 11,
                                         75: 12,
                                         80: 13,
                                         120: 14,
                                         60: 15},
                          'MSZoning': {'None': 0,
                                       'C (all)': 1,
                                       'RM': 2,
                                       'RH': 3,
                                       'RL': 4,
                                       'FV': 5},
                          'Street': {'Grvl': 0,
                                     'Pave': 1},
                          'LandContour': {'Bnk': 1,
                                          'Lvl': 2,
                                          'Low': 3,
                                          'HLS': 4},
                          'Neighborhood': {'MeadowV': 1,
                                           'IDOTRR': 2,
                                           'BrDale': 3,
                                           'OldTown': 4,
                                           'Edwards': 5,
                                           'BrkSide': 6,
                                           'Sawyer': 7,
                                           'Blueste': 8,
                                           'SWISU': 9,
                                           'NAmes': 10,
                                           'NPkVill': 11,
                                           'Mitchel': 12,
                                           'SawyerW': 13,
                                           'Gilbert': 14,
                                           'NWAmes': 15,
                                           'Blmngtn': 16,
                                           'CollgCr': 17,
                                           'ClearCr': 18,
                                           'Crawfor': 19,
                                           'Veenker': 20,
                                           'Somerst': 21,
                                           'Timber': 22,
                                           'StoneBr': 23,
                                           'NoRidge': 24,
                                           'NridgHt': 25},
                         'Condition1': {'Artery': 1,
                                        'Feedr': 2,
                                        'RRAe': 3,
                                        'Norm': 4,
                                        'RRAn': 5,
                                        'RRNe': 6,
                                        'PosN': 7,
                                        'PosA': 8,
                                        'RRNn': 9},
                         'HouseStyle': {'1.5Unf': 1,
                                        '1.5Fin': 2,
                                        '2.5Unf': 3,
                                        'SFoyer': 4,
                                        '1Story': 5,
                                        'SLvl': 6,
                                        '2Story': 7,
                                        '2.5Fin': 8},
                         'RoofStyle': {'Gambrel': 1,
                                       'Gable': 2,
                                       'Mansard': 3,
                                       'Hip': 4,
                                       'Flat': 5,
                                       'Shed': 6},
                         'Exterior1st': {'None': 0,
                                         'BrkComm': 1,
                                         'AsphShn': 2,
                                         'CBlock': 3,
                                         'AsbShng': 4,
                                         'WdShing': 5,
                                         'Wd Sdng': 6,
                                         'MetalSd': 7,
                                         'Stucco': 8,
                                         'HdBoard': 9,
                                         'BrkFace': 10,
                                         'Plywood': 11,
                                         'VinylSd': 12,
                                         'CemntBd': 13,
                                         'Stone': 14,
                                         'ImStucc': 15},
                         'Exterior2nd': {'None': 0,
                                         'CBlock': 1,
                                         'AsbShng': 2,
                                         'Wd Sdng': 3,
                                         'Wd Shng': 4,
                                         'MetalSd': 5,
                                         'AsphShn': 6,
                                         'Stucco': 7,
                                         'Brk Cmn': 8,
                                         'HdBoard': 9,
                                         'BrkFace': 10,
                                         'Plywood': 11,
                                         'Stone': 12,
                                         'ImStucc': 13,
                                         'VinylSd': 14,
                                         'CmentBd': 15,
                                         'Other': 16},
                         'MasVnrType': {'BrkCmn': 1,
                                        'None': 2,
                                        'BrkFace': 3,
                                        'Stone': 4},
                         'ExterQual': {'Fa': 1,
                                       'TA': 2,
                                       'Gd': 3,
                                       'Ex': 4},
                         'ExterCond': {'Po': 1,
                                       'Fa': 2,
                                       'Gd': 3,
                                       'Ex': 4,
                                       'TA': 5},
                         'BsmtQual': {'None': 0,
                                      'Fa': 1,
                                      'TA': 2,
                                      'Gd': 3,
                                      'Ex': 4},
                         'BsmtCond': {'None': 0,
                                      'Po': 1,
                                      'Fa': 2,
                                      'TA': 3,
                                      'Gd': 4},
                         'BsmtExposure': {'None': 0,
                                          'No': 1,
                                          'Mn': 2,
                                          'Av': 3,
                                          'Gd': 4},
                         'Heating': {'Floor': 1,
                                     'Grav': 2,
                                     'Wall': 3,
                                     'OthW': 4,
                                     'GasW': 5,
                                     'GasA': 6},
                         'HeatingQC': {'Po': 1,
                                       'Fa': 2,
                                       'TA': 3,
                                       'Gd': 4,
                                       'Ex': 5},
                         'Electrical': {'Unkown': 1,
                                        'Mix': 2,
                                        'FuseP': 3,
                                        'FuseF': 4,
                                        'FuseA': 5,
                                        'SBrkr': 6},
                         'KitchenQual': {'None': 0,
                                         'Fa': 1,
                                         'TA': 2,
                                         'Gd': 3,
                                         'Ex': 4},
                         'FireplaceQu': {'None': 0,
                                         'Po': 1,
                                         'Fa': 2,
                                         'TA': 3,
                                         'Gd': 4,
                                         'Ex': 5},
                         'GarageType': {'None': 0,
                                        'CarPort': 1,
                                        'Detchd': 2,
                                        'Basment': 3,
                                        '2Types': 4,
                                        'Attchd': 5,
                                        'BuiltIn': 6},
                         'GarageFinish': {'None': 0,
                                          'Unf': 1,
                                          'RFn': 2,
                                          'Fin': 3},
                         'GarageQual': {'None': 0,
                                        'Po': 1,
                                        'Fa': 2,
                                        'TA': 3,
                                        'Gd': 4,
                                        'Ex': 5},
                         'PavedDrive': {'N': 1,
                                        'P': 2,
                                        'Y': 3},
                         'SaleType': {'None': 0,
                                      'Oth': 1,
                                      'ConLI': 2,
                                      'COD': 3,
                                      'ConLD': 4,
                                      'ConLw': 5,
                                      'WD': 6,
                                      'CWD': 7,
                                      'New': 8,
                                      'Con': 9},
                         'SaleCondition': {'AdjLand': 1,
                                           'Abnorml': 2,
                                           'Family': 3,
                                           'Alloca': 4,
                                           'Normal': 5,
                                           'Partial': 6}}

        for column_name in label_encoding:
            self.apply_label_encoding_to(column_name, label_encoding)

        """ One-hot encoding """
        for column_name in ['Alley',
                            'LotShape',
                            'LotConfig',
                            'LandSlope',
                            'BldgType',
                            'Foundation',
                            'BsmtFinType1',
                            'BsmtFinType2',
                            'CentralAir',
                            'Functional',
                            'Fence']:
            self.apply_one_hot_encoding_to(column_name)

    def preprocess_output(self):
        """ Take the log of SalePrice, since ML algorithms usually behave better with normally distributed data """
        self.data['SalePrice'] = np.log(self.data['SalePrice'])

    def revert_output_preprocessing(self, output):
        """ Revert output preprocessing, since we are predicting prices, not log of prices """
        return np.exp(output)


class TrainingDataset(Dataset):

    def __init__(self, filename):
        Dataset.__init__(self, filename)
        self.prepare()

    def split(self, test_size = 0.20):
        """ Randomly split training data into two sets """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=test_size)

    def prepare(self):
        """ Sequence of steps to import and prepare dataset properly """
        self.delete_rows()
        self.substitute_missing_values()
        self.extract_new_features()
        self.delete_features()
        self.transform_features()
        self.preprocess_output()

        """ Make sure data is clean """
        self.data = self.data.dropna(axis=1)
        self.data = self.data.select_dtypes([np.number])

        self.X = self.data.ix[:, self.data.columns != 'SalePrice']
        self.y = self.data.ix[:, self.data.columns == 'SalePrice']


class SubmissionDataset(Dataset):

    def __init__(self, filename):
        Dataset.__init__(self, filename)
        self.prepare()

    def prepare(self):
        """ Sequence of steps to import and prepare dataset properly """
        self.delete_rows()
        self.substitute_missing_values()
        self.extract_new_features()
        self.delete_features()
        self.transform_features()

        self.X = self.data.ix[:, self.data.columns != 'SalePrice']


class Problem():

    def __init__(self, model, training_dataset, submission_dataset):
        self.model = model
        self.training_dataset = training_dataset
        self.submission_dataset = submission_dataset

    def solve(self, X, y):
        """ Train algorithm based on training data """
        self.model.fit(X,y)

    def predict(self, X):
        """ Predict output based on provided input """
        prediction = self.model.predict(X)
        return prediction

    def evaluate(self):
        """ Evaluate quality of prediction, based on untrained dataset """
        prediction = self.predict(self.training_dataset.X_test)
        error = np.sqrt(mean_squared_error(prediction, self.training_dataset.y_test))
        return error

    def preprocess_features(self):
        """ Preprocess numerical features """
        next
        """
        self.numeric_features = self.training_dataset.X_train.dtypes[self.training_dataset.X_train.dtypes == "float"].index
        self.scaler = StandardScaler()
        self.scaler.fit(self.training_dataset.X_train[self.numeric_features])

        scaled = self.scaler.transform(self.training_dataset.X_train[self.numeric_features])
        for i, col in enumerate(self.numeric_features):
            self.training_dataset.X_train[col] = scaled[:, i]

        scaled = self.scaler.transform(self.training_dataset.X_test[self.numeric_features])
        for i, col in enumerate(self.numeric_features):
            self.training_dataset.X_test[col] = scaled[:, i]
        """

    def select_features(self):
        """ Remove features to reduce probability of overfitting """
        self.selector = SelectKBest(k=150)
        self.selector.fit(self.training_dataset.X_train, self.training_dataset.y_train.values.ravel())

        self.training_dataset.X_train = self.selector.transform(self.training_dataset.X_train)
        self.training_dataset.X_test = self.selector.transform(self.training_dataset.X_test)

    def score(self, number_of_iterations=10):
        """ Perform Monte Carlo Cross-Validation """
        scores = []
        for iteration in range(number_of_iterations):
            self.training_dataset.split()
            self.preprocess_features()
            self.select_features()
            self.solve(X=self.training_dataset.X_train, y=self.training_dataset.y_train.values.ravel())
            score = self.evaluate()
            #if not np.isinf(score):
            scores.append(self.evaluate())
        return np.mean(np.array(scores))

    def submit(self):
        """ Run model on submission_dataset and export to .csv """
        """
        scaled = self.scaler.transform(self.training_dataset.X[self.numeric_features])
        for i, col in enumerate(self.numeric_features):
            self.training_dataset.X[col] = scaled[:, i]

        scaled = self.scaler.transform(self.submission_dataset.X[self.numeric_features])
        for i, col in enumerate(self.numeric_features):
            self.submission_dataset.X[col] = scaled[:, i]
        """
        self.training_dataset.X = self.selector.transform(self.training_dataset.X)
        self.submission_dataset.X = self.selector.transform(self.submission_dataset.X)

        self.solve(X=self.training_dataset.X, y=self.training_dataset.y.values.ravel())

        prediction = self.predict(self.submission_dataset.X)
        prediction = self.submission_dataset.revert_output_preprocessing(prediction)

        submission = self.submission_dataset.data
        submission['SalePrice'] = prediction
        submission.to_csv('submission.csv', index=False, columns=['Id','SalePrice'])


class Experiment():

    def __init__(self, algorithm, train_file, test_file):
        self.training_dataset = TrainingDataset(train_file)
        self.submission_dataset = SubmissionDataset(test_file)
        self.algorithm = algorithm

        if parameters[self.algorithm]:
            self.regressor = RandomizedSearchCV(regressor[self.algorithm], parameters[self.algorithm],
                                           scoring='neg_mean_squared_error', n_iter=10)
        else:
            self.regressor = regressor[self.algorithm]

        self.problem = Problem(self.regressor, self.training_dataset, self.submission_dataset)

    def run(self):
        self.score = self.problem.score()
        print(self.score)

    def submit(self):
        self.problem.submit()


if __name__ == "__main__":
    np.random.seed(12345)
    experiment = Experiment('xgboosting', 'train.csv', 'test.csv')
    experiment.run()
    experiment.submit()
