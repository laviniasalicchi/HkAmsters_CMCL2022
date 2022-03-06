import sys, textstat, argparse, os.path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
import numpy as np
from random import shuffle
import lightgbm as lgb

from sklearn.preprocessing import PolynomialFeatures

def train(X_train, Y_train, mapper, use_interaction):
    # feature fusion
    print('before interaction dim:', len(X_train[0]))
    if use_interaction:
        pl = PolynomialFeatures(interaction_only=True)
        X_train = pl.fit_transform(X_train)
    print('after interaction dim:', len(X_train[0]))

    if mapper == "plsr":
        model = PLSRegression(n_components=5, max_iter=2000).fit(X_train, Y_train)

    if mapper == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(5,5), activation='identity', solver='adam', early_stopping=True, max_iter=1000).fit(X_train, Y_train)

    if mapper == "rf":
        model = RandomForestRegressor(n_estimators=10).fit(X_train, Y_train)

    if mapper == "lr":
        model = LinearRegression().fit(X_train, Y_train)

    if mapper == "rr":
        model = Ridge(max_iter=2000).fit(X_train, Y_train)

    if mapper == 'svr':
        model = SVR(max_iter=2000).fit(X_train, Y_train)

    if mapper == "brr":
        model = BayesianRidge().fit(X_train, Y_train)

    if mapper == "elast":
        model = ElasticNet(max_iter=2000).fit(X_train, Y_train)

    if mapper == 'lgb':
        model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20,
                             verbosity=0).fit(X_train, Y_train, verbose=False)

    return model

def predict(X_test, model, use_interaction):
    print('before interaction dim:', len(X_test[0]))
    if use_interaction:
        pl = PolynomialFeatures(interaction_only=True)
        X_test = pl.fit_transform(X_test)
    print('after interaction dim:', len(X_test[0]))

    # evaluation
    Y_pred = model.predict(X_test)

    Y_pred = Y_pred.squeeze()

    return Y_pred

def test(X_test, model, use_interaction):
    print('before interaction dim:', len(X_test[0]))
    if use_interaction:
        pl = PolynomialFeatures(interaction_only=True)
        X_test = pl.fit_transform(X_test)
    print('after interaction dim:', len(X_test[0]))

    # evaluation
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred.squeeze().tolist()

    fpout = open('../output/pred.txt', 'w', encoding='utf-8')
    for item in Y_pred:
        fpout.writelines(str(item) + '\n')
