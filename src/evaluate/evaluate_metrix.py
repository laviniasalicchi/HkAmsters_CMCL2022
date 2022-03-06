from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy.stats import spearmanr
from scipy.stats import pearsonr

import logging

def evaluate(Y_pred, Y_true):
    logging.info ("The predictions of the model are ready for evaluation.")

    logging.info ("MAE: " + str(mean_absolute_error(Y_true, Y_pred)))
    logging.info ("MSE: " + str(mean_squared_error(Y_true, Y_pred)))
    logging.info ("R2: "+ str(r2_score(Y_true, Y_pred)))
    logging.info ("Pearson: " + str(pearsonr(Y_true, Y_pred)[0]))
    logging.info ("Spearman: " + str(spearmanr(Y_true, Y_pred).correlation))