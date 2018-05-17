_regression = False
_figs = False
if _figs:
    import matplotlib.pyplot as plt
    import seaborn as sns

from math import sqrt
import scipy.stats as stats
import numpy as np
if _regression:
    import statsmodels.api as sm

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


NPERIODS = 3

def metrics_deepar(y_true, y_pred):
    _fit = {}
    _fit['r2'] = r2_score(y_true, y_pred)
    _fit['rmse'] = sqrt(mean_squared_error(y_true, y_pred))
    _fit['explained_var'] = explained_variance_score(y_true, y_pred)
    _fit['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
    mod = sm.OLS(y_true, sm.add_constant(y_pred))
    mod = mod.fit()
    _fit['reg_params'] = list(mod.params)

    return _fit


def metrics(y_true, y_pred, _fit = {}):
    _fit['r2'] = r2_score(y_true, y_pred)
    _fit['rmse'] = sqrt(mean_squared_error(y_true, y_pred))
    _fit['explained_var'] = explained_variance_score(y_true, y_pred)
    _fit['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
    _fit['y_true'] = list(y_true)
    _fit['y_pred'] = list(y_pred)
    if _regression:
        mod = sm.OLS(y_true, sm.add_constant(y_pred))
        mod = mod.fit()
        _fit['reg_params'] = list(mod.params)
    else:
        _fit['reg_params'] = []

    x = list(range(NPERIODS+1))
    err = np.sqrt(_fit['mean_absolute_error'])
    y = _fit['forecasted_y']

    _fit['ci_low'] = [y[i]-x[i]*err for i in range(len(y))]
    _fit['ci_high'] = [y[i]+x[i]*err for i in range(len(y))]
    return _fit


def plot_fit_metrics(fit_metrics, path):

    # residual quantile distribution
    y_true = np.array(fit_metrics['y_true'])
    y_pred = np.array(fit_metrics['y_pred'])
    ax = plt.subplot(111)
    np.array(y_true)
    stats.probplot(y_true - y_pred, dist="norm", plot=plt)
    fig = ax.get_figure()
    plt.savefig(path + "/" + fit_metrics['model'] + "_residuals_quantile_against_normal.png", dpi=900)
    plt.close(fig)

    # residuals
    ax = plt.subplot(111)
    plt.ylabel("residual")
    # sns.residplot(y_true, y_pred)
    fig = ax.get_figure()
    plt.savefig(path + "/" + fit_metrics[
        'model'] + "_residuals.png", dpi = 300)
    plt.close(fig)

    # regplot
    ax = plt.subplot(111)
    sns.regplot(y_true, y_pred)
    fig = ax.get_figure()

    lims = [
        min(min(y_true), min(y_pred)),  # min of both axes
        max(max(y_true), max(y_pred))  # max of both axes
    ]

    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.plot(lims,lims, "--", color = 'black', linewidth = 0.5)

    plt.title(fit_metrics['model']+" model of {} with rho2 = {:.2f}, RMSE = {:.2f}.".format(
        fit_metrics['label'][:20], fit_metrics['r2'],fit_metrics['rmse']))
    plt.savefig(path + "/" + fit_metrics['model'] + "_regplot.png", dpi = 300)
    plt.close(fig)

    # temporal
    ax = plt.subplot(111)
    plt.plot(y_true, 'ro')
    plt.plot(y_pred, c="black", lw=2)

    fig = ax.get_figure()
    plt.savefig(path + "/" + fit_metrics['model'] + "_temporal_fit.png", dpi = 300)
    plt.close(fig)

    # forecast
    ax = plt.subplot(111)

    plt.plot(fit_metrics['actual_y'], 'ro', label = 'actual value')
    plt.plot(fit_metrics['forecasted_y'], c="black", lw=2, label = 'model forecasted')
    ax.fill_between(list(range(NPERIODS+1)), fit_metrics['ci_low'], fit_metrics['ci_high'], color = 'blue', alpha = 0.1)
    plt.plot(fit_metrics['ci_low'], c='blue', label= 'confidence interval')
    plt.plot(fit_metrics['ci_high'], c='blue')
    plt.legend()

    fig = ax.get_figure()
    plt.savefig(path + "/" + fit_metrics['model'] + "_forecast.png", dpi=900)
    plt.close(fig)




