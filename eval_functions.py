#%%
# Import packages
import pandas as pd
import numpy as np
import HydroErr as he

#%%
# categorical metrics
def binary_cat_metrics(df, test_label, pred_label,target_val,class_cat):
    '''
    Binary_cat_metrics fucntions is for computing categroical mertrics to evaluate classification models
    Can be used for evaluating multiclassification cases but this has to be applied for each class

    requires:
    df = dataframe containing observed and predicted labels

    test_label = columns name of the observed (test) data

    pred_label = columns name of the predictted/simulated (model preicted) data

    target value  = the value/class label in the observed to be evaluated 

    class_cat = the category/class been evaluated

    returns a dataframe of catagorical metrics Hits, Miss, False alarms, POD, FAR, POFD,
    ACC, CSI, HSS, ETS, HK

    since a multicalss is evaliuated this has to be run for each class
    '''

    df_cat = pd.DataFrame()
    
    a_hit = df.loc[(df[test_label] == target_val)  & (df[pred_label] == target_val)].count()[0]

    b_false = df.loc[(df[test_label] != target_val) & (df[pred_label] == target_val)].count()[0]

    c_miss = df.loc[(df[test_label] == target_val) & (df[pred_label] != target_val)].count()[0]

    # d_cor_neg = df.loc[(df[test_label] != target_val) & (df[pred_label] != target_val)].count()[0]
    # d_cor_neg = df.loc[((df[test_label] < target_val) & (df[pred_label] < target_val))].count()[0]

    # n = a_hit + b_false + c_miss + d_cor_neg # total number of calssifications

    #a_hit_rand_hss = ((a_hit + b_miss)*(a_hit + c_false) + (d_cor_false + b_miss)*(d_cor_false + c_false))/N # random hits for HSS

    #a_hit_rand_ets = (a_hit + b_miss)/N # random hits for ETS
    # a_ref = (a_hit + b_false)*(a_hit + c_miss) / n

    # TOB = a_hit + b_miss # total number of true classifications

    df_cat.loc[class_cat ,'Hits'] = a_hit #round((a_hit/TOB)*100,1) 

    df_cat.loc[class_cat ,'Miss'] = c_miss #round((c_miss/TOB)*100,1)

    df_cat.loc[class_cat,'False alarms'] = b_false #round((c_false/TOB)*100,1)

    df_cat.loc[class_cat ,'POD'] = round(a_hit/(a_hit + c_miss),3)

    df_cat.loc[class_cat,'FAR'] = round(b_false/(a_hit + b_false),3)

    # df_cat.loc[class_cat,'POFD'] = round(b_false/(d_cor_neg + b_false),3)

    # df_cat.loc[class_cat,'ACC'] = round((a_hit + d_cor_neg)/ n,2)

    df_cat.loc[class_cat,'Bias'] = round((a_hit + b_false)/(a_hit + c_miss),2)

    df_cat.loc[class_cat,'CSI'] = round(a_hit/(a_hit + b_false + c_miss),2)

    # df_cat.loc[class_cat,'HSS'] = round((2*((a_hit*d_cor_neg) - (b_false*c_miss)))/(((a_hit+c_miss)*(c_miss+d_cor_neg))+((a_hit+b_false)*(b_false+d_cor_neg))),2)
    #round(((a_hit + d_cor_false)-(a_hit_rand_hss))/(N - a_hit_rand_hss),3)

    # df_cat.loc[class_cat,'ETS'] = round((a_hit - a_ref)/(a_hit - a_ref + b_false + c_miss),2) 
    #round((a_hit - a_hit_rand_ets)/(a_hit + b_miss + c_false - a_hit_rand_ets),3)   

    return df_cat

#***********************************************************************************************************************************************************

# quantitative stats
# Note: always - model is simulated and obs is observed 
# bias ratio
def bias_ratio(obs,model):
    mu_obs = np.nanmean(obs)
    mu_model = np.nanmean(model)

    return mu_model/mu_obs

#***********************************************************************************************************************************************************

# mean normaised rmse (NRMSE)
# Compute the mean normalized root mean square error between the simulated and observed data.
# Range 0 NRMSE < inf, smaller is better.
# Notes: This metric is the RMSE normalized by the mean of the observed time series (x). Normalizing allows
# comparison between data sets with different scales.

def nrmsqe(obs,model):
    return he.nrmse_mean(model,obs)

#***********************************************************************************************************************************************************

# rmse
# Range 0 RMSE < inf, smaller is better.
# Notes: The standard deviation of the residuals. A lower spread indicates that the points are better concentrated
# around the line of best fit (linear). Random errors do not cancel. This metric will highlights larger errors.

def rmsqe(obs,model):
    return he.rmse(model,obs)

#***********************************************************************************************************************************************************
# pearson correlation

# Range: -1 R (Pearson) 1. 1 indicates perfect postive correlation, 0 indicates complete randomness, -1 indicate
# perfect negative correlation.
# Notes: The pearson r coefficient measures linear correlation. It is sensitive to outliers.

def p_corr(obs,model):
    return he.pearson_r(model,obs)


#***********************************************************************************************************************************************************

#Kling-Gupta efficiency (2012).

# Range: -inf < KGE (2009) < 1, larger is better.

# Notes: The modified version of the KGE (2009). Kling proposed this version to avoid cross-correlation between
# bias and variability ratios.
def kge2012(obs,model,how):
    if how == 'all':
        return he.kge_2012(model,obs,return_all=True)
    elif how == 'kge':
        return he.kge_2012(model,obs,return_all=False)

#***********************************************************************************************************************************************************

# relative bias
def relative_bias(obs,model):
    mu_residuals = np.nanmean(model - obs)
    mu_obs = np.nanmean(obs)

    return mu_residuals/mu_obs