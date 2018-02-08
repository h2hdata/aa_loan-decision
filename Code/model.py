"""
Fitting model 
"""



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from scipy.stats import expon
import pandas as pd
import helper

global param_dict 


#fitting to algos
def fit_algo(features,target,test_data=None):
    '''
    comparing different algos with different evaluation
    and predicting test data 
    '''

    l = len(features)
    tr = features[0:int(l*.7)]
    ts = features[int(l*.7):]
    ttr = target[0:int(l*.7)]
    tts = target[int(l*.7):]

    clf_logistic = LogisticRegression(warm_start = True)
    clf_svm = SVC()
    clf_mlp = MLPClassifier(hidden_layer_sizes = (10,4) ,warm_start = True)
    clf_GB = GradientBoostingClassifier(n_estimators=100,warm_start = True)

    print  cross_val_score(clf_logistic,features,target),'\tlogistic'
    print  cross_val_score(clf_svm,features,target),'\tsvm'
    print  cross_val_score(clf_mlp,features,target), '\tmlp'
    print  cross_val_score(clf_GB,features,target),'\tGBM'
    print '\t \t \t f1 score \t \t \t'
    print  cross_val_score(clf_logistic,features,target,scoring = 'f1'),'\t logistc'
    print  cross_val_score(clf_svm,features,target,scoring = 'f1'),'\tsvm'
    print  cross_val_score(clf_mlp,features,target,scoring = 'f1'),'\tmlp'
    print  cross_val_score(clf_GB,features,target,scoring = 'f1') ,'\tGBM'
    print '\t \t \t roc_auc \t \t \t'
    print  cross_val_score(clf_logistic,features,target,scoring = 'roc_auc'),'\t logistc'
    print  cross_val_score(clf_svm,features,target,scoring = 'roc_auc'),'\tsvm'
    print  cross_val_score(clf_mlp,features,target,scoring = 'roc_auc'),'\tmlp'
    print  cross_val_score(clf_GB,features,target,scoring = 'roc_auc') ,'\tGBM'

    clf_logistic = clf_logistic.fit(features,target)
    clf_svm = clf_svm.fit(features,target)
    clf_mlp = clf_mlp.fit(features,target)
    clf_GB = clf_GB.fit(features,target)

    try:
        feature_test = test_data
        predict('GBM',clf_GB,feature_test)
    except:pass



    
#functio to fit algo with random search optimization
def fit_algo_random(features,target,test=None):
    '''
    comparing different algos with different evaluation and doing hyperparameter optimization
    '''
    global param_dict
    param_dict = {}
    clf_logistic = LogisticRegression()
    clf_svm = SVC()
    clf_mlp = MLPClassifier()
    clf_GB = GradientBoostingClassifier()

    ################  Accuracy ########################
    random = RandomizedSearchCV(clf_logistic,{'penalty':['l1','l2'],'C': expon(scale=100), 'fit_intercept':[True,False]}, cv = 5)
    random.fit(features,target)
    print '\n logistic\t',random.best_score_
    print 'logistic\t',random.best_params_
    param_dict['logistic'] = random.best_params_

    random = RandomizedSearchCV(clf_GB,{'max_depth':[3,5,8,10], 'learning_rate':expon(scale=1), 'max_features': ['auto',None] }, cv = 5)
    random.fit(features,target)
    print '\n GBM\t', random.best_score_
    print 'GBM\t',random.best_params_
    param_dict['gbm'] = random.best_params_


    print '\t \t \t f1 score \t \t \t'
    ###################3  f1 ######################
    random = RandomizedSearchCV(clf_logistic,{'penalty':['l1','l2'],'C': expon(scale=100), 'fit_intercept':[True,False]},scoring = 'f1', cv = 5)
    random.fit(features,target)
    print '\n logistic\t',random.best_score_
    print 'logistic\t',random.best_params_

    random = RandomizedSearchCV(clf_GB,{'max_depth':[3,5,8,10], 'learning_rate':expon(scale=1), 'max_features': ['auto',None] },scoring = 'f1', cv = 5)
    random.fit(features,target)
    print '\n GBM\t', random.best_score_
    print 'GBM\t',random.best_params_


def predict(name,clf,feature_test):
    test = feature_test
    df = pd.DataFrame(clf.predict(test),columns=['SeriousDlqin2yrs'])
    df[list(feature_test.columns)] =test 
    df.to_csv('../Output/'+name+'.csv',index=False)