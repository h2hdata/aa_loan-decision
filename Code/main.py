"""
# -*- coding: utf-8 -*-

#POC Code  :  PROBLEM CREIDT DEFAULTER :  H2H DATA
This code is for identifying propablem defaulter.
Data: Kaggel lone deafulter data is used
Models: many models are used to get comparesion between them.


#Copyright@ H2H DATA

#The entire prcess occurs in seven stages-
# 1. DATA INGESTION
# 2. DATA ANALYSIS 
# 3. DATA MUNGING
# 4. DATA EXPLORATION
# 5. DATA MODELING
# 6. HYPER-PARAMETERS OPTIMIZATION
# 7. PREDICTION
# 8. VISUAL ANALYSIS
# 9. RESULTS


Used library
1. pandas
2. numpy
3. time
4. sklearn
5. matplotlib
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import helper
import model



########################################## data ingestion ###############################
"""
Data Description
	target:SeriousDlqin2yrs 
	value: Y/N

	features:
	Name:RevolvingUtilizationOfUnsecuredLines		value: percentage
	Name:age										value: integer
	Name:NumberOfTime30-59DaysPastDueNotWorse		value: integer
	Name:DebtRatio									value: percentage
	Name:MonthlyIncome								value: real
	Name:NumberOfOpenCreditLinesAndLoans			value: integer
	Name:NumberOfTimes90DaysLate					value: integer
	Name:NumberRealEstateLoansOrLines				value: integer
	Name:NumberOfTime60-89DaysPastDueNotWorse		value: integer
	Name:NumberOfDependents							value: integer
"""
def read_data():
	'''
	return pandas data frame
	'''
	data = pd.read_csv('../Data/cs1-training.csv')
	return data
####################################### data ingestion  ends ###############################



####################################### data analysis ###################################
'''
  DATA ANALYSIS
-----------------

 Here,  data is  analysed and visualised to look for patterns 
 and   anamolies   using   graphs.  bar  graphs,   histograms,  
 scatter plots, etc. will be used

'''
def data_plot(data):
	"""
	   This function analyzes the data and looks for patterns
	   to help understand the data more clearly.

	   Returns:
	   --------
	   Charts, graphs for various Features.

	"""
	helper._data(data = data)
	helper._scatter_plot(data,'SeriousDlqin2yrs','MonthlyIncome')
	helper._histogram_plot(data,'SeriousDlqin2yrs','age')
	helper._histogram_plot(data,'SeriousDlqin2yrs','NumberOfTime30-59DaysPastDueNotWorse')
	helper._histogram_plot(data,'SeriousDlqin2yrs','NumberOfOpenCreditLinesAndLoans')
	helper._histogram_plot(data,'SeriousDlqin2yrs','NumberOfTimes90DaysLate')
	helper._histogram_plot(data,'SeriousDlqin2yrs','NumberRealEstateLoansOrLines')
	helper._histogram_plot(data,'SeriousDlqin2yrs','NumberOfTime60-89DaysPastDueNotWorse')
	helper._histogram_plot(data,'SeriousDlqin2yrs','NumberOfDependents')
	helper._box_plot(data,'SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines')
	helper._box_plot(data,'SeriousDlqin2yrs','DebtRatio')
	helper._box_plot(data,'SeriousDlqin2yrs','MonthlyIncome')
	helper._bar_chart(data,'SeriousDlqin2yrs')
	helper._bar_chart(data,'age')
	helper._bar_chart(data,'NumberOfTime30-59DaysPastDueNotWorse')
	helper._bar_chart(data,'NumberOfOpenCreditLinesAndLoans')
	helper._bar_chart(data,'NumberOfTimes90DaysLate')
	helper._bar_chart(data,'NumberRealEstateLoansOrLines')
	helper._bar_chart(data,'NumberOfTime60-89DaysPastDueNotWorse')
	helper._bar_chart(data,'NumberOfDependents')
####################################### data analysis ###################################



####################################### data exploration ###################################
def prepocessing(data):
	"""
	returns feature and target after doing following stpes:
	1. misssing value imputation
	2. outlier treatment
	3. balancing data
	"""
	ID = data['ID']
	data = data.drop(['ID'],axis =1)
	data = helper.fill_missing(data,0)
	data = helper.treat_outlier(data)
	features,target = helper.oversampling(data)
	return features,target	
########################################## data exploration ends #############################


########################################## modeling ##########################################

def training(features,target):
	"""
	fit to model also apply hyperparameter optimization
	"""
	print 'reading test data'
	test_data = pd.read_csv('../Data/cs1-test.csv')
	test_data = test_data.drop(['SeriousDlqin2yrs','ID'],axis =1)
	test_data = test_data.dropna()

	print 'fit algo'
	model.fit_algo(features,target,test_data = test_data)
	model.fit_algo_random(features,target)
########################################## modeling ends ######################################

def main():
	"""
	main code to call all functions
	"""
	data = read_data()
	print 'reading data cmpleted'
	data_plot(data)
	features,target = prepocessing(data)
	print 'preprocessing done and fitting model'
	training(features,target)
	print 'code completed'



if __name__ == "__main__":

	main()


