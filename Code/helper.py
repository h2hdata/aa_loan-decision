import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#functio to find fill missing values
def fill_missing(data,method = 0):
	"""
	Returns:
	--------
	returns data(pandas data frame) after imputing missing value 
	default method is fill 0
	"""
	if method == 'ffill' or method == 'pad':
		data = data.fillna(method = 'pad')

	if method == 'bfill':
		data = data.fillna(method = 'bfill')

	if method == 'mean':
		data = data.fillna(data.mean())

	if method == 0:
		data = data.fillna(0)
	return data
 

 #function to treat outlier
def treat_outlier(data):
	"""
	Returns:
	--------
	Pandas data frame after treating outlier
	1. for value very less replace with mean-3std
	2. for value very high replace with mean+3std
	"""
	mean_columns = map(lambda x: data[x].mean(),data)
	std_columns = map(lambda x: data[x].std(),data)

	for i,col in enumerate(list(data.columns)):
		data.loc[data[col]> mean_columns[i-1] + 3*std_columns[i-1],col] = mean_columns[i-1] + 3*std_columns[i-1]
		data.loc[data[col]< mean_columns[i-1] - 3*std_columns[i-1],col] = mean_columns[i-1] - 3*std_columns[i-1]
	return data


#function to remove imbalance using oversampling and undersampling technque
def oversampling(data):
	"""
	Returns:
	--------
	feature and target after balancing data wrt to target
	"""
	target = data['SeriousDlqin2yrs']
	features = data.drop(['SeriousDlqin2yrs'],axis = 1)
	# from imblearn.over_sampling import SMOTE
	# features,target = SMOTE().fit_sample(features,target)

	########### below is both ver and undersampling ###############
	from imblearn.combine import SMOTEENN
	features,target = SMOTEENN().fit_sample(features,target)
	return features,target




def _data(data=0,features=0,target =0):
	data = data.dropna()
	try:
		target = data['SeriousDlqin2yrs']
		features = data.drop(['SeriousDlqin2yrs'],axis = 1)
	except:pass
	from sklearn.decomposition import PCA
	pca = PCA(n_components = 2)
	f_r = pca.fit(features).transform(features)
	plt.figure()
	plt.scatter( f_r[target == 0,0], f_r[target == 0,1], alpha = 0.8, color='navy', label = 'SeriousDlqin2yrs' )
	plt.scatter( f_r[target == 1,0], f_r[target == 1,1],alpha = 0.8, color='turquoise', label = 'SeriousDlqin2yrs' )
	plt.draw()
	plt.pause(0.01)



def _scatter_plot(data,x, y):
	"""
	   Function to create a scatter plot of one column versus
	   another.

	   Returns:
	   --------
	   scatter plot between x and y.

	"""
	ax = data.plot(x=x, y=y, kind='scatter')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	# raw_input("Press enter to continue")

def _histogram_plot(data,x, y):
	"""
	   Function  to  create  a  histogram  plot of one column 
	   versus another.

	   Returns:
	   --------
	   histogram plot between x and y.

	"""
	ax = data.plot(x=x, y=y, kind='hist')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	# raw_input("Press enter to continue")

def _box_plot(data,x, y):
	"""
	   Function  to  create  a  box plot of one column versus 
	   another.

	   Returns:
	   --------
	   box plot between x and y.

	"""
	ax = data.plot(x=x, y=y, kind='box')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	# raw_input("Press enter to continue")

def _bar_chart(data,x):
	"""
	   Function  to  create  a bar chart of one column versus 
	   another.

	   Returns:
	   --------
	   bar chart.

	"""
	if x is not None:
		ax = data.groupby(x).count().plot(kind='bar')
		ax.set_xlabel(x)
		ax.set_title(x)
		plt.draw()
		plt.pause(0.01)
		# raw_input("Press enter to continue")
	else:
		ax = data.plot(kind='bar')
		plt.draw()
		plt.pause(0.01)
		# raw_input("Press enter to continue")




