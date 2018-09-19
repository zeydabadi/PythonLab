# performing logisticregression on heart disee
# import required modules
#%matplotlib inline # sets the backend of matplotlib to the 'inline' backend
import pandas as pd # Python Data Analysis Library
import numpy as np  # NumPy is the fundamental package for scientific computing with Python
import matplotlib.pyplot as plt # Matplotlib is a Python 2D plotting library
from collections import Counter 
from pprint import pprint 


#Reading data
#Setting the column names
columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
#Reading the data fron CSV file using pandas and creating dataframe.
df0 = pd.read_table("processed.cleveland.data.csv", sep=',', header=None, names=columns)

#df0 = df0.iloc[0:100] #Added by Mahmoud Zeydabadinezhad

#To find mean,count,etc.
df0.describe()

#To displys first few rows of dataset
df0.head()


#To find no of rows and columns in the dataset
df0.shape


#Checking feature information
df0.info()


 

# calculates the intrinsic discrepancy (a symmetrized Kullback-Leibler distance) between them.

def intrinsic_discrepancy(x,y):

    assert len(x)==len(y)

    sumx = sum(xval for xval in x)

    sumy = sum(yval for yval in y)

    id1  = 0.0

    id2  = 0.0

    for (xval,yval) in zip(x,y):

        if (xval>0) and (yval>0):

            id1 += (float(xval)/sumx) * np.log((float(xval)/sumx)/(float(yval)/sumy))

            id2 += (float(yval)/sumy) * np.log((float(yval)/sumy)/(float(xval)/sumx))

    return min(id1,id2)




# Compute intrinsic discrepancies between disease and no-disease feature distributions

int_discr = {}

hist,bin_edges   = np.histogram(df0.age,density=False)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].age,bins=bin_edges,density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].age,bins=bin_edges,density=False)

int_discr["age"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].sex,bins=(-0.5,0.5,1.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].sex,bins=(-0.5,0.5,1.5),density=False)

int_discr["sex"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].cp,bins=(0.5,1.5,2.5,3.5,4.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].cp,bins=(0.5,1.5,2.5,3.5,4.5),density=False)

int_discr["cp"] = intrinsic_discrepancy(hist1,hist2)

hist,bin_edges   = np.histogram(df0.restbp,density=False)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].restbp,bins=bin_edges,density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].restbp,bins=bin_edges,density=False)

int_discr["restbp"] = intrinsic_discrepancy(hist1,hist2)

hist,bin_edges   = np.histogram(df0.chol,density=False)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].chol,bins=bin_edges,density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].chol,bins=bin_edges,density=False)

int_discr["chol"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].fbs,bins=(-0.5,0.5,1.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].fbs,bins=(-0.5,0.5,1.5),density=False)

int_discr["fbs"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].restecg,bins=(-0.5,0.5,1.5,2.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].restecg,bins=(-0.5,0.5,1.5,2.5),density=False)

int_discr["restecg"] = intrinsic_discrepancy(hist1,hist2)

hist,bin_edges   = np.histogram(df0.thalach,density=False)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].thalach,bins=bin_edges,density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].thalach,bins=bin_edges,density=False)

int_discr["thalach"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].exang,bins=(-0.5,0.5,1.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].exang,bins=(-0.5,0.5,1.5),density=False)

int_discr["exang"] = intrinsic_discrepancy(hist1,hist2)

hist,bin_edges   = np.histogram(df0.oldpeak,density=False)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].oldpeak,bins=bin_edges,density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].oldpeak,bins=bin_edges,density=False)

int_discr["oldpeak"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].slope,bins=(0.5,1.5,2.5,3.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].slope,bins=(0.5,1.5,2.5,3.5),density=False)

int_discr["slope"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].ca,bins=(-0.5,0.5,1.5,2.5,3.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].ca,bins=(-0.5,0.5,1.5,2.5,3.5),density=False)

int_discr["ca"] = intrinsic_discrepancy(hist1,hist2)

hist1,bin_edges1 = np.histogram(df0[df0.num>0].thal,bins=(2.5,3.5,6.5,7.5),density=False)

hist2,bin_edges2 = np.histogram(df0[df0.num==0].thal,bins=(2.5,3.5,6.5,7.5),density=False)

int_discr["thal"] = intrinsic_discrepancy(hist1,hist2)

id_list = Counter(int_discr).most_common()

print ('Intrinsic discrepancies between disease and no-disease, in decreasing order: ')

for item in id_list:

    print ('   %f  (%s)' % (item[1],item[0]))




#Converting categorial values into discrete values
#Note that feature ca is discrete but not categorical, so we don't convert it.
df = df0.copy()
dummies = pd.get_dummies(df["cp"],prefix="cp")
df = df.join(dummies)
del df["cp"]
del df["cp_4.0"]
df = df.rename(columns = {"cp_1.0":"cp_1","cp_2.0":"cp_2","cp_3.0":"cp_3"})

dummies = pd.get_dummies(df["restecg"],prefix="recg")
df = df.join(dummies)
del df["restecg"]
del df["recg_0.0"]
df = df.rename(columns = {"recg_1.0":"recg_1","recg_2.0":"recg_2"})

dummies = pd.get_dummies(df["slope"],prefix="slope")
df = df.join(dummies)
del df["slope"]
del df["slope_2.0"]
df = df.rename(columns = {"slope_1.0":"slope_1","slope_3.0":"slope_3"})

dummies = pd.get_dummies(df["thal"],prefix="thal")
df = df.join(dummies)
del df["thal"]
del df["thal_3.0"]
df = df.rename(columns = {"thal_6.0":"thal_6","thal_7.0":"thal_7"})

#Replace response variable values and renaming it hd
#The num variable values of 1,2,3,4 are replaced with 1 in order to signify that the patient has heart disease
df["num"].replace(to_replace=[1,2,3,4],value=1,inplace=True)
df = df.rename(columns = {"num":"hd"})

new_columns_1 = ["age", "sex", "restbp", "chol", "fbs", "thalach", 
                 "exang", "oldpeak", "ca", "hd", "cp_1", "cp_2",
                 "cp_3", "recg_1", "recg_2", "slope_1", "slope_3",
                 "thal_6", "thal_7"]

print ('\nNumber of patients in dataframe: %i, with disease: %i, without disease: %i\n' \
      % (len(df.index),len(df[df.hd==1].index),len(df[df.hd==0].index)))
print (df.head())
print (df.describe())

# Standardize the dataframe
stdcols = ["age","restbp","chol","thalach","oldpeak"]
nrmcols = ["ca"]
stddf = df.copy()
stddf[stdcols] = stddf[stdcols].apply(lambda x: (x-x.mean())/x.std())
stddf[nrmcols] = stddf[nrmcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))

new_columns_2 = new_columns_1[:9] + new_columns_1[10:]
new_columns_2.insert(0,new_columns_1[9])
stddf = stddf.reindex(columns=new_columns_2)

# Convert dataframe into lists for use by classifiers
yall = stddf["hd"]
Xall = stddf[new_columns_2[1:]].values
print(Xall)


'''
Note about logistic regression with scikit-learn:
-> In scikit-learn we can specify penalty="l1" or penalty="l2", with
  an associated C=c, which is the *inverse* of the regularization strength.
  Thus, for zero regularization specify a high value of c.  Scikit-learn
  does not calculate uncertainties on the fit coefficients.
'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

lasso = False

nfeatures = len(stddf.columns)
if lasso:           # lasso regularization (least absolute shrinkage and selection operator)
    penalty = "l1"
    cval    = 1.0
    alpha   = [1.0]*nfeatures
else:               # no regularization
    penalty = "l1"
    cval    = 1000.0
    alpha   = 0.0

model = LogisticRegression(fit_intercept=True,penalty=penalty,dual=False,C=cval)
print(model)
lrfit = model.fit(Xall,yall)
print('\nLogisticRegression score on full data set: %f\n' % lrfit.score(Xall,yall))
ypred = model.predict(Xall)
print ('\nClassification report on full data set:')
print(metrics.classification_report(yall,ypred))
print ('\nConfusion matrix:')
print(metrics.confusion_matrix(yall,ypred))
print ('\nLogisticRegression coefficients:')
coeff = model.coef_.tolist()[0]
for index in range(len(coeff)):
    print ('%s : %8.5f' % (new_columns_2[index+1].rjust(9),coeff[index]))
print( 'Intercept : %f' %model.intercept_)

