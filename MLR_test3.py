'''
Midterm project Multivariate statistic Analyasis
By:Victor Andrean (D10702808)
'''
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.decomposition import PCA
import statsmodels.api as sm
from numpy import linalg as LA

data = pd.read_csv("kc_house_data.csv")
conv_dates = []
for k in range(0,len(data.date)):
    conv_dates.append(data['date'][k][:4])
data['date']= conv_dates  
data['date']=data['date'].astype(float)

'''Correlation Matrix'''
correlation = data.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.savefig('correlation_after transformation.png', format='png')

##Plot of the most correlated parameters###
sns.set(font_scale=1.5)
g=sns.jointplot(x='grade',y='price',data=data, kind='reg',joint_kws={'line_kws':{'color':'red'}}, size=10)
g.savefig('Grade vs Price.png', format='png')

sns.set(font_scale=1.5)
g=sns.jointplot(x='bathrooms',y='price',data=data, kind='reg',joint_kws={'line_kws':{'color':'red'}}, size=10)
g.savefig('Bathrooms vs Price.png', format='png')


sns.set(font_scale=1.5)
g=sns.jointplot(x='sqft_living',y='price',data=data, kind='reg',joint_kws={'line_kws':{'color':'red'}}, size=10)
g.savefig('sqft_living vs Price.png', format='png')

sns.set(font_scale=1.5)
g=sns.jointplot(x='sqft_living15',y='price',data=data, kind='reg',joint_kws={'line_kws':{'color':'red'}}, size=10)
g.savefig('sqft_living15 vs Price.png', format='png')


sns.set(font_scale=1.5)
g=sns.jointplot(x='sqft_above',y='price',data=data, kind='reg',joint_kws={'line_kws':{'color':'red'}}, size=10)
#plt.grid(linestyle='--')
g.savefig('sqft_above vs Price.png', format='png')

###Latitude and longitude plot
g=sns.jointplot(x='lat',y='long',data=data, size=10)
g.savefig('lat vs long.png', format='png')


'''Split the data into train and test set'''
y=data.iloc[:, 2].values
#X=data.drop(['id','date','price'],axis=1)
X=data.drop(['id','date','price','zipcode','yr_renovated'],axis=1)

'''Put the data into linear regression'''

score1=np.zeros((1,3))

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.20,random_state =0)
reg = LinearRegression()
# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# Applying PCA
pca = PCA(n_components = 5) 
x_test = pca.fit_transform(x_test)
x_train = pca.transform(x_train)
explained_variance = pca.explained_variance_
explained_variance_ratio=pca.explained_variance_ratio_
explained_variance_ratio_total=pca.explained_variance_ratio_.cumsum()

OLSreg = LinearRegression()
OLSreg.fit(x_train,y_train)
y_pred=OLSreg.predict(x_train)

absOLS_err=abs(y_pred-y_train )

err_reg = LinearRegression()
err_reg.fit(y_pred.reshape(-1,1),absOLS_err)
errPred=err_reg.predict(y_pred.reshape(-1,1)).squeeze()

wls_model = sm.WLS(y_train, x_train, weights=1/(errPred**2)).fit()
y_hat = wls_model.predict(x_test).reshape(-1,1)
y_hat=sc_y.inverse_transform(y_hat)

  
def evaluation(Ytrue,Ypred):
    '''R2'''
    R2=r2_score(Ytrue,Ypred)
    '''adjusted R2'''
    N=sum((Ytrue-Ypred)**2)/(x_test.shape[0]-x_test.shape[1]-1)
    D=sum((Ytrue-np.mean(Ytrue))**2)/(len(x_test)-1)
    adjR2=1-(N/D)
    '''MAE'''
    MAE=mean_absolute_error(Ytrue,Ypred)
    return R2,adjR2,MAE
score1[0,:]=np.array([evaluation(y_test.squeeze(),y_hat.squeeze())])


A=np.dot(np.transpose(X),X)
eigenA=LA.eig(A)
cond_A=max(eigenA[0])/min(eigenA[0])

Apca=np.dot(np.transpose(x_train),x_train)
eigenPCA=LA.eig(Apca)
cond_Apca=max(eigenPCA[0])/min(eigenPCA[0])



eRes=y_hat-y_test
y_hat=y_hat.squeeze()
eRes=eRes.squeeze()
plt.figure(figsize = (9,5))
plt.grid(linestyle='--')
b, m = polyfit(y_hat,eRes, 1)
plt.scatter(y_hat,eRes,color='b')
plt.plot(y_hat, b + m * y_hat, '--',color='r')
plt.xlabel("y_hat", fontsize=14)
plt.ylabel("eRes", fontsize=14)
plt.savefig('eRes vs y_hat.png', format='png')

score1[0,:]=np.array([evaluation(y_test.squeeze(),y_hat.squeeze())])


covB=np.linalg.inv(np.dot(np.transpose(x_train),x_train))
aa=1/(errPred**2)
covB_WLS=np.linalg.inv(np.dot(np.transpose(x_train),np.dot(np.diag(aa),x_train)))


plt.figure(figsize = (10,5)) 
plt.scatter(range(16),explained_variance)
plt.plot(explained_variance)
plt.plot(range(len(explained_variance)), explained_variance, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
loadings =np.square(pca.components_* np.sqrt(pca.explained_variance_))
l_2=loadings.sum(axis=0)


plt.figure(figsize = (15,5))
plt.grid(linestyle='--')
plt.plot(range(200),y_hat[:200],color='r',label='Predicted value')
plt.plot(range(200),y_test[:200],color='b',label='Ground truth')
plt.xlabel("Sample", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.legend()
plt.savefig('Yhat vs Sample.png', format='png')


#scores = cross_val_score(reg, X, y, scoring='r2', cv=10)
#print("Cross-validation scores: {}".format(scores))
#print("Average cross-validation score: {:.2f}".format(scores.mean()))
# Encoding categorical data
#oneHotDate=pd.get_dummies(conv_dates).drop(['2015'],axis=1) 
#oneHotZip=pd.get_dummies(data.zipcode).drop([max(data.zipcode)],axis=1)
#data['date'] = oneHotDate
#X= pd.concat([oneHotZip, X], axis=1)

#X_err_reg = LinearRegression()
#X_err_reg.fit(x_train,absOLS_err)
#errPred=X_err_reg.predict(x_train).squeeze()

#WLS_reg = LinearRegression()
#WLS_reg.fit(x_train,y_train, sample_weight=1/(weights**2))
#WLS_reg.fit(x_train,y_train, sample_weight=1/(errPred**2))
#y_hat=WLS_reg.predict(x_test)