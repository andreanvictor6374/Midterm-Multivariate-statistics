'''
Midterm project Multivariate statistic Analyasis
By:Victor Andrean (D10702808)
April 2019
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error

data = pd.read_csv("kc_house_data.csv")
conv_dates = []
for k in range(0,len(data.date)):
    conv_dates.append(data['date'][k][:4])
data['date']= conv_dates  
data['date']=data['date'].astype(float)

'''Transformation to near normality'''
data['price']=np.log10(data['price'])
#data['bedrooms']=np.log10(data['bedrooms'])
#data['bathrooms']=np.log10(data['bathrooms'])
data['sqft_living']=np.log10(data['sqft_living'])
data['sqft_lot']=np.log10(data['sqft_lot'])
#data['floors']=np.log10(data['floors'])
#data['waterfront']=np.log10(data['waterfront'])
#data['view']=np.log10(data['view'])
#data['condition']=np.log10(data['condition'])
#data['grade']=np.log10(data['grade'])
data['sqft_above']=np.log10(data['sqft_above'])
#data['sqft_basement']=np.log10(data['sqft_basement'])
#data['yr_built']=np.log10(data['yr_built'])
#data['yr_renovated']=np.log10(data['yr_renovated'])
#data['zipcode']=np.log10(data['zipcode'])
#data['lat']=np.log10(data['lat'])
#data['long']=np.log10(data['long'])
data['sqft_living15']=np.log10(data['sqft_living15'])
data['sqft_lot15']=np.log10(data['sqft_lot15'])


'''Correlation Matrix'''
correlation = data.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.savefig('correlation_after transformation.png', format='png')

'''Histogram of price'''
bins=50
rangeHist=(max(data.price)-min(data.price))/bins
fig = plt.figure(figsize = (10,5))
#plt.grid(linestyle='--')
plt.xlabel("Price", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
ax = fig.gca()
data.price.hist(ax = ax,grid=False,bins=bins,xlabelsize=12, ylabelsize=12,color='b', edgecolor='black', linewidth=1.2)
#plt.savefig('hist ori price.png', format='png')
plt.savefig('hist price.png', format='png')

'''Histogram of sqft_living'''
bins=50
rangeHist=(max(data.sqft_living)-min(data.sqft_living))/bins
fig = plt.figure(figsize = (10,5))
#plt.grid(linestyle='--')
plt.xlabel("sqft_living", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
ax = fig.gca()
data.sqft_living.hist(ax = ax,grid=False,bins=bins,xlabelsize=12, ylabelsize=12,color='b', edgecolor='black', linewidth=1.2)
#plt.savefig('hist ori sqft_living.png', format='png')
plt.savefig('hist sqft_living.png', format='png')

'''Histogram of sqft_lot'''
bins=50
rangeHist=(max(data.sqft_lot)-min(data.sqft_lot))/bins
fig = plt.figure(figsize = (10,5))
#plt.grid(linestyle='--')
plt.xlabel("sqft_lot", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
ax = fig.gca()
data.sqft_lot.hist(ax = ax,grid=False,bins=bins,xlabelsize=12, ylabelsize=12,color='b', edgecolor='black', linewidth=1.2)
#plt.savefig('hist ori sqft_lot.png', format='png')
plt.savefig('hist sqft_lot.png', format='png')

'''Histogram of sqft_above'''
bins=50
rangeHist=(max(data.sqft_above)-min(data.sqft_above))/bins
fig = plt.figure(figsize = (10,5))
#plt.grid(linestyle='--')
plt.xlabel("sqft_above", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
ax = fig.gca()
data.sqft_above.hist(ax = ax,grid=False,bins=bins,xlabelsize=12, ylabelsize=12,color='b', edgecolor='black', linewidth=1.2)
#plt.savefig('hist ori sqft_above.png', format='png')
plt.savefig('hist sqft_above.png', format='png')

'''Histogram of sqft_living15'''
bins=50
rangeHist=(max(data.sqft_living15)-min(data.sqft_living15))/bins
fig = plt.figure(figsize = (10,5))
#plt.grid(linestyle='--')
plt.xlabel("sqft_living15", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
ax = fig.gca()
data.sqft_living15.hist(ax = ax,grid=False,bins=bins,xlabelsize=12, ylabelsize=12,color='b', edgecolor='black', linewidth=1.2)
#plt.savefig('hist ori sqft_living15.png', format='png')
plt.savefig('hist sqft_living15.png', format='png')

'''Histogram of sqft_lot15'''
bins=50
rangeHist=(max(data.sqft_lot15)-min(data.sqft_lot15))/bins
fig = plt.figure(figsize = (10,5))
#plt.grid(linestyle='--')
plt.xlabel("sqft_lot15", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
ax = fig.gca()
data.sqft_lot15.hist(ax = ax,grid=False,bins=bins,xlabelsize=12, ylabelsize=12,color='b', edgecolor='black', linewidth=1.2)
#plt.savefig('hist ori sqft_lot15.png', format='png')
plt.savefig('hist sqft_lot15.png', format='png')


fig, ax = plt.subplots()
plt.grid(linestyle='--')
plt.xlabel("")
plt.ylabel("Price")
ax.boxplot(data.price, labels=[''])
plt.savefig('priceBoxPlot.png', format='png')


yearHouseBuilt=pd.get_dummies(data.yr_built).sum()
# Fit with polyfit
b, m = polyfit(yearHouseBuilt.index, yearHouseBuilt, 1)
plt.grid(linestyle='--')
plt.scatter(yearHouseBuilt.index, yearHouseBuilt,color='lightsalmon')
plt.plot(yearHouseBuilt.index, yearHouseBuilt,color='b')
plt.plot(yearHouseBuilt.index, b + m * yearHouseBuilt.index, '--',color='r')
plt.xlabel('year')
plt.ylabel("number of house being built")
plt.savefig('number of house being built.png', format='png')

##Plot of the most correlated parameters###
plt.figure(figsize = (9,5))
b, m = polyfit(data.grade,data.price, 1)
plt.grid(linestyle='--')
plt.scatter(data.grade,data.price,color='b')
plt.plot(data.grade, b + m * data.grade, '--',color='r')
plt.xlabel("Grade", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.savefig('Grade vs Price.png', format='png')

plt.figure(figsize = (9,5))
b, m = polyfit(data.bathrooms,data.price, 1)
plt.grid(linestyle='--')
plt.scatter(data.bathrooms,data.price,color='b')
plt.plot(data.bathrooms, b + m * data.bathrooms, '--',color='r')
plt.xlabel("Bathrooms", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.savefig('Bathrooms vs Price.png', format='png')


plt.figure(figsize = (9,5))
b, m = polyfit(data.sqft_living,data.price, 1)
plt.grid(linestyle='--')
plt.scatter(data.sqft_living,data.price,color='b')
plt.plot(data.sqft_living, b + m * data.sqft_living, '--',color='r')
plt.xlabel("sqft_living", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.savefig('sqft_living vs Price.png', format='png')

plt.figure(figsize = (9,5))
b, m = polyfit(data.sqft_living15,data.price, 1)
plt.grid(linestyle='--')
plt.scatter(data.sqft_living15,data.price,color='b')
plt.plot(data.sqft_living15, b + m * data.sqft_living15, '--',color='r')
plt.xlabel("sqft_living15", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.savefig('sqft_living15 vs Price.png', format='png')

plt.figure(figsize = (9,5))
b, m = polyfit(data.sqft_above,data.price, 1)
plt.grid(linestyle='--')
plt.scatter(data.sqft_above,data.price,color='b')
plt.plot(data.sqft_above, b + m * data.sqft_above, '--',color='r')
plt.xlabel("sqft_above", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.savefig('sqft_above vs Price.png', format='png')

###Latitude and longitude plot
plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.savefig('lat vs long.png', format='png')


'''Split the data into train and test set'''
y=data.iloc[:, 2].values
#X=data.drop(['id','date','price'],axis=1)
X=data.drop(['id','date','price','zipcode','yr_renovated'],axis=1)

'''Put the data into linear regression'''
numCV=10
score1=np.zeros((numCV,3))
score2=np.zeros((numCV,3))
for i in range(0,numCV):
    x_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.20,random_state =i)
    reg = LinearRegression()
    def regressionFuncFit(inputt,target):
        inputt,target=np.matrix(inputt),np.matrix(target).reshape((-1,1))
        sc_X = StandardScaler()
        inputt = sc_X.fit_transform(inputt)
        sc_y = StandardScaler()
        target = sc_y.fit_transform(target)    
        matX=np.append(np.ones((len(inputt),1)),inputt,axis=1)
        temp=np.linalg.inv(np.dot(np.transpose(matX),matX))
        b_hat=np.dot(np.dot(temp,np.transpose(matX)),target)
        return b_hat,sc_X,sc_y
    
    def regressionFuncPredict(inputt,target,sc_X,sc_y,b_hat):
        inputt,target=np.matrix(inputt),np.matrix(target).reshape((-1,1))
        inputt = sc_X.transform(inputt)
        matX=np.append(np.ones((len(inputt),1)),inputt,axis=1)
        Yhat=sc_y.inverse_transform(np.dot(matX,b_hat))                         
        return Yhat
    
    reg.fit(x_train,y_train)
    y_pred1=reg.predict(x_test)
    y_pred1=10**y_pred1
    
'''y_pred2 is calculated step by step using a code built from scratch to validate the result yield by sikit-learn package'''      
    b_hat,sc_X,sc_y=regressionFuncFit(x_train,y_train)
    y_pred2=regressionFuncPredict(x_test,y_test,sc_X,sc_y,b_hat)[:,0]
    y_pred2=10**y_pred2
    
#    y_testInv=y_test
    y_testInv=10**y_test
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
    score1[i,:]=np.array([evaluation(y_testInv,y_pred1)])
    score2[i,:]=np.array([evaluation(y_testInv,y_pred2)])

eHat1=y_pred1-y_testInv 
eHat2=y_pred2-y_testInv 

np.mean(eHat1)
np.mean(eHat2)

plt.figure(figsize = (9,5))
plt.grid(linestyle='--')
b, m = polyfit(y_pred1,eHat1, 1)
plt.scatter(y_pred1,eHat1,color='b')
plt.plot(y_pred1, b + m * y_pred1, '--',color='r')
plt.xlabel("y_pred1", fontsize=14)
plt.ylabel("eHat1", fontsize=14)
plt.savefig('eHat1 vs y_pred1.png', format='png')

plt.figure(figsize = (9,5))
plt.grid(linestyle='--')
b, m = polyfit(y_pred2,eHat2, 1)
plt.scatter(y_pred2,eHat2,color='b')
plt.plot(y_pred2, b + m * y_pred2, '--',color='r')
plt.xlabel("y_pred2", fontsize=14)
plt.ylabel("eHat2", fontsize=14)
plt.savefig('eHat2 vs y_pred2.png', format='png')
    
score1=np.append(score1,np.mean(score1,axis=0).reshape((1,-1)),axis=0)  
score2=np.append(score2,np.mean(score2,axis=0).reshape((1,-1)),axis=0)  

score1Pd = pd.DataFrame({'R2':score1[:,0],'adjR2':score1[:,1],'MAE':score1[:,2]})
score1Pd.index.names = ['trial']
score2Pd = pd.DataFrame({'R2':score2[:,0],'adjR2':score2[:,1],'MAE':score2[:,2]})
score2Pd.index.names = ['trial']

def changeIndex(df,numInd):
    as_list = df.index.tolist()
    idx = as_list.index(numInd)
    as_list[idx] = 'Average'
    df.index = as_list
    return df

score1Pd=changeIndex(score1Pd,numCV)
score2Pd=changeIndex(score2Pd,numCV)
        
score1Pd.to_csv('Score1Pd.csv', sep=',')
score2Pd.to_csv('Score2Pd.csv', sep=',')


plt.figure(figsize = (15,5))
plt.grid(linestyle='--')
plt.plot(range(200),y_pred1[:200],color='r',label='Predicted value')
plt.plot(range(200),y_testInv[:200],color='b',label='Ground truth')
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
