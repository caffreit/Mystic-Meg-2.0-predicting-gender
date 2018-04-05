# Can we predict the Gender of an athlete?

First we use pairplot from Seaborn
It seems it should be possible to separate both genders. It seems like Age, Finish Position, Total Runs and Time are important.
Both Gen_Pos (Gender Position) and [Age_Grade](https://support.parkrun.com/hc/en-us/articles/200565263-What-is-age-grading-) have gender built into them. This is useful but makes the problem trivial. So we'll leave them both out.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
%matplotlib inline
```


```python
path_to_file = 'C:\Users\Administrator\Documents\Python Scripts\examplepark.csv'
data = pd.read_csv(path_to_file)
data['Time'] = ((pd.to_numeric(data['Time'].str.slice(0,2)))*60)+(pd.to_numeric\
(data['Time'].str.slice(3,5)))+((pd.to_numeric(data['Time'].str.slice(6,8)))/60)
data['Date'] = pd.to_datetime(data['Date'],errors='coerce', format='%d-%m-%Y')
data['Age_Cat'] = pd.to_numeric(data['Age_Cat'].str.slice(2,4),errors='coerce', downcast='integer')
data['Age_Grade'] = pd.to_numeric(data['Age_Grade'].str.slice(0,5),errors='coerce')
data['Club_Coded'] = data['Club'].isnull()
```


```python
def converter(Club):
    if Club==True:
        return 0
    else:
        return 1
```


```python
def converter(Gender):
    if Gender=='M':
        return 0
    else:
        return 1
```


```python
data['Club_Coded'] = data['Club_Coded'].apply(converter)
data['Gender_Coded'] = data['Gender'].apply(converter)
```


```python

```


```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pos</th>
      <th>Name</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Age_Grade</th>
      <th>Gender</th>
      <th>Gen_Pos</th>
      <th>Club</th>
      <th>Note</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
      <th>Gender_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-11-10</td>
      <td>1</td>
      <td>Michael MCSWIGGAN</td>
      <td>18.316667</td>
      <td>35.0</td>
      <td>73.43</td>
      <td>M</td>
      <td>1.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>First Timer!</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-11-10</td>
      <td>2</td>
      <td>Alan FOLEY</td>
      <td>18.433333</td>
      <td>30.0</td>
      <td>71.16</td>
      <td>M</td>
      <td>2.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>99.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-11-10</td>
      <td>3</td>
      <td>Matt SHIELDS</td>
      <td>18.533333</td>
      <td>55.0</td>
      <td>85.07</td>
      <td>M</td>
      <td>3.0</td>
      <td>North Belfast Harriers</td>
      <td>First Timer!</td>
      <td>274.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-11-10</td>
      <td>4</td>
      <td>David GARGAN</td>
      <td>18.650000</td>
      <td>40.0</td>
      <td>73.73</td>
      <td>M</td>
      <td>4.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>107.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-11-10</td>
      <td>5</td>
      <td>Paul SINTON-HEWITT</td>
      <td>18.900000</td>
      <td>50.0</td>
      <td>79.28</td>
      <td>M</td>
      <td>5.0</td>
      <td>Ranelagh Harriers</td>
      <td>First Timer!</td>
      <td>369.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
df = data[:5000]
df = df.drop('Club',1)
df = df.dropna()
df = df.drop('Gender',1)
df = df.drop('Note',1)
df = df.drop('Name',1)
df = df.drop('Date',1)
df = df.drop('Gen_Pos',1)
df = df.drop('Age_Grade',1)
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
      <th>Gender_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>18.316667</td>
      <td>35.0</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>18.433333</td>
      <td>30.0</td>
      <td>99.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>18.533333</td>
      <td>55.0</td>
      <td>274.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18.650000</td>
      <td>40.0</td>
      <td>107.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18.900000</td>
      <td>50.0</td>
      <td>369.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>20.250000</td>
      <td>40.0</td>
      <td>342.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>20.283333</td>
      <td>20.0</td>
      <td>40.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>20.450000</td>
      <td>40.0</td>
      <td>9.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>20.533333</td>
      <td>45.0</td>
      <td>296.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>20.816667</td>
      <td>30.0</td>
      <td>87.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_feat = df.drop('Gender_Coded', axis=1)

y = df['Gender_Coded']
X = df_feat
```

## We'll demonstrate several methods
### First up is the Support Vector Machine. 


```python
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
```

    [[713  35]
     [525 100]]
    
    
                 precision    recall  f1-score   support
    
              0       0.58      0.95      0.72       748
              1       0.74      0.16      0.26       625
    
    avg / total       0.65      0.59      0.51      1373
    
    


```python
from sklearn.grid_search import GridSearchCV

param_grid = {'C':[10000,1000],'gamma':[0.0001,0.00001]}
grid = GridSearchCV(SVC(),param_grid,verbose=1)
grid.fit(X_train,y_train)
```

    Fitting 3 folds for each of 4 candidates, totalling 12 fits
    

    [Parallel(n_jobs=1)]: Done  12 out of  12 | elapsed:  1.1min finished
    




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [10000, 1000], 'gamma': [0.0001, 1e-05]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=1)




```python
grid.best_estimator_
```




    SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
grid.best_params_
```




    {'C': 10000, 'gamma': 0.0001}




```python
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))
```

    [[579 169]
     [141 484]]
    
    
                 precision    recall  f1-score   support
    
              0       0.80      0.77      0.79       748
              1       0.74      0.77      0.76       625
    
    avg / total       0.78      0.77      0.77      1373
    
    

## Next we'll try Logistic Regression


```python
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
log_predictions = logmodel.predict(X_test)

print(confusion_matrix(y_test,log_predictions))
print('\n')
print(classification_report(y_test,log_predictions))
```

    [[594 154]
     [168 457]]
    
    
                 precision    recall  f1-score   support
    
              0       0.78      0.79      0.79       748
              1       0.75      0.73      0.74       625
    
    avg / total       0.77      0.77      0.77      1373
    
    


```python

```

## K Nearest Neighbours


```python

```


```python
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler.fit(df.drop('Gender_Coded',1))
scaled_features = scaler.transform(df.drop('Gender_Coded',1))
df_feat_scal = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat_scal.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.485405</td>
      <td>-1.464473</td>
      <td>-0.111670</td>
      <td>-0.180906</td>
      <td>-1.864933</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.474235</td>
      <td>-1.444179</td>
      <td>-0.613021</td>
      <td>1.106861</td>
      <td>-1.864933</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.463065</td>
      <td>-1.426784</td>
      <td>1.893737</td>
      <td>4.326277</td>
      <td>-1.864933</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.451895</td>
      <td>-1.406491</td>
      <td>0.389682</td>
      <td>1.254034</td>
      <td>-1.864933</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.440725</td>
      <td>-1.363004</td>
      <td>1.392385</td>
      <td>6.073960</td>
      <td>-1.864933</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.neighbors import KNeighborsClassifier

X = df_feat_scal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_predictions = knn.predict(X_test)

print(confusion_matrix(y_test,knn_predictions))
print('\n')
print(classification_report(y_test,knn_predictions))
```

    [[560 188]
     [174 451]]
    
    
                 precision    recall  f1-score   support
    
              0       0.76      0.75      0.76       748
              1       0.71      0.72      0.71       625
    
    avg / total       0.74      0.74      0.74      1373
    
    


```python
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.plot(range(1,40),error_rate,color='blue', marker='o',markerfacecolor='red')
plt.xlabel('K')
plt.ylabel('error rate')
plt.title('Error rate vs K')
plt.show()
```


![png](/img/output_31_0.png)



```python
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
knn_predictions = knn.predict(X_test)

print(confusion_matrix(y_test,knn_predictions))
print('\n')
print(classification_report(y_test,knn_predictions))
```

    [[615 133]
     [172 453]]
    
    
                 precision    recall  f1-score   support
    
              0       0.78      0.82      0.80       748
              1       0.77      0.72      0.75       625
    
    avg / total       0.78      0.78      0.78      1373
    
    

## Decision Trees and Random Forests


```python
from sklearn.tree import DecisionTreeClassifier

X = df.drop('Gender_Coded',1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_preds = dtree.predict(X_test)

print(confusion_matrix(y_test,dtree_preds))
print('\n')
print(classification_report(y_test,dtree_preds))
```

    [[563 185]
     [173 452]]
    
    
                 precision    recall  f1-score   support
    
              0       0.76      0.75      0.76       748
              1       0.71      0.72      0.72       625
    
    avg / total       0.74      0.74      0.74      1373
    
    


```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))
print('\n')
print(classification_report(y_test,rfc_preds))
```

    [[596 152]
     [122 503]]
    
    
                 precision    recall  f1-score   support
    
              0       0.83      0.80      0.81       748
              1       0.77      0.80      0.79       625
    
    avg / total       0.80      0.80      0.80      1373
    
    

#### The more data the more random forest does better than a single tree


```python

```

## K Means Clustering
### This is an unsupervised technique. We have labels so we can check<br> the performance of the method.


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Gender_Coded',1))

print(confusion_matrix(df['Gender_Coded'],kmeans.labels_))
print('\n')
print(classification_report(df['Gender_Coded'],kmeans.labels_))
```

    [[2037  457]
     [ 765 1315]]
    
    
                 precision    recall  f1-score   support
    
              0       0.73      0.82      0.77      2494
              1       0.74      0.63      0.68      2080
    
    avg / total       0.73      0.73      0.73      4574
    
    


```python
kmeans.cluster_centers_
```




    array([[  74.2159172 ,   23.56467999,   36.2426838 ,   46.70842256,
              10.68486795,    1.        ],
           [ 228.48871332,   31.75015989,   35.90970655,   26.38148984,
              12.95654628,    1.        ]])




```python
## so there seems to be two kinda distinct clusters
```


```python

```

## PCA, Principal Component Analysis


```python
from sklearn.decomposition import PCA

pca_scaler = StandardScaler()
pca_scaler.fit(df.drop('Gender_Coded',1))
scaled_data = scaler.transform(df.drop('Gender_Coded',1))
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
```


```python
x_pca.shape
```




    (4574L, 2L)




```python
scaled_data.shape
```




    (4574L, 6L)




```python
pca_scaler.fit(df.drop('Gender_Coded',1))

plt.scatter(x_pca[:,0],x_pca[:,1],c=df['Gender_Coded'],cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
```









![png](/img/output_47_1.png)



```python
df_comp = pd.DataFrame(pca.components_,columns=df.drop('Gender_Coded',1).columns.values)

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
```









![png](/img/output_48_1.png)


#### Seems to say that Pos, Time, Age and Total Runs are the important variables.
#### Could do SVM, LogReg etc. on the PCA'd data


```python

```


```python
# Add the global variables.
```


```python

```


```python
ser_count = data.groupby(['Date']).count()['Name']
Rel_Pos = []
c=0
for i in range(len(ser_count)):
    for j in range(ser_count[i]):
        rel_pos = float(data['Pos'][c])/float(ser_count[i])
        Rel_Pos.append(rel_pos)
        c+=1
data['Rel_Pos'] = Rel_Pos
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pos</th>
      <th>Name</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Age_Grade</th>
      <th>Gender</th>
      <th>Gen_Pos</th>
      <th>Club</th>
      <th>Note</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
      <th>Gender_Coded</th>
      <th>Rel_Pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-11-10</td>
      <td>1</td>
      <td>Michael MCSWIGGAN</td>
      <td>18.316667</td>
      <td>35.0</td>
      <td>73.43</td>
      <td>M</td>
      <td>1.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>First Timer!</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.006289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-11-10</td>
      <td>2</td>
      <td>Alan FOLEY</td>
      <td>18.433333</td>
      <td>30.0</td>
      <td>71.16</td>
      <td>M</td>
      <td>2.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>99.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.012579</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-11-10</td>
      <td>3</td>
      <td>Matt SHIELDS</td>
      <td>18.533333</td>
      <td>55.0</td>
      <td>85.07</td>
      <td>M</td>
      <td>3.0</td>
      <td>North Belfast Harriers</td>
      <td>First Timer!</td>
      <td>274.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.018868</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-11-10</td>
      <td>4</td>
      <td>David GARGAN</td>
      <td>18.650000</td>
      <td>40.0</td>
      <td>73.73</td>
      <td>M</td>
      <td>4.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>107.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.025157</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-11-10</td>
      <td>5</td>
      <td>Paul SINTON-HEWITT</td>
      <td>18.900000</td>
      <td>50.0</td>
      <td>79.28</td>
      <td>M</td>
      <td>5.0</td>
      <td>Ranelagh Harriers</td>
      <td>First Timer!</td>
      <td>369.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.031447</td>
    </tr>
  </tbody>
</table>
</div>




```python
ser_std = data.groupby(['Date']).std()['Time']
Std_time = []
c=0
for i in range(len(ser_std)):
    for j in range(ser_count[i]):
        std_time = float(ser_std[i])
        Std_time.append(std_time)
        c+=1
data['Std_time'] = Std_time
```


```python
ser_mtime = data.groupby(['Date']).mean()['Time']
Mean_time = []
c=0
for i in range(len(ser_mtime)):
    for j in range(ser_count[i]):
        mean_time = float(ser_mtime[i])
        Mean_time.append(mean_time)
        c+=1
data['Mean_time'] = Mean_time
```


```python
ser_mage = data.groupby(['Date']).mean()['Age_Grade']
Mean_age = []
c=0
for i in range(len(ser_mage)):
    for j in range(ser_count[i]):
        mean_age = float(ser_mage[i])
        Mean_age.append(mean_age)
        c+=1
data['Mean_age_grade'] = Mean_age
```


```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Pos</th>
      <th>Name</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Age_Grade</th>
      <th>Gender</th>
      <th>Gen_Pos</th>
      <th>Club</th>
      <th>Note</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
      <th>Gender_Coded</th>
      <th>Rel_Pos</th>
      <th>Std_time</th>
      <th>Mean_time</th>
      <th>Mean_age_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-11-10</td>
      <td>1</td>
      <td>Michael MCSWIGGAN</td>
      <td>18.316667</td>
      <td>35.0</td>
      <td>73.43</td>
      <td>M</td>
      <td>1.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>First Timer!</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.006289</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-11-10</td>
      <td>2</td>
      <td>Alan FOLEY</td>
      <td>18.433333</td>
      <td>30.0</td>
      <td>71.16</td>
      <td>M</td>
      <td>2.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>99.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.012579</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-11-10</td>
      <td>3</td>
      <td>Matt SHIELDS</td>
      <td>18.533333</td>
      <td>55.0</td>
      <td>85.07</td>
      <td>M</td>
      <td>3.0</td>
      <td>North Belfast Harriers</td>
      <td>First Timer!</td>
      <td>274.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.018868</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-11-10</td>
      <td>4</td>
      <td>David GARGAN</td>
      <td>18.650000</td>
      <td>40.0</td>
      <td>73.73</td>
      <td>M</td>
      <td>4.0</td>
      <td>Raheny Shamrock AC</td>
      <td>First Timer!</td>
      <td>107.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.025157</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-11-10</td>
      <td>5</td>
      <td>Paul SINTON-HEWITT</td>
      <td>18.900000</td>
      <td>50.0</td>
      <td>79.28</td>
      <td>M</td>
      <td>5.0</td>
      <td>Ranelagh Harriers</td>
      <td>First Timer!</td>
      <td>369.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.031447</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
df = data[:5000]
df = df.drop('Club',1)
df = df.dropna()
#df = df.drop('Gender',1)
df = df.drop('Note',1)
df = df.drop('Name',1)
df = df.drop('Date',1)
df = df.drop('Gen_Pos',1)
df = df.drop('Age_Grade',1)

df_feat = df.drop('Gender_Coded', axis=1)

y = df['Gender_Coded']
X = df_feat
```


```python
df_feat.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Gender</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
      <th>Rel_Pos</th>
      <th>Std_time</th>
      <th>Mean_time</th>
      <th>Mean_age_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>18.316667</td>
      <td>35.0</td>
      <td>M</td>
      <td>29.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.006289</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>18.433333</td>
      <td>30.0</td>
      <td>M</td>
      <td>99.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.012579</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>18.533333</td>
      <td>55.0</td>
      <td>M</td>
      <td>274.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.018868</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18.650000</td>
      <td>40.0</td>
      <td>M</td>
      <td>107.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.025157</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18.900000</td>
      <td>50.0</td>
      <td>M</td>
      <td>369.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.031447</td>
      <td>5.49527</td>
      <td>27.830778</td>
      <td>54.34449</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
```

    [[691  57]
     [468 157]]
    
    
                 precision    recall  f1-score   support
    
              0       0.60      0.92      0.72       748
              1       0.73      0.25      0.37       625
    
    avg / total       0.66      0.62      0.57      1373
    
    


```python
from sklearn.grid_search import GridSearchCV

param_grid = {'C':[10000,1000],'gamma':[0.0001,0.00001]}
grid = GridSearchCV(SVC(),param_grid,verbose=1)
grid.fit(X_train,y_train)
```

    Fitting 3 folds for each of 4 candidates, totalling 12 fits
    

    [Parallel(n_jobs=1)]: Done  12 out of  12 | elapsed:  1.0min finished
    




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [10000, 1000], 'gamma': [0.0001, 1e-05]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=1)




```python
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))
```

    [[577 171]
     [140 485]]
    
    
                 precision    recall  f1-score   support
    
              0       0.80      0.77      0.79       748
              1       0.74      0.78      0.76       625
    
    avg / total       0.77      0.77      0.77      1373
    
    


```python
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
log_predictions = logmodel.predict(X_test)

print(confusion_matrix(y_test,log_predictions))
print('\n')
print(classification_report(y_test,log_predictions))
```

    [[586 162]
     [158 467]]
    
    
                 precision    recall  f1-score   support
    
              0       0.79      0.78      0.79       748
              1       0.74      0.75      0.74       625
    
    avg / total       0.77      0.77      0.77      1373
    
    


```python

```


```python
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler.fit(df.drop('Gender_Coded',1))
scaled_features = scaler.transform(df.drop('Gender_Coded',1))
df_feat_scal = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat_scal.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pos</th>
      <th>Time</th>
      <th>Age_Cat</th>
      <th>Total_Runs</th>
      <th>Run_No.</th>
      <th>Club_Coded</th>
      <th>Gender_Coded</th>
      <th>Rel_Pos</th>
      <th>Std_time</th>
      <th>Mean_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.485405</td>
      <td>-1.464473</td>
      <td>-0.111670</td>
      <td>-0.180906</td>
      <td>-1.864933</td>
      <td>0.0</td>
      <td>-1.681895</td>
      <td>-0.2772</td>
      <td>1.203644</td>
      <td>-1.327526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.474235</td>
      <td>-1.444179</td>
      <td>-0.613021</td>
      <td>1.106861</td>
      <td>-1.864933</td>
      <td>0.0</td>
      <td>-1.659992</td>
      <td>-0.2772</td>
      <td>1.203644</td>
      <td>-1.327526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.463065</td>
      <td>-1.426784</td>
      <td>1.893737</td>
      <td>4.326277</td>
      <td>-1.864933</td>
      <td>0.0</td>
      <td>-1.638089</td>
      <td>-0.2772</td>
      <td>1.203644</td>
      <td>-1.327526</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.451895</td>
      <td>-1.406491</td>
      <td>0.389682</td>
      <td>1.254034</td>
      <td>-1.864933</td>
      <td>0.0</td>
      <td>-1.616186</td>
      <td>-0.2772</td>
      <td>1.203644</td>
      <td>-1.327526</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.440725</td>
      <td>-1.363004</td>
      <td>1.392385</td>
      <td>6.073960</td>
      <td>-1.864933</td>
      <td>0.0</td>
      <td>-1.594283</td>
      <td>-0.2772</td>
      <td>1.203644</td>
      <td>-1.327526</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.neighbors import KNeighborsClassifier

X = df_feat_scal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_predictions = knn.predict(X_test)

print(confusion_matrix(y_test,knn_predictions))
print('\n')
print(classification_report(y_test,knn_predictions))
```

    [[523 225]
     [202 423]]
    
    
                 precision    recall  f1-score   support
    
              0       0.72      0.70      0.71       748
              1       0.65      0.68      0.66       625
    
    avg / total       0.69      0.69      0.69      1373
    
    


```python
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.plot(range(1,40),error_rate,color='blue', marker='o',markerfacecolor='red')
plt.xlabel('K')
plt.ylabel('error rate')
plt.title('Error rate vs K')
plt.show()
```


![png](/img/output_68_0.png)



```python
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
knn_predictions = knn.predict(X_test)

print(confusion_matrix(y_test,knn_predictions))
print('\n')
print(classification_report(y_test,knn_predictions))
```

    [[573 175]
     [139 486]]
    
    
                 precision    recall  f1-score   support
    
              0       0.80      0.77      0.78       748
              1       0.74      0.78      0.76       625
    
    avg / total       0.77      0.77      0.77      1373
    
    


```python
from sklearn.tree import DecisionTreeClassifier

X = df.drop('Gender_Coded',1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_preds = dtree.predict(X_test)

print(confusion_matrix(y_test,dtree_preds))
print('\n')
print(classification_report(y_test,dtree_preds))
```

    [[554 194]
     [175 450]]
    
    
                 precision    recall  f1-score   support
    
              0       0.76      0.74      0.75       748
              1       0.70      0.72      0.71       625
    
    avg / total       0.73      0.73      0.73      1373
    
    


```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))
print('\n')
print(classification_report(y_test,rfc_preds))
```

    [[585 163]
     [149 476]]
    
    
                 precision    recall  f1-score   support
    
              0       0.80      0.78      0.79       748
              1       0.74      0.76      0.75       625
    
    avg / total       0.77      0.77      0.77      1373
    
    


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Gender_Coded',1))

print(confusion_matrix(df['Gender_Coded'],kmeans.labels_))
print('\n')
print(classification_report(df['Gender_Coded'],kmeans.labels_))
```

    [[2037  457]
     [ 765 1315]]
    
    
                 precision    recall  f1-score   support
    
              0       0.73      0.82      0.77      2494
              1       0.74      0.63      0.68      2080
    
    avg / total       0.73      0.73      0.73      4574
    
    


```python

```


```python
from sklearn.decomposition import PCA

pca_scaler = StandardScaler()
pca_scaler.fit(df.drop('Gender_Coded',1))
scaled_data = scaler.transform(df.drop('Gender_Coded',1))
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
```


```python
pca_scaler.fit(df.drop('Gender_Coded',1))

plt.scatter(x_pca[:,0],x_pca[:,1],c=df['Gender_Coded'],cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
```








![png](/img/output_75_1.png)



```python
df_comp = pd.DataFrame(pca.components_,columns=df.drop('Gender_Coded',1).columns.values)

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
```








![png](/img/output_76_1.png)



```python

```


```python

```


```python

```


```python

```


```python
sns.pairplot(df, hue="Gender")
```









![png](/img/output_81_1.png)



```python

```
