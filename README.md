[Homepage of blog-](https://caffreit.github.io/blog-/)


# Can we predict the Gender of a Runner?

In a previous post I compared Males and Females, I showed that there are clear differences in the finish time and position. Given this it should be possible to use these features to predict the Gender of an athlete.

Both Gen_Pos (Gender Position) and [Age_Grade](https://support.parkrun.com/hc/en-us/articles/200565263-What-is-age-grading-) have gender built into them. This is useful but makes the problem trivial. So we'll leave them both out.

I'll first walk through recoding and normalising the data. Then I'll select the features we want to focus on. I'm going to demonstrate several Machine Learning (ML) techniques as well optimise their hyperparameters.

My computer is fairly slow so I'm going to use only a small number (5000) of records for the demonstrations. I found that the random forest performed the best so I trained it on 75,000 records leaving roughly 10,000 records as holdout data to test against after tuning of the hyperparameters.

Finally I'll demonstrate two unsupervised techniques; K-Means Clustering and Principal Component Analysis.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
%matplotlib inline
```


```python
path_to_file = 'C:\Users\Administrator\Desktop\Python Scripts\examplepark.csv'
data = pd.read_csv(path_to_file)
data['Time'] = ((pd.to_numeric(data['Time'].str.slice(0,2)))*60)+(pd.to_numeric\
(data['Time'].str.slice(3,5)))+((pd.to_numeric(data['Time'].str.slice(6,8)))/60)
data['Date'] = pd.to_datetime(data['Date'],errors='coerce', format='%d-%m-%Y')
data['Age_Cat'] = pd.to_numeric(data['Age_Cat'].str.slice(2,4),errors='coerce', downcast='integer')
data['Age_Grade'] = pd.to_numeric(data['Age_Grade'].str.slice(0,5),errors='coerce')
data['Club_Coded'] = data['Club'].isnull()
```

### Recoding and Shuffling the data.
We have to convert the strings 'M' and 'F' to a number for the Machine Learning to work.
<br>

Classes are roughly equal in number so no need to rebalance.<br>

I'm splitting my data into initial testing set and then a holdout set. The data is ordered by date. Since I'd like to avoid any selection bias and keep the model general I'm going to shuffle the data first. In other words make sure the testing and training data look the same.


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
from sklearn.utils import shuffle
data = shuffle(data)
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
      <th>71005</th>
      <td>2017-02-11</td>
      <td>294</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>223</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23645</th>
      <td>2014-04-12</td>
      <td>58</td>
      <td>Brendan O DONNELL</td>
      <td>22.000000</td>
      <td>20.0</td>
      <td>58.64</td>
      <td>M</td>
      <td>55.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>PB stays at 00.21.41</td>
      <td>5.0</td>
      <td>75</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75723</th>
      <td>2017-05-20</td>
      <td>280</td>
      <td>Celine BRENNAN</td>
      <td>35.683333</td>
      <td>45.0</td>
      <td>45.87</td>
      <td>F</td>
      <td>115.0</td>
      <td>Portmarnock Athletic Club</td>
      <td>PB stays at 00.24.42</td>
      <td>149.0</td>
      <td>237</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37998</th>
      <td>2015-01-17</td>
      <td>321</td>
      <td>Jason BLAKE</td>
      <td>30.533333</td>
      <td>40.0</td>
      <td>45.36</td>
      <td>M</td>
      <td>214.0</td>
      <td>NaN</td>
      <td>PB stays at 00.22.49</td>
      <td>48.0</td>
      <td>115</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82984</th>
      <td>2017-11-25</td>
      <td>127</td>
      <td>Kristï¿½na PETIï¿½KOVï¿½</td>
      <td>27.200000</td>
      <td>20.0</td>
      <td>54.41</td>
      <td>F</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>New PB!</td>
      <td>2.0</td>
      <td>264</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




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
df = df.drop('Club_Coded',1)
```

### Normalise the data. 
Avoid overweighting any feature. Mainly important for K-Means since it's based on distance between points.


```python
cols_to_norm = ['Pos', 'Time', 'Age_Cat', 'Total_Runs', 'Run_No.']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df.head()
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
      <th>Gender_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23645</th>
      <td>0.099303</td>
      <td>0.134001</td>
      <td>0.142857</td>
      <td>0.009390</td>
      <td>0.270073</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75723</th>
      <td>0.486063</td>
      <td>0.399096</td>
      <td>0.500000</td>
      <td>0.347418</td>
      <td>0.861314</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37998</th>
      <td>0.557491</td>
      <td>0.299322</td>
      <td>0.428571</td>
      <td>0.110329</td>
      <td>0.416058</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82984</th>
      <td>0.219512</td>
      <td>0.234743</td>
      <td>0.142857</td>
      <td>0.002347</td>
      <td>0.959854</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7582</th>
      <td>0.790941</td>
      <td>0.415886</td>
      <td>0.285714</td>
      <td>0.004695</td>
      <td>0.091241</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Assigning our feature and target column.


```python
features = df.drop('Gender_Coded', axis=1)

X = features
y = df['Gender_Coded']
```

# Machine Learning
### Importing some useful functions for; splitting, validation and optimisation.

The dataset is split into training and testing sets so that we can validate the performance of the models on data it has not yet seen.


```python
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


    

## First up is Support Vector Machine. 
It uses an n-1 hyperplane to separate labelled classes, where n is the number of features. It selects the plane which best separates the classes and with the largest distance to the nearest points


```python
from sklearn.svm import SVC

model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
```

    [[447 153]
     [160 494]]
    
    
                 precision    recall  f1-score   support
    
              0       0.74      0.74      0.74       600
              1       0.76      0.76      0.76       654
    
    avg / total       0.75      0.75      0.75      1254
    
    

## Optimise hyperparameters
We'll use cross validation so that we don't over fit to our test set. Cross validation splits the training set into *K* folds. The data is trained on K-1 of the folds and tested against the Kth fold. The model is trained K times on this data (once on each fold), the final score of the model is the average of the K scores. 

This procedure is repeated for as many models as we specify. For example we may have 3 different hyperparameters we wish to tune. We give each 4 possible values this gives 4 x 4 x 4 = 12 different models to test.


```python
from sklearn.grid_search import GridSearchCV

param_grid = {'C':[100,1000,10000],'gamma':[1,0.1,0.01]}
grid = GridSearchCV(SVC(),param_grid,verbose=0)
grid.fit(X_train,y_train)
```







After two rounds of fine-tuning these were the best hyperparameters.


```python
grid.best_params_
```




    {'C': 10000, 'gamma': 1}



The high cost value C of 10000 could indicate overfitting as the cost to misclassifying points is high. 
The gamma value dictates how close a point needs to be to the separating plane to be considered.


```python
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))
```

    [[580 168]
     [137 488]]
    
    
                 precision    recall  f1-score   support
    
              0       0.81      0.78      0.79       748
              1       0.74      0.78      0.76       625
    
    avg / total       0.78      0.78      0.78      1373
    
    

An extra __3%__ across the board. Or (78/75)x100 = 104%, a 4% improvement. __*Thumbs up!*__

**Precision:** Ratio of the correctly predicted Males to total number of predicted Males. TM/TM + FM

**Recall:** Ratio of corrctly predicted Males to the total number of actual Males. TM/TM + FF

**F1-Score:** 2 x (Recall x Precision)/(Recall+Precision)

# Logistic Regression


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

    [[597 151]
     [173 452]]
    
    
                 precision    recall  f1-score   support
    
              0       0.78      0.80      0.79       748
              1       0.75      0.72      0.74       625
    
    avg / total       0.76      0.76      0.76      1373
    
    

## Optimise hyperparameters


```python
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
gs = GridSearchCV(LogisticRegression(), hyperparameters, cv=5, verbose=0)
gs.fit(X_train,y_train)
```









```python
print('Best Penalty:', gs.best_estimator_.get_params()['penalty'])
print('Best C:', gs.best_estimator_.get_params()['C'])
```

    ('Best Penalty:', 'l1')
    ('Best C:', 7.7426368268112693)
    


```python
grid_predictions = gs.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))
```

    [[593 155]
     [172 453]]
    
    
                 precision    recall  f1-score   support
    
              0       0.78      0.79      0.78       748
              1       0.75      0.72      0.73       625
    
    avg / total       0.76      0.76      0.76      1373
    
    

Didn't get any improvement here. However it's not worth the time to squeeze a bit more performance from it when there are other techniques that will perform better.

# K Nearest Neighbours
Have to scale the features for K-Means as it works by using the Euclidean (as the crow flies) distance between a point and its neighbours.


```python
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
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
    </tr>
    <tr>
      <th>1</th>
      <td>-1.474235</td>
      <td>-1.444179</td>
      <td>-0.613021</td>
      <td>1.106861</td>
      <td>-1.864933</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.463065</td>
      <td>-1.426784</td>
      <td>1.893737</td>
      <td>4.326277</td>
      <td>-1.864933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.451895</td>
      <td>-1.406491</td>
      <td>0.389682</td>
      <td>1.254034</td>
      <td>-1.864933</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.440725</td>
      <td>-1.363004</td>
      <td>1.392385</td>
      <td>6.073960</td>
      <td>-1.864933</td>
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
    
    

### Optimise hyperparameters


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


![png](/img/output_34_0.png)



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
    
    

__An extra 4%!__

# Decision Trees and Random Forests
A decision tree splits the data along a feature at a particular value. The value to split along is chosen by maximising the homogeneity of the classes either said of the split. 

A tree makes multiple splits of the data eventually giving an easily interpretable prediction model. Trees can over fit if too many splits are made or the number of samples in a split are too small.


```python
from sklearn.tree import DecisionTreeClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_preds = dtree.predict(X_test)

print(confusion_matrix(y_test,dtree_preds))
print('\n')
print(classification_report(y_test,dtree_preds))
```

    [[477 171]
     [223 386]]
    
    
                 precision    recall  f1-score   support
    
              0       0.68      0.74      0.71       648
              1       0.69      0.63      0.66       609
    
    avg / total       0.69      0.69      0.69      1257
    
    

## Random Forest
It is a ensemble method. A load (200 in this case) of trees vote on the classification of a point. It's a bit like Wisdom of the Crowd.


```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))
print('\n')
print(classification_report(y_test,rfc_preds))
```

    [[512 136]
     [147 462]]
    
    
                 precision    recall  f1-score   support
    
              0       0.78      0.79      0.78       648
              1       0.77      0.76      0.77       609
    
    avg / total       0.77      0.77      0.77      1257
    
    

### Optimise hyperparameters


```python
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
```

### Random Search

I'm using random search over a wide range of hyperparameters to find a rough location of the global minimum. I'll then fine tune the hyperparameters further with grid search.

I'd like to keep the depth low and min sample split high to minimise overfitting. If we wanted we could use Lasso or Ridge Regression to penalise model complexity.

n_estimators = Number of trees in the forest<br>
max_features = Number of features to consider at every split<br>
max_depth = Number of splits in the tree<br>
min_samples_split = Minimum number of samples required to split a node<br>
min_samples_leaf = Minimum number of samples required at each leaf node<br>
bootstrap = Resample with replacement of records<br>


```python
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 5)]
max_features = ['auto']
max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
max_depth.append(None)
min_samples_split = [5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
```


```python
pprint(random_grid)
```

    {'bootstrap': [True, False],
     'max_depth': [5, 10, 15, 20, 25, None],
     'max_features': ['auto'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [5, 10],
     'n_estimators': [100, 575, 1050, 1525, 2000]}
    

Using 3 fold cross validation, searching across 100 different combinations, and using two cores so I can still use my machine.


```python
rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=0, random_state=42, n_jobs = 2)
rf_random.fit(X_train,y_train)
```







```python
rf_random.best_params_
```




    {'bootstrap': False,
     'max_depth': 15,
     'max_features': 'auto',
     'min_samples_leaf': 1,
     'min_samples_split': 10,
     'n_estimators': 1525}




```python
grid_predictions = rf_random.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))
```

    [[594 154]
     [121 504]]
    
    
                 precision    recall  f1-score   support
    
              0       0.83      0.79      0.81       748
              1       0.77      0.81      0.79       625
    
    avg / total       0.80      0.80      0.80      1373
    
    

### Refining hyperparams further with Grid Search


```python
param_grid = {
    'bootstrap': [True],
    'max_depth': [13,15,17],
    'max_features': ['auto'],
    'min_samples_leaf': [1,2],
    'min_samples_split': [8,10,12],
    'n_estimators': [1300,1500,1700]
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = 2, verbose = 0)
```


```python
grid_search.fit(X_train,y_train)
grid_search.best_params_
```




    {'bootstrap': True,
     'max_depth': 15,
     'max_features': 'auto',
     'min_samples_leaf': 1,
     'min_samples_split': 8,
     'n_estimators': 1500}



Unfortunately it's quite a deep tree with 15 splits, coupled with the min samples per leaf of 1 I'd be concerned the model was fitting to noise in the data and won't generalise. Also the high number of trees in the forest (1500) makes the model very slow.


```python
grid_predictions = grid_search.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))
```

    [[590 158]
     [124 501]]
    
    
                 precision    recall  f1-score   support
    
              0       0.83      0.79      0.81       748
              1       0.76      0.80      0.78       625
    
    avg / total       0.80      0.79      0.79      1373
    
    

The reduction in its scores is probably due to an increase in overfitting caused the redcution in the minimum number of samples per split from 10 to 8. Unfortunately searching for optimum parameters takes over half an hour on my machine so I don't want to spend all day fiddling with it. 

Instead I'll throw way more data at it. There is no way I'm going to optimise the hyperparameters on 75000 records, it would take over 12 hours for even a small GridSearch (using all four cores).

We're keeping some holdout data to test against after finetuning the hyparams. No need to scale the data as it doesn't affect random forest.


```python
df1 = data[:75000]
df1 = df1.drop('Club',1)
df1 = df1.dropna()
df1 = df1.drop('Gender',1)
df1 = df1.drop('Note',1)
df1 = df1.drop('Name',1)
df1 = df1.drop('Date',1)
df1 = df1.drop('Gen_Pos',1)
df1 = df1.drop('Age_Grade',1)
df1 = df1.drop('Club_Coded',1)
df1.tail(10)
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
      <th>Gender_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>82632</th>
      <td>67</td>
      <td>23.033333</td>
      <td>35.0</td>
      <td>16.0</td>
      <td>263</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79986</th>
      <td>39</td>
      <td>22.216667</td>
      <td>11.0</td>
      <td>104.0</td>
      <td>254</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24216</th>
      <td>200</td>
      <td>25.616667</td>
      <td>40.0</td>
      <td>134.0</td>
      <td>76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>72043</th>
      <td>308</td>
      <td>33.650000</td>
      <td>50.0</td>
      <td>13.0</td>
      <td>226</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6488</th>
      <td>301</td>
      <td>30.700000</td>
      <td>45.0</td>
      <td>7.0</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13194</th>
      <td>152</td>
      <td>25.566667</td>
      <td>30.0</td>
      <td>7.0</td>
      <td>41</td>
      <td>1</td>
    </tr>
    <tr>
      <th>66101</th>
      <td>186</td>
      <td>30.766667</td>
      <td>55.0</td>
      <td>153.0</td>
      <td>207</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34712</th>
      <td>115</td>
      <td>26.050000</td>
      <td>50.0</td>
      <td>156.0</td>
      <td>105</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41638</th>
      <td>289</td>
      <td>28.083333</td>
      <td>50.0</td>
      <td>14.0</td>
      <td>124</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22665</th>
      <td>6</td>
      <td>17.916667</td>
      <td>35.0</td>
      <td>99.0</td>
      <td>73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = df1.drop('Gender_Coded', axis=1)

X = features
y = df1['Gender_Coded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
rfc = RandomForestClassifier(n_estimators=1500, max_depth=15, max_features='auto', 
                             min_samples_leaf=1, min_samples_split=8) ### ad in the good parmas
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))
print('\n')
print(classification_report(y_test,rfc_preds))
```

    [[7810 1802]
     [1316 8084]]
    
    
                 precision    recall  f1-score   support
    
              0       0.86      0.81      0.83      9612
              1       0.82      0.86      0.84      9400
    
    avg / total       0.84      0.84      0.84     19012
    
    


```python
rfc = RandomForestClassifier(n_estimators=1500, max_depth=15, max_features='auto', 
                             min_samples_leaf=1, min_samples_split=8) ### ad in the good parmas
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_preds))
print('\n')
print(classification_report(y_test,rfc_preds))
```

Good improvement. If I had a more powerful machine I'd do a more complete GridSearch and try to bring the number of estimators down to improve prediction speed.


```python
holdout_data = data[75001:]
holdout_data = holdout_data.drop('Club',1)
holdout_data = holdout_data.dropna()
holdout_data = holdout_data.drop('Gender',1)
holdout_data = holdout_data.drop('Note',1)
holdout_data = holdout_data.drop('Name',1)
holdout_data = holdout_data.drop('Date',1)
holdout_data = holdout_data.drop('Gen_Pos',1)
holdout_data = holdout_data.drop('Age_Grade',1)
holdout_data = holdout_data.drop('Club_Coded',1)
holdout_data.head(10)
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
      <th>Gender_Coded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22792</th>
      <td>133</td>
      <td>25.450000</td>
      <td>65.0</td>
      <td>86.0</td>
      <td>73</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20642</th>
      <td>66</td>
      <td>25.216667</td>
      <td>30.0</td>
      <td>57.0</td>
      <td>67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22327</th>
      <td>69</td>
      <td>23.016667</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>72</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62002</th>
      <td>192</td>
      <td>28.916667</td>
      <td>40.0</td>
      <td>19.0</td>
      <td>189</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23323</th>
      <td>278</td>
      <td>27.766667</td>
      <td>50.0</td>
      <td>41.0</td>
      <td>74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>80891</th>
      <td>224</td>
      <td>34.433333</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>257</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56569</th>
      <td>146</td>
      <td>28.583333</td>
      <td>35.0</td>
      <td>124.0</td>
      <td>172</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6477</th>
      <td>290</td>
      <td>30.300000</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>68544</th>
      <td>179</td>
      <td>29.183333</td>
      <td>45.0</td>
      <td>149.0</td>
      <td>216</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49864</th>
      <td>185</td>
      <td>27.066667</td>
      <td>50.0</td>
      <td>92.0</td>
      <td>147</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
holdout_data_feat = holdout_data.drop('Gender_Coded', axis=1)

y_holdout = holdout_data['Gender_Coded']
X_holdout = holdout_data_feat
```


```python
rfc_preds = rfc.predict(X_holdout)

print(confusion_matrix(y_holdout,rfc_preds))
print('\n')
print(classification_report(y_holdout,rfc_preds))
```

    [[3833  916]
     [ 809 3641]]
    
    
                 precision    recall  f1-score   support
    
              0       0.83      0.81      0.82      4749
              1       0.80      0.82      0.81      4450
    
    avg / total       0.81      0.81      0.81      9199
    
    

We do slightly worse on the holdout data, this shows the overfitting we suspected we had after optimising the hyperparameters. To be fair I don't think it's possible to completely separate Males from Females as there is always going to be some overlap in Times, Position etc. About 80% seems about right given how simialr they are.


```python
feature_list = ['Pos', 'Time', 'Age_Cat', 'Total_Runs', 'Run_No.']
importances = list(rfc.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print feature_importances
```

    [('Time', 0.32), ('Pos', 0.26), ('Total_Runs', 0.17), ('Run_No.', 0.15), ('Age_Cat', 0.11)]
    

# K Means Clustering
This is an unsupervised technique. We have labels so we can check<br> the performance of the method.


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



There seems to be two distinct clusters.


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(df1.drop('Gender_Coded',1))

print(confusion_matrix(df1['Gender_Coded'],kmeans.labels_))
print('\n')
print(classification_report(df1['Gender_Coded'],kmeans.labels_))
```

    [[25507  6599]
     [11451 19814]]
    
    
                 precision    recall  f1-score   support
    
              0       0.69      0.79      0.74     32106
              1       0.75      0.63      0.69     31265
    
    avg / total       0.72      0.72      0.71     63371
    
    

# PCA, Principal Component Analysis

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








![png](/img/output_73_1.png)



```python
df_comp = pd.DataFrame(pca.components_,columns=df.drop('Gender_Coded',1).columns.values)

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
```









![png](/img/output_74_1.png)


Seems to say that Position, Time, Age and Total Runs are the important variables.

# Conclusion:
So that's it for another post. We looked at several methods for predicting the gender of a runner from the Postion, Finish Time etc. We optimised the hyperparameters of each. 

The Random Forest method was found to be the most successful. We gave it a lot more data and managed to get a score of **0.84** for Precision, Recall and the F1-score. We then tested the optimised model against holdout data and found scores of **0.81**.

In a future post we will apply this method to predicting membership of a club.

We might also do some feature engineering to see if we can improve the prediction ability of our models.

