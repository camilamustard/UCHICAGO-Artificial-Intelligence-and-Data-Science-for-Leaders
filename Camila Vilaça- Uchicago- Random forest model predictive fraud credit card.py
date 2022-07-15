#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
#df=pd.read_csv('creditcard.csv',nrows=100000)
#df.to_csv('creditsample.csv')
df=pd.read_csv('creditsample.csv',index_col=[0])
df.head(30)
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	0	-1.359807	-0.072781	2.536347	1.378155	-0.338321	0.462388	0.239599	0.098698	0.363787	...	-0.018307	0.277838	-0.110474	0.066928	0.128539	-0.189115	0.133558	-0.021053	149.62	0
1	0	1.191857	0.266151	0.166480	0.448154	0.060018	-0.082361	-0.078803	0.085102	-0.255425	...	-0.225775	-0.638672	0.101288	-0.339846	0.167170	0.125895	-0.008983	0.014724	2.69	0
2	1	-1.358354	-1.340163	1.773209	0.379780	-0.503198	1.800499	0.791461	0.247676	-1.514654	...	0.247998	0.771679	0.909412	-0.689281	-0.327642	-0.139097	-0.055353	-0.059752	378.66	0
3	1	-0.966272	-0.185226	1.792993	-0.863291	-0.010309	1.247203	0.237609	0.377436	-1.387024	...	-0.108300	0.005274	-0.190321	-1.175575	0.647376	-0.221929	0.062723	0.061458	123.50	0
4	2	-1.158233	0.877737	1.548718	0.403034	-0.407193	0.095921	0.592941	-0.270533	0.817739	...	-0.009431	0.798278	-0.137458	0.141267	-0.206010	0.502292	0.219422	0.215153	69.99	0
5	2	-0.425966	0.960523	1.141109	-0.168252	0.420987	-0.029728	0.476201	0.260314	-0.568671	...	-0.208254	-0.559825	-0.026398	-0.371427	-0.232794	0.105915	0.253844	0.081080	3.67	0
6	4	1.229658	0.141004	0.045371	1.202613	0.191881	0.272708	-0.005159	0.081213	0.464960	...	-0.167716	-0.270710	-0.154104	-0.780055	0.750137	-0.257237	0.034507	0.005168	4.99	0
7	7	-0.644269	1.417964	1.074380	-0.492199	0.948934	0.428118	1.120631	-3.807864	0.615375	...	1.943465	-1.015455	0.057504	-0.649709	-0.415267	-0.051634	-1.206921	-1.085339	40.80	0
8	7	-0.894286	0.286157	-0.113192	-0.271526	2.669599	3.721818	0.370145	0.851084	-0.392048	...	-0.073425	-0.268092	-0.204233	1.011592	0.373205	-0.384157	0.011747	0.142404	93.20	0
9	9	-0.338262	1.119593	1.044367	-0.222187	0.499361	-0.246761	0.651583	0.069539	-0.736727	...	-0.246914	-0.633753	-0.120794	-0.385050	-0.069733	0.094199	0.246219	0.083076	3.68	0
10	10	1.449044	-1.176339	0.913860	-1.375667	-1.971383	-0.629152	-1.423236	0.048456	-1.720408	...	-0.009302	0.313894	0.027740	0.500512	0.251367	-0.129478	0.042850	0.016253	7.80	0
11	10	0.384978	0.616109	-0.874300	-0.094019	2.924584	3.317027	0.470455	0.538247	-0.558895	...	0.049924	0.238422	0.009130	0.996710	-0.767315	-0.492208	0.042472	-0.054337	9.99	0
12	10	1.249999	-1.221637	0.383930	-1.234899	-1.485419	-0.753230	-0.689405	-0.227487	-2.094011	...	-0.231809	-0.483285	0.084668	0.392831	0.161135	-0.354990	0.026416	0.042422	121.50	0
13	11	1.069374	0.287722	0.828613	2.712520	-0.178398	0.337544	-0.096717	0.115982	-0.221083	...	-0.036876	0.074412	-0.071407	0.104744	0.548265	0.104094	0.021491	0.021293	27.50	0
14	12	-2.791855	-0.327771	1.641750	1.767473	-0.136588	0.807596	-0.422911	-1.907107	0.755713	...	1.151663	0.222182	1.020586	0.028317	-0.232746	-0.235557	-0.164778	-0.030154	58.80	0
15	12	-0.752417	0.345485	2.057323	-1.468643	-1.158394	-0.077850	-0.608581	0.003603	-0.436167	...	0.499625	1.353650	-0.256573	-0.065084	-0.039124	-0.087086	-0.180998	0.129394	15.99	0
16	12	1.103215	-0.040296	1.267332	1.289091	-0.735997	0.288069	-0.586057	0.189380	0.782333	...	-0.024612	0.196002	0.013802	0.103758	0.364298	-0.382261	0.092809	0.037051	12.99	0
17	13	-0.436905	0.918966	0.924591	-0.727219	0.915679	-0.127867	0.707642	0.087962	-0.665271	...	-0.194796	-0.672638	-0.156858	-0.888386	-0.342413	-0.049027	0.079692	0.131024	0.89	0
18	14	-5.401258	-5.450148	1.186305	1.736239	3.049106	-1.763406	-1.559738	0.160842	1.233090	...	-0.503600	0.984460	2.458589	0.042119	-0.481631	-0.621272	0.392053	0.949594	46.80	0
19	15	1.492936	-1.029346	0.454795	-1.438026	-1.555434	-0.720961	-1.080664	-0.053127	-1.978682	...	-0.177650	-0.175074	0.040002	0.295814	0.332931	-0.220385	0.022298	0.007602	5.00	0
20	16	0.694885	-1.361819	1.029221	0.834159	-1.191209	1.309109	-0.878586	0.445290	-0.446196	...	-0.295583	-0.571955	-0.050881	-0.304215	0.072001	-0.422234	0.086553	0.063499	231.71	0
21	17	0.962496	0.328461	-0.171479	2.109204	1.129566	1.696038	0.107712	0.521502	-1.191311	...	0.143997	0.402492	-0.048508	-1.371866	0.390814	0.199964	0.016371	-0.014605	34.09	0
22	18	1.166616	0.502120	-0.067300	2.261569	0.428804	0.089474	0.241147	0.138082	-0.989162	...	0.018702	-0.061972	-0.103855	-0.370415	0.603200	0.108556	-0.040521	-0.011418	2.28	0
23	18	0.247491	0.277666	1.185471	-0.092603	-1.314394	-0.150116	-0.946365	-1.617935	1.544071	...	1.650180	0.200454	-0.185353	0.423073	0.820591	-0.227632	0.336634	0.250475	22.75	0
24	22	-1.946525	-0.044901	-0.405570	-1.013057	2.941968	2.955053	-0.063063	0.855546	0.049967	...	-0.579526	-0.799229	0.870300	0.983421	0.321201	0.149650	0.707519	0.014600	0.89	0
25	22	-2.074295	-0.121482	1.322021	0.410008	0.295198	-0.959537	0.543985	-0.104627	0.475664	...	-0.403639	-0.227404	0.742435	0.398535	0.249212	0.274404	0.359969	0.243232	26.43	0
26	23	1.173285	0.353498	0.283905	1.133563	-0.172577	-0.916054	0.369025	-0.327260	-0.246651	...	0.067003	0.227812	-0.150487	0.435045	0.724825	-0.337082	0.016368	0.030041	41.88	0
27	23	1.322707	-0.174041	0.434555	0.576038	-0.836758	-0.831083	-0.264905	-0.220982	-1.071425	...	-0.284376	-0.323357	-0.037710	0.347151	0.559639	-0.280158	0.042335	0.028822	16.00	0
28	23	-0.414289	0.905437	1.727453	1.473471	0.007443	-0.200331	0.740228	-0.029247	-0.593392	...	0.077237	0.457331	-0.038500	0.642522	-0.183891	-0.277464	0.182687	0.152665	33.00	0
29	23	1.059387	-0.175319	1.266130	1.186110	-0.786002	0.578435	-0.767084	0.401046	0.699500	...	0.013676	0.213734	0.014462	0.002951	0.294638	-0.395070	0.081461	0.024220	12.99	0
30 rows × 31 columns

get_ipython().set_next_input('First question to ask in Anomaly Detection is: "Do we have labeled data? In other words, do we have any previous observations we detected as fraud in our data');get_ipython().run_line_magic('pinfo', 'data')
get_ipython().set_next_input("If yes, what's next");get_ipython().run_line_magic('pinfo', 'next')

get_ipython().set_next_input("If no, what's next");get_ipython().run_line_magic('pinfo', 'next')

df['Class'].describe()
count    100000.00000
mean          0.00223
std           0.04717
min           0.00000
25%           0.00000
50%           0.00000
75%           0.00000
max           1.00000
Name: Class, dtype: float64
df['Class'].value_counts()
0    99777
1      223
Name: Class, dtype: int64
# Explore the features available in your dataframe
print(df.info())

# Count the occurrences of fraud and no fraud and print them
occ = df['Class'].value_counts()
print(occ)

# Print the ratio of fraud cases
print(occ / len(df.index))
<class 'pandas.core.frame.DataFrame'>
Int64Index: 100000 entries, 0 to 99999
Data columns (total 31 columns):
Time      100000 non-null int64
V1        100000 non-null float64
V2        100000 non-null float64
V3        100000 non-null float64
V4        100000 non-null float64
V5        100000 non-null float64
V6        100000 non-null float64
V7        100000 non-null float64
V8        100000 non-null float64
V9        100000 non-null float64
V10       100000 non-null float64
V11       100000 non-null float64
V12       100000 non-null float64
V13       100000 non-null float64
V14       100000 non-null float64
V15       100000 non-null float64
V16       100000 non-null float64
V17       100000 non-null float64
V18       100000 non-null float64
V19       100000 non-null float64
V20       100000 non-null float64
V21       100000 non-null float64
V22       100000 non-null float64
V23       100000 non-null float64
V24       100000 non-null float64
V25       100000 non-null float64
V26       100000 non-null float64
V27       100000 non-null float64
V28       100000 non-null float64
Amount    100000 non-null float64
Class     100000 non-null int64
dtypes: float64(29), int64(2)
memory usage: 24.4 MB
None
0    99777
1      223
Name: Class, dtype: int64
0    0.99777
1    0.00223
Name: Class, dtype: float64
# Define a function to create a scatter plot of our data and labels
import matplotlib.pyplot as plt  
import pandas as pd  
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
def plot_data(X, y):
	plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
	plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='y')
	plt.legend()
	return plt.show()

# Create X and y from our above defined function
from sklearn.model_selection import train_test_split
y = df.iloc[:,-1].values
X=df.iloc[:, :-1].values


# Plot our data by running our plot data function on X and y
plot_data(X, y)

Yes we have some labels
get_ipython().set_next_input('223 data labels -out of 100K- sufficient enough to run a classification learning algorithm');get_ipython().run_line_magic('pinfo', 'algorithm')
'''
We applied an upsampling technique to balance minory class in our data
Almost doubled the size of our data by generating generic samples
'''
from imblearn.over_sampling import SMOTE

#https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html


# Define the resampling method
method = SMOTE(kind='regular')

# Create the resampled feature set
X_resampled, y_resampled = method.fit_sample(X, y)

# Plot the resampled data
plot_data(X_resampled, y_resampled)

Conventional Rule-Based Statistical Methods
# Run a groupby command on our labels and obtain the mean for each feature
df.groupby('Class').mean()

# Implement a rule for stating which cases are flagged as fraud
df['flag_as_fraud'] = np.where(np.logical_and(df['V1'] < -3, df['V3'] < -5), 1, 0)

# Create a crosstab of flagged fraud cases versus the actual fraud cases
print(pd.crosstab(df.Class, df.flag_as_fraud, rownames=['Actual Fraud'], colnames=['Flagged Fraud']))
Flagged Fraud      0    1
Actual Fraud             
0              99374  403
1                134   89
a=df.groupby('Class').mean()
a
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	flag_as_fraud
Class																					
0	94838.202258	0.008258	-0.006271	0.012171	-0.007860	0.005453	0.002419	0.009637	-0.000987	0.004467	...	-0.001235	-0.000024	0.000070	0.000182	-0.000072	-0.000089	-0.000295	-0.000131	88.291022	0.004312
1	80746.806911	-4.771948	3.623778	-7.033281	4.542029	-3.151225	-1.397737	-5.568731	0.570636	-2.581123	...	0.713588	0.014049	-0.040308	-0.105130	0.041449	0.051648	0.170575	0.075667	122.211321	0.345528
2 rows × 31 columns

Machine Learning on Imbalanced Data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y = df.iloc[:,-1].values
X=df.iloc[:, :-1].values
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit a logistic regression model to our data
model = LogisticRegression()
model.fit(X_train, y_train)

# Obtain model predictions
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)
# Print the classifcation report and confusion matrix
#print('Classification report:\n', metrics.classification_report(y_test, predicted))
conf_mat = metrics.confusion_matrix(y_true=y_test, y_pred=predicted)
print(roc_auc_score(y_test, probs[:,1]))
print(classification_report(y_test, predicted))
print('Confusion matrix:\n', conf_mat)
Users(/utkupamuksuz/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433:, FutureWarning:, Default, solver, will, be, changed, to, 'lbfgs', in, 0.22., Specify, a, solver, to, silence, this, warning.)
  FutureWarning)
0.9725126081293209
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     29941
           1       0.76      0.64      0.70        59

   micro avg       1.00      1.00      1.00     30000
   macro avg       0.88      0.82      0.85     30000
weighted avg       1.00      1.00      1.00     30000

Confusion matrix:
 [[29929    12]
 [   21    38]]
import numpy as np
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
<matplotlib.axes._subplots.AxesSubplot at 0x11d22aa50>

Machine Learning with improved balance
# This is the pipeline module we need for this from imblearn
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
#https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html#smote-adasyn
# Define which resampling method and which ML model to use in the pipeline
resampling = SMOTE(kind='borderline2')
model = LogisticRegression()

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])
# Split your data X and y, into a training and a test set and fit the pipeline onto the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
Users(/utkupamuksuz/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433:, FutureWarning:, Default, solver, will, be, changed, to, 'lbfgs', in, 0.22., Specify, a, solver, to, silence, this, warning.)
  FutureWarning)
Pipeline(memory=None,
     steps=[('SMOTE', SMOTE(k_neighbors=5, kind='borderline-2', m_neighbors=10, n_jobs=1,
   out_step='deprecated', random_state=None, ratio=None,
   sampling_strategy='auto', svm_estimator='deprecated')), ('Logistic Regression', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False))])
predicted = model.predict(X_test)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')


'''
Logistic Regression Results for Resampled Data (~200K samples instead of 100K)

'''
<matplotlib.axes._subplots.AxesSubplot at 0x13faab450>

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
y_true=y_test
y_pred=predicted
labels = [0,1]
cm = confusion_matrix(y_true, y_pred, labels)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

'''
Random Forest Results for Resampled Data (~200K samples instead of 100K)

'''
Users(/utkupamuksuz/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246:, FutureWarning:, The, default, value, of, n_estimators, will, change, from, 10, in, version, 0.20, to, 100, in, 0.22.)
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
<matplotlib.axes._subplots.AxesSubplot at 0x140f60250>

from sklearn.svm import SVC
model = SVC(C=0.5, kernel='linear')
model.fit(X_train, y_train)
predicted = model.predict(X_test)
y_true=y_test
y_pred=predicted
labels = [0,1]
cm = confusion_matrix(y_true, y_pred, labels)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

'''
Support Vector Machine results for Resampled Data (~200K samples instead of 100K)
'''
<matplotlib.axes._subplots.AxesSubplot at 0x141126dd0>

from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
model = DecisionTreeClassifier(max_depth=6, random_state=0)
from sklearn.svm import SVC
model = SVC(C=0.5, kernel='linear')
model.fit(X_train, y_train)
predicted = model.predict(X_test)
y_true=y_test
y_pred=predicted
labels = [0,1]
cm = confusion_matrix(y_true, y_pred, labels)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

'''
Decision Tree Classifier results for Resampled Data (~200K samples instead of 100K)



'''
<matplotlib.axes._subplots.AxesSubplot at 0x11d2bf710>

Calculate Precision and Recall for each model
Compare the Confusion Matrices for each model and determine which model will be a better fit for this business problem
Linear & Non-linear Decision Boundaries
import base64, io, IPython
from PIL import Image as PILImage

image = PILImage.open("monns.png")

output = io.BytesIO()
image.save(output, format='PNG')
encoded_string = base64.b64encode(output.getvalue()).decode()

html = '<img src="data:monns.png;base64,{}"/>'.format(encoded_string)
IPython.display.HTML(html)

import base64, io, IPython
from PIL import Image as PILImage

image = PILImage.open("irs.png")

output = io.BytesIO()
image.save(output, format='PNG')
encoded_string = base64.b64encode(output.getvalue()).decode()

html = '<img src="data:irs.png;base64,{}"/>'.format(encoded_string)
IPython.display.HTML(html)

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(X_resampled)
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance with 2 components: pca
pca = PCA(n_components=2)

pca_features = pca.fit_transform(X_resampled)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]
plt.scatter(xs, ys)
plt.axis()
plt.show()

orthogonal

Based on decision boundary shapes above, we think either decision tree or random forest decision boundaries would do a good work
We may be biased though since precision/recall and confusion matrix results also showed the strength of Random Forest
# Count the total number of observations from the length of y
total_obs = len(y)

# Count the total number of non-fraudulent observations 
non_fraud = [i for i in y if i == 0]
count_non_fraud = non_fraud.count(0)

# Calculate the percentage of non fraud observations in the dataset
percentage = (float(count_non_fraud)/float(total_obs)) * 100

# Print the percentage: this is our "natural accuracy" by doing nothing
print(percentage)
99.777
With Random Forests you create random subsets of the features and build smaller trees using these subsets. Afterwards, the algorithm combines the subtrees of subsamples of features, and averages the results.

# Import the packages to get the different performance metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Obtain the predictions from our random forest model 
predicted = model.predict(X_test)

# Predict probabilities
probs = model.predict_proba(X_test)

# Print the ROC curve, classification report and confusion matrix
print(roc_auc_score(y_test, probs[:,1]))
#print(classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))
Users(/utkupamuksuz/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246:, FutureWarning:, The, default, value, of, n_estimators, will, change, from, 10, in, version, 0.20, to, 100, in, 0.22.)
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
0.9829548960413107
[[29938     3]
 [    3    56]]
We monitor Random Forest algorithm provided less accuracy than natural guess by using a general performance accuracy metric. However, it seems more reliable to us and performs significantly better than a traditional rule-based approach
 

