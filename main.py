import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import preprocessing

data_t = pd.read_csv('transactions_final_ex.csv')
data_c = pd.read_csv('Customers_final_ex.csv')
print(data_t.info())
print(data_c.info())

data = pd.merge(data_t, data_c, on='Customer_id', how='outer')
print(data.info())

""" 
We have 1000 records in total, 33 of them have missing data. Since this is less than 5% of the total data,
we can remove the incomplete records. transaction_id and Customer_id are identifiers that are not potential predictors.
payment_t, count_dev_is_shop, product_type, Day_of_week, Preferred_col, and Gender are categorical variables. 
num_s and num_dev are categorical ordinal variables. All others are numeric (except the target).

Let's verify this by calculating the values for each of the predictors:
"""

data = data.dropna()

for col in data.drop(columns=['transaction_id', 'Customer_id', 'Fraudulent_trans']).columns:
    print()
    print(data[col].value_counts())

"""
In addition to the typical male and female genders, we see that for some clients the gender is unknown, 
or d is indicated.
I would not transform the data, perhaps it is the scammers who mostly do not indicate the gender, 
or choose the third option.
"""

identification_cols = ['transaction_id', 'Customer_id']
categorical_cols = ['payment_t', 'count_dev_is_shop', 'product_type', 'Day_of_week', 'Preferred_col', 'Gender']
categorical_ordered_cols = ['num_s', 'num_dev']
numerical_cols = ['Dist', 'basket', 'age']
responce_var = 'Fraudulent'

# Adding a variable that is True when the transaction is fraudulent for convenience
data['Fraudulent'] = data['Fraudulent_trans']==1

sns.scatterplot(x=data['Dist'], y=data['basket'], hue=data['Fraudulent'])
plt.show()

"""
For large purchase values (basket) and large differences between the delivery address and the home address (Dist), 
the risk of a transaction being fraudulent increases. At the same time, distance is more likely to be 
a better predictor, because the difference in distance is more clearly correlated with the target variable.

The first graph also shows the correlation of two predictors with fraudulent transactions. Let's try to take this 
interaction into account by adding a new variable to the product of these two predictors.

We also notice a large number of outliers, which can negatively affect the quality of our future model. And the scale of
these variables, their weight in the model will be greater due to the larger scale, so let's try to normalize the data 
to reduce this bias.

But before the transformation, let's also build graphs for categorical variables.
"""

fig, ax = plt.subplots(1,len(numerical_cols),figsize=(len(numerical_cols)*4,5))
for col in numerical_cols:
    sns.boxplot(ax=ax[numerical_cols.index(col)],x=responce_var, y=col,
            data=data,
            hue=responce_var,
            legend=False,
            showfliers=True)
    ax[numerical_cols.index(col)].set_title(f'{col} by Fraudulent_trans')
sns.despine(offset=10, trim=True)
plt.show()

fig, ax = plt.subplots(2, int(len(categorical_cols + categorical_ordered_cols) / 2), figsize=(len(categorical_cols + categorical_ordered_cols)*1.5, 10))
ax = ax.flatten()

for i, col in enumerate(categorical_cols + categorical_ordered_cols):
    sns.countplot(ax=ax[i], x=col, hue='Fraudulent', data=data)
    ax[i].set_title(f'{col} \nby Fraudulent_trans')
    ax[i].tick_params(axis='x', rotation=30)

for i in range(len(categorical_cols + categorical_ordered_cols), len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout()
plt.show()

"""
We can see that transactions made from abroad are more risky. It is also typical for a larger number of
orders and a larger number of devices used to have a higher risk of fraud. Other indicators do not express such an
obvious dependence. But we will check this by calculation later.
"""

data['Dist&basket'] = data['Dist'] * data['basket']
numerical_cols = numerical_cols + ['Dist&basket']

scaler = preprocessing.StandardScaler().fit(data[numerical_cols + categorical_ordered_cols])

numerical_cols_scaled = scaler.transform(data[numerical_cols + categorical_ordered_cols])
data[numerical_cols + categorical_ordered_cols] = numerical_cols_scaled

# Visualising the distribution of the numerical features after the transformation.

fig, ax = plt.subplots(1,len(numerical_cols),figsize=(len(numerical_cols)*5,5))
for col in numerical_cols:
    sns.boxplot(ax=ax[numerical_cols.index(col)],x=responce_var, y=col,
            data=data,
            hue=responce_var,
            legend=False,
            showfliers=True)
    ax[numerical_cols.index(col)].set_title(f'{col} by Fraudulent_trans')
sns.despine(offset=10, trim=True)
plt.tight_layout()
plt.show()

correlation_matrix = data[numerical_cols + [responce_var]].corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
plt.tight_layout()
plt.show()

"""
We can see that there is a positive correlation between the target variable and all numeric variables except age. 
Let's test separately the hypotheses that increasing the parameter also increases the probability that the transaction 
is fraudulent:
H0: the average value of the parameter for fraudulent transactions is NOT MORE than for non-fraudulent ones;
H1: the average value of the parameter for fraudulent transactions is MORE than for non-fraudulent ones.

For age, the hypotheses will be reversed (LESS)
We will conduct a right-tailed t-test, since we do not know the variance.
"""

data_1 = data[data['Fraudulent'] == 1]
data_0 = data[data['Fraudulent'] == 0]


for param in numerical_cols:
    param_mean_1 = data_1[param].mean()
    param_mean_2 = data_0[param].mean()

    param_var_1 = data_1[param].var(ddof=1)
    param_var_2 = data_0[param].var(ddof=1)

    n_1 = len(data_1[param])
    n_2 = len(data_0[param])

    se1, se2 = param_var_1/n_1, param_var_2/n_2
    se = np.sqrt(se1 + se2)

    t_stat = (param_mean_1 - param_mean_2) / se

    df = (se1 + se2)**2 / ((se1**2 / (n_1 - 1)) + (se2**2 / (n_2 - 1)))

    if param != 'age':
        p_value = 1 - stats.t.cdf(abs(t_stat), df=df)
    else:
        p_value = stats.t.cdf(abs(t_stat), df=df)

    print(f"{param} \n t-statistic: {t_stat}, p-value: {p_value}\n\n")

"""
All parameters are statistically significant at the 5% confidence level except age.
Now we will perform the same calculation for categorical variables using chi-square statistic.
H0: Categorical variable and target variable are not related.
H1: There is a relationship between categorical variable and Fraudulent.
"""

for param in categorical_cols + categorical_ordered_cols:

    contingency_table = pd.crosstab(data[param], data['Fraudulent'])

    observed = contingency_table.values

    row_sums = observed.sum(axis=1).reshape(-1, 1)
    col_sums = observed.sum(axis=0).reshape(1, -1)

    total = observed.sum()
    expected = (row_sums * col_sums) / total

    chi2_stat = np.sum((observed - expected)**2 / expected)

    df = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    print(f"{param} \n χ² statistic: {chi2_stat:.3f}, p-value: {p_value:.3f}\n\n")

"""
As we can see, count_dev_is_shop, num_s, num_dev are indeed statistically significant characteristics for identifying 
fraudulent transactions, as is payment_t, which was not so obvious from visual observations.
"""

exclude_from_prediction_cols = ['product_type', 'Day_of_week', 'Preferred_col', 'Gender', 'age']

"""
Before creating the model, we will check the distribution of the target variable to finally decide on the choice of 
model and metrics that we will use to evaluate the effectiveness.
"""

sns.countplot(x=responce_var, data=data)
plt.title('Distribution of the Responce Variable')
plt.show()

print(data[responce_var].value_counts(normalize=True))

"""
The target variable, although stored as a float, is essentially a binary classification. Logistic regression would be a
reasonable choice for the model. We see that the data is unbalanced, only 12.6% of transactions were fraudulent.
An accuracy score would not be enough.
"""

# Defining the predictors and the target variable
pred_cols = [var for var in list(data.columns) if var not in identification_cols + exclude_from_prediction_cols + [responce_var] + ['Fraudulent_trans']]
X = data[pred_cols]
y = data[responce_var]

# Dividing the data into train and test in a ratio of 80/20
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Checking whether the distribution of shares well represents the main sample
sns.countplot(x=y_train)
plt.title('Distribution of the Train Data')
plt.show()
print(y_train.value_counts(normalize=True))
print()

sns.countplot(x=y_test)
plt.title('Distribution of the Test Data')
plt.show()
print(y_test.value_counts(normalize=True))

# Training the model with the training data
clf = LogisticRegression(random_state=42).fit(X_train, y_train)

# To evaluate the model's performance, we make predictions on training and test data.
results_train = clf.predict(X_train)
results_test = clf.predict(X_test)

# Building a bar plot to visualize the results
train_scores = [accuracy_score(y_train, results_train), precision_score(y_train,results_train), recall_score(y_train,results_train)]
test_scores = [accuracy_score(y_test, results_test), precision_score(y_test,results_test), recall_score(y_test,results_test)]

metrics = ['Accuracy', 'Precision', 'Recall']

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, train_scores, width, label='Train 80%')
bars2 = ax.bar(x + width/2, test_scores, width, label='Test 20%')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Train and Test Scores')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

ax.bar_label(bars1, fmt='%.2f', padding=3)
ax.bar_label(bars2, fmt='%.2f', padding=3)

plt.tight_layout()
plt.show()

"""
Accuracy has a high score, is generally less informative, as the data are highly unbalanced.
Precision is important to avoid false positives.
Recall is important to minimize missing fraudulent transactions. We see that the estimated ability of the
model to predict fraudulent transactions is not very high. But this is the result of training on only 80% of the data.
Let's try to improve these scores by training the model on all the data and doing cross-validation to compare the results.
"""

# Training the model with all the data
clf = LogisticRegression(random_state=42).fit(X, y)

# Saving the names of the predictors and their coefficients (weights) in a separate dictionary
feature_names = X.columns
coefficients = clf.coef_[0]
coef_dict = dict(zip(feature_names, coefficients))

# Sorting the dictionary by the modulus values of the coefficients
sorted_coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True)}

# Building a barplot showing the weight of each predictor
coefs_df = pd.DataFrame(list(sorted_coef_dict.items()), columns=['Predictor', 'Weight'])

plt.figure(figsize=(10, 5))
sns.barplot(data=coefs_df, x='Predictor', y='Weight', hue='Predictor')

plt.xticks(rotation=30, ha='right')
plt.title('Logistic Regression Coefficients')
plt.xlabel('Predictors')
plt.ylabel('Weight')

plt.tight_layout()
plt.show()

# Adding predictors to the model one by one and saving the estimated metrics after adding each one.
recall, precision, accuracy = [], [], []
predictors, recall_predictors, precision_predictors, accom_predictors = [], [], [], []

for i in np.arange(1, len(sorted_coef_dict)+1):
    X = data[list(sorted_coef_dict.keys())[:i]]
    log_model = LogisticRegression()
    res = cross_validate(log_model, X, y, cv=10, scoring=['recall','precision','accuracy'])
    recall.append(res['test_recall'].mean())
    precision.append(res['test_precision'].mean())
    accuracy.append(res['test_accuracy'].mean())
    predictors.append('+ ' + X.columns[i-1])
    if i == 1:
        predictors, recall_predictors, precision_predictors, accom_predictors = [X.columns[i-1]], [X.columns[i-1]], [X.columns[i-1]], [X.columns[i-1]]
    else:
        if recall[-1] > recall[-2]:
            recall_predictors.append(X.columns[i-1])
        if precision[-1] > precision[-2]:
            precision_predictors.append(X.columns[i-1])
        if recall[-1] >= recall[-2]:
            accom_predictors.append(X.columns[i-1])

# Building a graph of evaluation metrics
fig, ax = plt.subplots(figsize=(15,5))
plt.grid()
plt.plot(predictors, accuracy, label='accuracy')
plt.plot(predictors, precision, label='precision')
plt.plot(predictors, recall, label='recall')
plt.title('Metrics after Adding Each Parameter One by One')
plt.xticks(rotation=30)
plt.legend()
plt.show()

"""
This graph does not give a clear picture of how the metrics will change, because if you change the order of adding 
predictors, the values may vary due to existing correlations, but it should help to determine the most helpful predictors.

As we can see, the best predictor that helps to increase the sensitivity of the model is count_dev_is_shop — a parameter 
that determines whether the country where the device from which the transaction took place is located is identical to 
the country of the store. num_s also positively affects this metric. If it is important for us to have a model that is 
as sensitive as possible to potential fraudulent activities, we can use only those predictors that positively affect 
this metric. For example, if we can implement additional verification for suspicious accounts and this will not 
significantly worsen the user experience.

If it is important for us to increase the accuracy as much as possible in order to disturb the smallest number of users 
that the model mistakenly assigned to the risk group, then we can take only predictors that positively affect the 
accuracy parameter.

But, if we can afford to sacrifice 2 percent of sensitivity to avoid false positives and not force users to go through 
unnecessary checks, we can make the model quite accurate (76%) by taking all predictors that DO NOT directly reduce the 
sensitivity level as they are added to the model. (Remember that the sensitivity will still decrease slightly due to the 
correlation of the predictors. Hence the 2 percent difference).

Let's train all three models and visualize their metrics for comparison.
"""

X_1 = data[recall_predictors]
X_2 = data[precision_predictors]
X_3 = data[accom_predictors]

log_model_1 = LogisticRegression()
res_1 = cross_validate(log_model_1, X_1, y, cv=10, scoring=['recall','precision','accuracy'])

log_model_2 = LogisticRegression()
res_2 = cross_validate(log_model_2, X_2, y, cv=10, scoring=['recall','precision','accuracy'])

log_model_3 = LogisticRegression()
res_3 = cross_validate(log_model_3, X_3, y, cv=10, scoring=['recall','precision','accuracy'])

results_1 = [res_1['test_accuracy'].mean(), res_1['test_precision'].mean(), res_1['test_recall'].mean()]
results_2 = [res_2['test_accuracy'].mean(), res_2['test_precision'].mean(), res_2['test_recall'].mean()]
results_3 = [res_3['test_accuracy'].mean(), res_3['test_precision'].mean(), res_3['test_recall'].mean()]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize = (8, 6))
bars1 = ax.bar(x - width/2, results_1, width/2, label=' + '.join(recall_predictors))
bars2 = ax.bar(x, results_2, width/2, label=' + '.join(precision_predictors))
bars3 = ax.bar(x + width/2, results_3, width/2, label=' + '.join(accom_predictors))

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Two Models')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

ax.bar_label(bars1, fmt='%.2f', padding=3)
ax.bar_label(bars2, fmt='%.2f', padding=3)
ax.bar_label(bars3, fmt='%.2f', padding=3)

plt.grid(axis='y')
plt.legend(loc='upper left', bbox_to_anchor=(0, -0.1))
plt.show()

