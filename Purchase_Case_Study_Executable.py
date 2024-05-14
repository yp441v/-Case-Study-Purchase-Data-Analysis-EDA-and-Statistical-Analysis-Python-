#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:c131f3d0-5b23-46b7-907c-1d2c8cc843de.png)

# Problem Statement:
# 
# You have been provided purchase data for various customers across a vertical. You need to apply your learnings from Data Manipulation, Data Visualization, and statistical analysis to come up with actionable insights about the data.
# 
# 
# Tasks To Be Performed:
# 
# 1. Perform a Detailed EDA for the Data with inferences from each of the actions.
# 2. Using Statistical Analysis, find out statistical evidence for the following:
#     - a. It was observed that the average purchase made by the Men of the age 18-25 was 10000. Is it still the same?
#     - b. It was observed that the percentage of women of the age that spend more than 10000 was 35%. Is it still the same?
#     - c. Is the average purchase made by men and women of thellge 18-25 same?
#     - d. Is the percentage of men who have spent more than 10000 the same for the ages 18-25 and 26-35?
#     - e. Is Purchase independent of Product_Category_1?

# ## 1.  Exploratory Data Analysis (EDA):
# 
# Steps we will follow: 
# 
# 1. Data Collection
# 2. Data Cleant
# 3. Descriptive Analysis3cs
# 4. Univariate Analysis
# 5. Bivariate Analysis
# 6. Multivariate Analysis
# 7. Data Visualization
# 8. Feature Engineering
# 9. Outlier Detection
# 10. Missing Value Imputation
# 11. Dimensionality Reduction
# 12. Correlation Analysis

# In[1]:


# Import Neccesarry Liabraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import data 

data = pd.read_csv('Purchase.csv')


# In[2]:


data.head(3)


# In[5]:


data.info()


# In[20]:


list(data.columns)


# In[ ]:





# #### label Encoding :
#                         Label Encoding assigns numerical labels to categories based on their alphabetical order. For example, if you have categories like "A," "B," and "C," they might be encoded as 0, 1, and 2 respectively.
# 
# 
# 

# In[6]:


# import 
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[7]:


# label Encoding

data['User_ID'] = le.fit_transform(data['User_ID'])

data['Product_ID'] = le.fit_transform(data['Product_ID'])

data['Gender'] = le.fit_transform(data['Gender'])

data['Age'] = le.fit_transform(data['Age'])

data['City_Category'] = le.fit_transform(data['City_Category'])


# In[35]:


data['Gender'].unique() 


# In[36]:


print('User_ID', le.classes_)


# In[37]:


print('City_Category',le.classes_)


# In[40]:


data['User_ID'].unique() 


# In[41]:


data['Stay_In_Current_City_Years'].unique()



# In[8]:


# changing 4+  to 4

data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].replace('4+' , '4')


# now we will convert datatype from object to integer using typecasting 

data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].astype(int)



# In[9]:


data.head()


# In[10]:


#  Checking fro null  values 

data.isna().sum()


# This seams to be  big number : Generally we do not drop that big numbers ;
# However we are applying statistical Analysis on this so we only need smaller sample. 
# 
# 
#     **so we will drop  rather that data Imputation**

# In[11]:


data = data.dropna()

data.info()


# In[55]:


# descriptive stats summary 

data.describe().T


# In[57]:


data.head()


# ## Visualisation

# In[59]:


# ignore warnings 

import warnings 

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[61]:


sns.catplot(data=data, x='Gender', y='User_ID', kind='bar')


#  Amongst all the purshases made, majority of the purchases were made by Women/Female.


# In[ ]:





# In[70]:


sns.catplot(data=data, x='Age', y='User_ID', hue='Gender', kind='bar', height=6)



# In[71]:


data['Age'].unique()


# In[74]:


sns.catplot(data=data, x='Marital_Status', y='User_ID', hue='Gender', kind='bar', height=6)


# In[ ]:


le.classes_


# ### count plot 

# ### checking purchages with respect to product catagories 

# In[112]:


sns.countplot(data=data,x=data['Product_Category_1'])


# In[113]:


data['Product_Category_2'].unique()


# In[75]:


sns.countplot(data=data,x=data['Product_Category_2'])


# In[76]:


sns.countplot(data=data,x=data['Product_Category_3'])


# In[78]:


data.head()


# ### Dist Plot == talks about desity of the data .. 

# In[81]:


sns.distplot(data['Purchase'])


# ### data is not normally distributed Rather it varies in different intervals 

# In[ ]:





# ## Correlation 

# In[82]:


#checking the correlation of the features

corr_spearman = data.corr(numeric_only = True)
corr_pearson = data.corr(method='pearson')
corr_kendall = data.corr(method='kendall')


# In[ ]:





# In[83]:


data.corr()


# In[84]:


corr_spearman


# In[85]:


corr_pearson


# In[88]:


#spearman correlation


plt.figure(figsize=(15,5))
sns.heatmap(corr_spearman, annot = True, cmap='BuPu')


# In[ ]:





# ## 2. Statistical Analysis
# 
# 
# 
# | **Statistical Test**     | **Python Libraries & Functions**                               | **R Functions**                  |
# |--------------------------|-----------------------------------------------------------------|----------------------------------|
# | T-tests                  | `scipy.stats.ttest_1samp`, `scipy.stats.ttest_ind`, `scipy.stats.ttest_rel` | `t.test()`                       |
# | ANOVA                    | `scipy.stats.f_oneway`                                          | `aov()`                          |
# | Chi-squared test         | `scipy.stats.chi2_contingency`                                  | `chisq.test()`                   |
# | Linear Regression        | `statsmodels.api.OLS`, `sklearn.linear_model.LinearRegression`    | `lm()`                           |
# | Logistic Regression      | `statsmodels.api.Logit`, `sklearn.linear_model.LogisticRegression` | `glm()`                          |
# | Correlation              | `numpy.corrcoef`, `scipy.stats.pearsonr`, `scipy.stats.spearmanr` | `cor()`, `cor.test()`            |
# | Non-parametric tests     | `scipy.stats.wilcoxon`, `scipy.stats.mannwhitneyu`                 | `wilcox.test()`, `wilcox.test()` |
# 

# #### a. It was observed that the average purchase made by the Men of the age 18-25 was 10000. Is it still the same?
# 
# Therefore assumptions : 
# 
# - null hypothesis - The mean is 10000
# - alternate hypothesis - The mean is not 10000

# In[12]:


new_data = data.loc[(data['Age'] == 1) & (data['Gender'] == 1)]
new_data


# In[13]:


new_data.shape 


# In[ ]:





# In[15]:


original_dataSet = pd.read_csv('Purchase.csv')


# In[16]:


original_dataSet.head()


# In[19]:


print(original_dataSet['Age'].unique())

print(original_dataSet['Gender'].unique())


# In[36]:


from scipy.stats import ttest_1samp
new_data
sample_size = 1000
sample= new_data.sample(sample_size, random_state = 0)
sample_mean = sample['Purchase'].mean()
print('sample_mean : ' , sample_mean)

pop_mean = 100000
print('pop_mean: ', pop_mean)

t_stat , p_value = ttest_1samp(sample['Purchase'],pop_mean )
print('t_stat:', t_stat )
print('p_value:', p_value )

alpha =  0.05
print('alpha:', alpha )


if p_value < alpha:
    print('Reject Null Hyp, the sample mean is significantly diff')
else:
    print('Fail to Reject/ accept null Hyp ')


# #### Comclusion:  P-value is less than 0.05, reject the null hypothesis. therefore, the mean purchase for men aged 18-25 is not 10000.

# 

# ### Two Sample test for Means
# #### Is the average purchase made by men and women of the age 18-25 same?

# null hypothesis =  average purchase are equal
# alternat hypothesis = average purchase are not equal
# 
# 

# In[49]:


from scipy.stats import ttest_ind


new_men = data.loc[(data['Age'] == 1) & (data['Gender'] == 1)]
new_women  = data.loc[(data['Age'] == 1) & (data['Gender'] == 0)]

#creating samples
sample_men = new_men.sample(500, random_state = 0 )
sample_women = new_women.sample(500, random_state = 0 )



# check for varience : If we have unequal variences we go for Independent T-test
print('sample_men_varience :', sample_men['Purchase'].var() )
print('sample_women_varience :', sample_women['Purchase'].var() )



t_stats , p_value = ttest_ind(sample_men['Purchase'],sample_women['Purchase'], equal_var = False)
print('t_stat:', t_stat )
print('p_value:', p_value )


if p_value < alpha:
    print('Reject Null Hyp, the sample mean is significantly diff')
else:
    print('Fail to Reject/ accept null Hyp ')


# #### Conclusion : therefore the average purchases are not the same 

# In[ ]:





# ### one Sample test for Proportion
# Let's Suppose, It was observed that the percentage of women that spend more than 10000 was 35%. Is it still the same?

# null hypothesis = Proportion is 35%.
# alternate hypothesis = Proportion is not 35%.

# In[97]:


data1= data.loc[(data['Purchase'] > 10000)]
# we can't extract only women data because we need to compare it with men so that we can  get the prportion 

from statsmodels.stats.proportion import proportions_ztest


# from proportion z test we need count, nobs , observed value 
# therefore 

# count of women in the data: 
count = data1['Gender'].value_counts()[0]
print('count:' , count)

# number of obersevation: entire length of the gender column 
nobs = len(data1['Gender'])
print('nobs:', nobs)

# hypothesis value/observed p value 
p0 = 0.35

z_stats , p_value = proportions_ztest(count= count,
                                     nobs =nobs,
                                     value = p0,
                                     alternative = 'two-sided',
                                     prop_var = False) # as  we  do not have variance  
print('z_stats:', z_stats )
print('p_value:', p_value )   

if p_value < alpha:
    print('Reject Null Hyp, the sample mean is significantly diff')
else:
    print('Fail to Reject/ accept null Hyp ')


# In[98]:


data1['Gender'].value_counts()[0]


# In[99]:


data1['Gender'].value_counts()


# 
# 
# ### Two Sample test for Proportion
# Is the percentage of men who have spend more than 10000 same for the ages 18-25 and 26-35

# In[101]:


# since we have  to calculate the percentage we will go with Ztest for proportion 

data_age1 = data.loc[(data['Age'] == 1) & (data['Purchase'] > 10000)]
data_age2 = data.loc[(data['Age'] == 2) & (data['Purchase']> 10000)]


data_age1_sample = data_age1.sample(1000, random_state = 0)
data_age2_sample = data_age2.sample(1000, random_state = 0)

count = [(data_age1_sample['Gender'] == 1).sum(), (data_age2_sample['Gender'] == 1).sum()]
print('count:' , count)

nobs = [(len(data_age1_sample)), (len(data_age2_sample))]
print('nobs:' , nobs)


from statsmodels.stats.proportion import proportions_ztest


zstats, p_value = proportions_ztest(count = count,
                                   nobs = nobs, 
                                   value = 0,
                                   alternative= 'two-sided',
                                   prop_var = False)

print('z_stats:', z_stats )
print('p_value:', p_value )   

if p_value < alpha:
    print('Reject Null Hyp, the sample mean is significantly diff')
else:
    print('Fail to Reject/ accept null Hyp ')


#  p value is more than 0.05, cannot reject the null hypthesis.
#  
#  therefore, Percentage of the men in the age groups is same

# In[ ]:





# In[ ]:





# In[ ]:




