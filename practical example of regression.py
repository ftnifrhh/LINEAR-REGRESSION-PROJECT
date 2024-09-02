# practical / real life example of linear regression
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.linear_model import LinearRegression


# -------- open the data ----------------
raw_data_path = r"C:\Users\Fatini\OneDrive - Universiti Malaya\Documents\DATA ANALYTICS COURSES\PHYTON\regression analysis\1.04.+Real-life+example.csv"
raw_data = pd.read_csv(raw_data_path)


#------------------------STEP 1----------------------------------------------
# check whether your data is clean or not
print(raw_data.describe(include='all')) # it means that we can see the descriptive of all the data

# after detecting process, delete unuseful column
data=raw_data.drop( ['Model'],axis=1)
print(data.describe() )

# handling missing value data
print( data.isnull().sum() )  #sum = summation of true # true stands for missing # false stands for available and not missing

# delete the missing values (not the column of it) that have missing value ( if the sum of the data is less than 5% of overall data)
data_no_missingvalues=data.dropna(axis=0)

# exploring the pdf of all variables, in this way we will see the outliers too
#--- they should be in normal distribution to make it easy to do regression. if not, we may remove the top 1% observation
#----pdf of price : we corrext it by deleting the outliers
print(sns.displot(data_no_missingvalues['Price']) ) # to see the pdf of it
q=data_no_missingvalues['Price'].quantile(0.99) # a point/value that really stands on the 99 percentile
data_1=data_no_missingvalues[ data_no_missingvalues['Price']<q ]
print(data_1.describe(include='all'))
#----pdf of mileage : we corrext it by deleting the outliers
print(sns.displot(data_1['Mileage']) ) # to see the pdf of it
q=data_1['Mileage'].quantile(0.99) # a point/value that really stands on the 99 percentile
data_2=data_1[ data_1['Mileage']<q ]
print(data_2.describe(include='all'))
#---pdf of engine volume : we correct it by deleting the useless data
#   in this case we know that the engines voulmes is not more than 6.5
data_3=data_2[  data_2['EngineV']<6.5 ]
#---pdf of year
q=data_3[ 'Year'].quantile(0.01)
data_4=data_3[ data_3['Year'] >q ]

data_cleaned = data_4.reset_index(drop=True) # drop=True is for really delete what we want to delete and change based waht we want to change above
print( data_cleaned.describe(include='all') )

#----------------------STEP 2----------------------------------------------
# we check the regression assumption

# 1 : check the linearity
f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, figsize=(15,3))
ax1.scatter( data_cleaned['Year'],data_cleaned['Price'] )
ax1.set_title('Price and Year')
ax2.scatter( data_cleaned['EngineV'],data_cleaned['Price'] )
ax2.set_title('Price and EngineV')
ax3.scatter( data_cleaned['Mileage'],data_cleaned['Price'] )
ax3.set_title('Price and Mileage')

plt.show() # note that, after we do the scatter plot, we realize that the scatter plot will not create a regression line. Thus, we need to do some log transformation to the data
#   log transformation
log_price=np.log( data_cleaned['Price'] ) #transform the data of price by log transformation
data_cleaned['log_price']=log_price # added the new transformation data into the data frame

#   plot the scatter plot again
f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, figsize=(15,3))
ax1.scatter( data_cleaned['Year'],data_cleaned['log_price'] )
ax1.set_title('New Price and Year')
ax2.scatter( data_cleaned['EngineV'],data_cleaned['log_price'] )
ax2.set_title(' New Price and EngineV')
ax3.scatter( data_cleaned['Mileage'],data_cleaned['log_price'] )
ax3.set_title(' New Price and Mileage')

plt.show()

#2 : check the endogeinty assumption
#..... check others too
#5 : check multicollinearity
from statsmodels.stats.outliers_influence  import variance_inflation_factor
variables = data_cleaned[ ['Mileage','Year','EngineV']]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor ( variables.values,i) for i in range(variables.shape[1])]
vif["features"]=variables.columns
print(vif)

data_no_multi=data_cleaned.drop( ['Year','Price'] , axis=1)

# make dummy variable for categoricals

data_with_dummies = pd.get_dummies(data_no_multi,drop_first=True)
print(data_with_dummies.columns.values)
cols=['log_price','Mileage' ,'EngineV' ,'Brand_BMW' ,'Brand_Mercedes-Benz',
 'Brand_Mitsubishi','Brand_Renault', 'Brand_Toyota' ,'Brand_Volkswagen',
 'Body_hatch' ,'Body_other' ,'Body_sedan' ,'Body_vagon' ,'Body_van',
 'Engine Type_Gas' ,'Engine Type_Other', 'Engine Type_Petrol',
 'Registration_yes']
data_preprocessed = data_with_dummies[cols]
print(data_preprocessed)

# ----------------------------STEP 3: log transformation for all the data-----
# noted : targets=dependent variable
# noted : inputs=independent variable
targets=data_preprocessed['log_price']
inputs=data_preprocessed.drop(['log_price'],axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(inputs)
inputs_scaled=scaler.transform(inputs)
print(inputs_scaled)

#----------------------------STEP 4 : step to do to handle overfitting issue
# remember that all data is now have done the log transformation
# --- now, we are doin the regression of the data for training data set , mwe want to see if the prediction of our y is same with the outcome of real y

#     in this step we also need to do the summary of regression between training data set ( x_train, y train)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs_scaled,targets, test_size=0.2, random_state=365)

reg=LinearRegression()
reg.fit(x_train,y_train) # we fit the training regression line for x_train and y_train set as it should
y_outcome = reg.predict(x_train) # y outcome = y in real life ( y observation) , y train = y prediction
plt.scatter(y_train,y_outcome)
plt.xlabel('Targets(y_train)',size=18)
plt.ylabel('Predictions(y_outcome)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

#     in this step we also need to see the resideual between y expected and y outcome , they should be in normal probability
sns.distplot(y_train - y_outcome)
plt.title("Residual PDF", size=18)

#     in this step we also need to do the summary of regression between training data set ( x_train, y train) 
reg_summary=pd.DataFrame(inputs.columns.values , columns=['Features'])
reg_summary['Weights']=reg.coef_
print(reg_summary) # note that , in this section we already declare the reg above

# --- so now we need to test the regression of the data to test data
y_outcome2=reg.predict(x_test)
plt.scatter( y_test , y_outcome2 )
plt.xlabel('Targets(y_test)',size=18)
plt.ylabel('Predictions(y_outcome2)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

# --- data frame of expected result and real result from testing data adn find the residual of them to know how far of the,
df_pf=pd.DataFrame( np.exp (y_outcome2) , columns=['Real'] )
df_pf['Target'] = np.exp(y_test.reset_index(drop=True)) # if index diorg mcm tak betul, susun balik
df_pf['Residual']=df_pf['Real']-df_pf['Target']
df_pf['Diferrences %' ] = np.absolute ( df_pf['Residual'] / df_pf ['Target'] * 100 )
pd.set_option('display.float_format', lambda x : '%.2f' % x)
df_pf.sort_values(by=['Diferrences %' ] ) # we  may see the differences in percent 
print(df_pf)