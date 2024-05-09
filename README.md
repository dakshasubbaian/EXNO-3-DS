EX NO 3

```
NAME : DAKSHA SUBBAIAN
REG NO : 212223230036
```

# AIM:  
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Done by-
Name- DAKSHA SUBBAIAN
Reg No- 212223230036
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![316974447-03f8eed4-910f-4b36-952a-e42f04af61d2](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/b61c749f-7791-4eeb-9494-a5956f61d7a1)

## Ordinal Encoding 
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![316974700-18d9f738-4373-4195-86c6-1867c0537e42](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/eb43f28e-1c9e-44b8-89cb-013af252152a)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![316974937-3c39cd75-cbc4-4be0-b6e1-77885f2157aa](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/cf881163-bfdc-40bc-94e8-2abf4b00c378)

## Label Encoder
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![316975220-d6b15fec-dc0f-4335-aa5e-b7d9c8456359](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/5fd26668-0654-4eff-8526-b1dd2484f7bf)


## OneHot Encoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![316975574-68e576e2-502c-4219-9c2f-82c0cc4be899](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/64a7af03-a9a3-4c4c-a96a-374dd2904a09)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![316976459-850cfb3a-1268-4876-a209-53f393e11e07](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/9a4f5c0a-d182-4d70-bb68-f43086b066cf)

## Binary Encoder
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![316976459-850cfb3a-1268-4876-a209-53f393e11e07](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/9dbd917e-011b-4a7f-9214-2bee9d1bf6c9)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![316976734-48f270d4-9112-4ed1-a819-966fbc9e253c](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/82892b16-9e36-47ef-872e-332021bd7a8d)

## Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![316976963-4f88289a-59ac-41b4-a069-362e4ef02f9c](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/bb3077c3-10ac-4cd4-9f57-7bd987aae23b)

## Data Transformation
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![316977331-e534fb21-83c5-4e3d-b308-0e35d7b97b5a](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/af9f0b63-8581-4fbf-8c76-e4a1b4214724)

```
df.skew()
```
![316977432-6fa9e262-5939-4ed8-9431-2f53e5df0249](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/029f9dfb-36b3-4096-a07d-2e5853eefce1)

```
np.log(df["Highly Positive Skew"])
```
![316977539-ebdff44e-bf3c-4d23-8370-47de6dafc4e1](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/445bd7ed-d945-4269-9952-76cf3846fce9)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![316977690-e1545b48-ab16-47da-84bc-f6121b46f43e](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/19586994-7153-43fe-9bc9-009df351ee31)

```
np.sqrt(df["Highly Positive Skew"])
```
![316977841-1d427996-1049-4286-8703-4352034c1abf](https://github.com/dakshasubbaian/EXNO-3-DS/assets/112880924/fce6d62a-f24a-46ce-98e6-f76ddda4c802)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/00c18009-d540-4d86-b440-39377bde361c)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/f3ef2b2b-9eeb-4062-888b-d118f395436b)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/ea63ad5a-cb65-4c02-8393-25bfe84571d8)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/a704ece5-fa6b-4ee8-8c2f-292747d6bcd8)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/a33675eb-6c8f-41fa-b235-c28ee2f6648b)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/977f0b7b-add6-4cff-a014-d46936459ff9)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/381b2529-69d1-449f-9268-acd090addfc1)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/d06eb8f6-3e0e-46e7-b946-262bb767b38d)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/a9f1e12e-82e6-4009-ac21-ecce9a0376cc)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/fb14ea2a-4cdd-4cc3-bbe4-7c91931be99d)
```
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/f62a8f38-65c6-468f-820f-9a2d94bc9bec)
```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/671fb915-43b8-4a79-9817-23a1cf973983)


# RESULT:
Thus perform Feature Encoding and Transformation process is executed successfully.

       
