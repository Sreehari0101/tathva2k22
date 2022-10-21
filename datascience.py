import pandas as pd

df=pd.read_csv('titanic.csv')

df.head(10)

df.info()

import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False)

sns.countplot(df['Survived'])

sns.countplot(df['Survived'],hue=df['Sex'])

sns.countplot(df['Survived'],hue=df["Pclass"])

df.drop('PassengerId',axis=1,inplace=True)

df.info()

df.drop(["Name","Ticket","Fare","Embarked"],axis=1,inplace=True)

df.drop('Cabin',axis=1,inplace=True)

df.head()

df.info()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['Sex']=le.fit_transform(df['Sex'])

df.head()

df.info()

df['Age'].mean()

df['Age'].median()

sns.boxplot(df['Pclass'],df['Age'])

def calculate(cols):
  Age=cols[0]
  Pclass=cols[1]
  if pd.isnull(Age):
    if Pclass==1:
      return 37
    elif Pclass==2:
      return 29

    else:
      return 25
  else:
    return Age

df['Age']=df[['Age','Pclass']].apply(calculate,axis=1)

df.head()

df.info()

df=pd.read_csv('bhpmod.csv')

df.head()

df.info()

pd.options.display.float_format='{:,.2f}'.format

df.describe()

df['price_per_sqft'].quantile(0.999)

