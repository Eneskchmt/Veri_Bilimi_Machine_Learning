#!/usr/bin/env python
# coding: utf-8

# **Aykırı Gözlem Analizi**

# **Aykırı Değerleri Yakalamak**

# In[1]:


import seaborn as sns 
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include = ['float64', 'int64'])
df = df.dropna()
df.head()


# In[2]:


df_table = df["table"]


# In[3]:


df_table.head()


# In[4]:


sns.boxplot(x = df_table)


# In[5]:


Q1 = df_table.quantile(0.25) 
Q3 = df_table.quantile(0.75)
IQR = Q3-Q1


# In[6]:


Q1


# In[7]:


Q3


# In[8]:


IQR


# In[9]:


alt_sinir = Q1 - 1.5*IQR
ust_sinir = Q3 + 1.5*IQR


# In[10]:


alt_sinir


# In[11]:


ust_sinir


# In[12]:


(df_table < alt_sinir) | (df_table > ust_sinir)


# In[13]:


aykiri_tf = (df_table < alt_sinir)


# In[14]:


aykiri_tf.head()


# In[15]:


df_table[aykiri_tf]


# In[16]:


df_table[aykiri_tf].index


# **Aykırı Değer Problemini Çözmek**

# In[17]:


df_table[aykiri_tf]


# **Silme**

# In[18]:


import pandas as pd


# In[19]:


type(df_table)


# In[20]:


df_table = pd.DataFrame(df_table)


# In[21]:


df_table.shape


# In[22]:


t_df = df_table[~((df_table < (alt_sinir)) | (df_table > (ust_sinir))).any(axis = 1)]


# In[23]:


t_df.shape


# **Ortalama ile Doldurma**

# In[24]:


import seaborn as sns 
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include = ['float64', 'int64'])
df = df.dropna()
df.head()


# In[25]:


df_table = df["table"]


# In[26]:


aykiri_tf.head()


# In[27]:


df_table[aykiri_tf]


# In[28]:


df_table.mean()


# In[29]:


df_table[aykiri_tf] = df_table.mean()


# In[30]:


df_table[aykiri_tf]


# **Baskılama Yöntemi**

# In[31]:


import seaborn as sns 
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include = ['float64', 'int64'])
df = df.dropna()
df.head()


# In[32]:


df_table = df["table"]


# In[33]:


df_table[aykiri_tf]


# In[34]:


alt_sinir


# In[35]:


df_table[aykiri_tf] = alt_sinir


# In[36]:


df_table[aykiri_tf]


# **Çok Değişkenli Aykırı Gözlem Analizi**

# **Local Outlier Factor**

# In[37]:


import seaborn as sns
diamonds = sns.load_dataset('diamonds')
diamonds = diamonds.select_dtypes(include = ['float64', 'int64'])
df = diamonds.copy()
df = df.dropna()
df.head()


# In[38]:


import numpy as np
from sklearn.neighbors import LocalOutlierFactor


# In[39]:


clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)


# In[40]:


clf.fit_predict(df)


# In[41]:


df_scores = clf.negative_outlier_factor_


# In[42]:


df_scores[0:10]


# In[43]:


np.sort(df_scores)[0:20]


# In[44]:


esik_deger = np.sort(df_scores)[13]


# In[45]:


aykiri_tf = df_scores > esik_deger


# In[46]:


aykiri_tf


# In[47]:


### silme yöntemi


# In[48]:


yeni_df = df[df_scores > esik_deger]


# In[49]:


yeni_df


# In[50]:


df[df_scores < esik_deger]


# In[51]:


df[df_scores == esik_deger]


# In[52]:


### baskilama


# In[53]:


baski_deger = df[df_scores == esik_deger]


# In[54]:


aykirilar = df[~aykiri_tf]


# In[55]:


aykirilar


# In[56]:


aykirilar.to_records(index = False)


# In[57]:


res = aykirilar.to_records(index = False)


# In[58]:


res[:] = baski_deger.to_records(index = False)


# In[59]:


res


# In[60]:


df[~aykiri_tf]


# **Eksik Veri Analizi**

# **Hızlı Çözüm**

# In[61]:


import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df


# In[62]:


df.isnull().sum()


# In[63]:


df.notnull().sum()


# In[64]:


df.isnull().sum().sum()


# In[65]:


df.isnull()


# In[66]:


df[df.isnull().any(axis = 1)]


# In[67]:


df[df.notnull().all(axis = 1)]


# In[68]:


df[df["V1"].notnull() & df["V2"].notnull()&df["V3"].notnull()]


# In[69]:


#eksik degerlerin direk silinmesi


# In[70]:


df.dropna()


# In[71]:


df


# In[72]:


#basit deger atama


# In[73]:


df["V1"].mean()


# In[74]:


df["V1"].fillna(df["V1"].mean())


# In[75]:


df["V2"].fillna(0)


# In[76]:


df.apply(lambda x: x.fillna(x.mean()), axis = 0)


# **Eksik Değerlerin Saptanması**

# In[77]:


#değişkenlerdeki tam değer sayısı
df.notnull().sum()


# In[78]:


#değişkenlerdeki eksik değer sayısı
df.isnull().sum()


# In[79]:


#en az bir eksik değere sahip gözlemler
df.isnull().sum().sum()


# In[80]:


#tüm değerleri tam olan gözlemler
df[df.notnull().all(axis=1)]


# **Eksik Veri Yapısının Görselleştirilmesi**

# In[81]:


get_ipython().system('pip install missingno')


# In[82]:


import missingno as msno


# In[83]:


msno.bar(df)


# In[84]:


msno.matrix(df)


# In[85]:


df


# In[86]:


import seaborn as sns 
df = sns.load_dataset('planets')
df.head()


# In[87]:


df.isnull().sum()


# In[88]:


msno.matrix(df)


# In[89]:


msno.heatmap(df)


# **Silme Yöntemleri**

# In[90]:


import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df


# In[91]:


df.dropna()


# In[92]:


df


# In[93]:


df.dropna(how = "all")


# In[94]:


df.dropna(axis = 1)


# In[95]:


df.dropna(axis = 1, how = "all")


# In[96]:


df["sil_beni"] = np.nan


# In[97]:


df


# In[98]:


df.dropna(axis = 1, how = "all")


# In[99]:


df


# In[100]:


df.dropna(axis = 1, how = "all", inplace = True)


# In[101]:


df


# **Değer Atama Yöntemleri**

# In[102]:


import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}        
)

df


# In[103]:


#sayısal degiskenlerde atama


# In[104]:


df["V1"].fillna(0)


# In[105]:


df


# In[106]:


df["V1"].fillna(df["V1"].mean())


# In[107]:


#tum degiskenler icin birinci yol
df.apply(lambda x: x.fillna(x.mean()), axis = 0)


# In[108]:


#ikinci yol


# In[109]:


df.fillna(df.mean()[:])


# In[110]:


df.fillna(df.mean()["V1":"V2"])


# In[111]:


df["V3"].fillna(df["V3"].median())


# In[112]:


#ucuncu yol


# In[113]:


df.where(pd.notna(df), df.mean(), axis = "columns")


# **Kategorik Değişken Kırılımında Değer Atama**

# In[114]:


V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])

df = pd.DataFrame(
        {"maas" : V1,
         "V2" : V2,
         "V3" : V3,
        "departman" : V4}        
)

df


# In[115]:


df.groupby("departman")["maas"].mean()


# In[116]:


df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))


# **Kategorik Değişkenler için Eksik Değer Atama**

# In[117]:


import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V4 = np.array(["IT",np.nan,"IK","IK","IK","IK","IK","IT","IT"], dtype=object)

df = pd.DataFrame(
        {"maas" : V1,
         "departman" : V4}
)
df


# In[118]:


df["departman"].mode()[0]


# In[119]:


df["departman"].fillna(df["departman"].mode()[0])


# In[120]:


df


# In[121]:


df["departman"].fillna(method = "bfill")


# In[122]:


df["departman"].fillna(method = "ffill")


# **Tahmine Dayalı Değer Atama Yöntemleri**

# In[123]:


import seaborn as sns 
import missingno as msno
df = sns.load_dataset('titanic')
df = df.select_dtypes(include = ['float64','int64'])
print(df.head())
df.isnull().sum()


# In[124]:


get_ipython().system('pip install ycimpute')


# In[125]:


from ycimpute.imputer import knnimput


# In[126]:


var_names = list(df)


# In[127]:


import numpy as np
n_df = np.array(df)


# In[128]:


n_df[0:10]


# In[129]:


n_df.shape


# In[130]:


dff = knnimput.KNN(k = 4).complete(n_df)


# In[131]:


type(dff)


# In[132]:


import pandas as pd
dff = pd.DataFrame(dff, columns = var_names)


# In[133]:


type(dff)


# In[134]:


dff.isnull().sum()


# In[135]:


#random forests


# In[136]:


import seaborn as sns
import missingno as msno
df = sns.load_dataset('titanic')
df = df.select_dtypes(include = ['float64', 'int64'])


# In[137]:


df.isnull().sum()


# In[138]:


var_names = list(df)


# In[139]:


import numpy as np
n_df = np.array(df)


# In[140]:


from ycimpute.imputer import iterforest
dff = iterforest.IterImput().complete(n_df)


# In[141]:


dff = pd.DataFrame(dff, columns = var_names)


# In[142]:


dff.isnull().sum()


# In[143]:


#EM


# In[144]:


import seaborn as sns
import missingno as msno
df = sns.load_dataset('titanic')
df = df.select_dtypes(include = ['float64', 'int64'])


# In[145]:


from ycimpute.imputer import EM


# In[146]:


var_names = list(df)


# In[147]:


import numpy as np
n_df = np.array(df)


# In[148]:


dff = EM().complete(n_df)


# In[149]:


dff = pd.DataFrame(dff, columns = var_names)


# In[150]:


dff.isnull().sum()


# **Değişken Standardizasyonu (Veri Standardizasyonu)**

# In[151]:


import numpy as np
import pandas as pd
V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])
df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3})
df = df.astype(float)
df


# **Standardizasyon**

# In[152]:


from sklearn import preprocessing


# In[153]:


preprocessing.scale(df)


# In[154]:


df


# **Normalizasyon**

# In[155]:


preprocessing.normalize(df)


# **Min-Max Dönüşümü**

# In[157]:


scaler = preprocessing.MinMaxScaler(feature_range = (100,200))


# In[158]:


scaler.fit_transform(df)


# **Değişken Dönüşümleri**

# In[159]:


import seaborn as sns 
df = sns.load_dataset('tips')
df.head()


# **0-1 Dönüşümü**

# In[160]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()


# In[161]:


lbe.fit_transform(df["sex"])


# In[162]:


df["yen,_sex"] = lbe.fit_transform(df["sex"])


# In[163]:


df


# **"1 ve Diğerleri(0)" Dönüşümü**

# In[164]:


df.head()


# In[165]:


df["day"].str.contains("Sun")


# In[166]:


import numpy as np
df["yeni_day"] = np.where(df["day"].str.contains("Sun"), 1, 0)


# In[167]:


df


# **Çok Sınıflı Dönüşüm**

# In[169]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()


# In[170]:


lbe.fit_transform(df["day"])


# **One-Hot Dönüşümü ve Dummy Değişken Tuzağı**

# In[171]:


df.head()


# In[172]:


df_one_hot = pd.get_dummies(df, columns = ["sex"], prefix = ["sex"])


# In[174]:


df_one_hot.head()


# In[175]:


pd.get_dummies(df, columns = ["day"], prefix = ["day"]).head()

