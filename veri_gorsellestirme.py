#!/usr/bin/env python
# coding: utf-8

# # **1 - Veriye İlk Bakış**

# **1 - Veri Seti Hikayesi ve Yapısının İncelenmesi**

# In[3]:


import seaborn as sns 
planets = sns.load_dataset("planets")
planets.head()


# In[3]:


#veri setinin hikayesi nedir? 


# In[4]:


df = planets.copy()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


#veri seti yapısal bilgileri


# In[8]:


df.info()


# In[9]:


df.dtypes


# In[10]:


import pandas as pd 
df.method = pd.Categorical(df.method)


# In[11]:


df.dtypes


# In[12]:


df.head()


# **2 - Veri Setinin Betimlenmesi**

# In[13]:


import seaborn as sns 
planets = sns.load_dataset("planets")
df = planets.copy()


# In[14]:


df.head()


# In[15]:


df.shape


# In[16]:


df.columns


# In[17]:


df.describe().T


# In[18]:


df.describe(include = "all").T


# **3 - Eksik Değerlerin İncelenmesi**

# In[19]:


import seaborn as sns 
planets = sns.load_dataset("planets")
df = planets.copy()
df.head()


# In[20]:


#hiç eksik gözlem(değer) var mı?
df.isnull().values.any()


# In[21]:


#hangi değişkende kaçar tane var?


# In[22]:


df.isnull().sum()


# In[23]:


df["orbital_period"].fillna(0, inplace = True)


# In[24]:


df.isnull().sum()


# In[25]:


df["mass"].fillna(df.mass.mean(), inplace = True)


# In[26]:


df.isnull().sum()


# In[27]:


df["distance"].fillna(df.distance.mean(), inplace = True)


# In[28]:


df.isnull().sum()


# In[29]:


df = planets.copy()
df.head()


# In[30]:


df.isnull().sum()


# **4 - Kategorik Değişken Özetleri**

# In[31]:


import seaborn as sns 
planets = sns.load_dataset("planets")
df = planets.copy()
df.head()


# *Sadece Kategorik Değişkenler ve Özetleri*

# In[32]:


kat_df = df.select_dtypes(include = ["object"])


# In[33]:


kat_df.head(5)


# *Kategorik Değişkenin Sınıflarına ve Sınıf Sayısına Erişmek*

# In[34]:


kat_df.method.unique()


# In[35]:


kat_df["method"].value_counts().count()


# *Kategorik Değişkenin Sınıflarının Frekanslarına Erişmek*

# In[36]:


kat_df["method"].value_counts()


# In[37]:


df["method"].value_counts().plot.barh()


# **5 - Sürekli Değişken Özetleri**

# In[38]:


import seaborn as sns 
planets = sns.load_dataset("planets")
df = planets.copy()
df.head()


# In[39]:


df_num = df.select_dtypes(include = ["float64", "int64"])


# In[40]:


df_num.head()


# In[41]:


df_num.describe().T


# In[42]:


df_num["distance"].describe()


# In[43]:


print("Ortalama: " + str(df_num["distance"].mean()))
print("Dolu Gözlem Sayısı: " + str(df_num["distance"].count()))
print("Maksimum Değer: " + str(df_num["distance"].max()))
print("Minimum Değer: " + str(df_num["distance"].min()))
print("Medyan: " + str(df_num["distance"].median()))
print("Standart Sapma: " + str(df_num["distance"].std()))


# # **2 - Dağılım Grafikleri**

# **A)Barplot**

# **1 - Veri Seti Hikayesi**

# In[44]:


import seaborn as sns 
diamonds = sns.load_dataset('diamonds')
df = diamonds.copy()
df.head()


# **2 - Veri Setine Hızlı Bakış**

# In[45]:


df.info()


# In[46]:


df.describe().T


# In[47]:


df.head()


# In[48]:


df["cut"].value_counts()


# In[49]:


df["color"].value_counts()


# In[50]:


#ordinal tanımlama
from pandas.api.types import CategoricalDtype


# In[51]:


df.cut.head()


# In[52]:


df.cut = df.cut.astype(CategoricalDtype(ordered = True))


# In[53]:


df.dtypes


# In[54]:


df.cut.head(1)


# In[55]:


cut_kategoriler = ["Fair","Good","Very Good","Premium","Ideal"]


# In[56]:


df.cut = df.cut.astype(CategoricalDtype(categories = cut_kategoriler, ordered = True))


# In[57]:


df.cut.head(1)


# In[58]:


#barplot


# In[59]:


df["cut"].value_counts().plot.barh().set_title("Cut Değişkeninin Sınıf Frekansları")


# In[60]:


(df["cut"]
.value_counts()
.plot.barh()
.set_title("Cut Değişkeninin Sınıf Frekansları"))


# In[61]:


sns.barplot(x = "cut", y = df.cut.index, data = df)


# **B)Çaprazlamalar**

# In[62]:


import seaborn as sns 
from pandas.api.types import CategoricalDtype
diamonds = sns.load_dataset('diamonds')
df = diamonds.copy()
cut_kategoriler = ["Fair","Good","Very Good","Premium","Ideal"]
df.cut = df.cut.astype(CategoricalDtype(categories = cut_kategoriler, ordered = True))
df.head()


# In[63]:


sns.catplot(x = "cut", y = "price", data = df)


# In[64]:


sns.barplot(x = "cut", y= "price", hue = "color", data = df)


# In[65]:


df.groupby(["cut","color"])["price"].mean()


# **C)Histogram ve Yoğunluk**

# In[66]:


import seaborn as sns 
diamonds = sns.load_dataset('diamonds')
df = diamonds.copy()
df.head()


# In[67]:


sns.distplot(df.price, kde = False)


# In[68]:


df["price"].describe()


# In[69]:


get_ipython().run_line_magic('pinfo', 'sns.distplot')


# In[70]:


sns.distplot(df.price, bins = 10, kde = False)


# In[71]:


sns.distplot(df.price)


# In[72]:


sns.distplot(df.price, hist = False)


# In[73]:


sns.kdeplot(df.price, shade = True)


# **D)Çaprazlamalar**

# In[74]:


import seaborn as sns 
diamonds = sns.load_dataset('diamonds')
df = diamonds.copy()
df.head()


# In[75]:


sns.kdeplot(df.price, shade = True)


# In[76]:


(sns
    .FacetGrid(df,
                  hue = "cut",
                  height = 5,
                  xlim = (0, 1000))
 .map(sns.kdeplot, "price", shade = True)
 .add_legend()
)


# In[77]:


sns.catplot(x = "cut", y = "price", hue = "color", kind = "point", data = df)


# **E)Boxplot**

# In[78]:


import seaborn as sns 
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()


# In[79]:


df.describe().T


# In[80]:


df["sex"].value_counts()


# In[81]:


df["smoker"].value_counts()


# In[82]:


df["day"].value_counts()


# In[83]:


df["time"].value_counts()


# *Boxplot*

# In[84]:


import seaborn as sns 
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()


# In[85]:


sns.boxplot(x = df["total_bill"])


# In[86]:


sns.boxplot(x = df["total_bill"], orient = "v")


# *Çaprazlamalar*

# In[87]:


df.describe().T


# In[88]:


#Hangi gunler daha fazla kazanıyoruz?


# In[89]:


sns.boxplot(x = "day", y = "total_bill", data = df)


# In[90]:


#sabah mı akşam mı daha çok kazanıyoruz?


# In[91]:


sns.boxplot(x = "time", y = "total_bill", data = df)


# In[92]:


#kisi sayısı kazanc


# In[93]:


sns.boxplot(x = "size", y = "total_bill", data = df)


# In[94]:


sns.boxplot(x = "day", y = "total_bill", hue = "sex", data = df)


# *Violin*

# In[95]:


df.head()


# In[96]:


sns.catplot(y = "total_bill", kind = "violin", data = df)


# *Çaprazlamalar*

# In[97]:


sns.catplot(x = "day", y = "total_bill", kind = "violin", data = df)


# In[98]:


sns.catplot(x = "day", y = "total_bill", hue = "sex", kind = "violin", data = df)


# # **3 - Korelasyon Grafikleri**

# **1 - Scatterplot**

# In[99]:


import seaborn as sns
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()


# In[100]:


sns.scatterplot(x = "total_bill", y = "tip", data = df)


# **2 - Çaprazlamalar**

# In[101]:


sns.scatterplot(x = "total_bill", y = "tip", hue = "time", data = df)


# In[102]:


sns.scatterplot(x = "total_bill", y = "tip", hue = "time", style = "time", data = df)


# In[103]:


sns.scatterplot(x = "total_bill", y = "tip", hue = "day", style = "day", data = df)


# In[104]:


sns.scatterplot(x = "total_bill", y = "tip", hue = "size", size = "size", data = df)


# # **4 - Heatmap**

# **1 - Veri Seti Hikayesi**

# In[105]:


import seaborn as sns 
flights = sns.load_dataset('flights')
df = flights.copy()
df.head()


# In[106]:


df.shape


# In[107]:


df["passengers"].describe()


# In[108]:


df = df.pivot("month", "year", "passengers")


# In[109]:


df


# In[110]:


sns.heatmap(df)


# In[111]:


sns.heatmap(df, annot = True, fmt = "d")


# In[112]:


sns.heatmap(df, annot = True, fmt = "d", linewidths = .5)


# In[113]:


sns.heatmap(df, annot = True, fmt = "d", linewidths = .5, cbar = False)


# # **5 - Çizgi Grafik**

# **1 - Veri Seti Hikayesi**

# In[114]:


import seaborn as sns 
fmri = sns.load_dataset("fmri")
df = fmri.copy()
df.head()


# In[115]:


df.shape


# In[116]:


df["timepoint"].describe()


# In[117]:


df["signal"].describe()


# In[118]:


df.groupby("timepoint")["signal"].count()


# In[119]:


df.groupby("signal").count()


# In[120]:


df.groupby("timepoint")["signal"].describe()


# **2 - Çizgi Grafik ve Çaprazlamalar**

# In[121]:


sns.lineplot(x = "timepoint", y = "signal", data = df)


# In[122]:


sns.lineplot(x = "timepoint", y = "signal", hue = "event", data = df)


# In[123]:


sns.lineplot(x = "timepoint", y = "signal", hue = "event", style = "event", data = df)


# In[124]:


sns.lineplot(x = "timepoint",
             y = "signal",
             hue = "event",
             style = "event",
             markers = True, dashes = False, data = df)


# In[125]:


sns.lineplot(x = "timepoint",
             y = "signal",
             hue = "region",
             style = "event",
             data = df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




