#!/usr/bin/env python
# coding: utf-8

# # **Veri Bilimi İçin İstatistik**

# **Örnek Teorisi**

# In[1]:


import numpy as np


# In[2]:


populasyon = np.random.randint(0, 80, 10000)


# In[3]:


populasyon[0:10]


# In[4]:


#orneklem cekimi
np.random.seed(115)
orneklem = np.random.choice(a = populasyon, size = 100)
orneklem[0:10]


# In[5]:


orneklem.mean()


# In[6]:


populasyon.mean()


# In[7]:


#orneklem dagılımı


# In[8]:


np.random.seed(10)
orneklem1 = np.random.choice(a = populasyon, size = 100)
orneklem2 = np.random.choice(a = populasyon, size = 100)
orneklem3 = np.random.choice(a = populasyon, size = 100)
orneklem4 = np.random.choice(a = populasyon, size = 100)
orneklem5 = np.random.choice(a = populasyon, size = 100)
orneklem6 = np.random.choice(a = populasyon, size = 100)
orneklem7 = np.random.choice(a = populasyon, size = 100)
orneklem8 = np.random.choice(a = populasyon, size = 100)
orneklem9 = np.random.choice(a = populasyon, size = 100)
orneklem10 = np.random.choice(a = populasyon, size = 100)


# In[9]:


(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean() 
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


# In[10]:


orneklem1.mean()


# In[11]:


orneklem2.mean()


# **Betimsel İstatistikler**

# In[60]:


import seaborn as sns
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()


# In[61]:


df.describe().T


# In[62]:


get_ipython().system('pip install researchpy')
import researchpy as rp


# In[63]:


rp.summary_cont(df[["total_bill","tip","size"]])


# In[64]:


rp.summary_cat(df[["sex","smoker","day"]])


# In[66]:


df[["tip","total_bill"]].cov()


# In[68]:


df[["tip","total_bill"]].corr()


# **İş Uygulaması: Fiyat Stratejisi Karar Destek**

# In[22]:


import numpy as np
fiyatlar = np.random.randint(10,110,1000)


# In[23]:


fiyatlar.mean()


# In[24]:


import statsmodels.stats.api as sms


# In[25]:


sms.DescrStatsW(fiyatlar).tconfint_mean()


# In[27]:


def yazdir(metin):
    print(metin, "program ögrenilecek")

yazdir("0")⍷


# In[28]:


print("metin", "program öğrenilecek")


# **Olasılık Dağılımları**

# **Bernoulli Dağılımı**

# In[30]:


from scipy.stats import bernoulli


# In[31]:


p = 0.6


# In[32]:


rv = bernoulli(p)
rv.pmf(k = 0)


# **Büyük Sayılar Yasası**

# In[72]:


rng.randint(0,2,size = 5)


# In[70]:


import numpy as np
rng = np.random.RandomState(123)
for i in np.arange(1,21):
    deney_sayisi = 2**i
    yazi_turalar = rng.randint(0 ,2, size = deney_sayisi)
    yazi_olasiliklari = np.mean(yazi_turalar)
    print("Atış Sayısı:",deney_sayisi,"---",'Yazı Olasılığı: %.2f' % (yazi_olasiliklari * 100))


# In[71]:


1/5


# **Binom Dağılımı**

# In[73]:


from scipy.stats import binom


# In[74]:


p = 0.01
n = 100
rv = binom(n, p)
print(rv.pmf(1))
print(rv.pmf(5))
print(rv.pmf(10))


# **Poisson Dağılımı**

# In[75]:


from scipy.stats import poisson


# In[76]:


lambda_ = 0.1


# In[77]:


rv = poisson(mu = lambda_)
print(rv.pmf(k = 0))


# In[78]:


print(rv.pmf(k = 3))


# In[79]:


print(rv.pmf(k = 5))


# **Normal Dağılım**

# In[80]:


from scipy.stats import norm


# In[81]:


#90'dan fazla olması
1-norm.cdf(90 ,80, 5)


# In[82]:


#70'den fazla olması
1-norm.cdf(70, 80, 5)


# In[83]:


#73'den az olması
norm.cdf(73, 80, 5)


# In[84]:


#85 ile 90 arasında olması


# In[85]:


norm.cdf(90, 80, 5) - norm.cdf(85, 80, 5)


# **Tek Örneklem T Testi**

# In[95]:


import numpy as np
import pandas as pd

olcumler = np.array([17, 160, 234, 149, 145, 107, 197, 75, 201, 225, 211, 119, 
              157, 145, 127, 244, 163, 114, 145,  65, 112, 185, 202, 146,
              203, 224, 203, 114, 188, 156, 187, 154, 177, 95, 165, 50, 110, 
              216, 138, 151, 166, 135, 155, 84, 251, 173, 131, 207, 121, 120])


# In[87]:


olcumler[0:10]


# In[88]:


import scipy.stats as stats


# In[90]:


stats.describe(olcumler)


# **Varsayımlar**

# In[91]:


#normallik varsayımı


# In[96]:


#histogram
pd.DataFrame(olcumler).plot.hist()


# In[97]:


#qqlot
import pylab
stats.probplot(olcumler, dist="norm", plot=pylab)
pylab.show()


# **Shapiro-Wilks Testi**

# In[98]:


from scipy.stats import shapiro


# In[99]:


shapiro(olcumler)


# In[100]:


print("T Hesap İstatistiği: " + str(shapiro(olcumler)[0]))
print("Hesaplanan P-value: " + str(shapiro(olcumler)[1]))


# **Hipotez Testinin Uygulanması**

# In[101]:


stats.ttest_1samp(olcumler, popmean = 170)


# **Nonparametrik Tek Örneklem Testi**

# In[102]:


from statsmodels.stats.descriptivestats import sign_test


# In[103]:


sign_test(olcumler, 170)


# **Tek Örneklem Oran Testi**

# In[104]:


from statsmodels.stats.proportion import proportions_ztest


# In[105]:


count = 40
nobs = 500
value = 0.125


# In[106]:


proportions_ztest(count, nobs, value)


# **Bağımsız İki Örneklem T Testi**

# In[107]:


#veri tipi I


# In[108]:


import numpy as np
import pandas as pd


# In[109]:


A = pd.DataFrame([30,27,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,
        25,21,18,24,26,33,26,27,28,19,25])

B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,
        32,33,33,33,31,32,33,26,32,33,29])


A_B = pd.concat([A, B], axis = 1)
A_B.columns = ["A","B"]

A_B.head()


# In[ ]:


#veri tipi II


# In[115]:


A = pd.DataFrame([30,27,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,
        25,21,18,24,26,33,26,27,28,19,25])

B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,
        32,33,33,33,31,32,33,26,32,33,29])

#A ve A'nın grubu
GRUP_A = np.arange(len(A))
GRUP_A = pd.DataFrame(GRUP_A)
GRUP_A[:] = "A"
A = pd.concat([A, GRUP_A], axis = 1)

#B ve B'nin Grubu
GRUP_B = np.arange(len(B))
GRUP_B = pd.DataFrame(GRUP_B)
GRUP_B[:] = "B"
B = pd.concat([B, GRUP_B], axis = 1)

#Tum veri
AB = pd.concat([A,B])
AB.columns = ["gelir","GRUP"]
print(AB.head())
print(AB.tail())


# In[116]:


import seaborn as sns 
sns.boxplot(x = "GRUP", y = "gelir", data = AB)


# **Varsayım Kontrolü**

# In[118]:


A_B.head()


# In[119]:


AB.head()


# In[120]:


#normallik varsayımı


# In[121]:


from scipy.stats import shapiro


# In[122]:


shapiro(A_B.A)


# In[123]:


shapiro(A_B.B)


# In[124]:


#varyans homojenligi varsayımı


# In[125]:


stats.levene(A_B.A, A_B.B)


# **Hipotez Testi**

# In[126]:


stats.ttest_ind(A_B["A"], A_B["B"], equal_var = True)


# In[128]:


test_istatistigi, pvalue = stats.ttest_ind(A_B["A"], A_B["B"], equal_var=True)
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# **Nonparametrik Bağımsız İki Örneklem Testi**

# In[129]:


stats.mannwhitneyu(A_B["A"], A_B["B"])


# In[131]:


test_istatistigi, pvalue = stats.mannwhitneyu(A_B["A"], A_B["B"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# **Bağımlı İki Örneklem T Testi**

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


oncesi = pd.DataFrame([123,119,119,116,123,123,121,120,117,118,121,121,123,119,
            121,118,124,121,125,115,115,119,118,121,117,117,120,120,
            121,117,118,117,123,118,124,121,115,118,125,115])

sonrasi = pd.DataFrame([118,127,122,132,129,123,129,132,128,130,128,138,140,130,
             134,134,124,140,134,129,129,138,134,124,122,126,133,127,
             130,130,130,132,117,130,125,129,133,120,127,123])


# In[5]:


oncesi[0:5]


# In[6]:


sonrasi[0:5]


# In[7]:


np.arange(len(oncesi))


# In[9]:


#BIRINCI VERI SETI
AYRIK = pd.concat([oncesi, sonrasi], axis = 1)
AYRIK.columns = ["ONCESI", "SONRASI"]
print("'AYRIK' Veri Seti: \n\n ", AYRIK.head(), "\n\n")

#IKINCI VERI SETI
#ONCESI FLAG/TAG'INI OLUSTURMA
GRUP_ONCESI = np.arange(len(oncesi))
GRUP_ONCESI = pd.DataFrame(GRUP_ONCESI)
GRUP_ONCESI[:] = "ONCESI"
#FLAG VE ONCESI DEGERLERINI BIR ARAYA GETIRME
A = pd.concat([oncesi, GRUP_ONCESI], axis = 1)
#SONRASI FLAG/TAG'INI OLUSTURMA
GRUP_SONRASI = np.arange(len(sonrasi))
GRUP_SONRASI = pd.DataFrame(GRUP_SONRASI)
GRUP_SONRASI[:] = "SONRASI"

#FLAG VE SONRASI DEGERLERINI BIR ARAYA GETIRME
B = pd.concat([sonrasi, GRUP_SONRASI], axis = 1)

#TUM VERIYI BIR ARAYA GETIRME
BIRLIKTE = pd.concat([A,B])
BIRLIKTE

#ISIMLENDIRME
BIRLIKTE.columns = ["PERFORMANS","ONCESI_SONRASI"]
print("'BIRLIKTE' Veri Seti: \n\n", BIRLIKTE.head(), "\n")


# In[10]:


import seaborn as sns
sns.boxplot(x = "ONCESI_SONRASI", y = "PERFORMANS", data = BIRLIKTE)


# **Varsayım Kontrolleri**

# In[12]:


from scipy.stats import shapiro


# In[13]:


shapiro(AYRIK.ONCESI)


# In[15]:


shapiro(AYRIK.SONRASI)


# In[16]:


import scipy.stats as stats
stats.levene(AYRIK.ONCESI, AYRIK.SONRASI)


# **Hipotez Testi**

# In[17]:


stats.ttest_rel(AYRIK.ONCESI, AYRIK.SONRASI)


# In[18]:


test_istatistigi, pvalue = stats.ttest_rel(AYRIK["ONCESI"], AYRIK["SONRASI"])
print('Test İstatistiği = %.5f, p-değeri = %.5f' % (test_istatistigi, pvalue))


# **Nonparametrik Bağımlı İki Örneklem Testi**

# In[19]:


stats.wilcoxon(AYRIK.ONCESI, AYRIK.SONRASI)


# In[20]:


test_istatistigi, pvalue = stats.wilcoxon(AYRIK["ONCESI"], AYRIK["SONRASI"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# **İki Örneklem Oran Testi**

# In[21]:


from statsmodels.stats.proportion import proportions_ztest


# In[22]:


import numpy as np
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])


# In[23]:


proportions_ztest(count = basari_sayisi, nobs = gozlem_sayilari)


# **Varyans Analizi**

# In[24]:


A = pd.DataFrame([28,33,30,29,28,29,27,31,30,32,28,33,25,29,27,31,31,30,31,34,30,32,31,34,28,32,31,28,33,29])

B = pd.DataFrame([31,32,30,30,33,32,34,27,36,30,31,30,38,29,30,34,34,31,35,35,33,30,28,29,26,37,31,28,34,33])

C = pd.DataFrame([40,33,38,41,42,43,38,35,39,39,36,34,35,40,38,36,39,36,33,35,38,35,40,40,39,38,38,43,40,42])

dfs = [A, B, C]

ABC = pd.concat(dfs, axis = 1)
ABC.columns = ["GRUP_A","GRUP_B","GRUP_C"]
ABC.head()


# **Varsayım Kontrolü**

# In[25]:


from scipy.stats import shapiro 


# In[26]:


shapiro(ABC["GRUP_A"])


# In[27]:


shapiro(ABC["GRUP_B"])


# In[28]:


shapiro(ABC["GRUP_C"])


# In[29]:


stats.levene(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])


# **Hipotez Testi**

# In[31]:


from scipy.stats import f_oneway


# In[32]:


f_oneway(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])


# In[33]:


print('{:.5f}'.format(f_oneway(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])[1]))


# In[34]:


ABC.describe().T


# **Nonparametrik Hipotez Testi**

# In[35]:


from scipy.stats import kruskal


# In[36]:


kruskal(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])


# **Korelasyon Analizi**

# In[37]:


import seaborn as sns 
tips = sns.load_dataset('tips')
df = tips.copy()
df.head()


# In[38]:


df["total_bill"] = df["total_bill"] - df["tip"]


# In[39]:


df.head()


# In[40]:


df.plot.scatter("tip","total_bill")


# **Varsayım Kontrolü**

# In[41]:


from scipy.stats import shapiro


# In[43]:


test_istatistigi, pvalue = shapiro(df["tip"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

test_istatistigi, pvalue = shapiro(df["total_bill"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# **Hipotez Testi**

# **Korelasyon Katsayısı**

# In[44]:


df["tip"].corr(df["total_bill"])


# In[45]:


df["tip"].corr(df["total_bill"], method = "spearman")


# **Korelasyonun Anlamlılığının Testi**

# In[46]:


from scipy.stats.stats import pearsonr


# In[47]:


test_istatistigi, pvalue = pearsonr(df["tip"],df["total_bill"])

print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# **Nonparametrik Hipotez Testi**

# In[49]:


from scipy.stats import stats
stats.spearmanr(df["tip"],df["total_bill"])


# In[50]:


test_istatistigi, pvalue = stats.spearmanr(df["tip"],df["total_bill"])

print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# In[51]:


test_istatistigi, pvalue = stats.kendalltau(df["tip"],df["total_bill"])

print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

