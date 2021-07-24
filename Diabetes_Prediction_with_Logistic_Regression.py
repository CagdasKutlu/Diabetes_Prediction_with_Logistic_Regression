import pandas as pd
from datetime import date

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action = "ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


# Data
def load():
    data = pd.read_csv("datasets/diabetes.csv")
    return data

df = load()

# 1) Keşifçi Veri Analizi (EDA)

print("##################### Shape #####################")
print(df.shape)
print("##################### Types #####################")
print(df.dtypes)
print("##################### Head #####################")
print(df.head())
print("##################### Tail #####################")
print(df.tail())
print("##################### NA #####################")
print(df.isnull().sum())
print("##################### Describe ##############")
print(df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T)

# Veri setinde her ne kadar NA değer gözlemlenmiyor olsa da değişkenlerin çoğunda sıfır değeri bol miktarda gözlemlenmektedir.
# "min" ve "quantile" değerlerini gözlemleyecek olursak; SkinThickness, BMI ve Insulin gibi değişkenlerde sıfır değerinin
# ağırlığını gözlemleyebiliriz. İnsan vücudu için imkansıza yakın bir durum olduğu için biz sıfır değerlerine ilerleyen
# süreçlerde NA muamelesi yapacağız.

# Hedef değişken değerleri
df["Outcome"].value_counts()

print(100 * df["Outcome"].value_counts() / len(df))

#Sayısal ve Aynı Zamanda KAtegorik olmayan Değişkenleri Yakalıyoruz
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in num_but_cat]
num_cols

# Sayısal Değişkenler için Histogram Grafikleri
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0])
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1])
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0])
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1])
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0])
sns.distplot(df.BMI, bins = 20, ax=ax[3,1])
plt.show()

# Değişkenlerin Hedef Değişkenin 0 ve 1 olma Durumuna Göre Ortalamaları
for col in num_cols:
    print(df.groupby("Outcome").agg({col: "mean"}), end = "\n\n")

# Değişkenlerin Birbirleri Arası Korelasyonu
corr = df.corr()
sns.heatmap(corr)
plt.show()

# 2) Veri Ön İşleme (Data Preprocessing)
# 2.1) Eksik Değerler (Missing Values)

# Eksik Değerler
df.isnull().sum()

# 0 değerlerinin NA değerlere dönüştürülmesi
zero_columns = ["Glucose","BloodPressure","Insulin","BMI","SkinThickness"]

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

# Eksik Değerler
df.isnull().sum()

# Eksik Değerlerin Görselleştirilmesi
msno.matrix(df)
plt.show()

# Oluşturduğumuz NA değerlere meadian ataması yapıyoruz.
for col in num_cols:
    df[(df["Outcome"] == 0) & (df[col].isnull() == True)] = df[(df["Outcome"] == 0) &
                                                               (df[col].isnull() == True)].fillna(df[df["Outcome"] == 0].median())
    df[(df["Outcome"] == 1) & (df[col].isnull() == True)] = df[(df["Outcome"] == 1) &
                                                               (df[col].isnull() == True)].fillna(df[df["Outcome"] == 1].median())

# Eksik Değerler ve Betimsel İstatistikler
df.isnull().sum()
df.describe().T

# 2.2) Aykırı Değerler (Outliers)
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    if df[df[col] > up_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    elif df[df[col] < low_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    else:
        print(col,"NO",(low_limit,up_limit))

# Replace with Thresholds
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    df.loc[(df[col] < low_limit), col] = low_limit
    df.loc[(df[col] > up_limit), col] = up_limit
    if df[df[col] > up_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    elif df[df[col] < low_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    else:
        print(col,"NO",(low_limit,up_limit))

df.describe().T

# 3) Özellik Mühendisliği(Feature Engineering)
def fea_eng(df):
    df["INPEDIGREE"] = df["Insulin"] * df["DiabetesPedigreeFunction"]
    df["SKINBMI"] = df["SkinThickness"] * df["BMI"]

    df.loc[(df["BloodPressure"] > 60) & (df["BloodPressure"] <= 80),"NEW_BP_CAT"]= "Ideal"
    df.loc[(df["BloodPressure"] > 80) & (df["BloodPressure"] < 90),"NEW_BP_CAT"]= "NormalBP"
    df.loc[(df["BloodPressure"] >= 90) & (df["BloodPressure"] < 120),"NEW_BP_CAT"]= "Hyper"
    df.loc[(df["BloodPressure"] >= 120),"NEW_BP_CAT"]= "Hypertensive crisis"
    df.loc[(df["BloodPressure"] <= 60),"NEW_BP_CAT"]= "Hypo"

    df.loc[(df["Pregnancies"] == 0) ,"NEW_P_CAT"]= "No_Child"
    df.loc[(df["Pregnancies"] > 0) & (df["Pregnancies"] <= 3 ),"NEW_P_CAT"]= "Child_03"
    df.loc[(df["Pregnancies"] > 3) & (df["Pregnancies"] <= 6 ),"NEW_P_CAT"]= "Child_36"
    df.loc[(df["Pregnancies"] > 6) ,"NEW_P_CAT"]= "Too_much"

    df.loc[(df["BMI"] < 18.5) ,"NEW_BMI_CAT"]= "UnderWeight"
    df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9 ),"NEW_BMI_CAT"]= "NormalBMI"
    df.loc[(df["BMI"] >= 25) & (df["BMI"] <= 30 ),"NEW_BMI_CAT"]= "Overweight"
    df.loc[(df["BMI"] > 30) & (df["BMI"] <= 35 ),"NEW_BMI_CAT"]= "Type1_Obese"
    df.loc[(df["BMI"] > 35) & (df["BMI"] <= 40 ),"NEW_BMI_CAT"]= "Type2_Obese"
    df.loc[(df["BMI"] > 40) ,"NEW_BMI_CAT"]= "Morbid_Obese"

    df.loc[(df["Insulin"] < 70) ,"NEW_I_CAT"]= "Hypoglycemia"
    df.loc[(df["Insulin"] >= 70) & (df["Insulin"] < 100 ),"NEW_I_CAT"]= "NormalI"
    df.loc[(df["Insulin"] >= 100) & (df["Insulin"] <= 125 ),"NEW_I_CAT"]= "PrediabetityI"
    df.loc[(df["Insulin"] >= 126) ,"NEW_I_CAT"]= "DiabetesI"

    df.loc[(df["Glucose"] < 100) ,"NEW_G_CAT"]= "Lower"
    df.loc[(df["Glucose"] >= 100) & (df["Glucose"] < 140 ),"NEW_G_CAT"]= "NormalG"
    df.loc[(df["Glucose"] >= 140) & (df["Glucose"] <= 199 ),"NEW_G_CAT"]= "PrediabetityG"
    df.loc[(df["Glucose"] >= 200) ,"NEW_G_CAT"]= "DiabetesG"

    df.loc[(df["Age"] < 30) ,"NEW_AGE_CAT"]= "Young"
    df.loc[(df["Age"] >= 30) & (df["Age"] < 40 ),"NEW_AGE_CAT"]= "Middle"
    df.loc[(df["Age"] >= 40) & (df["Age"] < 50 ),"NEW_AGE_CAT"]= "Old"
    df.loc[(df["Age"] >= 50) ,"NEW_AGE_CAT"]= "Elder"
    return df

df = fea_eng(df)

# değişken İsimlerinin Büyütülmesi
df.columns = [col.upper() for col in df.columns]

# Feature Engineering sonucunda oluşan eksik değer var mı? Kontrol edelim.
df.isnull().sum()

# Feature Engineering Sonrası Değişken Tutma İşlemini Tekrarlamalıyız
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in num_but_cat]
num_cols

# Kategorik Değişkenler Oluşturduğumuz için Onları da Yakalıyoruz.
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_cols = cat_cols + num_but_cat
cat_cols

# 4) Encoding and Scale
# 4.1) Label Encoding
binary_col = [col for col in df.columns if
                  df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_col:
    labelencoder = LabelEncoder()
    df[col] = labelencoder.fit_transform(df[col])

# 4.2) One-Hot Encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = pd.get_dummies(df, columns=ohe_cols, drop_first = True)

# 4.3) Robust Scaler
for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])

# 5) Model ve Final
y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_

log_model.coef_


# Tahmin'lerin oluşturulması ve kaydedilmesi
y_pred = log_model.predict(X_train)
y_pred[0:10]
y_train[0:10]

# Sınıf olasılıkları (0. indextekiler 0 a ait olma 1. indekstekiler 1 sınıfına ait olma)
log_model.predict_proba(X_train)[0:10]

# 1. sınıfa ait olma olasılıkları:
y_prob = log_model.predict_proba(X_train)[:, 1]

# Final Başarı Değelendirme
# Train Accuracy
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred);

# Test
# AUC Score için y_prob
y_prob = log_model.predict_proba(X_test)[:, 1]

# Diğer metrikler için y_pred
y_pred = log_model.predict(X_test)

# ACCURACY
accuracy_score(y_test, y_pred)

# PRECISION
precision_score(y_test, y_pred)

# RECALL
recall_score(y_test, y_pred)

# F1
f1_score(y_test, y_pred)

# Roc Eğrisi
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

# Modeli oluşturduğumuz değişkenlerin etki düzeylerini gösteren feature importance grafiğini oluşturuyoruz.
import math as mt
feature_importance = pd.DataFrame(X_train.columns, columns = ["feature"])
feature_importance["importance"] = pow(mt.e, log_model.coef_[0])
feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
ax = feature_importance.plot.barh(x='feature', y='importance', figsize=(15,10), fontsize=10)
plt.xlabel('Importance', fontsize=20)
plt.ylabel('Features', fontsize=10)
# plt.yticks(rotation=50)
plt.show()