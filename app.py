
# import warnings filter
import pandas as pd

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('heart10.csv', header = None)

data.columns = ['age', 'chest_pain_type', 'rest_blood_pressure', 'blood_sugar', 'rest_electro',
              'max_heart_rate', 'exercice_angina', 'disease']

data.head(5)


#######################



#copy how many rows and columns i have
"(Rows, columns): " + str(data.shape)

#return the names of columns

data.columns

#return how many every att has values
data.nunique(axis=0)

#describe data
data.describe()

#if data has null values
data.isnull().sum()
data.head(5)
len(data)

data=data.loc[(data['rest_electro'] != '?')]
len(data)

######################################
data['disease'] = data.disease.map({'negative': 0, 'positive': 1})
data['disease'].unique()

data['chest_pain_type'] =data.chest_pain_type.map({'asympt': 0, 'atyp_angina': 1, 'non_anginal': 2, 'typ_angina': 3})
data['chest_pain_type'].unique()

data['rest_electro'] = data.rest_electro.map({'normal': 0, 'left_vent_hyper': 1, 'st_t_wave_abnormality': 2})
data['rest_electro'].unique()
data['exercice_angina'] = data.exercice_angina.map({'yes': 1, 'no': 0})
data['exercice_angina'].unique()


data.dtypes
import matplotlib.pyplot as plt
import seaborn as sns
corr = data.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot=True,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))
#plt.title(' ')
#plt.show()
#############################


###################################
##############################################

pos_data = data[data['disease']==1]
pos_data.describe()

neg_data = data[data['disease']==0]
neg_data.describe()

#"(Positive Patients ST chest): " + str(pos_data['chest_pain_type'].mean())
#"(Negative Patients ST chest): " + str(neg_data['chest_pain_type'].mean())


#"(Positive Patients blood_sugar: " + str(pos_data['blood_sugar'].mean())
#"(Negative Patients blood_sugar): " + str(neg_data['blood_sugar'].mean())

##################################
######################################

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler as ss

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
########################id3
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
classification_report(y_test, y_pred)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

# get importance
importance = model.feature_importances_

# summarize feature importance
#for i,v in enumerate(importance):
   # print('Feature: %0d, Score: %.5f' % (i,v))

index= data.columns[:-1]
importance = pd.Series(model.feature_importances_, index=index)
importance.nlargest(13).plot(kind='barh', colormap='winter')
print("Desicion Tree")
print('The result is:')
if print(model.predict(sc.transform([[66,0,140,0,0,160,0]]))) ==data.iloc[1, 7]:
    print("positive")
else:
    print("negative")

# if the values which we expect is right or not, the first value is the expected one and the second is the real one
y_pred = model.predict(X_test)
np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)


#############33BAYES

from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

model2 = GaussianNB()
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
classification_report(y_test, y_pred2)


print("Bayes")


# if the values which we expect is right or not, the first value is the expected one and the second is the real one
y_pred2 = model.predict(X_test)
np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1)

from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
app = Flask(__name__)

# Flask-WTF requires an encryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'

# Flask-Bootstrap requires this line
Bootstrap(app)
class NameForm(FlaskForm):
    name = StringField('Which actor is your favorite?', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
def home():
    return("ruselt")

@app.route('/form', methods=['GET', 'POST'])
def login():
    result = None
    if request.method == 'POST':
        age=request.form['age']
        chest_pain_type=request.form['chest_pain_type']
        rest_blood_pressure=request.form['rest_blood_pressure']
        blood_sugar=request.form['blood_sugar']
        rest_electro=request.form['rest_electro']
        max_heart_rate=request.form['max_heart_rate']
        exercice_angina=request.form['exercice_angina']

        if model2.predict(sc.transform([[age, chest_pain_type, rest_blood_pressure, blood_sugar, rest_electro, max_heart_rate, exercice_angina]])) == 1:
            result= "Positive"
        else:
           result= "Negative"
    return render_template('form.html', result=result)