from django.shortcuts import render
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Create your views here.

def home(request):
    data = pd.read_csv(os.path.join(BASE_DIR, 'static/credit_data.csv'))
    X = data.drop(['default', 'clientid','LTI'], axis = 1)
    y = data['default']
    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size = 0.3)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    rfc = RandomForestClassifier(n_estimators=200)
    model = rfc.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    pres = f1_score(y_test, prediction)
    if request.method =="GET":
        income = request.GET.get('income')
        age = request.GET.get('age')
        loan = request.GET.get('loan')
        pdct = model.predict(sc.transform([[income or 0, age or 0, loan or 0]]))
        prob = []
        prob = model.predict_proba(sc.transform([[income or 0, age or 0, loan or 0]]))
       
    if pdct == 0:
        prob = prob[0][0]
    else:
        prob = prob[0][1]
    msg = []
    if pdct == 0:
        msg = 'success'
    else:
        msg = 'danger'
    context = {
        "datas": accuracy*100,
        "precsion": pres*100,
        "prediction": pdct[0],
         "msg": msg,
         "prob": prob*100
  
    }
    return render(request, 'home.html', context)


