from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Path to your .pkl file
model=pickle.load(open('model\model.pkl','rb'))
preprocessor=pickle.load(open('model\preprocessor.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')




@app.route('/predict',methods=['GET','POST'])
def predectdata():
    if request.method == 'GET':
        return render_template('predict.html')

    else:
        data=CustomData(
        
            
        loan_id=(request.form.get('loan_id')),   
        Gender=(request.form.get('gender')),
        Married=(request.form.get('married')),	
        Dependents=(request.form.get('dependents')),	
        Education=(request.form.get('education')),
        Self_Employed=(request.form.get('self_employed')),	
        ApplicantIncome=(request.form.get('applicant_income')),
        CoapplicantIncome=(request.form.get('coapplicant_income')),
        LoanAmount=(request.form.get('loan_amount')),
        Loan_Amount_Term=(request.form.get('loan_amount_term')),
        Credit_History=(request.form.get('credit_history')),
        Property_Area=(request.form.get('property_area')),	
        DayOfBirth=(request.form.get('day_of_birth')),
        MonthOfBirth=(request.form.get('month_of_birth')),	
        YearOfBirth  =(request.form.get('year_of_birth'))
        
        
        )

        
        # loan_id,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,DayOfBirth,MonthOfBirth,YearOfBirth])
        new_data_scaled=preprocessor.transform([loan_id,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,DayOfBirth,MonthOfBirth,YearOfBirth])
        result=model.predict([loan_id,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,DayOfBirth,MonthOfBirth,YearOfBirth])
            
        return render_template('predict.html',final_result=result)
    
    
    

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)