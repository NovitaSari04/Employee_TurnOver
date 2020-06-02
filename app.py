from flask import Flask, render_template, redirect, request, jsonify, url_for
import json
import numpy as np 
import pandas as pd 

# SQL
import mysql.connector

# Model ML
import joblib

# Database
dbhr = mysql.connector.connect(
host = 'localhost',
port = 3306,
user = 'root',
passwd = 'novita',
use_pure=True,
database = 'employee_data')

db = dbhr.cursor()

# Function for categorical features

def travel (x) :
    if x == 'Travel_Frequently' :
        return [1,0]
    elif x == 'Travel_Rarely' :
        return [0,1]
    else :
        return [0,0]

def dept(x) :
    if x == 'Research & Development' :
        return [1,0]
    elif x == 'Sales' :
        return [0,1]
    else :
        return [0,0]

def education(x) :
    if x == 'Life Sciences' :
        return [1,0,0,0,0]
    elif x == 'Marketing' :
        return [0,1,0,0,0]
    elif x == 'Medical' :
        return [0,0,1,0,0]
    elif x == 'Other' :
        return [0,0,0,1,0]
    elif x == 'Technical Degree' :
        return [0,0,0,0,1]
    else :
        return [0,0,0,0,0]

def role(x) :
    if x == 'Human Resources' :
        return [1,0,0,0,0,0,0,0]
    elif x == 'Laboratory Technician' :
        return [0,1,0,0,0,0,0,0]
    elif x == 'Manager' :
        return [0,0,1,0,0,0,0,0]
    elif x == 'Manufacturing Director' :
        return [0,0,0,1,0,0,0,0]
    elif x == 'Research Director' :
        return [0,0,0,0,1,0,0,0]
    elif x == 'Research Scientist' :
        return [0,0,0,0,0,1,0,0]
    elif x == 'Sales Executive' :
        return [0,0,0,0,0,0,1,0]
    elif x == 'Sales Representative' :
        return [0,0,0,0,0,0,0,1]
    else :
        return [0,0,0,0,0,0,0,0]

def marital(x) :
    if x == 'Married' :
        return [1,0]
    elif x == 'Single' :
        return [0,1]
    else :
        return [0,0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/model')
def model():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        return redirect('/')

@app.route('/model-result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET':
        return redirect('/')
    else:
        user_data = request.form

        # Preprocessing form => Data
        user_test = pd.DataFrame([{
            'Age': int(user_data['age']),
            'DailyRate': int(user_data['drate']),
            'DistanceFromHome': int(user_data['distance']),
            'Education': int(user_data['edu']),
            'EnvironmentSatisfaction': int(user_data['enviro']),
            'Gender': int(user_data['gender']),
            'HourlyRate': int(user_data['hrate']),
            'JobInvolvement': int(user_data['jobinv']),
            'JobLevel': int(user_data['joblvl']),
            'JobSatisfaction': int(user_data['jobstf']),
            'MonthlyIncome': int(user_data['mincome']),
            'MonthlyRate': int(user_data['mrate']),
            'NumCompaniesWorked': int(user_data['numcomp']),
            'OverTime': int(user_data['ot']),
            'PercentSalaryHike': int(user_data['salhike']),
            'PerformanceRating': int(user_data['perf']),
            'RelationshipSatisfaction': int(user_data['relatio']),
            'StockOptionLevel': int(user_data['stock']),
            'TotalWorkingYears': int(user_data['workyear']),
            'TrainingTimesLastYear': int(user_data['training']),
            'WorkLifeBalance': int(user_data['balance']),
            'YearsAtCompany': int(user_data['compyear']),
            'YearsInCurrentRole': int(user_data['roleyear']),
            'YearsSinceLastPromotion': int(user_data['promotion']),
            'YearsWithCurrManager': int(user_data['mngyear'])
        }])

        # Handling Multiclass
        # Business Travel
        BT = ['BusinessTravel_Travel_Frequently',
        'BusinessTravel_Travel_Rarely']
        BusinessTravel = travel(str(user_data['travel']))
        dfBT = pd.DataFrame([dict(zip(BT, BusinessTravel))])
        user_test = pd.concat([user_test, dfBT], axis='columns')

        # Department
        DP = ['Department_Research & Development',
        'Department_Sales']
        Department = dept(str(user_data['dept']))
        dfDP = pd.DataFrame([dict(zip(DP, Department))])
        user_test = pd.concat([user_test, dfDP], axis='columns')

        # Education Field
        EF = ['EducationField_Life Sciences',
        'EducationField_Marketing',
        'EducationField_Medical',
        'EducationField_Other',
        'EducationField_Technical Degree']
        EducationField = education(str(user_data['education']))
        dfEF = pd.DataFrame([dict(zip(EF, EducationField))])
        user_test = pd.concat([user_test, dfEF], axis='columns')

        # Job Role
        JR = ['JobRole_Human Resources',
        'JobRole_Laboratory Technician',
        'JobRole_Manager',
        'JobRole_Manufacturing Director',
        'JobRole_Research Director',
        'JobRole_Research Scientist',
        'JobRole_Sales Executive',
        'JobRole_Sales Representative',]
        JobRole = role(str(user_data['role']))
        dfJR = pd.DataFrame([dict(zip(JR, JobRole))])
        user_test = pd.concat([user_test, dfJR], axis='columns')

        # Marital Status
        MS = ['MaritalStatus_Married',
        'MaritalStatus_Single']
        MaritalStatus = marital(str(user_data['marital']))
        dfMS = pd.DataFrame([dict(zip(MS, MaritalStatus))])
        user_test = pd.concat([user_test, dfMS], axis='columns')

        # DataBaseInput
        Age = int(user_data['age']),
        DailyRate = int(user_data['drate']),
        DistanceFromHome = int(user_data['distance']),
        Education = int(user_data['edu']),
        EnvironmentSatisfaction = int(user_data['enviro']),
        Gender = int(user_data['gender']),
        HourlyRate = int(user_data['hrate']),
        JobInvolvement = int(user_data['jobinv']),
        JobLevel = int(user_data['joblvl']),
        JobSatisfaction = int(user_data['jobstf']),
        MonthlyIncome = int(user_data['mincome']),
        MonthlyRate = int(user_data['mrate']),
        NumCompaniesWorked = int(user_data['numcomp']),
        OverTime = int(user_data['ot']),
        PercentSalaryHike = int(user_data['salhike']),
        PerformanceRating = int(user_data['perf']),
        RelationshipSatisfaction = int(user_data['relatio']),
        StockOptionLevel = int(user_data['stock']),
        TotalWorkingYears = int(user_data['workyear']),
        TrainingTimesLastYear = int(user_data['training']),
        WorkLifeBalance = int(user_data['balance']),
        YearsAtCompany = int(user_data['compyear']),
        YearsInCurrentRole = int(user_data['roleyear']),
        YearsSinceLastPromotion = int(user_data['promotion']),
        YearsWithCurrManager = int(user_data['mngyear'])

        # savedata = 'insert into new_data (Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, Gender, HourlyRate,JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, BusinessTravel, Department, EducationField, JobRole, MaritalStatus) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        # insertdata = (Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, Gender, HourlyRate,JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, BusinessTravel, Department, EducationField, JobRole, MaritalStatus)
        # db.execute(savedata, insertdata)
        # dbhr.commit()

        # Probability
        user_prob = finalmodel.predict_proba(user_test)
        probability = int(user_prob[:,1]*100)
        if probability <= 60 :
            return render_template('result.html', probability = probability, recommendation ="Low-Risk", color = '#00FF00')
        elif probability >= 80 :
            return render_template('result.html', probability = probability, recommendation ="High-Risk", color = '#ff0000' )
        else:
            return render_template('result.html', probability = probability, recommendation ="Medium-Risk", color = '#9CC0E7')

if __name__ == "__main__":
    finalmodel = joblib.load('finalmodel')

    app.run(debug=True, port=5000)