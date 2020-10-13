


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
from sklearn.preprocessing import StandardScaler


from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


#cred_pred=pickle.load(open('finalized_model2.pkl','rb'))
model123 = pickle.load(open('finalized_model2.pkl','rb'))
scaler_data = pickle.load(open('scaler_Train.pkl','rb'))

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_credit_balance(
         loan_amnt,
         terms,
         Rate_of_intrst,
         grade,
         home_ownership,
         annual_inc,
         verification_status,
         purpose,
         debt_income_ratio,
         delinq_2yrs,
         inq_last_6mths,
         mths_since_last_delinq,
         numb_credit,
         pub_rec,
         total_credits,
         initial_list_status,
         total_rec_int,
         total_rec_late_fee,
         recoveries,
         collection_recovery_fee,
         collections_12_mths_ex_med,
         acc_now_delinq,
         tot_colle_amt,
         tot_curr_bal,
         last_week_pay_nos,
         Experience_status): 


 
    annual_inc  = 120000 if int(annual_inc) >=120000 else annual_inc
    debt_income_ratio  = 125.25 if int(debt_income_ratio) >=125.25 else debt_income_ratio
    delinq_2yrs  = 10 if int(delinq_2yrs) >=10 else delinq_2yrs
    inq_last_6mths  = inq_last_6mths if int(inq_last_6mths) >=10 else inq_last_6mths
    mths_since_last_delinq  = 100 if int(mths_since_last_delinq) >=100 else mths_since_last_delinq
    numb_credit  = 40 if int(numb_credit) >=40 else numb_credit
    total_credits  = 75 if int(total_credits) >=75 else total_credits
    total_rec_int  = 15000 if int(total_rec_int) >=15000 else total_rec_int
    total_rec_late_fee  = 100 if int(total_rec_late_fee) >=100 else total_rec_late_fee
    recoveries  = 100 if int(recoveries) >=100 else recoveries

    verification_status= 1 if verification_status=='Verified' or verification_status=='Source Verified' else 0
    Exp=0
    if Experience_status == '9 years':
        Exp = 9
    elif Experience_status == '< 1 year':
        Exp = 0
    elif Experience_status == '10+ years':
        Exp = 10
    elif Experience_status == '5 years':
        Exp = 5
    elif Experience_status == '8 years':
        Exp = 8
    elif Experience_status == '7 years':
        Exp = 7
    elif Experience_status == '4 years':
        Exp = 4
    elif Experience_status == '1 year':
        Exp = 1
    elif Experience_status == '3 years':
        Exp = 3
    elif Experience_status == '6 years':
        Exp = 6      
    else:
        Exp = 0;
        
    Experience_status = int(Exp)
   
   
    if    purpose == 'other':
      purpose = 10 
    elif  purpose == 'credit_card':
        purpose = 1
    elif  purpose == 'home_improvement':
        purpose = 5
    elif  purpose == 'renewable_energy':
        purpose = 11    
    elif  purpose == 'debt_consolidation':
        purpose = 3 
    elif  purpose == 'medical':
        purpose = 8
    elif  purpose == 'car':
        purpose = 0
    elif  purpose == 'moving':
        purpose = 9
    elif  purpose == 'major_purchase':
        purpose = 7
    elif  purpose == 'small_business':
        purpose = 12     
    elif  purpose == 'house':
        purpose = 6
    elif  purpose == 'vacation':
        purpose = 13
    elif  purpose == 'wedding':
        purpose = 14    
    elif  purpose == 'educational':
        purpose = 4  
    else:
        purpose = 0
        
    purpose=int(purpose)    
             
    
   
    data_imp = {'loan_amnt': [loan_amnt]
       , 'terms': [terms] , 'Rate_of_intrst': [Rate_of_intrst], 'grade': [grade], 'home_ownership': [home_ownership],
       'annual_inc': [annual_inc], 'verification_status': [verification_status], 'purpose': [purpose], 'debt_income_ratio': [debt_income_ratio],
       'delinq_2yrs': [delinq_2yrs], 'inq_last_6mths': [inq_last_6mths], 'mths_since_last_delinq': [mths_since_last_delinq],
       'numb_credit': [numb_credit], 'pub_rec': [pub_rec], 'total_credits': [total_credits],
       'initial_list_status': [initial_list_status], 'total_rec_int': [total_rec_int], 'total_rec_late_fee': [total_rec_late_fee],
       'recoveries': [recoveries], 'collection_recovery_fee': [collection_recovery_fee], 'collections_12_mths_ex_med': [collections_12_mths_ex_med],
       'acc_now_delinq': [acc_now_delinq], 'tot_colle_amt': [tot_colle_amt], 'tot_curr_bal': [tot_curr_bal], 'last_week_pay_nos': [last_week_pay_nos],
       'Experience_status': [Experience_status]}
    df_test2 = pd.DataFrame(data=data_imp) 
   
    
   
    transformed_values=scaler_data.transform(df_test2)
     
  #  transformed_values=scaler_data.transform(480016,4000,60,20.25,5,0,65000.0,1,2,18.09,1.0,3.0,8.0,24.0,0.0,52.0,0,2357.45,0.0,0.0,0.0,0.0,0.0,232.0876107674302,160864.77786374473,23,10.0)

        
  #  n2=np.array([[1.82382,-0.65403,0.479622,0.161627,1.11764,0.726537,0.660216,-0.250186,1.08021,-0.383998,1.32643,1.47203,0.476981,-0.335833,0.415914,-0.973894,2.92744,-0.101825,-0.125747,-0.0874649,-0.109132,-0.0662241,0.117231,-0.034814,-0.807533,1.11919]])
  #  print(model123.predict(n2))
   
    #prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    prediction=model123.predict(transformed_values).astype(int)

    print(prediction)
    return prediction   
        
    
def predict_credit_balance_test():         
    
    n2=np.array([[1.82382,-0.65403,0.479622,0.161627,1.11764,0.726537,0.660216,-0.250186,1.08021,-0.383998,1.32643,1.47203,0.476981,-0.335833,0.415914,-0.973894,2.92744,-0.101825,-0.125747,-0.0874649,-0.109132,-0.0662241,0.117231,-0.034814,-0.807533,1.11919]])
    print(model123.predict(n2))
   
    #prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    prediction=model123.predict(n2).astype(int)

    print(prediction)
    return prediction



def main():
    st.title("Revolving Balance  Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> All details must be filled in whole numbers</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    annual_inc = (st.text_input("Enter the Annual Income "))
    debt_income_ratio = (st.text_input("Enter the debt Income Ratio"))
    delinq_2yrs = (st.text_input("Delinquency of past 2 years"))
    inq_last_6mths = (st.text_input("Inquiries made in past 6 months"))
    mths_since_last_delinq = (st.text_input("number of months since last delinq"))
    numb_credit = (st.text_input("number of credit Lines"))
    total_credits = (st.text_input("total_credits"))
    total_rec_int = (st.text_input("Total interest received till date"))
    total_rec_late_fee = (st.text_input("Late fee received till date"))
    recoveries = (st.text_input("Recoveries made"))
    collection_recovery_fee = (st.text_input("post charge off collection fee"))
    loan_amnt = (st.text_input("loan_amnt"))
    Rate_of_intrst = (st.text_input("Rate_of_intrst"))
    collections_12_mths_ex_med = (st.text_input("number of collections in last 12 months excluding medical collections"))
    tot_colle_amt = (st.text_input("tot_colle_amt"))
    tot_curr_bal = (st.text_input("tot_curr_bal"))
    last_week_pay_nos = (st.text_input("last_week_pay_nos"))
    acc_now_delinq = (st.text_input("acc_now_delinq"))
    pub_rec = (st.text_input("pub_rec"))

    grade = (st.selectbox(
     'Grade  (select  A-0,B-1,C-2,D-3,E-4,F-5,G-6) ' ,
    (1,2,3,4,5,6)))
    initial_list_status = (st.selectbox(
     '"unique listing status of the loan - W(Waiting) -1 ,F(Forwarded) - 0","Type Here' ,
    (0,1)))
    terms = (st.selectbox(
     'Terms - in months',
    (36,60)))
    home_ownership = (st.selectbox(
     'Home_onwership - Select 0-MORTGAGE  1-OTHER  OWN-2 RENT 3 ',
    (0,1,2,3)))
    verification_status = (st.selectbox(
     'Verification_status  Verified-2 Not Verified-0  Source Verified -1',
    (0,1,2)))
    purpose = st.selectbox(
     'Purpose',
    ('other', 'credit_card', 'home_improvement', 'renewable_energy',
       'debt_consolidation', 'medical', 'car', 'moving', 'major_purchase',
       'small_business', 'house', 'vacation', 'wedding', 'educational'))
    Experience_status = st.selectbox(
     'Experience Status',
    ('9 years', '8 years', '10+ years', '2 years', '3 years', '7 years',
       '< 1 year', '6 years', '4 years', '1 year', '5 years'))
  
    
    
    result=""
    if st.button("Predict"):
        #result=predict_credit_balance()
        result=predict_credit_balance(loan_amnt,
                                     terms,
                                     Rate_of_intrst,
                                     grade,
                                     home_ownership,
                                     annual_inc,
                                     verification_status,
                                     purpose,
                                     debt_income_ratio,
                                     delinq_2yrs,
                                     inq_last_6mths,
                                     mths_since_last_delinq,
                                     numb_credit,
                                     pub_rec,
                                     total_credits,
                                     initial_list_status,
                                     total_rec_int,
                                     total_rec_late_fee,
                                     recoveries,
                                     collection_recovery_fee,
                                     collections_12_mths_ex_med,
                                     acc_now_delinq,
                                     tot_colle_amt,
                                     tot_curr_bal,
                                     last_week_pay_nos,
                                     Experience_status)
                                    
    st.success('The Revolving Balance is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    