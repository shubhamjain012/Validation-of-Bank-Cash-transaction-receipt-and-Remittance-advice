import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix 
from flask import Flask
from flask import request
import json

app = Flask(__name__)

dataset = pd.read_csv("DataSet.csv")

from sklearn.preprocessing import LabelEncoder

dPayment_lb_conv = LabelEncoder()
dataset["date_of_Payment"] = dPayment_lb_conv.fit_transform(dataset["date_of_Payment"])

Branch_lb_conv = LabelEncoder()
dataset["branch_ID"] = Branch_lb_conv.fit_transform(dataset["branch_ID"])

Bank_lb_conv = LabelEncoder()
dataset["bank_ID"] = Bank_lb_conv.fit_transform(dataset["bank_ID"])

#Tran_lb_conv = LabelEncoder()
#dataset["transaction_ID"] = Tran_lb_conv.fit_transform(dataset["transaction_ID"])

mode1_lb_conv = LabelEncoder()
dataset["mode_of_Transfer1"] = mode1_lb_conv.fit_transform(dataset["mode_of_Transfer1"])

dSubmission_lb_conv = LabelEncoder()
dataset["date_of_Submission"] = dSubmission_lb_conv.fit_transform(dataset["date_of_Submission"])

dInvoice_lb_conv = LabelEncoder()
dataset["invoice_Due_Date"] = dInvoice_lb_conv.fit_transform(dataset["invoice_Due_Date"])

Remit_lb_conv = LabelEncoder()
dataset["remittance_Number"] = Remit_lb_conv.fit_transform(dataset["remittance_Number"])

mode2_lb_conv = LabelEncoder()
dataset["mode_of_Transfer2"] = mode2_lb_conv.fit_transform(dataset["mode_of_Transfer2"])

#Status_lb_conv = LabelEncoder()
#dataset["Status"] = Status_lb_conv.fit_transform(dataset["Status"])  

# dataset["Date_of_Payment"] = LbEncode.fit_transform(dataset["Date_of_Payment"])
# dataset["Branch_ID"] = LbEncode.fit_transform(dataset["Branch_ID"])
# dataset["Bank_ID"] = LbEncode.fit_transform(dataset["Bank_ID"])
# dataset["Transaction_ID"] = LbEncode.fit_transform(dataset["Transaction_ID"])
# dataset["Mode_of_Transfer1"] = LbEncode.fit_transform(dataset["Mode_of_Transfer1"])
# dataset["Date_of_Submission"] = LbEncode.fit_transform(dataset["Date_of_Submission"])
# dataset["Invoice_Due_Date"] = LbEncode.fit_transform(dataset["Invoice_Due_Date"])
# dataset["Remittance_Number"] = LbEncode.fit_transform(dataset["Remittance_Number"])
# dataset["Mode_of_Transfer2"] = LbEncode.fit_transform(dataset["Mode_of_Transfer2"])
# dataset["Approved/Rejected"] = LbEncode.fit_transform(dataset["Approved/Rejected"])

X = dataset.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,13]]
y = dataset.iloc[:, [14]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# from sklearn.svm import SVC
# classifier = SVC(C = 0.1, gamma = 0.1, kernel = "rbf")
# classifier.fit(X_train, y_train.values.ravel())

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train.values.ravel())


@app.route('/MLrecommendation', methods=['GET', 'POST'])
def MLrecommendation():
    a = request.get_json(force=True)
    dPay = a["date_of_Payment"]
    acc_No =a["account_Number"]
    b_ID =a["branch_ID"]
    ba_ID =a["bank_ID"]
    #trn_Id =a["transaction_ID"]
    trn_Amt =a["transaction_Amount"]
    mod1_Trnf =a["mode_of_Transfer1"]
    dSub =a["date_of_Submission"]
    dDue =a["invoice_Due_Date"]
    rem_No =a["remittance_Number"]
    amt_Due =a["total_due_Amount"]
    amt_Paid =a["amount_Paid"]
    amt_Bal =a["balance_Amount"]
    mod2_Trnf =a["mode_of_Transfer2"]
    brw = a["brw_value"]
    
    features   = [ dPayment_lb_conv.transform([dPay]), 
				acc_No,
	            Branch_lb_conv.transform([b_ID]) , 
                Bank_lb_conv.transform([ba_ID]),
                #Tran_lb_conv.transform([trn_Id]),
				trn_Amt,
                mode1_lb_conv.transform([mod1_Trnf]),
                dSubmission_lb_conv.transform([dSub]),
                dInvoice_lb_conv.transform([dDue]),
                Remit_lb_conv.transform([rem_No]),
				amt_Due,	
				amt_Paid,	
				amt_Bal,		
				mode2_lb_conv.transform([mod2_Trnf])
			]
    
    prediction = knn.predict([np.asarray(features)])
    
    #ML1-------------------------------------------
    trn = int(acc_No)
    ttl_trn = dataset.account_Number.value_counts()[trn]
    
    df2 = dataset[dataset["account_Number"]==trn]
    df3 = df2[['account_Number','status']] 
    apd_trn = df3.status.value_counts()['Approved']
    
    ML1 = apd_trn / ttl_trn
    
    #ML2-------------------------------------------
    trn_Amount = int(trn_Amt)
    df4 = dataset[dataset["account_Number"]==trn]
    df5 = df4[['account_Number','transaction_Amount']]
    
    m = df5['transaction_Amount'].mean()
    R = df5['transaction_Amount'].ptp()  
    cal1 = m + 5*R
    cal1
    cal2 = m + 4*R
    cal2
    cal3 = m + 3*R
    cal3
    cal4 = m + 2*R
    cal4
    cal5 = m + 1*R
    cal5
    
    if trn_Amount >= cal1:
        ML2= 0.1
    elif trn_Amount < cal1 and trn_Amount >= cal2:
        ML2= 0.3
    elif trn_Amount < cal2 and trn_Amount >= cal3:
        ML2= 0.5
    elif trn_Amount < cal3 and trn_Amount >= cal4:
        ML2= 0.7
    elif trn_Amount < cal4 and trn_Amount >= cal5:
        ML2= 0.9    
    else:
        ML2= 0.9
    
    #ML3-------------------------------------------
    brw1 = int(brw)
    ML3 = brw1/6
    
    #ML4-------------------------------------------
    knn_predictions = knn.predict(X_test)  
    cm = confusion_matrix(y_test, knn_predictions)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    ML4 = recall[0]
    #ML4 = 346 / (346+292)
    
    #Total_confidence-------------------------------------------
    wml1 = 0.3
    wml2 = 0.1
    wml3 = 0.4
    wml4 = 0.2
    total_confidence = wml1*ML1 + wml2*ML2 + wml3*ML3 + wml4*ML4
    
    #total_confidence
    #released = {"ML1":ML1,"ML2":ML2,"ML3":ML3,"ML4":ML4,"total_confidence":total_confidence}
    #json.dumps(released)
    return  '{} {:.2g} {:.2g} {:.2g} {:.2g} {:.2g}'.format(prediction[0], ML1, ML2, ML3, ML4, total_confidence)
    
if __name__ == "__main__":
    app.run(host= '0.0.0.0',debug = 'true' ,port = 5100)