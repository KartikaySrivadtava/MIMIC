#Import necessary libraries

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings('ignore')
from tkinter import *
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import shap
%matplotlib inline
from shap import Explanation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import psycopg2
import warnings
warnings.filterwarnings('ignore')

#Establishing connection to the source database

conn = psycopg2.connect(database="", user="", password="", host="", port="")
cur = conn.cursor()

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Retrieving data
cursor.execute('''Create an SQL query based on the table parameters''')

#Creating dataframe from this sql query

df = pd.read_sql_query("SQL query", conn)

#Extracting Features from the source database

list_headers_all = list(df.columns.values)
n_static = 4
list_headers_features = list_headers_all[n_static:]
list_headers_features.pop()

#Main Tkinter window

root = Tk()
root.title('Input window')
width= root.winfo_screenwidth()               
height= root.winfo_screenheight()     
root.geometry("%dx%d" % (width, height))

#Declare tree for opening window for showing Available Features
tree = ttk.Treeview(root)

tree.column('#0', width=0)
tree['columns'] = 'Available-Features'

for column in tree['columns']:
    tree.heading(column, text=column)

for item in list_headers_features:
    tree.insert('', 'end', values=item) 

#Declare 2nd window when everything in the first window is correct    
def second_win(selected_features):
   #Declare parameters for 2nd window
   new = Toplevel(root)
   width= new.winfo_screenwidth()               
   height= new.winfo_screenheight()     
   new.geometry("%dx%d" % (width, height))
   new.title("Welcome to Patient Tele-Monitoring Software")
    
   #Map mandatory columns 
   REL_DAY = df.columns[3]
   GROUP_ID = df.columns[2]
   HADM_ID = df.columns[1]
   SUBJECT_ID = df.columns[0]
   DOD_LABEL = df.columns[-1] 

   #Declare X and Y for Machine Learning
   X = df[selected_features]
   Y = df[DOD_LABEL] 

   #Scale the Data before Machine Learning
   scaler = StandardScaler()
   scaler.fit(X)
   X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
   
   #Train the Machine Learning Models
   lr = LogisticRegression(C=0.03359818286283781, multi_class='multinomial', penalty='l2', solver='newton-cg').fit(X, Y)
   rf = RandomForestClassifier(criterion='gini', max_depth=12, max_features='sqrt', min_samples_leaf=4, min_samples_split=10).fit(X, Y)
   xgb = XGBClassifier(learning_rate=0.1, max_depth=5, min_child_weight=4, n_estimators=20).fit(X, Y)
   rc = LogisticRegression(penalty='l2').fit(X, Y)  
   knn=KNeighborsClassifier(n_neighbors= 14).fit(X, Y)
   svm = SVC(C=0.1, gamma='scale', kernel='linear', probability=True).fit(X, Y)
   dt = DecisionTreeClassifier(criterion='entropy', max_depth=18, max_features='auto', min_samples_leaf=1, min_samples_split=2,
                              splitter='best').fit(X ,Y)
   ann=MLPClassifier(activation='tanh', hidden_layer_sizes=(5, 2), max_iter=20).fit(X, Y)
   
   #Create full header for main dataframe 
   full_header = selected_features
   full_header.insert(0, REL_DAY)
   full_header.insert(0, GROUP_ID)
   full_header.insert(0, HADM_ID)
   full_header.insert(0, SUBJECT_ID)
   full_header.append(DOD_LABEL)
    
   #Create main dataframe for the selected features for XAI part
   df_main = df[df.columns.intersection(full_header)] 
    
   #Create dataframes for each Machine Learning Algorithms
   df_lr = df_main.copy()
   df_rf = df_main.copy()
   df_xgb = df_main.copy()
   df_rc = df_main.copy() 
   df_knn = df_main.copy()
   df_dt = df_main.copy()
   df_svm = df_main.copy()
   df_ann = df_main.copy()  
    
   #Logistic Regression
   log_pred = lr.predict_proba(X)[:, 1]
   log_pred = log_pred*100
   df_lr['RISK'] = log_pred
   df_lr_max = df_lr.copy()
   #del df_lr['DOD_LABEL']
   df_lr = df_lr.loc[df_lr['DOD_LABEL'] == 1]
   df_lr=(df_lr.loc[df_lr.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details
   del df_lr['DOD_LABEL']

   #Random Forest 
   rf_pred = rf.predict_proba(X)[:, 1]
   rf_pred = rf_pred*100
   df_rf['RISK'] = rf_pred
   df_rf_max = df_rf.copy()
   #del df_rf['DOD_LABEL']
   df_rf = df_rf.loc[df_rf['DOD_LABEL'] == 1]
   df_rf=(df_rf.loc[df_rf.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details 
   del df_rf['DOD_LABEL']
    
   #XGB
   xgb_pred = xgb.predict_proba(X)[:, 1]
   xgb_pred = xgb_pred*100 
   df_xgb['RISK'] = xgb_pred
   df_xgb_max = df_xgb.copy()
   #del df_xgb['DOD_LABEL']
   df_xgb = df_xgb.loc[df_xgb['DOD_LABEL'] == 1]
   df_xgb=(df_xgb.loc[df_xgb.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details
   del df_xgb['DOD_LABEL'] 
   
   #Ridge CLassifier
   rc_pred = rc.predict_proba(X)[:, 1]
   rc_pred = rc_pred*100
   df_rc['RISK'] = rc_pred
   df_rc_max = df_rc.copy() 
   #del df_rc['DOD_LABEL']
   df_rc = df_rc.loc[df_rc['DOD_LABEL'] == 1]
   df_rc=(df_rc.loc[df_rc.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details
   del df_rc['DOD_LABEL'] 

   #KNN
   knn_pred = knn.predict_proba(X)[:, 1]
   knn_pred = knn_pred*100
   df_knn['RISK'] = knn_pred
   df_knn_max = df_knn.copy() 
   #del df_rc['DOD_LABEL']
   df_knn = df_knn.loc[df_knn['DOD_LABEL'] == 1]
   df_knn=(df_knn.loc[df_knn.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details
   del df_knn['DOD_LABEL'] 

   #Decision Tree
   dt_pred = dt.predict_proba(X)[:, 1]
   dt_pred = dt_pred*100
   df_dt['RISK'] = dt_pred
   df_dt_max = df_dt.copy() 
   #del df_rc['DOD_LABEL']
   df_dt = df_dt.loc[df_dt['DOD_LABEL'] == 1]
   df_dt=(df_dt.loc[df_dt.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details
   del df_dt['DOD_LABEL'] 

   #SVM
   svm_pred = svm.predict_proba(X)[:, 1]
   svm_pred = svm_pred*100
   df_svm['RISK'] = svm_pred
   df_svm_max = df_svm.copy() 
   #del df_rc['DOD_LABEL']
   df_svm = df_svm.loc[df_svm['DOD_LABEL'] == 1]
   df_svm=(df_svm.loc[df_svm.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details
   del df_svm['DOD_LABEL'] 

   #ANN
   ann_pred = ann.predict_proba(X)[:, 1]
   ann_pred = ann_pred*100
   df_ann['RISK'] = ann_pred
   df_ann_max = df_ann.copy() 
   #del df_rc['DOD_LABEL']
   df_ann = df_ann.loc[df_ann['DOD_LABEL'] == 1]
   df_ann=(df_ann.loc[df_ann.groupby(['GROUP_ID'])['REL_DAY'].idxmax()]).sort_values(by='RISK', ascending=False)
   #Delete DOD_LABEL as it is not required for showing patient details
   del df_ann['DOD_LABEL'] 
    
   #Start Designing the Software window
   # Create Heading for the System
   l = Label(new, text = "Welcome to Patient Tele-Monitoring System")
   l.config(font =("Times New Roman", 20)) 
   l.grid(row=0, column=0, padx=750)

   #Text for choosing ML type:
   l2 = Label(new, text = "Please choose a Machine Learning Algorithm:")
   l2.config(font =("Times New Roman", 16)) 
   l2.grid(row=1, column=0, sticky="W", padx=10) 

   #Get values of selected radio button and fill tree view accordingly
   def sel():
     if r.get()==0:
         print("Value of r is", r.get())
         fill_tree(df_lr)
     elif r.get()==1: 
         print("Value of r is", r.get())
         fill_tree(df_rf)
     elif r.get()==2: 
         print("Value of r is", r.get())
         fill_tree(df_xgb)
     elif r.get()==3: 
         print("Value of r is", r.get())
         fill_tree(df_rc)
     elif r.get()==4:
         print("Value of r is", r.get())
         fill_tree(df_knn)
     elif r.get()==5: 
         print("Value of r is", r.get())
         fill_tree(df_dt)
     elif r.get()==6: 
         print("Value of r is", r.get())
         fill_tree(df_svm)
     elif r.get()==7: 
         print("Value of r is", r.get())
         fill_tree(df_ann)    
    
   #Declare Radio buttons for choosing ML:
   r = IntVar(new)    
    
   log_reg_btn = Radiobutton(new, text="Logistic Regression", variable=r, value=0, highlightthickness=0, command=sel)
   log_reg_btn.config(font =("Times New Roman", 10)) 
   log_reg_btn.grid(row=2, column=0, sticky="W", padx=30, pady=0)

   rf_btn = Radiobutton(new, text="Random Forest", variable=r, value=1, highlightthickness=0, command=sel)
   rf_btn.config(font =("Times New Roman", 10)) 
   rf_btn.grid(row=3, column=0, sticky="W", padx=30, pady=0)

   xgb_btn = Radiobutton(new, text="XGBoost", variable=r, value=2, highlightthickness=0, command=sel)
   xgb_btn.config(font =("Times New Roman", 10)) 
   xgb_btn.grid(row=4, column=0, sticky="W", padx=30, pady=0)

   rc_btn = Radiobutton(new, text="Ridge Classifier", variable=r, value=3, highlightthickness=0, command=sel)
   rc_btn.config(font =("Times New Roman", 10)) 
   rc_btn.grid(row=5, column=0, sticky="W", padx=30, pady=0)
    
   knn_reg_btn = Radiobutton(new, text="K Nearest Neighbour", variable=r, value=4, highlightthickness=0, command=sel)
   knn_reg_btn.config(font =("Times New Roman", 10)) 
   knn_reg_btn.grid(row=6, column=0, sticky="W", padx=30, pady=0)

   dt_btn = Radiobutton(new, text="Decision Tree", variable=r, value=5, highlightthickness=0, command=sel)
   dt_btn.config(font =("Times New Roman", 10)) 
   dt_btn.grid(row=7, column=0, sticky="W", padx=30, pady=0)

   svm_btn = Radiobutton(new, text="Support Vector Machine", variable=r, value=6, highlightthickness=0, command=sel)
   svm_btn.config(font =("Times New Roman", 10)) 
   svm_btn.grid(row=8, column=0, sticky="W", padx=30, pady=0)

   ann_btn = Radiobutton(new, text="Artificial Neural Network", variable=r, value=7, highlightthickness=0, command=sel)
   ann_btn.config(font =("Times New Roman", 10)) 
   ann_btn.grid(row=9, column=0, sticky="W", padx=30, pady=0) 

   #The following label shows the details on how the patients have been sorted for treeview 

   l2 = Label(new, text = "Patient details are sorted in decreasing order of their risk based upon the outcome of the above selected algorithm. Please select a Patient to check outome based on Machine Learning:")
   l2.config(font =("Times New Roman", 14)) 
   l2.grid(row=10, column=0, sticky="W", padx=10, pady=15) 
   
   l3 = Label(new, text = "The plot on lower left shows the plot between Risk per day for a selected patient form the above list."+
           "\t\t\t\t The plot on the lower right shows the most important risk increasing or decresing factors based upon SHAP values.")
   l3.config(font =("Times New Roman", 14)) 
   l3.grid(row=12, column=0, sticky="W", padx=10) 
   
   #Declare All features Tree
   columns = df_lr.columns.tolist()
   tree3 = ttk.Treeview(new, selectmode="extended",columns=columns) 
   tree3['height']=10    
   tree3.column('#0', width=0)
   tree3["columns"] = columns
   # Defining headings, other option in tree
   # width of columns and alignment 
   for i in columns:
     tree3.column(i, width = 100, anchor ='c')
   # Headings of respective columns
   for i in columns:
     tree3.heading(i, text =i)
   
   #Get value of selected item from tree view entry        
   def item_selected2(event):
        for selected_item in tree3.selection():
            item = tree3.item(selected_item)
            record = item['values']
            group_id = np.float(record[2])
            rel_day = np.float(record[3])
            print("gorup id is", group_id)
            print("rel_day is", rel_day) 
            
            #Create Temporary dataframe for selected patient
            print("Value of radio button is ",r.get())
            if r.get()==0:
                #Logistic Regression
                df = df_lr
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                df_temp2 = df_lr_max[(df_lr_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
        
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                print("index is")
                print(index)
                index = index[0]
                print("Index Value is", index)
            
                #Plot Feature Importance 
                explainer = shap.Explainer(lr, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
            
                #Plot Relative day vs Risk for temporary patient
                df = df_lr_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('RISK')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10) 
                plt.close(fig1) 
            
            elif r.get()==1:
                #Random Forest
                df = df_rf
                df_temp2 = df_rf_max[(df_rf_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
                
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                index = index[0]
                print("Index Value is", index)
                        
                #Plot Feature Importance 
                explainer = shap.Explainer(rf, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w*1.1, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
            
                #Plot Relative day vs Risk for temporary patient
                df = df_rf_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('Risk')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10) 
                plt.close(fig1) 
            
            elif r.get()==2:
                #XGBoost
                df = df_xgb
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                df_temp2 = df_xgb_max[(df_xgb_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
                
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                index = index[0]
                print("Index Value is", index)
                
                #Plot Feature Importance 
                explainer = shap.Explainer(xgb, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w*1.1, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
            
                #Plot Relative day vs Risk for temporary patient
                df = df_xgb_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('Risk')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10) 
                plt.close(fig1) 
        
            elif r.get()==3:
                #Ridge Classifier
                df = df_rc
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                df_temp2 = df_rc_max[(df_xgb_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                index = index[0]
                print("Index Value is", index)
                
                #Plot Feature Importance 
                explainer = shap.Explainer(rc, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w*1.1, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
                    
                #Plot Relative day vs Risk for temporary patient
                df = df_rc_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('Risk')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10)
                plt.close(fig1)    
                
            elif r.get()==4:
                #KNN
                df = df_knn
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                df_temp2 = df_knn_max[(df_knn_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
                
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                index = index[0]
                print("Index Value is", index)
                
                #Plot Feature Importance 
                explainer = shap.KernelExplainer(knn, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w*1.1, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
                    
                #Plot Relative day vs Risk for temporary patient
                df = df_knn_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('Risk')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10)
                plt.close(fig1)
                
            elif r.get()==5:
                #Decision Tree
                df = df_dt
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                df_temp2 = df_dt_max[(df_dt_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
                
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                index = index[0]
                print("Index Value is", index)
                
                #Plot Feature Importance 
                explainer = shap.Explainer(dt, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w*1.1, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
                    
                #Plot Relative day vs Risk for temporary patient
                df = df_dt_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('Risk')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10)
                plt.close(fig1)
                
            elif r.get()==6:
                #SVM
                df = df_svm
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                df_temp2 = df_svm_max[(df_svm_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
                
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                index = index[0]
                print("Index Value is", index)
                explainer = shap.Explainer(rc, X)
                shap_values = explainer(X)   
                shap_values_local = shap_values[index]
                
                #Plot Feature Importance 
                explainer = shap.Explainer(svm, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w*1.1, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
                    
                #Plot Relative day vs Risk for temporary patient
                df = df_rc_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('Risk')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10)
                plt.close(fig1)     
                
            elif r.get()==7:
                #ANN
                df = df_ann
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                df_temp2 = df_ann_max[(df_ann_max['GROUP_ID'] == group_id)] 
                third_win(df_temp2) 
                
                index = df_main.index[(df_main['GROUP_ID'] == group_id) & (df_main['REL_DAY'] == rel_day)]
                index = index[0]
                print("Index Value is", index)
                
                #Plot Feature Importance 
                explainer = shap.Explainer(ann, X)
                shap_values = explainer(X)
                shap_values_local = shap_values[index]
                shap.plots.bar(shap_values_local, show=False, max_display=10)   
                fig2 = plt.gcf()
                w, _ = fig2.get_size_inches()
                fig2.set_size_inches(w*1.1, w*0.5)
                plt.tight_layout()        
                canvas = FigureCanvasTkAgg(fig2, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="E", padx=100, pady=10)
                plt.close(fig2) 
                    
                #Plot Relative day vs Risk for temporary patient
                df = df_rc_max
                df_temp = df[(df['GROUP_ID'] == group_id)] 
                fig1 = plt.figure()
                plt.plot(df_temp['REL_DAY'], df_temp['RISK'])
                plt.title('Risk vs Relative Day')
                plt.xlabel('Relative Day')
                plt.ylabel('Risk')
                plt.tight_layout()
                fig1.set_size_inches(w*1.1, w*0.5)
                canvas = FigureCanvasTkAgg(fig1, master=new)
                canvas.draw()
                canvas.get_tk_widget().grid(row=13, column=0, sticky="W", padx=10, pady=10)
                plt.close(fig1)      
        
   #Declare tree for patient details
   tree3.bind('<<TreeviewSelect>>', item_selected2)
   tree3.grid(row=11, column=0, columnspan=2, sticky="W", padx=10, pady=5)

   #Clear a tree view entry upon click 
   def clear_all():
        for item in tree3.get_children():
              tree3.delete(item) 
                
   #Fill a tree view based upon the selected radio button    
   def fill_tree(df_max): 
        clear_all()
        r_set=df_max.to_numpy().tolist()
        for dt in r_set:
            v=[r for r in dt] # collect the row data as list 
            tree3.insert("",'end',iid=v[0],values=v)   
            
#Declare Error window when the features have not been selected in the opening window            
def third_win(df_temp2):
   new2=Toplevel()
   width= new2.winfo_screenwidth()               
   height= new2.winfo_screenheight()     
   new2.geometry("%dx%d" % (width/2, height/2))
   new2.title("Patient Vitals")
   #Create a Label in New window
   Label(new2, text="Below are the most important vital signs for the patient:").grid(row=0, column=0)
   #print(df_temp2)
   #Declare All features Tree
   columns = df_temp2.columns.tolist()
   #print("columns are ")
   #print(columns)
   tree4 = ttk.Treeview(new2, selectmode="extended",columns=columns) 
   tree4['height']=5    
   tree4.column('#0', width=0)
   for item in tree4.get_children():
        tree4.delete(item) 
   tree4["columns"] = columns
   # Defining headings, other option in tree
   # width of columns and alignment 
   for i in columns:
     tree4.column(i, width = 100, anchor ='c')
   # Headings of respective columns
   for i in columns:
     tree4.heading(i, text=i) 
   r_set2=df_temp2.to_numpy().tolist()
   print("r_set is")
   print(r_set2) 
   #v2=['1','2'] 
   #tree4.insert("",'end',iid=v2[0],values=v2)
   for dt2 in r_set2:
     v2=[r2 for r2 in dt2] # collect the row data as list 
     tree4.insert("",'end',values=v2)  
   
   tree4.bind('<<TreeviewSelect>>')
   tree4.grid(row=1, column=0, sticky="W", padx=0, pady=10) 

#Declare Error window when the features have not been selected in the opening window            
def error_win():
   new=Toplevel(root)
   new.geometry("100x100")
   new.title("Empty Window")
   #Create a Label in New window
   Label(new, text="Input features are empty!!").grid(row=0, column=0)
   

feature_list = []  
    
#Get value of selected item from first tree view entry        

def item_selected(event):
    #Get selected features from first treeview
    for selected_item in tree.selection():
        item = tree.item(selected_item)
        record = item['values']
        feature_list.remove(record) if record in feature_list else feature_list.append(record)
        
        #Creat and add records for 2nd tree
        tree2 = ttk.Treeview(root)
        tree2['height']=20
        tree2.column('#0', width=0)
        tree2['columns'] = 'Selected-Features'
        
        tree2.grid(row=1, column=1, sticky="E", padx=10, pady=100)
        
        for column in tree2['columns']:
            tree2.heading(column, text=column)

        for item in feature_list:
            tree2.insert('', 'end', values=item)
        
        #Get selected features from second treeview
        selected_features = []
        for line in tree2.get_children():
            for value in tree2.item(line)['values']:
                selected_features.append(value)
        
        if len(selected_features) != 0:
            #Open second widnow if feature list is not empty
            ttk.Button(root, text="SUBMIT", command=(lambda: second_win(selected_features))).grid(row=2, column=2)
        else:
            #Open error message windo if feature list is empty
            ttk.Button(root, text="SUBMIT", command=error_win()).grid(row=0, column=2)
        
#Declare All features Tree
tree['height']=20
tree.bind('<<TreeviewSelect>>', item_selected)
tree.grid(row=1, column=0, sticky="W", padx=100, pady=100)

l = Label(root, text = "Please select features from the list on the left 'Available Features'. Once an item has been selected from this list it will be shown in the selected features list on the right. \n\n Click on a feature twice to remove it from theselected features list. Once satisfied with the selected features, press the submit button on extreme right.")
l.config(font =("Times New Roman", 15))
l.grid(row=0, column=0)

root.mainloop()
