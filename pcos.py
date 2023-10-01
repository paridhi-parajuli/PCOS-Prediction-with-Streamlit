import numpy as np
import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from PIL import Image


def main():
	
	st.title("Poly Cystic Ovarian Symdrome Prediction")
	st.sidebar.title("PCOS Prediction")
	st.sidebar.markdown("This is for changing the model parameters")
	@st.cache(persist=True)
	def load_data():
		no_inf=pd.read_csv("no_inf.csv")
		inf=pd.read_csv("inf.csv")
		data = pd.merge(no_inf,inf, on='Patient File No.', suffixes={'','_y'},how='left')
		data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].median(),inplace=True)
		data.drop(['PCOS (Y/N)_y','AMH(ng/mL)_y','Patient File No.','Unnamed: 42'],axis=1,inplace=True)
		corr_features=data.corrwith(data["PCOS (Y/N)"]).abs().sort_values(ascending=False)
		corr_features=corr_features[corr_features>0.25].index
		data=data[corr_features]
		#print(data.head())
		return data
		

	df=load_data()

	@st.cache(persist=True)
	def split(df):
		y = df['PCOS (Y/N)']
		x=df.drop(columns=['PCOS (Y/N)'])
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
		return x_train, x_test, y_train, y_test

	class_names = ['pcos', 'no pcos']
	x_train, x_test, y_train, y_test = split(df)

	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
			st.pyplot()

		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			plot_roc_curve(model, x_test, y_test)
			st.pyplot()
		
		if 'Precision-Recall Curve' in metrics_list:
			st.subheader('Precision-Recall Curve')
			plot_precision_recall_curve(model, x_test, y_test)
			st.pyplot()



	st.sidebar.subheader("Choose Classifier")
	classifier = st.sidebar.selectbox("Classifier", ("SVM", "Random Forest"))
	if classifier == 'SVM':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("Regularization (C)", 0.01, 10.0, step=0.01, key='C_SVM')
		kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
		gamma = st.sidebar.radio("Gamma (Kernel Cofficient)", ("scale", "auto"), key='gamma')
		metrics = st.sidebar.multiselect("Plot Metrices", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

		if st.sidebar.button("Classify", key='classify'):
			st.subheader("Results of SVM")
			model = SVC(C=C, kernel=kernel, gamma=gamma)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)


	if classifier == 'Random Forest':
		st.sidebar.subheader("Model Hyperparameters")
		n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
		max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
		bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
		metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

		if st.sidebar.button("Classify", key='classify'):
			st.subheader("Random Forest Results")
			model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)

	skin=st.number_input("Have you experienced skin darkening?(0/1)", key='skin')
	hair=st.number_input("Have you experienced hair growth?(0/1)", key='hair')
	weight=st.number_input("Have you experienced weight gain?(0/1)", key='weight')
	
	ff=st.number_input("Do you eat too much fast foods?(0/1)", key='ff')
	pimple=st.number_input("Do you get pimples?(0/1)", key='pimple')
	lf = st.number_input("No of left follicles", key='lf')
	rf=st.number_input("No of right follicle", key='rf')
	cycle=st.number_input("Cycle(R/I)", key='cycle')
	my_data=[[lf,rf,skin,hair,weight,cycle,ff,pimple]]
	pd.DataFrame.from_dict(my_data)
	if st.button("Predict", key='predict'):
			st.subheader("Your Result:")
			model = SVC(C=0.01, kernel="linear", gamma="auto")
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(my_data)
			if y_pred==1:
				st.write("Alert!!You are predicted to have PCOS")
			else:
				st.write("Congratulations!!You are predicted negative for PCOS")
			
			st.write("The prediction made has Accuracy: ", accuracy.round(2))
			
			#plot_metrics(metrics)
if __name__ == '__main__':
	main()
