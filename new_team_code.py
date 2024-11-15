import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy as sp
import scipy.stats
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgboost
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.svm import SVC
from helper_code import *
from preprocessing import *
import warnings

warnings.filterwarnings('ignore')



def extract_recording_feature(data, recordings):

	locations = get_locations(data)

	recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
	num_recording_locations = len(recording_locations)
	recording_features = np.zeros((num_recording_locations, 4), dtype=float)
	num_locations = len(locations)
	num_recordings = len(recordings)
	if num_locations==num_recordings:
		for i in range(num_locations):
			for j in range(num_recording_locations):
				if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
					recording_features[j, 0] = 1
					recording_features[j, 1] = np.mean(recordings[i])
					recording_features[j, 2] = np.var(recordings[i])
					recording_features[j, 3] = sp.stats.skew(recordings[i])

	recording_features = recording_features.flatten()

	return recording_features


def create_lists(patient_files, preprocessed_data, training_data_recording):

	murmur_dict = {0: 'Present', 1: 'Absent'}

	features = list()
	murmurs = list()

	#Extract recording's features and stack them with the rest
	for i, row in preprocessed_data.iterrows():

		current_patient_data_id = row["Patient ID"]

		current_patient_data = load_patient_data(patient_files[i])
		current_recordings = load_recordings(training_data_recording, current_patient_data)

		current_recordings_features = extract_recording_feature(current_patient_data, current_recordings)

		features.append(np.hstack(([row["Age"]], row["Sex"], [row["Height"]], [row["Weight"]], row["Pregnancy status"], current_recordings_features)))

		current_murmur = np.zeros(len(murmur_dict), dtype=int)
		current_murmur[row["Murmur"]] = 1
		murmurs.append(current_murmur)

	return features, murmurs

def eval(murmurs_test, murmur_prediction):

	conf_matrix_murmur = confusion_matrix(np.argmax(murmurs_test, axis=1), np.argmax(murmur_prediction, axis=1))

	class_report_murmur = classification_report(np.argmax(murmurs_test, axis=1), np.argmax(murmur_prediction, axis=1), target_names = ['Present', 'Absent'])

	print("Murmur confusion matrix")
	print(conf_matrix_murmur)

	print("Murmur classification report")
	print(class_report_murmur)

	#Overall F1-score
	print("Overall F1 score: ", f1_score(np.argmax(murmurs_test, axis=1), np.argmax(murmur_prediction, axis=1), average='macro'))

	#calculate AUC
	auc = roc_auc_score(np.argmax(murmurs_test, axis=1), np.argmax(murmur_prediction, axis=1))
	print('AUC: %.3f' % auc)

	# Plot confusion matrix
	plt.figure(figsize=(8, 6))
	sns.heatmap(conf_matrix_murmur, annot=True, fmt="d", cmap="Blues")
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.title('Confusion Matrix')
	plt.show()



if __name__ == "__main__":

	model = 1
	if len(sys.argv) > 1:
		if int(sys.argv[1]) >=1 and int(sys.argv[1]) <= 3:
			model = int(sys.argv[1])

	#Path to the data
	csv_file_path = "dataset/training_data.csv"
	training_data_recording = "dataset/train_data"
	val_file_path = "dataset/val_data.csv"
	validation_data_recording = "dataset/validation_data"
	test_file_path = "dataset/test_data.csv"
	testing_data_recording = "dataset/test_data"
	patient_files = find_patient_files(training_data_recording)
	val_files = find_patient_files(validation_data_recording)
	test_files = find_patient_files(testing_data_recording)

	#Data pre-processing
	data_preprocessor = DataPreprocessor(csv_file_path)
	preprocessed_data = data_preprocessor.preprocess()

	val_preprocessor = DataPreprocessor(val_file_path)
	preprocessed_val = val_preprocessor.train_val_preprocess()

	test_preprocessor = DataPreprocessor(test_file_path)
	preprocessed_test = test_preprocessor.train_val_preprocess()

	features = list()
	murmurs = list()

	feature, murmur = create_lists(patient_files, preprocessed_data, training_data_recording)

	features = np.vstack(feature)
	murmurs = np.vstack(murmur)

	features_val = list()
	murmurs_val = list()

	feature_val, murmur_val = create_lists(val_files, preprocessed_val, validation_data_recording)

	features_val = np.vstack(feature_val)
	murmurs_val = np.vstack(murmur_val)

	features_test = list()
	murmurs_test = list()

	feature_test, murmur_test = create_lists(test_files, preprocessed_test, testing_data_recording)

	features_test = np.vstack(feature_test)
	murmurs_test = np.vstack(murmur_test)


	murmurs_labels = np.argmax(murmurs, axis=1)

	if model == 1:

		#Define parameters for random forest classifier.
		max_depth = None
		min_samples_leaf = 4
		min_samples_split = 2
		n_estimators   = 100  # Number of trees in the forest.
		max_leaf_nodes = 75   # Maximum number of leaf nodes in each tree.

		murmur_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=6789, class_weight='balanced').fit(features, murmurs)

		murmur_prediction = murmur_classifier.predict(features_test)

		eval(murmurs_test, murmur_prediction)


	elif model == 2:

		scale_pos_weight = len(murmurs[murmurs == 0]) / len(murmurs[murmurs == 1])

		#Define parameters for XGBoost

		model = xgboost.XGBClassifier(
    		learning_rate =0.01,
    		n_estimators=1000,
    		max_depth = 3,
    		min_child_weight = 6,
    		gamma=0.4,
    		subsample=0.6,
    		colsample_bytree=0.8,
    		reg_alpha=0.005,
    		objective= 'binary:logistic',
    		nthread=4,
    		scale_pos_weight=scale_pos_weight,
    		seed=27
		)

		model.fit(features, murmurs)

		y_pred = model.predict(features_test)
		murmur_prediction = [np.round(value) for value in y_pred]

		eval(murmurs_test, murmur_prediction)


	elif model == 3:


		sm = SMOTE(random_state=1)
		X_res, y_res = sm.fit_resample(features, murmurs)

		clf = MLPClassifier(solver='adam', activation='relu', learning_rate_init=1e-4, alpha=1e-5, hidden_layer_sizes=(300, 110), random_state=1).fit(X_res, y_res)
		murmur_prediction = clf.predict(features_test)

		murmur_prediction_one_hot = np.eye(2)[murmur_prediction]

		eval(murmurs_test, murmur_prediction_one_hot)

	elif model == 4:

		svc = SVC(
    		kernel='rbf',
    		decision_function_shape='ovr',
    		max_iter=40,
    		verbose=False,
    		random_state=1,
    		class_weight='balanced',
    		C=0.7,
    		gamma='scale'
		)
		vector_model = OneVsRestClassifier(svc)
		vector_model.fit(features,murmurs)

		murmur_prediction = vector_model.predict(features_test)

		eval(murmurs_test, murmur_prediction)

	elif model == 5:
		dtc = DecisionTreeClassifier(criterion='entropy',max_features=int(math.sqrt(len(features))),random_state=1,splitter='best', class_weight='balanced', max_depth=None, min_samples_split=2, min_samples_leaf=1)
		dtc.fit(features,murmurs)
		y_pred = dtc.predict(features_test)

		eval(murmurs_test, y_pred)


	else:
		print("Something went wrong")



