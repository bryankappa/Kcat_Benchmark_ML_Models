import sys
sys.path.append('C:/Users/Gilbert/Documents/BCB_Research/Kcat_Benchmark_ML_Models/')
from utils.data_preprocessor import *
from benchmarks.randomforest_benchmark import *
from benchmarks.xgboost_benchmark import *
from benchmarks.svm_benchmark import *
from utils.performance_metrics import *

preprocessor, amino_encoded = load_data()
data = preprocessor.assign_column()
data_pre = preprocessor.apply_log_transform(data)

X = amino_encoded
y = data_pre['Kcat'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = customSVMregressor(verbose=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

evaluate_model(y_test, y_pred)