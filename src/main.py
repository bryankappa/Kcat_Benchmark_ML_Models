import sys
sys.path.append('C:/Users/Gilbert/Documents/BCB_Research/Kcat_Benchmark_ML_Models/')
from utils.data_preprocessor import *
from benchmarks.randomforest_benchmark import *
from benchmarks.xgboost_benchmark import *
from benchmarks.svm_benchmark import *
from utils.performance_metrics import *

preprocessor, jazzy_data= load_data()
data = preprocessor.assign_column()
data_pre = preprocessor.apply_log_transform(jazzy_data)
jazz_encoded = pd.read_csv(r'C:\Users\Gilbert\Documents\BCB_Research\Kcat_Benchmark_ML_Models\Data\jazz_encoded.csv')

data_pre = pd.DataFrame(data_pre)
jazz_encoded  = pd.DataFrame(jazz_encoded)
final_df = pd.concat([jazz_encoded, data_pre], axis=1)
# final_df = final_df.drop('Unnamed: 0', axis=1)
# print(final_df.head())
final_df.drop('Kcat', axis=1)

X = final_df.drop('Kcat', axis=1)
y = data_pre['Kcat'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#run the model that is available: CustomRandomForestRegressor, customSVMregressor, and customXGBregressor
model = CustomRandomForestRegressor(verbose=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

evaluate_model(y_test, y_pred)