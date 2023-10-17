import sys
sys.path.append('C:/Users/Gilbert/Documents/BCB_Research/Kcat_Benchmark_ML_Models/')
from utils.data_preprocessor import *
from benchmarks.randomforest_benchmark import *
from benchmarks.boosting_benchmark import *
from benchmarks.svm_benchmark import *
from utils.performance_metrics import *

preprocessor = load_data() # loads main data
data = preprocessor.assign_column() # assings columns to main data
amino_pca = PCA(r'C:\Users\Gilbert\Documents\BCB_Research\Kcat_Benchmark_ML_Models\Data\encoded_amino.csv', n_components=433)
amino_pca = amino_pca.apply_pca()

X = amino_pca
y = np.log10(data['Kcat'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#run the model that is available: CustomRandomForestRegressor, customSVMregressor, and customXGBregressor
# model = customXGBregressor(verbose=2)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# evaluate_model(y_test, y_pred)

rf = RandomForestRegressor()

# Initialize the tuner class
tuner = RandomForestTuner(rf)

# Fit the data and search for the best hyperparameters
best_params = tuner.fit(X_train, y_train)

# Retrieve the best parameters
print("Best parameters found: ", best_params)

# Or you could use the method to get the best parameters
print("Best parameters found: ", tuner.get_best_params())