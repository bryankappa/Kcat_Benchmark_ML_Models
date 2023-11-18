import sys
sys.path.append('C:/Users/Gilbert/Documents/BCB_Research/Kcat_Benchmark_ML_Models/')
from utils.data_preprocessor import *
from benchmarks.randomforest_benchmark import *
from benchmarks.boosting_benchmark import *
from benchmarks.svm_benchmark import *
from utils.performance_metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

preprocessor = load_data() # loads main data
data = preprocessor.assign_column() # assings columns to main data
aminoencoding = PCA("C:\\Users\Gilbert\Documents\BCB_Research\Kcat_Benchmark_ML_Models\Data\encoded_amino.csv", n_components=433)
amino_encoding = PCA.apply_pca(aminoencoding)


X = amino_encoding
y = np.log10(data['Kcat'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
print(y_train)

#run the model that is available: CustomRandomForestRegressor, customSVMregressor, and customXGBregressor
model = customXGBregressor(verbose=2) # you can add the specific tuning parameters needed.
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(evaluate_model(y_test, y_pred))

# this is for tuning 
# tuner = SVRTuner(model)
# best_params = tuner.fit(X_train, y_train)
# print(f"Best parameters: {best_params}")


# Seaborn styling
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 10))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5, 's':60}, line_kws={"color":"red", "linewidth":2}, ci=None)
r2 = r2_score(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--')
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.title(f"Actual vs Predicted (R² = {r2:.2f})", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
