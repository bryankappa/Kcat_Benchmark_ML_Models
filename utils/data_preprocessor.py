import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os


def load_data():
    preprocessor = DataPreprocessor(r'C:\Users\Gilbert\Documents\BCB_Research\Kcat_Benchmark_ML_Models\Data\kcat_transferase.csv')
    amino_encoded = pd.read_csv(r'C:\Users\Gilbert\Documents\BCB_Research\Kcat_Benchmark_ML_Models\Data\encoded_amino.csv')
    return preprocessor, amino_encoded

#creating inputs and assigning columns.
class DataPreprocessor:
    def __init__(self, csv_file_path): 
        self.csv_file_path = csv_file_path

    def assign_column(self):
        data = pd.read_csv(self.csv_file_path)
        data.columns = ["EC_number", "Species", "Compound", "Compound_name", "Amino_encoding", "Kcat", "unit"]
        data = pd.DataFrame(data)
        return data
    
    def compute_molecular_weight(self, data):
        def molecular_weight(compound):
            mol = Chem.MolFromSmiles(compound)
            if mol:
                return Descriptors.MolWt(mol)
            else:
                return None

        data["Molecular_Weight"] = data["Compound"].apply(molecular_weight)
        return data

    #applying log to the data
    def apply_log_transform(self, data):
        data["Kcat"] = np.log10(data["Kcat"])
        return data





