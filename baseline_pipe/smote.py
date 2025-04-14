import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def smote_gen(data_nm, subsample):
    train_path = f"../sample/{data_nm}/df{subsample}.csv"
    test_path = f"../sample/{data_nm}/df{subsample}_test.csv"

    seed_lst = list(range(5))
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
