# import libraries
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

def data_cleaning(split):
    df = pd.read_csv('breast_cancer_data.csv')
    # rename columns to for better coding practice
    df = df.rename(columns=lambda x: x.replace(' ', '_'))
    # map diagnosis to numbers
    df.diagnosis = df.diagnosis.map({'B': 0, 'M': 1})
    # drop columns not used
    df.drop(columns=['id', 'Unnamed:_32'], inplace=True)
    # let's check for scenarios where 0 might be a missing value
    missing_col = []
    for i in df.columns:
        if df[i].min() <= 0:
            missing_col.append(i)
    missing_col.remove('diagnosis')
    # Next we replace missing values (0) with mean
    for i in missing_col:
        mean = df[i].mean()
        df[i] = df[i].replace(0, mean)

    # split the data
    dep_var = df.drop(['diagnosis'], axis=1)
    indep_var = df['diagnosis']
    x_train, x_test, y_train, y_test = train_test_split(dep_var, indep_var, test_size=split, random_state=2)
    return x_train, x_test, y_train, y_test

def smote_balancing(split):
    smote_algorithm = SMOTE(random_state=2)
    x_train, x_test, y_train, y_test = data_cleaning(split)
    smote_data_x, smote_data_y = smote_algorithm.fit_resample(x_train, y_train)
    sc = StandardScaler()
    smote_data = sc.fit_transform(smote_data_x)
    smote_test = sc.fit_transform(x_test)
    smote_data = pd.DataFrame(smote_data, columns=x_train.columns)
    smote_test = pd.DataFrame(smote_test, columns=x_test.columns)
    return smote_data, smote_test, smote_data_y, y_test

def pca(data, count):
    pca = PCA(n_components=count).fit_transform(data)
    return pca

def system_pca(n_components, split):
    x_train, x_test, y_train, y_test = smote_balancing(split)
    # list to hold repeated words in columns
    repititive_list = []

    # loop through columns and get repeated words (view output of cell below to understand approach)
    # Loop through columns
    for i in x_train.columns:
        # identify columns that have the word 'mean'
        if 'mean' in i:
            # get the column name but remove the word 'mean'
            repititive_list.append(i.replace('_mean', ''))
    # store the data as dictionary of strings (keys) to lists (values)
    repititive_dict = {}
    for i in repititive_list:
        repititive_dict[i] = []
    # Now we group similarly worded column names
    repititive_dict = {key: [col for col in x_train.columns if key in col] for key, val in repititive_dict.items()}
    for key, val in repititive_dict.items():
        # val here is a list of column names
        # get an attribute that best represents all columns in list
        features = pca(x_train[val], n_components)
        # do the same for test
        feat_test = pca(x_test[val], n_components)
        if n_components > 1:
            new_keys = []
            for i in range(n_components):
                new_keys.append(key + '_1')
        elif n_components == 1:
            new_keys = key
        # replace the columns with the pca attribute
        x_train.drop(columns=val, inplace=True)
        x_test.drop(columns=val, inplace=True)
        x_train[new_keys] = features
        x_test[new_keys] = feat_test
    if n_components == 1:
        measurement_feat = pca(x_train[['radius', 'area', 'perimeter']], 1)
        measurement_feat_test = pca(x_test[['radius', 'area', 'perimeter']], 1)
        x_train.drop(columns=['radius', 'area', 'perimeter'], inplace=True)
        x_train['measurements'] = measurement_feat
        x_test.drop(columns=['radius', 'area', 'perimeter'], inplace=True)
        x_test['measurements'] = measurement_feat_test
    return x_train, x_test, y_train, y_test

def straight_forward_pca(n_components, split):
    x_train, x_test, y_train, y_test = smote_balancing(split)
    x_train_ = pca(x_train, n_components)
    # x_train_ = pd.DataFrame(x_train_, columns=x_train.columns)
    x_test_ = pca(x_test, n_components)
    # x_test_ = pd.DataFrame(x_test_, columns=x_test.columns)
    return x_train, x_test, y_train, y_test

def normal_dist(split, data, n_components):
    if data == 1:
        x_train, x_test, y_train, y_test = data_cleaning(split)
    if data == 2:
        x_train, x_test, y_train, y_test = smote_balancing(split)
    if data == 3:
        x_train, x_test, y_train, y_test = system_pca(n_components, split)
    elif data == 4:
        x_train, x_test, y_train, y_test = straight_forward_pca(n_components, split)
    # next we remove skewness in the data
    pt=PowerTransformer(method='yeo-johnson') 
    norm_dist=pt.fit_transform(x_train)
    norm_dist=pd.DataFrame(norm_dist,columns=x_train.columns)
    # perform same action to the test data
    norm_test=pt.fit_transform(x_test)
    norm_test=pd.DataFrame(norm_test,columns=x_test.columns)
    # return pre-processed data
    return norm_dist, norm_test, y_train, y_test
