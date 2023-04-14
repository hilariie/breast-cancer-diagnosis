import yaml
from preprocessing import data_cleaning, system_pca, straight_forward_pca, normal_dist
from modeling import *

with open('config.yaml', 'r') as file:
    yml = yaml.safe_load(file)
random_state = 2

# Data pre-processing
print("pre-processing data")
if yml['data_preprocessing'] == 1:
    x_train, x_test, y_train, y_test = data_cleaning(yml['split'])
elif yml['data_preprocessing'] == 2:
    x_train, x_test, y_train, y_test = system_pca(yml['system_pca'], yml['split'])
elif yml['data_preprocessing'] == 3:
    x_train, x_test, y_train, y_test = straight_forward_pca(yml['pca'], yml['split'])
elif yml['data_preprocessing'] == 4:
    x_train, x_test, y_train, y_test = normal_dist(yml['split'], yml['norm_data'], yml['norm_pca'])
else:
    raise ValueError(f"In config.yaml file, set 'data_preprocessing' between 1 and 4. Got\
{yml['data_preprocess']} instead.")
print('Done with preprocessing stage')

# Modeling
print('Creating models')
if yml['model'] == 1:
    xgb = XGBClassifier(random_state=random_state)
    rfc = RandomForestClassifier(random_state=random_state)
    lr = LogisticRegression(random_state=random_state)
    svc = svm.SVC(kernel='linear')
    # store models in a list so we can train them at a go
    models = [xgb, rfc, lr, svc]
    print('Training statistical models')
    train_models(x_train, x_test, y_train, y_test, models)
    for i in models:
        if i == xgb:
            print(f"classification report for XGBoost Classifier")
        else:
            print(f"classification report for {i}")
        classification_rep(i, x_test, y_test)
elif yml['model'] == 2:
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=yml['val_split'],
                                                      random_state=random_state)
    print("Training Neural network model")
    history, model = nn_compile(x_train,
                                x_val,
                                y_train,
                                y_val,
                                optimizer=yml['optimizer'],
                                blocks=yml['blocks'],
                                loss=yml['loss'],
                                epochs=yml['epochs'],
                                batch_size=yml['batch_size'],
                                dropout=yml['dropout'],
                                regu=yml['regu'],
                                activation=yml['activation'],
                                patience=yml['patience'],
                                first=yml['first'],
                                verbose=yml['verbose'],
                                monitor=yml['monitor'])
    if 'sparse' in yml['loss']:
        sparse = True
    else: sparse = False
    nn_evaluation(model, x_train, x_val, x_test, y_train, y_val, y_test, sparse=sparse)
    if yml['plot']:
        plot_train_hist(history)
else: raise ValueError(f"In yaml file set 'model' as 1 or 2 and not {yml['model']}")