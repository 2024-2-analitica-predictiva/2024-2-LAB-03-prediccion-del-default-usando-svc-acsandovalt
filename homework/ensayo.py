from sklearn.svm import SVC
import functions
from sklearn.metrics import balanced_accuracy_score
# Paso 1: Cargar los datos

data_test= functions.load_data("files/input/test_data.csv.zip")
data_train= functions.load_data("files/input/train_data.csv.zip")

# Paso 2: Limpiar los datos
data_test = functions.clean_data(data_test)
data_train = functions.clean_data(data_train)
# paso 3: dividir los datos 
x_test=data_test.drop("default", axis=1)
y_test=data_test[["default"]]
x_train=data_train.drop("default", axis=1)
y_train=data_train[["default"]]
# Paso 4: Crear un pipeline para el modelo de clasificación
pipeline = functions.make_pipeline(
    estimator=SVC(kernel='linear', random_state=42))
pipeline

# Paso 5: Definir los hiperparámetros para la búsqueda en cuadrícula


param_grid = {

    'pca__n_components': [2, 5, 10, 15, 20],
    'estimator__coef0': [0.0, 1.0],
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__class_weight': [None, 'balanced'],
    'estimator__max_iter': [1000, -1],
    
}

#Paso 6: Crear el objeto GridSearchCV

estimator = functions.make_grid_search(estimator=pipeline, param_grid=param_grid, cv=10)

# Paso 7: Ajustar el modelo a los datos de entrenamiento
estimator.fit(x_train, y_train)
print("Mejor modelo: ")
print(estimator)
