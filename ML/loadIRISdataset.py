from sklearn import datasets


#Load iris dataset return in dict obj.
iris_dataset =  datasets.load_iris()

#show keys
print(iris_dataset.keys())

#show data
print(iris_dataset['data'])

#show features name
print(iris_dataset['feature_names'])

#show target
print(iris_dataset['target'])

#show target name
print(iris_dataset['target_names'])

#show description
print(iris_dataset['DESCR'])

#access single datasets
print(iris_dataset['data'][0],iris_dataset['target'][0])

#dataset shape
print(iris_dataset['data'].shape)
