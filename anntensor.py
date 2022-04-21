#Importing necessary Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
""" 
NumPy is used for working with arrays.It also has functions for working in the domain of linear algebra, Fourier transform, 
and matrices.
Pandas is a tool used for data wrangling and analysis.
"""
#Loading Dataset
"""
you can see in the dataset, there are 13 independent variables and 1 dependent variable This dataset has Customer Id, 
Surname, **Credit Score, Geography, Gender, Age, Tenure, Balance, Num of Products they( use from the bank such as credit card or loan, etc), 
Has Credit card or not (1 means yes 0 means no), Is Active Member ( That means the customer is using the bank or not), estimated salary.
So these all are independent variables of the Churn Modelling dataset. The last feature is the dependent variable and 
that is customer exited or not from the bank in the future( 1 means the customer will exit the bank and 0 means the customer will stay in the bank.)
"""
data = pd.read_csv("Churn_Modelling.csv")
"""
x:features from credit_score to estimated_salary.
y:Dependent Variable
"""
#Generating Matrix of Features
X = data.iloc[:,3:-1].values
#Generating Dependent Variable Vectors
Y = data.iloc[:,-1].values

#Encoding Categorical Variable Gender
""" 
we can see, there are two categorical variables-Geography and Gender.
So we have to encode these categorical variables into some labels such as 0 and 1 for gender. 
And one hot encoding for geography variable.
"""
from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
X[:,2] = np.array(LE1.fit_transform(X[:,2]))
#Encoding Categorical variable Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))
"Spain will be encoded as 001, France will be 010 and germany 100"
#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#Performing Feature Scaling
"""
certain variables have very high values while certain variables have very low values. So there is a chance that during model creation, 
the variables having extremely high-value dominate variables having extremely low value. 
Because of this, there is a possibility that those variables with the low value might be neglected by
our model Standardization Normalization
Whenever standardization is performed, all values in the dataset will be converted into values ranging between -3 to +3. While in the case of normalization, all values will be converted into a range between -1 to +1.
Normalization is used only when our dataset follows a normal distribution while standardization is a universal technique that can be used
for any dataset irrespective of the distribution
we are going to use Standardization
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initialising ANN
"""
The Sequential class is a part of the models module of Keras library which is a part of the tensorflow library now
The Sequential class allows us to build ANN but as a sequence of layers. Here we are going to create a network that will have
2 hidden layers, 1 input layer, and 1 output layer. 
"""
ann = tf.keras.models.Sequential()
#Adding First Hidden Layer
"""
Here we have created our first hidden layer by using the Dense class which is part of the layers module.
This class accepts 2 inputs:-
1. units:- number of neurons that will be present in the respective layer
2. activation:- specify which activation function to be used we are always going to use “relu”[rectified linear unit] as an activation function for
hidden layers
"""
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#Adding Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#Adding Output Layer
"""
we are going to use the Dense class in order to create the output layer. 
1. In a binary classification problem(like this one) where we will be having only two classes as output (1 and 0), we will be allocating only one neuron to output this result. For the multiclass classification
problem, we have to use more than one neuron in the output layer.
2. For the binary classification Problems, the activation function that should always be used is sigmoid. For a multiclass classification problem, the activation function that should be used is softmax.
"""
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
#Compiling ANN
"""
Compile method accepts 
1. optimizer:- specifies which optimizer to be used in order to perform stochastic gradient descent
2. loss:- specifies which loss function should be used. 
3. metrics:- which performance metrics to be used in order to compute performance.
"""
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
#Fitting ANN
"""
Here we have used the fit method in order to train our ann. The fit method is accepting 4 inputs in this case:-
1.X_train:- Matrix of features for the training dataset
2.Y_train:- Dependent variable vectors for the training dataset
3.batch_size: how many observations should be there in the batch.
4.epochs: How many times neural networks will be trained"""
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)
y_pred = ann.predict(X_test)
"""
y_pred > 0.5 means if y-pred is in between 0 to 0.5, then this new y_pred
will become 0(False). And if y_pred is larger than 0.5, then new y_pred
will become 1(True).
"""
Y_pred = (y_pred > 0.5)
"""
The Confusion Matrix is a way to find how many predicted categories or classes were correctly predicted and how many were not.
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
#Saving created neural network
ann.save("ANN.h5")