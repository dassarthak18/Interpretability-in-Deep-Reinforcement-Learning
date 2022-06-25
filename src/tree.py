import graphviz
import numpy as np
from tqdm import tqdm
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from src.data import output

def build_tree(env,filename,split=0.33,num=None):
  # Defining parameters
  try:
    n = env.observation_space.shape[0]
  except:
    n = 1

  # Extracting data from csv
  data = read_csv(filename)
  X = []
  try:
    for i in range(n):
      temp = data[f'Input {i}'].tolist()
      X.append(temp)
  except:
    temp = data[f'Input'].tolist()
    X.append(temp)
  Y = data['Output'].tolist()
  
  X_encoded = np.array(X)
  Y_encoded = np.array(Y)

  # Building the Decision Tree
  Tree = DecisionTreeClassifier(max_depth=num)
  X1_encoded = []
  n = len(X_encoded[0])
  for i in range(n):
    arr = []
    for j in range(len(X_encoded)):
      arr.append(X_encoded[j][i])
    X1_encoded.append(arr)
  X_train, X_test, Y_train, Y_test = train_test_split(X1_encoded, Y_encoded, test_size=split, random_state=42)
  Tree.fit(X_train, Y_train.reshape(-1,1))

  # Testing the Decision Tree
  count = 0
  for i in tqdm(range(len(Y_test))):
    true = Y_test[i]
    pred = Tree.predict([X_test[i]], check_input=True)[0]
    if true == pred:
      count += 1
  print(f"Instances checked: {len(Y_test)}\nPredictions matched: {count}\nAccuracy: {float(count*100/len(Y_test))}%")

  return Tree

# Visualizing the Decision Tree
def visualize_tree(env,Tree):
  n = env.action_space.n
  class_names = []
  for i in range(n):
    class_names.append(str(i))
  data = export_graphviz(Tree,class_names=class_names,filled=True)
  graph = graphviz.Source(data, format="png")
  return graph