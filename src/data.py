import numpy as np
from tqdm import tqdm

def output(model,tup):
       # Returning output from ANN model
       x = np.array([list(tup)]).reshape(1,1,len(tup))
       return np.argmax(model.predict(x))

def dataset(env,model,filename="agent_data.csv",steps=100000):
       i = 0
       observation = env.reset()
       done = False
       f = open(filename, 'w')
       string = ''
       try:
              for j in range(len(observation)):
                     string = string + "Input " + str(j) + ','
              string = string + "Output\n"
              f.write(string)
              pbar = tqdm(total = steps)
              while i < steps:
                     action = output(model,observation)
                     string = ''
                     for j in range(len(observation)):
                            string = string + str(observation[j]) + ','
                     string = string + str(action) + "\n"
                     f.write(string)
                     if done == True:
                            observation = env.reset()
                            done = False
                     else:
                            observation, reward, done, info = env.step(action)
                     i = i + 1
                     pbar.update(1)
              pbar.close()
              f.close()
       except:
              string = string + "Input,Output\n"
              f.write(string)
              pbar = tqdm(total = steps)
              while i < steps:
                     action = np.argmax(model.predict([observation]))
                     string = ''
                     string = string + str(observation) + ','
                     string = string + str(action) + "\n"
                     f.write(string)
                     if done == True:
                            observation = env.reset()
                            done = False
                     else:
                            observation, reward, done, info = env.step(action)
                     i = i + 1
                     pbar.update(1)
              pbar.close()
              f.close()