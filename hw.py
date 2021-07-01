import statistics as st
import pandas as pd
import plotly.express as pe
import plotly.figure_factory as pf
import numpy as np
import matplotlib.pyplot as mp
from sklearn.linear_model import LogisticRegression

data= pd.read_csv("escape_velocity.csv")
velocity = data["Velocity"].tolist()
escape = data["Escaped"].tolist()
#Plotting the naked data
graph = pe.scatter(x=velocity, y=escape)
graph.show()

#Plotting the line of regression
arrayvelocity = np.array(velocity)
arrayescape = np.array(escape)

m,c = np.polyfit(arrayvelocity, arrayescape, 1)
y=[]

for x in arrayvelocity:
    newy=m*x+c
    y.append(newy)
graph1 = pe.scatter(x=arrayvelocity, y=arrayescape)
graph1.update_layout(shapes=[dict(type="line", y0=min(y), y1=max(y), x0=min(arrayvelocity), x1=max(arrayvelocity))])
print("This is a uselss graph because of the data given, but here it is anyway.")
graph1.show()
print("Graph done, now starting logistic regression. ")

#Plotting real data

X= np.reshape(velocity, (len(velocity),1 ))
Y= np.reshape(escape, (len(escape),1 ))
lr = LogisticRegression()
lr.fit(X,Y)
mp.figure()
mp.scatter(X.ravel(), Y.ravel(), color="turquoise", zorder=20)

#Plotting logistic regression
def model(x):
  # formula below
  return 1.0/(1+np.exp(-x))
xtest=np.linspace(0,5000,10000)
chances = model(xtest*lr.coef_+lr.intercept_).ravel()
mp.plot(xtest,chances,color="yellow", linewidth=3)
mp.axhline(y=0,color="green", linestyle="-")
mp.axhline(y=0.5, color="red", linestyle="-")
mp.axhline(y=1, color="blue", linestyle="-")
mp.axvline(x=xtest[23], color="black", linestyle="--")
mp.xlim(5,19)
mp.show()
print('Logistic regression and s curve completed. Onward!')

#Taking user input and acting on it
input = float(input("Enter a velocity(how fast the gerbil runs), and then I will tell you if it escapes it's cage or not."))
change= model(input*lr.coef_+lr.intercept_).ravel()
if(change>0.6):
  print("Your annoying gerbil will escape and traumatize the population.")
else:
  print("Your gerbil will stay in it's cage.")
