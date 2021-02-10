from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from django.conf import settings


def home(request):
	if request.method == 'POST':
		# for getting project base folder
		path = settings.BASE_DIR
		print('MY PATH:',path)

		# get value/input from html file
		getresult = request.POST.get('speed')

		df = pd.read_csv(r'{}\staticfiles\car_driving_risk_analysis.csv'.format(path))

		x = df[['speed']]
		y = df['risk']
		xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.20,random_state=1)
		reg = LinearRegression()
		reg.fit(xtrain,ytrain)
		result = reg.predict([[getresult]])
		result = result[0]

		context = {
			'result':result
		}
		return render(request,'home.html',context)
	else:
		return render(request,'home.html')