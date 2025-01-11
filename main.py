# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/01/09 20:40:49 by ggalon            #+#    #+#              #
#    Updated: 2025/01/11 01:58:55 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import matplotlib.pyplot as plt
import numpy as np

theta0 = 0
theta1 = 0

def predict():
	mileage = int(input("Give mileage: "))
	prediction = theta0 + theta1 * mileage	
	print(f"Predicted price for {mileage} km: {prediction}")
	
	plt.scatter(mileage, prediction, color='green', zorder=3)
	plt.show()

def train():
	
	global theta0, theta1

	with open('data.csv', 'r') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		
		km = []
		price = []

		for row in reader:
			km.append(int(row[0]))
			price.append(int(row[1]))

	plt.scatter(km, price)
	
	plt.xlabel('Mileage')
	plt.ylabel('Price')

	m = len(km)

	max_km = max(km)
	max_price = max(price)

	for i in range(m):
		km[i] /= max_km
		price[i] /= max_price

	learning_rate = 0.01
	
	prev_error = float('inf')

	while True:

		error = (1 / m) * sum( ( (theta0 + theta1 * km[i]) - price[i]) ** 2 for i in range(m) )
	
		if prev_error == error:
			break

		prev_error = error
		
		theta0 = theta0 - ( learning_rate * (1 / m) * sum( (theta0 + theta1 * km[i]) - price[i] for i in range(m) ) )
		theta1 = theta1 - ( learning_rate * (1 / m) * sum( ( (theta0 + theta1 * km[i]) - price[i] ) * km[i] for i in range(m) ) )

	theta0 = theta0 * max_price
	theta1 = theta1 * max_price / max_km

	x = np.linspace(0, max_km)
	y = theta0 + theta1 * x

	plt.plot(x, y, color='red')

def precision():
	return

train()
predict()