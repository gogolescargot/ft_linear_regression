# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    predict.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/01/11 02:34:08 by ggalon            #+#    #+#              #
#    Updated: 2025/01/11 15:01:53 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import pickle as pkl

mileage = int(input("Give mileage: "))

with open('data.pkl', 'rb') as file:
	data = pkl.load(file)

prediction = data['theta0'] + data['theta1'] * mileage

print(f"Predicted price for {mileage} km: {prediction}")
print(f"Precision of the algorithm: {data['error']}")

plt.xlabel('Mileage')
plt.ylabel('Price')
plt.plot(data['x'], data['y'], color='red')
plt.scatter(data['km'], data['price'])
plt.scatter(mileage, prediction, color='green', zorder=3)
plt.show()