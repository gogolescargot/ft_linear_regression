# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/01/11 02:34:05 by ggalon            #+#    #+#              #
#    Updated: 2025/06/06 13:54:12 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import pickle as pkl

import numpy as np

theta0 = 0
theta1 = 0

try:
    with open("data.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        km = []
        price = []

        for row in reader:
            km.append(int(row[0]))
            price.append(int(row[1]))

    m = len(km)

    if m == 0:
        raise ValueError("The dataset is empty.")

    max_km = max(km)
    max_price = max(price)

    for i in range(m):
        km[i] /= max_km
        price[i] /= max_price

    learning_rate = 0.01
    prev_error = float("inf")

    while True:
        error = (
            sum(((theta0 + theta1 * km[i]) - price[i]) ** 2 for i in range(m))
            / m
        )

        if prev_error == error:
            break

        prev_error = error

        theta0 = theta0 - (
            learning_rate
            * sum((theta0 + theta1 * km[i]) - price[i] for i in range(m))
            / m
        )
        theta1 = theta1 - (
            learning_rate
            * sum(
                ((theta0 + theta1 * km[i]) - price[i]) * km[i]
                for i in range(m)
            )
            / m
        )

    theta0 = theta0 * max_price
    theta1 = theta1 * max_price / max_km

    x = np.linspace(0, max_km)
    y = theta0 + theta1 * x

    data = {
        "theta0": theta0,
        "theta1": theta1,
        "x": x,
        "y": y,
        "price": [elem * max_price for elem in price],
        "km": [elem * max_km for elem in price],
        "max_price": max_price,
        "max_km": max_km,
        "error": error,
    }

    with open("data.pkl", "wb") as file:
        pkl.dump(data, file)

except FileNotFoundError:
    print("Error: 'data.csv' file not found.")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
