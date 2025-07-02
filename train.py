import csv
import pickle as pkl

import numpy as np

theta0 = 0
theta1 = 0

try:
    with open("data.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        kms = []
        prices = []

        for row in reader:
            kms.append(int(row[0]))
            prices.append(int(row[1]))

    m = len(kms)

    if m == 0:
        raise ValueError("The dataset is empty.")

    max_km = max(kms)
    max_price = max(prices)

    kms_norm = [elem / max_km for elem in kms]
    prices_norm = [elem / max_price for elem in prices]

    learning_rate = 0.01
    prev_error = float("inf")

    while True:
        error = (
            sum(
                ((theta0 + theta1 * kms_norm[i]) - prices_norm[i]) ** 2
                for i in range(m)
            )
            / m
        )

        if prev_error == error:
            break

        prev_error = error

        theta0 = theta0 - (
            learning_rate
            * sum(
                (theta0 + theta1 * kms_norm[i]) - prices_norm[i]
                for i in range(m)
            )
            / m
        )
        theta1 = theta1 - (
            learning_rate
            * sum(
                ((theta0 + theta1 * kms_norm[i]) - prices_norm[i])
                * kms_norm[i]
                for i in range(m)
            )
            / m
        )

    theta0 = theta0 * max_price
    theta1 = theta1 * max_price / max_km

    price_mean = sum(prices) / m

    ss_res = sum(
        ((theta0 + theta1 * kms[i]) - prices[i]) ** 2 for i in range(m)
    )
    ss_tot = sum((p - price_mean) ** 2 for p in prices)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    precision = r2_score * 100

    x = np.linspace(0, max_km)
    y = theta0 + theta1 * x

    data = {
        "theta0": theta0,
        "theta1": theta1,
        "x": x,
        "y": y,
        "prices": prices,
        "kms": kms,
        "max_price": max_price,
        "max_km": max_km,
        "precision": precision,
    }

    with open("data.pkl", "wb") as file:
        pkl.dump(data, file)

except FileNotFoundError:
    print("Error: 'data.csv' file not found.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
