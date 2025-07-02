import pickle as pkl
import signal

import matplotlib.pyplot as plt

try:
    signal.signal(
        signal.SIGINT,
        lambda *_: (
            print("\033[2Dft_linear_regression: CTRL+C sent by user."),
            exit(1),
        ),
    )

    mileage = int(input("Give mileage: "))

    if mileage <= 0:
        raise ValueError("Mileage must be greater than 0.")

    with open("data.pkl", "rb") as file:
        data = pkl.load(file)

    prediction = data["theta0"] + data["theta1"] * mileage

    print(f"Predicted price for {mileage} km: {prediction}")
    print(f"Precision of the algorithm: {data['error']}")

    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.plot(data["x"], data["y"], color="red")
    plt.scatter(data["km"], data["price"])
    plt.scatter(mileage, prediction, color="green", zorder=3)
    plt.show()

except FileNotFoundError:
    print("Error: The file 'data.pkl' was not found.")
except KeyError as e:
    print(f"Error: Missing key in data file: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
