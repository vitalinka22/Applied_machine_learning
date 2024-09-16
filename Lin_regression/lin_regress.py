import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def lin_regress(x:np.array, y: np.array) -> tuple[float, float]:

    # number of training samples
    N = x.shape[0]

    # normal form
    # A = [N, sum(x); sum(x), sum(x ** 2)]
    # b = [sum(y), sum(y*x)]

    A = np.array([[N, np.sum(x)], [np.sum(x), np.sum(x ** 2)]])
    b = np.expand_dims(np.array([np.sum(y), np.sum(y * x)]), axis = -1)

    # solve Ax = b
    theta = np.linalg.solve(A, b).flatten()
    print(f"theta0 = {theta[0]}, theta1 = {theta[1]}")

    return (theta[0], theta[1])

# simple test case

x = np.arange(0, 10, 1)
y_hat = x + 0.1 * np.random.randn(10)
theta = lin_regress(x, y_hat)
theta_0, theta_1 = theta[0], theta[1]

plt.figure()
plt.scatter(x, y_hat)
plt.plot(x, theta_0 + theta_1 * x, color = "red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["data", "lin. fit"])
plt.title(rf'$\theta_0={theta_0:.3f}$, $\theta_1={theta_1:.3f}$')
plt.show()
"""
--- BEGINNING OF DRIVING DATA CASE STUDY ---
"""
data = np.genfromtxt("driving_data.csv", delimiter=",")

# extract data
velocity = data[:, 0]  # m/s
power = data[:, 1]  # W

# some constants as given in the exercise sheet
CW, A, RHO, G, M = 0.4, 1.5, 1.2, 9.81, 2400  # -, m^2, kg/m^3, m/s^2, kg

plt.figure()
plt.plot(velocity * 3.6, power / 1000, linestyle='none', marker='.', markersize=6)
plt.xlabel(r'driving velocity $v$ [km/h]')
plt.ylabel(r'Enginge power $P_{\mathrm{engine}}$ [kW]')
plt.show()

def wind_resistance(v:np.ndarray)->np.ndarray:
    return CW * A * (RHO * v ** 2) / 2


power_wo_wind = power - velocity * wind_resistance(v=velocity)

y = np.expand_dims(power_wo_wind, axis=1)
X = np.expand_dims(velocity * M * G, axis=1)

# simple implementation
theta = lin_regress(x=X, y=y)
theta_0, theta_1 = theta[0], theta[1]

print(f'\nlinear regression model: \ty = {theta_0:.4f} + {theta_1:.4f} * x')
print(f'rolling resistance \t\t\tcR = {theta_1:.4f}')

plt.figure()
plt.plot(X, y / 1000, linestyle='none', marker='.', markersize=6)
plt.plot(X, (theta_0 + X * theta_1) / 1000, color='red')
plt.xlabel(r'$v\cdot M \cdot g$')
plt.ylabel('Rolling power resistance [kW]')
plt.legend(['data', 'lin. fit'])
plt.show()

def wind_resistance(v:np.ndarray)->np.ndarray:
    return CW * A * (RHO * v ** 2) / 2

data = np.genfromtxt("driving_data.csv", delimiter=",")

# extract data
velocity = data[:, 0]  # m/s
power = data[:, 1]  # W

# some constants as given in the exercise sheet
CW, A, RHO, G, M = 0.4, 1.5, 1.2, 9.81, 2400  # -, m^2, kg/m^3, m/s^2, kg

plt.figure()
plt.plot(velocity * 3.6, power / 1000, linestyle='none', marker='.', markersize=6)
plt.xlabel(r'driving velocity $v$ [km/h]')
plt.ylabel(r'Enginge power $P_{\mathrm{engine}}$ [kW]')
plt.show()

"""
Step 1: subtract the wind resistance
- assumption: driving velocity == relative velocity (no wind)
"""



power_wo_wind = power - velocity * wind_resistance(v=velocity)

plt.figure()
plt.plot(velocity * 3.6, power_wo_wind / 1000, linestyle='none', marker='.', markersize=6)
plt.xlabel(r'Driving velocity $v$ [km/h]')
plt.ylabel('Power w/o wind [kW]')
plt.legend(['rolling resistance power'])
plt.show()

"""
Step 2: obtain rolling resistance force

rolling force must be the engine power without wind energy, then divided by the velocity
"""

# f_roll = power_wo_wind / velocity

"""
Step 3: set up linear regression model 

P_roll = v * cR * M * g * cos(alpha)

--> we want to find cR:
y = theta_0 + theta_1 * (v * M * g * cos(alpha))

alpha (inclination) is unknown -> set it to zero
theta_1 will be cR
theta_0 will be some offset of the power (additional users, heating, etc.)
"""
y = np.expand_dims(power_wo_wind, axis=1)
X = np.expand_dims(velocity * M * G, axis=1)

# simple implementation
theta = lin_regress(x=X, y=y)
theta_0, theta_1 = theta[0], theta[1]

print(f'\nlinear regression model: \ty = {theta_0:.4f} + {theta_1:.4f} * x')
print(f'rolling resistance \t\t\tcR = {theta_1:.4f}')

plt.figure()
plt.plot(X, y / 1000, linestyle='none', marker='.', markersize=6)
plt.plot(X, (theta_0 + X * theta_1) / 1000, color='red')
plt.xlabel(r'$v\cdot M \cdot g$')
plt.ylabel('Rolling power resistance [kW]')
plt.legend(['data', 'lin. fit'])
plt.show()
