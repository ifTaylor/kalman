import numpy as np

class KF:
    def __init__(self, initial_x: float, initial_v: float, accel_variance: float) -> None:
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance
        self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        F = np.array([[1, dt], [0, 1]])
        new_x = F @ self._x

        G = np.array([[0.5 * dt ** 2], [dt]])
        new_P = F @ self._P @ F.T + G @ G.T * self._accel_variance

        self._x = new_x
        self._P = new_P

    def update(self, meas_value: float, meas_variance: float):
        H = np.array([[1, 0]])
        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H @ self._x
        S = H @ self._P @ H.T + R

        K = self._P @ H.T @ np.linalg.inv(S)

        new_x = self._x + K @ y
        new_P = (np.eye(2) - K @ H) @ self._P

        self._x = new_x
        self._P = new_P

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[0]

    @property
    def vel(self) -> float:
        return self._x[1]
