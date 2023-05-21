import numpy as np
import matplotlib.pyplot as plt
from KF import KF

def initialize_kf():
    return KF(initial_x=0.0, initial_v=1.0, accel_variance=0.01)


def generate_noise(variance):
    return np.random.normal(scale=np.sqrt(variance)) 

def simulate_motion(real_x, real_v, dt):
    return real_x + dt * real_v

def plot_results(mus, covs, real_xs, real_vs, plt_noise):
    plt.subplot(2, 1, 1)
    plt.title('Position')
    plt.plot([mu[0] for mu in mus], 'r', label='Estimate')
    plt.plot(real_xs, 'b', label='True')
    plt.plot(plt_noise, 'g', label='Measurement')
    plt.plot([mu[0] - 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--', label='Estimate +/- 2*sqrt(cov)')
    plt.plot([mu[0] + 2*np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Velocity")
    plt.plot([mu[1] for mu in mus], 'r', label='Estimate')
    plt.plot(real_vs, 'b', label='True')
    plt.plot([mu[1] - 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--', label='Estimate +/- 2*sqrt(cov)')
    plt.plot([mu[1] + 2*np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
    plt.legend()

    plt.show()

def main():
    plt.ion()
    plt.figure()

    kf = initialize_kf()

    DT = 0.01
    NUM_STEPS = 1000
    MEAS_EVERY_STEPS = 5

    real_x = 0.0
    meas_variance = 0.2 ** 2
    real_v = 0.5

    mus = []
    covs = []
    real_xs = []
    real_vs = []
    plt_noise = []

    for step in range(NUM_STEPS):
        if step > 500:
            real_v *= 0.9
        
        covs.append(kf.cov)
        mus.append(kf.mean)

        real_x = simulate_motion(real_x, real_v, DT)

        kf.predict(dt=DT)

        noise = generate_noise(meas_variance)
        plt_noise.append(real_x + generate_noise(meas_variance))

        if step != 0 and step % MEAS_EVERY_STEPS == 0:
            kf.update(meas_value=real_x + noise, meas_variance=meas_variance)
            
        real_xs.append(real_x)
        real_vs.append(real_v)

    plot_results(mus, covs, real_xs, real_vs, plt_noise)

if __name__ == "__main__":
    main()
    # plt.ginput(1)
