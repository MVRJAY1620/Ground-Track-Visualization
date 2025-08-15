import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

MU = 398600.4418  # Earth's gravitational parameter (km^3/s^2)

# Function to compute time in days since J2000 epoch
def time_since_j2000(dt):
    y, m, d = dt.year, dt.month, dt.day
    hr, mn, sec = dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
    term1 = 367 * y - np.floor(7 * (y + np.floor((m + 9) / 12)) / 4)
    term2 = np.floor(275 * m / 9)
    day_frac = d + (hr + mn/60 + sec/3600) / 24
    return term1 + term2 + day_frac - 730531.5

# Compute Greenwich Sidereal Angle for a given time (Earth rotation)
def greenwich_sidereal_angle(days_since_2000):
    theta_deg = 280.46061837 + 360.9856473 * days_since_2000
    return np.radians(theta_deg % 360)

# Convert ECI coordinates to ECEF frame
def convert_eci_to_ecef(r_eci, t_seconds, epoch_dt):
    """
    r_eci : np.array
        Satellite position in ECI frame
    t_seconds : float
        Time elapsed since epoch (s)
    epoch_dt : datetime
        Epoch start time
    """
    dt = epoch_dt + timedelta(seconds=t_seconds)
    d2000 = time_since_j2000(dt)
    theta = greenwich_sidereal_angle(d2000)

    # Rotation matrix for ECI -> ECEF
    R = np.array([
        [np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0,              0,             1]
    ])
    return R @ r_eci

# Convert ECEF to latitude and longitude
def ecef_to_latlon(r_ecef):
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    return np.degrees(lat), np.degrees(lon)

# Two-body acceleration function for RK4 propagation
def acceleration(state, t):
    pos = state[:3]
    vel = state[3:]
    r_mag = np.linalg.norm(pos)
    acc = -MU * pos / r_mag**3
    return np.concatenate((vel, acc))

# Single RK4 integration step
def rk4_integrate_step(f, current_state, t, h):
    k1 = f(current_state, t)
    k2 = f(current_state + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(current_state + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(current_state + h * k3, t + h)
    return current_state + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Main function to simulate satellite ground track
def simulate_ground_track(r_init, v_init, label, epoch_str, sim_time=86400, step=10):
    """
    r_init : np.array
        Initial position in ECI
    v_init : np.array
        Initial velocity in ECI
    label : str
        Satellite label for plotting
    epoch_str : str
        Epoch in YYDDD.ddddd format
    sim_time : int
        Total simulation time in seconds
    step : int
        RK4 step size in seconds
    """
    # Parse epoch string
    year = 2000 + int(epoch_str[:2])
    doy = float(epoch_str[2:])
    epoch = datetime(year, 1, 1) + timedelta(days=doy - 1)
    
    # Initialize state vector
    num_steps = int(sim_time / step)
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = np.concatenate((r_init, v_init))
    
    # Propagate orbit using RK4
    for i in range(1, num_steps):
        trajectory[i] = rk4_integrate_step(acceleration, trajectory[i-1], i*step, step)

    # Convert positions to lat/lon for plotting
    lats, lons = [], []
    for j in range(num_steps):
        r_ecef = convert_eci_to_ecef(trajectory[j, :3], j*step, epoch)
        lat, lon = ecef_to_latlon(r_ecef)
        lats.append(lat)
        lons.append(lon)

    # Plot the ground track
    plt.figure(figsize=(10, 5))
    plt.plot(lons, lats, '.', markersize=0.8, label=f'{label} Track')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.title(f'{label} Ground Track (24 hrs)')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
