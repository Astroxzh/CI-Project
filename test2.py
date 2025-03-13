import jax.numpy as jnp
from jax import grad, jit
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt
from chex import assert_equal_shape

# Data reading functions (unchanged)
def probe_read(filepath=r'Papercode\reconstructions\e17965_1_00678_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        dataset = f['probe']
        probe = jnp.array(dataset[:])  # Shape: (2, 264, 264) for two modes
        return probe

def data_read(filepath=r'Papercode\cxi_files\e17965_1_00677.cxi'):
    with h5py.File(filepath, 'r') as f:
        data = f['entry_1/data_1/data'][:]
        data = jnp.mean(data, axis=0)  # Shape: (264, 264)
    return data

def background_read(filepath=r'Papercode\reconstructions\e17965_1_00677_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        dataset = f['background']
        background = jnp.array(dataset[:])  # Shape: (264, 264)
        return background

# Padding function (unchanged)
def pad_array(array, target_shape):
    pad_m = (target_shape[0] - array.shape[0]) // 2
    pad_n = (target_shape[1] - array.shape[1]) // 2
    return jnp.pad(array, ((pad_m, target_shape[0] - array.shape[0] - pad_m), 
                           (pad_n, target_shape[1] - array.shape[1] - pad_n)), mode='constant')

# Forward model with phase-only constraint
@jit
def forward_model(T_low, probe):
    m, n = probe.shape[1], probe.shape[2]  # e.g., 264, 264
    T_fre_low = jnp.fft.fftshift(jnp.fft.fft2(T_low))  # FFT of low-res phase
    T_fre_padded = pad_array(T_fre_low, (m, n))  # Upsample to probe size
    T_high = jnp.fft.ifft2(jnp.fft.ifftshift(T_fre_padded))  # High-res phase
    obj_high = jnp.exp(1j * T_high)  # Phase-only object
    diff_pattern0 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(probe[0, :, :] * obj_high)))**2
    diff_pattern1 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(probe[1, :, :] * obj_high)))**2
    simulated = diff_pattern0 * 0.756 + diff_pattern1 * 0.244  # Weighted sum
    return simulated

# Loss function (unchanged)
@jit
def loss_function(simulated: jnp.ndarray, background: jnp.ndarray, measured: jnp.ndarray) -> float:
    assert_equal_shape([simulated, background, measured])
    factor = 1.0 / (jnp.sum(measured) + 1e-10)
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(measured)
    return factor * jnp.sum((simulated_amp - measured_amp) ** 2)

# Derivative with respect to T_low
def derivative_loss_function_wrt_T(T_low: jnp.ndarray, probe: jnp.ndarray, 
                                  measured: jnp.ndarray, background: jnp.ndarray) -> jnp.ndarray:
    def loss_wrt_T(t):
        simulated = forward_model(t, probe)
        return loss_function(simulated, background, measured)
    return grad(loss_wrt_T)(T_low)

# Adam optimization
def adam_optimization(init_T: jnp.ndarray, measured: jnp.ndarray, background: jnp.ndarray, 
                      probe: jnp.ndarray, alpha: float, num_iterations: int) -> jnp.ndarray:
    optimizer = optax.adam(alpha)
    opt_state = optimizer.init(init_T)
    T = init_T
    best_loss = float('inf')
    plateau_counter = 0
    current_alpha = alpha

    for _ in range(num_iterations):
        grad_T = derivative_loss_function_wrt_T(T, probe, measured, background)
        updates, opt_state = optimizer.update(grad_T, opt_state, T)
        T = optax.apply_updates(T, updates)
        simulated = forward_model(T, probe)
        loss = loss_function(simulated, background, measured)
        
        if loss < best_loss * 0.99:
            best_loss = loss
            plateau_counter = 0
        else:
            plateau_counter += 1

        if plateau_counter >= 10:
            current_alpha *= 0.1
            print(f"Alpha reduced to {current_alpha}")
            optimizer = optax.adam(current_alpha)
            opt_state = optimizer.init(T)
            plateau_counter = 0

        if _ % 50 == 0:
            print(f"Iteration: {_}, Loss: {loss}")

        if loss < 1e-9 or current_alpha < 1e-9:
            print("Converged below threshold.")
            break
    return T

# Example usage
data = data_read()  # Shape: (264, 264)
probe = probe_read()  # Shape: (2, 264, 264)
background = background_read()  # Shape: (264, 264)

# Initialize low-resolution phase
low_res_shape = (70, 70)  # Band-limited size, R ≈ 70/264 ≈ 0.265 < 0.4
key = jran.PRNGKey(0)
init_T = jran.normal(key, shape=low_res_shape) * 0.1  # Small random phases

# Optimize
update_T = adam_optimization(init_T, data, background, probe, 0.4, 1000)

# Compute high-resolution object for visualization
m, n = probe.shape[1], probe.shape[2]
T_fre_low = jnp.fft.fftshift(jnp.fft.fft2(update_T))
T_fre_padded = pad_array(T_fre_low, (m, n))
T_high = jnp.fft.ifft2(jnp.fft.ifftshift(T_fre_padded))
obj_high = jnp.exp(1j * T_high)

# Visualize the phase
plt.imshow(jnp.angle(obj_high), cmap='gray')
plt.title("Reconstructed Phase")
plt.colorbar()
plt.show()