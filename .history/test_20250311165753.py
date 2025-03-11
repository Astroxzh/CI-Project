import jax.numpy as jnp
from jax import grad
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt

def probe_read(filepath=r'Papercode\reconstructions\e17965_1_00678_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        dataset = f['probe']
        probe = jnp.array(dataset[:])  # Reads both probe modes
        return probe
    
def data_read(filepath=r'Papercode\cxi_files\e17965_1_00677.cxi'):
    with h5py.File(filepath, 'r') as f:
        data = f['entry_1/data_1/data'][:]
        data = jnp.sum(data, axis=0)  # Sum over frames if necessary
    return data

def background_read(filepath=r'Papercode\reconstructions\e17965_1_00677_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        dataset = f['background']
        background = jnp.array(dataset[:])
        return background

def pad_array(array, target_shape):
    pad_m = (target_shape[0] - array.shape[0]) // 2
    pad_n = (target_shape[1] - array.shape[1]) // 2
    return jnp.pad(array, 
                   ((pad_m, target_shape[0] - array.shape[0] - pad_m), 
                    (pad_n, target_shape[1] - array.shape[1] - pad_n)), 
                   mode='constant')

def forward_model(obj_low, probe):
    m, n = probe.shape[1], probe.shape[2]
    obj_fre_low = jnp.fft.fftshift(jnp.fft.fft2(obj_low))
    obj_fre_padded = pad_array(obj_fre_low, (m, n))
    obj_high = jnp.fft.ifft2(jnp.fft.ifftshift(obj_fre_padded))
    
    # Compute exit waves for each probe mode
    exit_wave_0 = probe[0] * obj_high
    exit_wave_1 = probe[1] * obj_high
    
    # Calculate diffraction patterns
    diff_pattern0 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave_0)))**2
    diff_pattern1 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave_1)))**2
    
    # Dynamically determine coefficients based on probe intensity
    weight0 = jnp.sum(jnp.abs(probe[0])**2)
    weight1 = jnp.sum(jnp.abs(probe[1])**2)
    total_weight = weight0 + weight1
    coeff0 = weight0 / total_weight
    coeff1 = weight1 / total_weight
    
    simulated = diff_pattern0 * coeff0 + diff_pattern1 * coeff1
    return simulated

def loss_function(simulated: jnp.ndarray, background: jnp.ndarray, measured: jnp.ndarray) -> float:
    factor = 1.0 / (jnp.sum(measured) + 1e-10)
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(measured)
    return factor * jnp.sum((simulated_amp - measured_amp) ** 2)

def derivative_loss_function_wrt_obj(obj_low: jnp.ndarray, probe: jnp.ndarray, measured: jnp.ndarray, background: jnp.ndarray) -> jnp.ndarray:
    def loss_wrt_obj(o):
        simulated = forward_model(o, probe)
        return loss_function(simulated, background, measured)
    return grad(loss_wrt_obj)(obj_low)

def adam_optimization(init_obj_low, measured, background, probe, alpha=0.4, num_iterations=1000, patience=10):
    # Initialize optimizer with initial learning rate
    optimizer = optax.adam(alpha)
    opt_state = optimizer.init(init_obj_low)
    obj_low = init_obj_low
    best_loss = float('inf')
    patience_counter = 0
    current_alpha = alpha

    for _ in range(num_iterations):
        grad_obj = derivative_loss_function_wrt_obj(obj_low, probe, measured, background)
        updates, opt_state = optimizer.update(grad_obj, opt_state, obj_low)
        obj_low = optax.apply_updates(obj_low, updates)
        simulated = forward_model(obj_low, probe)
        current_loss = loss_function(simulated, background, measured)

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Reduce learning rate
                current_alpha *= 0.1
                print(f"Reducing learning rate to {current_alpha}")
                # Reinitialize optimizer with new alpha
                optimizer = optax.adam(current_alpha)
                opt_state = optimizer.init(obj_low)
                patience_counter = 0

        if current_loss < 1e-9 or current_alpha < 1e-4:
            print(f"Converged with loss {current_loss}.")
            break

    return obj_low

# Example usage
data = data_read()
probe = probe_read()
background = background_read()

# Initialize object with complex random values
key = jran.PRNGKey(0)
key_real, key_imag = jran.split(key)
real_part = jran.normal(key_real, shape=data.shape)
imag_part = jran.normal(key_imag, shape=data.shape)
initial_obj_guess = real_part + 1j * imag_part

# Run optimization with parameters from the paper
update_obj = adam_optimization(initial_obj_guess, data, background, probe, alpha=0.4, num_iterations=1000)

# Visualization
plt.imshow(jnp.angle(update_obj))
plt.colorbar()
plt.show()
