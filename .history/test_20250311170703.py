import jax.numpy as jnp
from jax import grad
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt

def probe_read(filepath):
    with h5py.File(filepath, 'r') as f:
        dataset = f['probe']
        probe = jnp.array(dataset[:], dtype=jnp.complex64)  # Ensure complex type
        return probe
    
def data_read(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['entry_1/data_1/data'][:]
        # Convert to float32 and ensure non-negative
        data = jnp.array(data, dtype=jnp.float32)
        data = jnp.sum(data, axis=0)
        data = jnp.maximum(data, 0.0)  # Clip negative values
    return data

def background_read(filepath):
    with h5py.File(filepath, 'r') as f:
        dataset = f['background']
        background = jnp.array(dataset[:], dtype=jnp.float32)
        background = jnp.maximum(background, 0.0)  # Ensure non-negative
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
    
    exit_wave_0 = probe[0] * obj_high
    exit_wave_1 = probe[1] * obj_high
    
    diff_pattern0 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave_0)))**2
    diff_pattern1 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave_1)))**2
    
    weight0 = jnp.sum(jnp.abs(probe[0])**2)
    weight1 = jnp.sum(jnp.abs(probe[1])**2)
    total_weight = weight0 + weight1 + 1e-10
    coeff0 = weight0 / total_weight
    coeff1 = weight1 / total_weight
    
    return diff_pattern0 * coeff0 + diff_pattern1 * coeff1

def loss_function(simulated, background, measured):
    # Add validation checks
    simulated = jnp.maximum(simulated, 0.0)  # Ensure non-negative
    background = jnp.maximum(background, 0.0)
    
    factor = 1.0 / (jnp.sum(measured) + 1e-10)
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(jnp.maximum(measured, 0.0))
    return factor * jnp.sum((simulated_amp - measured_amp) ** 2)

def derivative_loss_function_wrt_obj(obj_low, probe, measured, background):
    def loss_wrt_obj(o):
        simulated = forward_model(o, probe)
        return loss_function(simulated, background, measured)
    return grad(loss_wrt_obj)(obj_low)

def adam_optimization(init_obj_low, measured, background, probe, alpha=0.4, num_iterations=1000):
    # Data validation
    print("Data range:", jnp.min(measured), jnp.max(measured))
    print("Background range:", jnp.min(background), jnp.max(background))
    
    optimizer = optax.adam(alpha)
    opt_state = optimizer.init(init_obj_low)
    obj_low = init_obj_low
    
    for i in range(num_iterations):
        grad_obj = derivative_loss_function_wrt_obj(obj_low, probe, measured, background)
        updates, opt_state = optimizer.update(grad_obj, opt_state, obj_low)
        obj_low = optax.apply_updates(obj_low, updates)
        
        # Calculate loss every 100 iterations
        if i % 100 == 0:
            simulated = forward_model(obj_low, probe)
            current_loss = loss_function(simulated, background, measured)
            print(f"Iteration {i}, Loss: {current_loss:.4e}")

    return obj_low

# Example usage
data = data_read(r'Papercode\cxi_files\e17965_1_00677.cxi')
probe = probe_read(r'Papercode\reconstructions\e17965_1_00678_ptycho_reconstruction.h5')
background = background_read()

# Initialize with proper complex type
key = jran.PRNGKey(0)
key_real, key_imag = jran.split(key)
real_part = jran.normal(key_real, shape=data.shape)
imag_part = jran.normal(key_imag, shape=data.shape)
initial_obj_guess = real_part.astype(jnp.complex64) + 1j*imag_part.astype(jnp.complex64)

update_obj = adam_optimization(initial_obj_guess, data, background, probe)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(jnp.angle(update_obj), cmap='viridis')
plt.title("Reconstructed Phase")
plt.colorbar()

plt.subplot(122)
plt.imshow(jnp.abs(update_obj), cmap='gray')
plt.title("Reconstructed Amplitude")
plt.colorbar()
plt.show()
