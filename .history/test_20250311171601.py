import jax.numpy as jnp
from jax import grad, value_and_grad
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt

def probe_read(filepath):
    with h5py.File(filepath, 'r') as f:
        dataset = f['probe']
        probe = jnp.array(dataset[:], dtype=jnp.complex64)
        return probe
    
def data_read(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['entry_1/data_1/data'][:]
        data = jnp.array(data, dtype=jnp.float32)
        data = jnp.sum(data, axis=0)
        data = jnp.maximum(data, 0.0)
    return data

def background_read(filepath):
    with h5py.File(filepath, 'r') as f:
        dataset = f['background']
        background = jnp.array(dataset[:], dtype=jnp.float32)
        background = jnp.maximum(background, 0.0)
    return background

def normalize_probe(probe, measured):
    """Scale probe intensity to match measured data"""
    probe_intensity = jnp.sum(jnp.abs(probe)**2)
    data_intensity = jnp.sum(measured)
    scale = jnp.sqrt(data_intensity / (probe_intensity + 1e-12))
    return probe * scale

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
    simulated = jnp.maximum(simulated, 0.0)
    background = jnp.maximum(background, 0.0)
    
    # Add intensity matching term
    total_simulated = jnp.sum(simulated)
    total_measured = jnp.sum(measured)
    intensity_match = jnp.abs(total_simulated - total_measured) / total_measured
    
    factor = 1.0 / (total_measured + 1e-10)
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(jnp.maximum(measured, 0.0))
    diff_term = factor * jnp.sum((simulated_amp - measured_amp) ** 2)
    
    return diff_term + 0.1 * intensity_match

def adam_optimization(init_obj_low, measured, background, probe, alpha=0.01, num_iterations=1000):
    # Normalize data
    max_val = jnp.max(measured)
    measured = measured / max_val
    background = background / max_val
    
    # Normalize probe
    probe = normalize_probe(probe, measured)
    
    # Initialize with proper phase (magnitude=1)
    obj_magnitude = jnp.ones_like(init_obj_low)
    obj_phase = jnp.angle(init_obj_low)
    obj_low = obj_magnitude * jnp.exp(1j * obj_phase)
    
    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(alpha)
    )
    opt_state = optimizer.init(obj_low)
    
    # Use value_and_grad to compute loss and gradient together
    loss_grad_fn = value_and_grad(lambda x: loss_function(forward_model(x, probe), background, measured))

    best_loss = float('inf')
    best_obj = obj_low
    
    for i in range(num_iterations):
        current_loss, grad_obj = loss_grad_fn(obj_low)
        
        # Early stopping check
        if current_loss < best_loss:
            best_loss = current_loss
            best_obj = obj_low
        
        # Apply updates
        updates, opt_state = optimizer.update(grad_obj, opt_state, obj_low)
        obj_low = optax.apply_updates(obj_low, updates)
        
        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {current_loss:.4e}")

        if current_loss < 1e-6:
            break

    return best_obj

# Load and prepare data
data = data_read()
probe = probe_read()
background = background_read()

# Initialize object with phase-only random initialization
key = jran.PRNGKey(0)
_, key_phase = jran.split(key)
initial_phase = jran.uniform(key_phase, shape=data.shape, minval=0, maxval=2*jnp.pi)
initial_obj_guess = jnp.exp(1j * initial_phase)

# Run optimization
update_obj = adam_optimization(initial_obj_guess, data, background, probe)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(jnp.angle(update_obj), cmap='hsv')
plt.title("Reconstructed Phase")
plt.colorbar()

plt.subplot(122)
plt.imshow(jnp.abs(update_obj), cmap='gray', vmin=0.9, vmax=1.1)
plt.title("Reconstructed Amplitude")
plt.colorbar()
plt.show()
