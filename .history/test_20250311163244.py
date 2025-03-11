import jax.numpy as jnp
from jax import grad, jit
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt

def probe_read(filepath='reconstructions/e17965_1_00678_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        probe = jnp.array(f['probe'][:])  # Shape: (2, m, n)
        return probe

def data_read(filepath='cxi_files/e17965_1_00677.cxi'):
    with h5py.File(filepath, 'r') as f:
        data = jnp.sum(f['entry_1/data_1/data'][:], axis=0)
    return data

def background_read(filepath='reconstructions/e17965_1_00677_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        return jnp.array(f['background'][:])

def pad_array(array, target_shape):
    pad_m = (target_shape[0] - array.shape[0]) // 2
    pad_n = (target_shape[1] - array.shape[1]) // 2
    return jnp.pad(array, ((pad_m, target_shape[0] - array.shape[0] - pad_m), 
                          (pad_n, target_shape[1] - array.shape[1] - pad_n)), mode='constant')

def forward_model(obj_low, probe):
    m, n = probe.shape[1], probe.shape[2]
    obj_fre_low = jnp.fft.fftshift(jnp.fft.fft2(obj_low))
    obj_fre_padded = pad_array(obj_fre_low, (m, n))
    obj_high = jnp.fft.ifft2(jnp.fft.ifftshift(obj_fre_padded))
    # Sum contributions from all probe modes (no fixed weights)
    diff_pattern = jnp.zeros((m, n), dtype=jnp.float32)
    for mode in range(probe.shape[0]):
        exit_wave = probe[mode] * obj_high
        diff_pattern += jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave)))**2
    return diff_pattern

def loss_function(simulated, background, measured):
    factor = 1.0 / (jnp.sum(measured) + 1e-10)
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(measured)
    return factor * jnp.sum((simulated_amp - measured_amp)**2)

@jit
def update_step(obj_low, opt_state, optimizer, probe, measured, background):
    def loss_fn(o):
        simulated = forward_model(o, probe)
        return loss_function(simulated, background, measured)
    grad_obj = grad(loss_fn)(obj_low)
    updates, new_opt_state = optimizer.update(grad_obj, opt_state, obj_low)
    new_obj_low = optax.apply_updates(obj_low, updates)
    return new_obj_low, new_opt_state

def adam_optimization(init_obj_low, measured, background, probe, num_iterations=1000):
    # Initialize optimizer with parameters from the first paper
    optimizer = optax.adam(learning_rate=0.4, b1=0.9, b2=0.999)
    opt_state = optimizer.init(init_obj_low)
    obj_low = init_obj_low
    prev_loss = jnp.inf
    for epoch in range(num_iterations):
        obj_low, opt_state = update_step(obj_low, opt_state, optimizer, probe, measured, background)
        simulated = forward_model(obj_low, probe)
        loss = loss_function(simulated, background, measured)
        # Reduce learning rate if loss plateaus (simplified)
        if epoch % 10 == 0 and loss >= prev_loss:
            optimizer = optax.adam(learning_rate=0.4 * 0.1 ** (epoch // 10), b1=0.9, b2=0.999)
            opt_state = optimizer.init(obj_low)
        prev_loss = loss
        if loss < 1e-9:
            break
    return obj_low

# Example usage
probe = probe_read()
data = data_read()
background = background_read()

# Initialize low-resolution object with random phases (band-limited)
key = jran.PRNGKey(42)
obj_low_real = jran.normal(key, (64, 64))  # Example low-res shape
obj_low_imag = jran.normal(key, (64, 64)) * 0.1  # Small initial imaginary part
initial_obj_low = obj_low_real + 1j * obj_low_imag

# Run optimization
reconstructed_obj = adam_optimization(initial_obj_low, data, background, probe)

# Visualize result
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('Amplitude')
plt.imshow(jnp.abs(reconstructed_obj))
plt.subplot(122)
plt.title('Phase')
plt.imshow(jnp.angle(reconstructed_obj))
plt.show()