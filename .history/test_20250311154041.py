import jax.numpy as jnp
from jax import grad
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt

# File reading functions
def probe_read(filepath=r'Papercode\reconstructions\e17965_1_00678_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        probe = jnp.array(f['probe'][:])  # Shape: (2, m, n) for two modes
    return probe

def data_read(filepath=r'Papercode\cxi_files\e17965_1_00677.cxi'):
    with h5py.File(filepath, 'r') as f:
        data = jnp.sum(f['entry_1/data_1/data'][:], axis=0)  # Shape: (m, n)
    return data

def background_read(filepath=r'Papercode\reconstructions\e17965_1_00677_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        bcg = jnp.array(f['background'][:])  # Shape: (m, n)
    return bcg

# Utility function for padding in Fourier space
def pad_array(array, target_shape):
    pad_m = (target_shape[0] - array.shape[0]) // 2
    pad_n = (target_shape[1] - array.shape[1]) // 2
    return jnp.pad(array, ((pad_m, target_shape[0] - array.shape[0] - pad_m),
                           (pad_n, target_shape[1] - array.shape[1] - pad_n)), mode='constant')

# Upsampling function
def upsample(low_res, target_shape):
    fre_low = jnp.fft.fftshift(jnp.fft.fft2(low_res))
    fre_padded = pad_array(fre_low, target_shape)
    high_res = jnp.fft.ifft2(jnp.fft.ifftshift(fre_padded))
    return high_res

# Forward model with optional phase-only constraint
def forward_model(input_low, probe, target_shape, phase_only=True):
    if phase_only:
        # Input is real-valued phase T, object is exp(i T)
        high_res = upsample(input_low, target_shape)
        obj_high = jnp.exp(1j * high_res)
    else:
        # Input is complex object
        obj_high = upsample(input_low, target_shape)
    
    diff_pattern0 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(probe[0] * obj_high)))**2
    diff_pattern1 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(probe[1] * obj_high)))**2
    simulated = diff_pattern0 * 0.756 + diff_pattern1 * 0.244  # Weights from Optics Letters
    return simulated

# Loss function
def loss_function(simulated, background, measured):
    factor = 1.0 / (jnp.sum(measured) + 1e-10)
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(measured)
    return factor * jnp.sum((simulated_amp - measured_amp) ** 2)

# Derivative computation
def derivative_loss(input_low, probe, measured, background, target_shape, phase_only):
    def loss_fn(input):
        simulated = forward_model(input, probe, target_shape, phase_only)
        return loss_function(simulated, background, measured)
    return grad(loss_fn)(input_low)

# Adam optimization
def adam_optimization(init_input, measured, background, probe, target_shape, alpha, num_iterations, phase_only):
    optimizer = optax.adam(alpha)
    opt_state = optimizer.init(init_input)
    input_low = init_input
    for _ in range(num_iterations):
        grad_input = derivative_loss(input_low, probe, measured, background, target_shape, phase_only)
        updates, opt_state = optimizer.update(grad_input, opt_state, input_low)
        input_low = optax.apply_updates(input_low, updates)
        simulated = forward_model(input_low, probe, target_shape, phase_only)
        loss = loss_function(simulated, background, measured)
        if loss < 1e-9:
            break
    return input_low

# Main execution
data = data_read()
probe = probe_read()
background = background_read()

m, n = probe.shape[1], probe.shape[2]  # Probe dimensions, e.g., 1024x1024
R = 0.5  # Resolution ratio, adjustable (e.g., 0.6 from Optics Express, 400/1024 ~ 0.39 from Optics Letters)
m_o, n_o = int(R * m), int(R * n)  # Low-resolution object shape, e.g., 512x512
target_shape = (m, n)

# Initialization
key = jran.PRNGKey(0)
phase_only = True  # Set to False for complex object (Optics Express style)
if phase_only:
    # Real-valued phase T for phase-only object (Optics Letters)
    init_input = jran.normal(key, shape=(m_o, n_o))
else:
    # Complex object (Optics Express)
    key_real, key_imag = jran.split(key)
    real_part = jran.normal(key_real, shape=(m_o, n_o))
    imag_part = jran.normal(key_imag, shape=(m_o, n_o))
    init_input = real_part + 1j * imag_part

# Optimization
updated_input = adam_optimization(init_input, data, background, probe, target_shape, alpha=0.1, num_iterations=200, phase_only=phase_only)

# Compute high-resolution object for visualization
obj_high = forward_model(updated_input, probe, target_shape, phase_only)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(jnp.abs(obj_high), cmap='gray')
plt.title("Magnitude")
plt.subplot(1, 2, 2)
plt.imshow(jnp.angle(obj_high), cmap='hsv')
plt.title("Phase")
plt.show()