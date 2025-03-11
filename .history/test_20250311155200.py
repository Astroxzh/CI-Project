import jax.numpy as jnp
from jax import grad
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt

# File reading functions
def probe_read(filepath=r'Papercode\reconstructions\e17965_1_00678_ptycho_reconstruction.h5'):
    """Read the probe array from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        probe = jnp.array(f['probe'][:])  # Shape: (num_modes, m, n), e.g., (2, 1024, 1024)
    return probe

def data_read(filepath=r'Papercode\cxi_files\e17965_1_00677.cxi'):
    """Read the measured diffraction pattern from a CXI file."""
    with h5py.File(filepath, 'r') as f:
        data = jnp.sum(f['entry_1/data_1/data'][:], axis=0)  # Shape: (m, n), e.g., (1024, 1024)
    return data

def background_read(filepath=r'Papercode\reconstructions\e17965_1_00677_ptycho_reconstruction.h5'):
    """Read the background pattern from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        bcg = jnp.array(f['background'][:])  # Shape: (m, n), e.g., (1024, 1024)
    return bcg

# Utility function for Fourier padding
def pad_array(array, target_shape):
    """Pad an array in Fourier space to match target shape."""
    pad_m = (target_shape[0] - array.shape[0]) // 2
    pad_n = (target_shape[1] - array.shape[1]) // 2
    return jnp.pad(array, ((pad_m, target_shape[0] - array.shape[0] - pad_m),
                           (pad_n, target_shape[1] - array.shape[1] - pad_n)), mode='constant')

# Upsampling function
def upsample(low_res, target_shape):
    """Upsample a low-resolution object to target shape via Fourier padding."""
    fre_low = jnp.fft.fftshift(jnp.fft.fft2(low_res))
    fre_padded = pad_array(fre_low, target_shape)
    high_res = jnp.fft.ifft2(jnp.fft.ifftshift(fre_padded))
    return high_res

# Forward model
def forward_model(obj_low, probe, target_shape):
    """Simulate the diffraction pattern from the object guess."""
    # Upsample the low-resolution object to probe resolution
    obj_high = upsample(obj_low, target_shape)
    
    # Compute diffraction patterns for each probe mode and sum intensities
    simulated = 0
    num_modes = probe.shape[0]
    weights = [0.756, 0.244] if num_modes == 2 else [1.0]  # Example weights for two modes
    for i in range(num_modes):
        exit_wave = probe[i] * obj_high
        diff_pattern = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave)))**2
        simulated += weights[i] * diff_pattern if num_modes > 1 else diff_pattern
    return simulated

# Loss function
def loss_function(simulated, background, measured):
    """Compute normalized mean squared error between simulated and measured patterns."""
    factor = 1.0 / (jnp.sum(measured) + 1e-10)  # Normalization factor
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(measured)
    return factor * jnp.sum((simulated_amp - measured_amp) ** 2)

# Gradient computation
def derivative_loss(obj_low, probe, measured, background, target_shape):
    """Compute the gradient of the loss with respect to the object."""
    def loss_fn(obj):
        simulated = forward_model(obj, probe, target_shape)
        return loss_function(simulated, background, measured)
    return grad(loss_fn)(obj_low)

# Adam optimization
def adam_optimization(init_obj, measured, background, probe, target_shape, alpha=0.1, num_iterations=200):
    """Optimize the object guess using the Adam algorithm."""
    optimizer = optax.adam(alpha)
    opt_state = optimizer.init(init_obj)
    obj_low = init_obj
    for iteration in range(num_iterations):
        grad_obj = derivative_loss(obj_low, probe, measured, background, target_shape)
        updates, opt_state = optimizer.update(grad_obj, opt_state, obj_low)
        obj_low = optax.apply_updates(obj_low, updates)
        simulated = forward_model(obj_low, probe, target_shape)
        loss = loss_function(simulated, background, measured)
        print(f"Iteration {iteration + 1}, Loss: {loss:.6f}")
        if loss < 1e-9:  # Convergence criterion
            break
    return obj_low

# Main execution
def main():
    # Load data
    measured = data_read()
    probe = probe_read()
    background = background_read()

    # Define shapes
    m, n = probe.shape[1], probe.shape[2]  # Detector/probe resolution, e.g., 1024x1024
    R = 0.5  # Resolution ratio for initial guess, e.g., 0.5 gives 512x512
    m_o, n_o = int(R * m), int(R * n)  # Low-resolution object shape
    target_shape = (m, n)

    # Initialize object with Gaussian distribution
    key = jran.PRNGKey(0)
    key_real, key_imag = jran.split(key)
    real_part = jran.normal(key_real, shape=(m_o, n_o))
    imag_part = jran.normal(key_imag, shape=(m_o, n_o))
    init_obj = real_part + 1j * imag_part  # Complex Gaussian initial guess

    # Optimize
    reconstructed_obj_low = adam_optimization(
        init_obj, measured, background, probe, target_shape, alpha=0.1, num_iterations=200
    )

    # Upsample for visualization
    reconstructed_obj_high = upsample(reconstructed_obj_low, target_shape)

    # Visualize the reconstructed object
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(jnp.abs(reconstructed_obj_high), cmap='gray')
    plt.title("Magnitude of Reconstructed Object")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(jnp.angle(reconstructed_obj_high), cmap='hsv')
    plt.title("Phase of Reconstructed Object")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()