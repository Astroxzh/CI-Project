import jax.numpy as jnp
from jax import grad
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt

def probe_read(filepath=r'Papercode\reconstructions\e17965_1_00678_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        # print("Keys in the file:", list(f.keys()))
        dataset = f['probe']
        probe = jnp.array(dataset[:])  # 读取整个数据集
        # there are two probe modes
        return probe
    
def data_read(filepath=r'Papercode\cxi_files\e17965_1_00677.cxi'):
    with h5py.File(filepath, 'r') as f:
        data = f['entry_1/data_1/data'][:]
        data = jnp.sum(data,axis=0)
    return data

def background_read(filepath = r'Papercode\reconstructions\e17965_1_00677_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        dataset = f['background']
        background = jnp.array(dataset[:])  # 读取整个数据集
        return background

def pad_array(array, target_shape):
    pad_m = (target_shape[0] - array.shape[0]) // 2
    pad_n = (target_shape[1] - array.shape[1]) // 2
    return jnp.pad(array, ((pad_m, target_shape[0] - array.shape[0] - pad_m), (pad_n, target_shape[1] - array.shape[1] - pad_n)), mode='constant')

def forward_model(obj_low, probe, target_shape):
    """Simulate diffraction pattern from the object guess."""
    # Placeholder: Upsample obj_low to target_shape, multiply by probe, compute FFT
    # For multiple modes, sum intensities with weights
    num_modes = probe.shape[0]
    upsampled = upsample(obj_low, target_shape)  # Define upsample separately
    simulated = 0
    weights = [0.756, 0.244]  # Adjust based on experiment
    for i in range(num_modes):
        exit_wave = upsampled * probe[i]
        diffraction = jnp.abs(jnp.fft.fft2(exit_wave))**2
        simulated += weights[i] * diffraction
    return simulated

def loss_function(simulated, background, measured):
    """Compute normalized MSE between amplitude patterns."""
    return jnp.mean((jnp.sqrt(simulated + background) - jnp.sqrt(measured))**2) / (jnp.sum(measured) + 1e-10)

def derivative_loss(obj_low, probe, measured, background, target_shape):
    """Compute gradient of loss with respect to object."""
    return grad(loss_function)(forward_model(obj_low, probe, target_shape), background, measured)

def adam_optimization(init_obj, measured, background, probe, target_shape, alpha=0.01, num_iterations=1000):
    """Optimize the object guess using Adam with adjusted parameters."""
    optimizer = optax.adam(learning_rate=alpha)
    opt_state = optimizer.init(init_obj)
    obj_low = init_obj
    
    for iteration in range(num_iterations):
        grad_obj = derivative_loss(obj_low, probe, measured, background, target_shape)
        updates, opt_state = optimizer.update(grad_obj, opt_state, obj_low)
        obj_low = optax.apply_updates(obj_low, updates)
        
        simulated = forward_model(obj_low, probe, target_shape)
        loss = loss_function(simulated, background, measured)
        print(f"Iteration {iteration + 1}, Loss: {loss:.6f}")
        
        if loss < 1e-9:
            print("Converged below threshold.")
            break
            
    return obj_low

# Example usage

data = data_read()
probe = probe_read()
background = background_read()

#initilize
# initial_data = data * np.exp(1j * np.random.uniform(0, 2*np.pi, size=data.shape))
key = jran.PRNGKey(0)
key_real, key_imag = jran.split(key)
real_part = jran.normal(key_real, shape=jnp.shape(data))
imag_part = jran.normal(key_imag, shape=jnp.shape(data))
initial_obj_guess = real_part + 1j * imag_part
obj = jnp.copy(initial_obj_guess)

update_obj = adam_optimization(obj, data, background, probe, 0.1, 1000)

plt.imshow(jnp.abs(update_obj))
plt.show()