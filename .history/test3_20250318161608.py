import jax.numpy as jnp
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt
import jax

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
        data = jnp.mean(data,axis=0)
    return data

def background_read(filepath = r'Papercode\reconstructions\e17965_1_00677_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        dataset = f['background']
        background = jnp.array(dataset[:])  # 读取整个数据集
        return background

def complex_parameterization(params: jnp.ndarray) -> jnp.ndarray:
    """Convert dual-channel parameters [log_amp, phase] into complex fields (simulating CDTools' complex processing logic)[^3]"""
    log_amp, phase = params[...,0], params[...,1]
    return jnp.exp(log_amp) * jnp.exp(1j * phase)  # Amplitude > 0

def scaled_fft(field: jnp.ndarray) -> jnp.ndarray:
    """Forward far-field propagation simulation (replacing CDTools' propagators.far_field) [^6]"""
    # Convert shape tuple to JAX array before taking product
    scaling_factor = jnp.array(field.shape[-2:])
    scaling = 1.0 / jnp.sqrt(jnp.prod(scaling_factor))
    return scaling * jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(field)))

def forward_model(params: jnp.ndarray, probe: jnp.ndarray) -> jnp.ndarray:
    """Complex-to-real conversion and diffraction intensity calculation (consistent with neutron wave diffraction physics logic in the tutorial)[^3]"""
    obj = complex_parameterization(params)
    exit_wave = probe[0] * obj  # Assume probe is multimodal [^2]
    diff_field = scaled_fft(exit_wave)
    return jnp.abs(diff_field)**2  # Light intensity calculation

def loss_function(
    params: jnp.ndarray, 
    probe: jnp.ndarray, 
    measurements: jnp.ndarray,
    beta: float = 0.01  # Phase smoothness regularization strength[^6]
) -> float:
    """Combine intensity error and phase smoothness constraints (similar to constraints in the tutorial)[^1][^3]"""
    # Calculate diffraction intensity residual
    predicted = forward_model(params, probe)
    intensity_error = jnp.mean((jnp.sqrt(predicted) - jnp.sqrt(measurements))**2)
    
    # Phase smoothness constraint
    phase = params[...,1]
    phase_x_grad = jnp.diff(phase, axis=-1)
    phase_y_grad = jnp.diff(phase, axis=-2)
    
    # Pad gradients to original shape [H, W]
    phase_x_grad_padded = jnp.pad(phase_x_grad, ((0, 0), (0, 1)), mode='constant')  # [624, 624]
    phase_y_grad_padded = jnp.pad(phase_y_grad, ((0, 1), (0, 0)), mode='constant')  # [624, 624]
    
    # Compute smoothness loss
    smoothness_loss = jnp.mean(phase_x_grad_padded**2 + phase_y_grad_padded**2)
    return intensity_error + beta * smoothness_loss

'''
def adam_optimization(
    initial_params: jnp.ndarray,  # Shape [H,W,2]
    measurements: jnp.ndarray,
    probe: jnp.ndarray, 
    lr: float = 0.005, 
    epochs: int = 100,
    batch_size: int = 50,
) -> jnp.ndarray:
    
    # Adam optimizer configuration (standard β/ε)
    optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)  
    opt_state = optimizer.init(initial_params)
    
    # Batch generation
    key = jax.random.PRNGKey(42)
    num_samples = measurements.shape[0]
    
    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num_samples)
        
        # Update in batches
        for batch_idx in jnp.array_split(perm, num_samples//batch_size):
            batch_meas = measurements[batch_idx]
            
            # Automatic differentiation to compute gradients
            grad_fn = jax.grad(lambda p: loss_function(p, probe, batch_meas))
            grads = grad_fn(initial_params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            initial_params = optax.apply_updates(initial_params, updates)
        
        # Log output (simulating model.report())
        if epoch % 10 == 0:
            epoch_loss = loss_function(initial_params, probe, measurements)
            print(f"Epoch {epoch}: Loss={epoch_loss:.3e}")
        
        # Learning rate stage-wise decay (example: reduce to 0.2x after 50 epochs[^2][^4])
        if epoch in [50, 80]:  
            lr *= 0.2
            optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
            print(f"Learning rate reduced to {lr}")
    
    return complex_parameterization(initial_params)  # Return complex object
'''

def adam_optimization(
    initial_params: jnp.ndarray,  # Shape [H, W, 2]
    measurements: jnp.ndarray,    # Shape [H, W]
    probe: jnp.ndarray,
    lr: float = 0.005,
    epochs: int = 100,
) -> jnp.ndarray:
    """
    Optimize parameters using the full measurement without batching.
    """
    optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = optimizer.init(initial_params)

    for epoch in range(epochs):
        # Compute gradients using the full measurements
        grad_fn = jax.grad(lambda p: loss_function(p, probe, measurements))
        grads = grad_fn(initial_params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        initial_params = optax.apply_updates(initial_params, updates)

        # Log progress
        if epoch % 10 == 0:
            epoch_loss = loss_function(initial_params, probe, measurements)
            print(f"Epoch {epoch}: Loss={epoch_loss:.3e}")

        # Learning rate decay
        if epoch in [50, 80]:
            lr *= 0.2
            optimizer = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
            print(f"Learning rate reduced to {lr}")

    return initial_params

data = data_read()
probe = probe_read()
background = background_read()

key_real = jran.PRNGKey(0)
key_imag = jran.PRNGKey(1)
real_part = jran.normal(key_real, shape=jnp.shape(data))
imag_part = jran.normal(key_imag, shape=jnp.shape(data)) 
initial_obj_guess = jnp.stack([real_part, imag_part], axis=2)
obj = initial_obj_guess

update_obj = adam_optimization(obj, data, probe, 0.5, 500)

update_obj_1 = adam_optimization(update_obj, data, probe, 0.05, 750)

update_obj_2 = adam_optimization(update_obj_1, data, probe, 0.005, 1000)

plt.imshow(jnp.angle(complex_parameterization(update_obj_2)))
plt.show()