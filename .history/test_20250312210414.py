import jax.numpy as jnp
from jax import grad, jit
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
        data = jnp.mean(data,axis=0)
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

@jit
def forward_model(obj_low, probe):
    m, n = probe.shape[1], probe.shape[2]
    m_d, n_d = obj_low.shape
    obj_fre_low = jnp.fft.fftshift(jnp.fft.fft2(obj_low))
    obj_fre_padded = pad_array(obj_fre_low, (m, n))
    obj_high = jnp.fft.ifft2(jnp.fft.ifftshift(obj_fre_padded))
    diff_pattern0 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(probe[0, :, :] * obj_high)))**2
    diff_pattern1 = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(probe[1, :, :] * obj_high)))**2
    simulated = diff_pattern0 * 0.756 + diff_pattern1 * 0.244
    return simulated

@jit
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

def adam_optimization(init_obj_low: jnp.ndarray, measured: jnp.ndarray, background: jnp.ndarray, probe: jnp.ndarray, alpha: float, num_iterations: int) -> jnp.ndarray:
    
    optimizer = optax.adam(alpha)
    
    opt_state = optimizer.init(init_obj_low)
    obj_low = init_obj_low
    
    best_loss = float('inf')
    plateau_counter = 0
    current_alpha = alpha

    for _ in range(num_iterations):
        grad_obj = derivative_loss_function_wrt_obj(obj_low, probe, measured, background)
        
        updates, opt_state = optimizer.update(grad_obj, opt_state, obj_low)
        obj_low = optax.apply_updates(obj_low, updates)
        
        simulated = forward_model(obj_low, probe)
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
            opt_state = optimizer.init(obj_low)
            plateau_counter = 0

        if _ % 50 == 0:
            print(f"Iteration: {_}, Loss: {loss}")
        
        if loss < 1e-9 or alpha < 1e-4:
            print("Converged below threshold.")
            break
    return obj_low

# Example usage

data = data_read()
probe = probe_read()
background = background_read()

#initilize
# initial_data = data * np.exp(1j * np.random.uniform(0, 2*np.pi, size=data.shape))
key_real = jran.PRNGKey(0)
key_imag = jran.PRNGKey(1)
real_part = jran.normal(key_real, shape=jnp.shape(data))
#imag_part = jran.normal(key_imag, shape=jnp.shape(data)) 
initial_obj_guess = real_part #+ 1j * imag_part
obj = jnp.copy(initial_obj_guess)

update_obj = adam_optimization(obj, data, background, probe, 0.4, 200)

update_obj_1 = adam_optimization(update_obj, data, background, probe, 0.4, 500)

update_obj_2 = adam_optimization(update_obj_1, data, background, probe, 0.4, 1000)

plt.imshow(jnp.angle(update_obj_2))
plt.show()
