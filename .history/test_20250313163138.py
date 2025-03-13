import jax.numpy as jnp
from jax import grad, jit
import optax
import h5py
import jax.random as jran
import matplotlib.pyplot as plt
from chex import assert_equal_shape

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

def pad_array(array, pad_array):
    m_pad = pad_array.shape[0]  # Python int
    n_pad = pad_array.shape[1]  # Python int
    m_array = array.shape[0]    # Python int
    n_array = array.shape[1]    # Python int
    pad_factortb = abs((m_pad - m_array) // 2)  # Python int
    pad_factorrl = abs((n_pad - n_array) // 2)  # Python int
    paded_array = jnp.pad(array, ((pad_factortb, pad_factortb), (pad_factorrl, pad_factorrl)))
    return paded_array

def down_sampling_fre(obj, f=2):
    '''
        down sampling the obj with factor f in Fourier space
    '''
    m, n = obj.shape
    obj_fre = jnp.fft.fftshift(jnp.fft.fft2(obj))
    m_d, n_d = m // f, n // f  # downsampling size
    m_d = int(m_d)
    n_d = int(n_d)

    m_center, n_center = m // 2, n // 2  # original center
    m_d_half, n_d_half = n_d // 2, m_d // 2

    #index of cropped area
    m_start = m_center - m_d_half
    m_end = m_start + m_d
    n_start = n_center - n_d_half
    n_end = n_start + n_d

    obj_fre_cropped = obj_fre[m_start:m_end, n_start:n_end]

    obj_downSampled = jnp.fft.ifft2(jnp.fft.ifftshift(obj_fre_cropped))
    return obj_downSampled

def down_sampling_spa(obj,f=2):
    '''
        down sampling the obj with factor f in spatial domain
        block averaging is employed, average amplitude and phase
    '''
    m, n = obj.shape

    # crop obj to ensure integer times with the factor
    m_crop = (m // f) * f
    n_crop = (n // f) * f
    obj_cropped = obj[:m_crop, :n_crop]

    reshaped = obj_cropped.reshape(m_crop // f, f, n_crop // f, f)
    obj_downSampled = reshaped.mean(axis=(1, 3))

    return obj_downSampled


def forward_model(obj, probe, f=1.5):
    '''
        the forward model generates the update diffraction field from the down sampled obj
        in this simulation assumes only one probe 
    '''
    obj_downSampled = down_sampling_fre(obj, f)
    obj_fre = jnp.fft.fftshift(jnp.fft.fft2(obj_downSampled))
    obj_frepad = pad_array(obj_fre, probe[0,:,:])
    obj_pad = jnp.fft.ifft2(jnp.fft.ifftshift(obj_frepad))
    update_diff_pattern0 = (jnp.abs(
        jnp.fft.fftshift(jnp.fft.fft2((probe[0, :, :]) * obj_pad))))**2 # 
    update_diff_pattern1 = (jnp.abs(
        jnp.fft.fftshift(jnp.fft.fft2((probe[1, :, :]) * obj_pad))))**2 # 
    update_diff_pattern = update_diff_pattern0 * 0.756 + update_diff_pattern1 * 0.244

    # return (jnp.abs(update_diff_pattern))**2
    return update_diff_pattern


def loss_function(simulated: jnp.ndarray, background: jnp.ndarray, measured: jnp.ndarray) -> float:
    
    assert_equal_shape([simulated, background, measured])
    
    factor = 1.0 / (jnp.sum(measured) + 1e-10)
    simulated_amp = jnp.sqrt(simulated + background)
    measured_amp = jnp.sqrt(measured)
    return factor * jnp.sum((simulated_amp - measured_amp) ** 2)

def derivative_loss_function_wrt_obj(obj_low: jnp.ndarray, probe: jnp.ndarray, measured: jnp.ndarray, background: jnp.ndarray) -> jnp.ndarray:
    def loss_wrt_obj(o):
        simulated = forward_model(o, probe)
        return loss_function(simulated, background, measured)
    return grad(loss_wrt_obj)(obj_low)

def adam_optimization(init_obj: jnp.ndarray, measured: jnp.ndarray, background: jnp.ndarray, probe: jnp.ndarray, alpha: float, num_iterations: int) -> jnp.ndarray:
    
    optimizer = optax.adam(alpha)
    
    opt_state = optimizer.init(init_obj)
    obj = init_obj
    
    best_loss = float('inf')
    plateau_counter = 0
    current_alpha = alpha

    for _ in range(num_iterations):
        grad_obj = derivative_loss_function_wrt_obj(obj, probe, measured, background)
        
        updates, opt_state = optimizer.update(grad_obj, opt_state, obj)
        obj = optax.apply_updates(obj, updates)
        
        simulated = forward_model(obj, probe)
        loss = loss_function(simulated, background, measured)
        
        if loss < best_loss:
            best_loss = loss
            plateau_counter = 0
        else:
            plateau_counter += 1

        if plateau_counter >= 10:
            current_alpha *= 0.1
            print(f"Alpha reduced to {current_alpha}")
            optimizer = optax.adam(current_alpha)
            opt_state = optimizer.init(obj)
            plateau_counter = 0

        if _ % 50 == 0:
            print(f"Iteration: {_}, Loss: {loss}")
        
        if loss < 1e-9 or current_alpha < 1e-9:
            print("Converged below threshold.")
            break
    return obj

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

update_obj = adam_optimization(obj, data, background, probe, 0.4, 500)

update_obj_1 = adam_optimization(update_obj, data, background, probe, 0.4, 750)

update_obj_2 = adam_optimization(update_obj_1, data, background, probe, 0.4, 1000)

m, n = probe.shape[1], probe.shape[2]
T_fre_low = jnp.fft.fftshift(jnp.fft.fft2(update_obj_2))
T_fre_padded = pad_array(T_fre_low, (m, n))
T_high = jnp.fft.ifft2(jnp.fft.ifftshift(T_fre_padded))
obj_high = jnp.exp(1j * T_high)

plt.imshow(jnp.angle(obj_high))
plt.show()
