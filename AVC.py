# Dynamic Vekua Cascade
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from functools import partial

# ==========================================
# CONFIGURATION
# ==========================================
jax.config.update("jax_enable_x64", True)

# ==========================================
# 0. UTILITIES
# ==========================================

def count_params(params):
    """Counts total trainable scalars in a Pytree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# ==========================================
# 1. MODELS
# ==========================================

class Siren:
    """Baseline A: SIREN."""
    def __init__(self, key, in_dim, layers=[64, 64, 64, 1], w0=30.0):
        self.params = []
        self.w0 = w0
        full_layers = [in_dim] + layers
        keys = jax.random.split(key, len(full_layers))
        
        for i, (n_in, n_out) in enumerate(zip(full_layers[:-1], full_layers[1:])):
            w_key, b_key = jax.random.split(keys[i])
            limit = jnp.sqrt(6 / n_in) / (w0 if i == 0 else 1.0)
            W = jax.random.uniform(w_key, (n_in, n_out), minval=-limit, maxval=limit)
            b = jnp.zeros((n_out,))
            self.params.append({'W': W, 'b': b})
            
    def forward(self, params, x):
        h = x
        for i, p in enumerate(params[:-1]):
            h = jnp.sin(self.w0 * (h @ p['W'] + p['b']))
        return h @ params[-1]['W'] + params[-1]['b']

class GridMLP:
    """Baseline B: N-Dimensional Hash/Grid Encoding + MLP.
    Now supports 1D, 2D, and 3D grids dynamically.
    """
    def __init__(self, key, in_dim, grid_res=64, feat_dim=16):
        k1, k2 = jax.random.split(key)
        
        # --- DYNAMIC GRID SHAPE ---
        # If in_dim=2, shape is (64, 64, 16)
        # If in_dim=3, shape is (64, 64, 64, 16) -> Massive!
        grid_shape = (grid_res,) * in_dim + (feat_dim,)
        
        # Initialize small to avoid exploding gradients
        self.grid = jax.random.uniform(k1, grid_shape) * 0.01
        
        # Decoder parameters
        self.decoder_params = []
        layers = [feat_dim, 64, 64, 1]
        keys = jax.random.split(k2, len(layers))
        for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:])):
            k_w, _ = jax.random.split(keys[i])
            W = jax.random.normal(k_w, (n_in, n_out)) * jnp.sqrt(2/n_in)
            b = jnp.zeros((n_out,))
            self.decoder_params.append({'W': W, 'b': b})

    def forward(self, params, x):
        # x is (N, D). Map [-1, 1] to [0, grid_res-1]
        grid_shape = params['grid'].shape
        D = x.shape[1]
        coords = (x + 1) * 0.5 * (grid_shape[0] - 1)
        
        # --- N-DIMENSIONAL INTERPOLATION ---
        # We need to construct the coordinate list for map_coordinates.
        # For each dimension d, we extract coords[:, d] and reshape to (N, 1).
        # The last dimension is the feature channel, reshaped to (1, C).
        
        spatial_coords = [coords[:, d][:, None] for d in range(D)]
        channel_coords = [jnp.arange(grid_shape[-1])[None, :]]
        
        # Combine: [x_coords, y_coords, ..., z_coords, channel_indices]
        query_coords = spatial_coords + channel_coords
        
        feats = jax.scipy.ndimage.map_coordinates(
            params['grid'], 
            query_coords, 
            order=1, mode='nearest'
        )
        
        h = feats
        for i, p in enumerate(params['mlp'][:-1]):
            h = jax.nn.relu(h @ p['W'] + p['b'])
        return h @ params['mlp'][-1]['W'] + params['mlp'][-1]['b']

class VekuaCascade:
    """Ours: Adaptive Vekua Cascade with Hybrid Initialization."""
    def __init__(self, key):
        self.key = key
        self.blocks = []
        self.scalers = []
        
    def create_block(self, key, in_dim, freq_scale, is_first=False):
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Hybrid Init: Identity-like for first block
        warp_scale = 1e-5 if is_first else 0.1
        
        # Warping: Projects N-Dim input -> 2D Manifold
        W = jax.random.normal(k1, (in_dim, 32)) * warp_scale
        b = jnp.zeros((32,))
        W_out = jax.random.normal(k2, (32, 2)) * warp_scale
        
        # Analytic Basis
        r = jax.random.uniform(k3, (24,), minval=freq_scale/2, maxval=freq_scale*1.5)
        theta = jax.random.uniform(k3, (24,), minval=0, maxval=2*jnp.pi)
        freqs = r * jnp.exp(1j * theta)
        
        return {'W': W, 'b': b, 'W_out': W_out, 'freqs': freqs}
    
    def get_basis(self, params, x):
        # 1. Deep Coordinate Warp
        h = jnp.sin(x @ params['W'] + params['b'])
        uv = h @ params['W_out']
        
        # 2. Form Complex Variable z
        # If input has at least 2 dims, use them as residual base
        if x.shape[1] >= 2:
            z = (x[:,0]+uv[:,0]) + 1j * (x[:,1]+uv[:,1])
        else:
            z = uv[:,0] + 1j * uv[:,1]
        
        # 3. Analytic Expansion
        z_f = z[:, None] * jnp.conj(params['freqs'])[None, :]
        bs, bc = jnp.sin(z_f.real), jnp.cos(z_f.real)
        mag = jnp.abs(z)[:, None]
        return jnp.concatenate([bs, bc, bs*mag, bc*mag], axis=-1)

    def solve(self, phi, y, reg=1e-5):
        cov = phi.T @ phi + reg * jnp.eye(phi.shape[1])
        rhs = phi.T @ y
        return jax.scipy.linalg.solve(cov, rhs, assume_a='pos')

# ==========================================
# 2. DATA GENERATORS
# ==========================================

def gen_exp_A_helmholtz(N_train, N_test, seed):
    np.random.seed(seed)
    def wave_fn(x, y):
        return (np.sin(20*x)*np.cos(20*y) + 0.5*np.sin(35*x + 35*y))
    ls = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(ls, ls)
    Z_truth = wave_fn(X, Y)
    x_tr = np.random.uniform(-1, 1, (N_train, 2))
    y_tr = wave_fn(x_tr[:,0], x_tr[:,1])[:, None] + np.random.normal(0, 0.1, (N_train, 1))
    return x_tr, y_tr, (X, Y, Z_truth)

def gen_exp_B_sparse(N_train, N_test, seed):
    def phantom(x, y):
        z = 1.0 * ((x**2 + (y/1.5)**2) < 0.8)
        z -= 0.8 * (((x-0.2)**2 + (y-0.2)**2) < 0.05)
        z += 0.5 * np.exp(-50*((x)**2 + (y+0.4)**2))
        return z
    ls = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(ls, ls)
    Z_truth = phantom(X, Y)
    np.random.seed(seed)
    mask_idx = np.random.choice(len(X.flatten()), size=int(len(X.flatten())*0.02), replace=False)
    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    vals = Z_truth.flatten()[:, None]
    y_tr = vals[mask_idx] + np.random.normal(0, 0.05, (len(mask_idx), 1))
    return coords[mask_idx], y_tr, (X, Y, Z_truth)

def gen_exp_C_inverse(N_train, N_test, seed):
    x_test = np.linspace(-1, 1, N_test)
    u_true_test = np.exp(-x_test**2) * np.sin(15*x_test)
    d2 = (4*x_test**2 - 2)*np.exp(-x_test**2)*np.sin(15*x_test) - 60*x_test*np.exp(-x_test**2)*np.cos(15*x_test) - 225*np.exp(-x_test**2)*np.sin(15*x_test)
    k_sq_true = -d2 / (u_true_test + 1e-9)
    np.random.seed(seed)
    x_tr = np.sort(np.random.uniform(-1, 1, N_train))
    u_true_tr = np.exp(-x_tr**2) * np.sin(15*x_tr)
    u_noisy = u_true_tr + np.random.normal(0, 0.08, size=u_true_tr.shape)
    return x_tr[:, None], u_noisy[:, None], (x_test, k_sq_true, u_true_test)

def gen_exp_D_chirp(N_train, N_test, seed):
    x_test = np.linspace(-1, 1, N_test)
    u_true_test = np.sin(30 * x_test**2)
    np.random.seed(seed)
    x_tr = np.random.uniform(-1, 1, N_train)
    u_tr = np.sin(30 * x_tr**2)
    u_noisy = u_tr + np.random.normal(0, 0.1, size=u_tr.shape)
    return x_tr[:, None], u_noisy[:, None], (x_test, u_true_test, u_true_test)

def gen_exp_E_navier_stokes(N_train, N_test, seed):
    nu = 0.05
    def get_field(x, y, t):
        F = np.exp(-2 * nu * t)
        u = np.sin(x) * np.cos(y) * F
        v = -np.cos(x) * np.sin(y) * F
        return np.sqrt(u**2 + v**2)
    ls = np.linspace(0, 2*np.pi, 200)
    X, Y = np.meshgrid(ls, ls)
    Z_truth = get_field(X, Y, t=2.5)
    np.random.seed(seed)
    raw_x = np.random.uniform(0, 2*np.pi, (N_train, 1))
    raw_y = np.random.uniform(0, 2*np.pi, (N_train, 1))
    raw_t = np.random.uniform(0, 5.0, (N_train, 1))
    vals = get_field(raw_x, raw_y, raw_t)
    norm_x = (raw_x - np.pi) / np.pi
    norm_y = (raw_y - np.pi) / np.pi
    norm_t = (raw_t - 2.5) / 2.5
    inputs = np.concatenate([norm_x, norm_y, norm_t], axis=1)
    y_tr = vals + np.random.normal(0, 0.05, vals.shape)
    return inputs, y_tr, (X, Y, Z_truth, 2.5)

# ==========================================
# 3. TRAINING ROUTINES
# ==========================================

def train_siren(key, x, y, steps=2000):
    in_dim = x.shape[1]
    model = Siren(key, in_dim)
    opt = optax.adam(1e-4)
    state = opt.init(model.params)
    @jax.jit
    def step(p, s):
        def loss(w): return jnp.mean((model.forward(w, x) - y)**2)
        l, g = jax.value_and_grad(loss)(p)
        u, ns = opt.update(g, s)
        return optax.apply_updates(p, u), ns, l
    start = time.time()
    for _ in range(steps):
        model.params, state, l = step(model.params, state)
    return model, time.time()-start, count_params(model.params)

def train_grid(key, x, y, steps=2000):
    in_dim = x.shape[1]
    # Use smaller grid for 3D to prevent OOM, or keep 64 to show param explosion
    # Let's use 32 for 3D to be kind to memory, but it will still be huge.
    res = 64 if in_dim < 3 else 32 
    
    model = GridMLP(key, in_dim, grid_res=res)
    params = {'grid': model.grid, 'mlp': model.decoder_params}
    opt = optax.adam(1e-2)
    state = opt.init(params)
    @jax.jit
    def step(p, s):
        def loss(w): return jnp.mean((model.forward(w, x) - y)**2)
        l, g = jax.value_and_grad(loss)(p)
        u, ns = opt.update(g, s)
        return optax.apply_updates(p, u), ns, l
    start = time.time()
    for _ in range(steps):
        params, state, l = step(params, state)
    return model, params, time.time()-start, count_params(params)

def train_vekua_cascade(key, x, y, max_blocks=3):
    model = VekuaCascade(key)
    residual = y
    freq_scales = [5.0, 15.0, 30.0]
    in_dim = x.shape[1]
    
    def global_pred(x_in):
        res = jnp.zeros((x_in.shape[0], 1))
        for blk, w in zip(model.blocks, model.scalers):
            res += model.get_basis(blk, x_in) @ w
        return res

    start = time.time()
    total_params = 0
    for b in range(max_blocks):
        k, subk = jax.random.split(jax.random.PRNGKey(b), 2)
        block_p = model.create_block(subk, in_dim, freq_scales[b], is_first=(b==0))
        opt = optax.adam(0.01)
        state = opt.init(block_p)
        @jax.jit
        def train_step(p, s, res_target):
            def loss_fn(curr_p):
                phi = model.get_basis(curr_p, x)
                w = model.solve(phi, res_target)
                pred = phi @ w
                return jnp.mean((pred - res_target)**2) + 1e-6*jnp.sum(w**2), w
            (l, w_optimal), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
            u, ns = opt.update(g, s)
            return optax.apply_updates(p, u), ns, l, w_optimal
        final_w = None
        for _ in range(300):
            block_p, state, l, final_w = train_step(block_p, state, residual)
        model.blocks.append(block_p)
        model.scalers.append(final_w)
        total_params += count_params(block_p) + final_w.size
        preds = global_pred(x)
        residual = y - preds
        if jnp.mean(residual**2) < 1e-5: break
    return model, time.time() - start, total_params

# ==========================================
# 4. BENCHMARK RUNNER
# ==========================================

def run_benchmark():
    experiments = [
        ("A: Noisy Helmholtz", gen_exp_A_helmholtz),
        ("B: Sparse Phantom", gen_exp_B_sparse),
        ("C: Inverse Param", gen_exp_C_inverse),
        ("D: Noisy Chirp", gen_exp_D_chirp),
        ("E: Navier-Stokes (3D)", gen_exp_E_navier_stokes)
    ]
    
    results = []
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))
    
    for i, (exp_name, gen_fn) in enumerate(experiments):
        print(f"--- Running {exp_name} ---")
        n_samples = 10000 if "Navier" in exp_name else 3000
        x_tr, y_tr, truth_tuple = gen_fn(n_samples, 200, 42)
        x_tr_jax = jnp.array(x_tr)
        y_tr_jax = jnp.array(y_tr)
        
        if x_tr_jax.shape[1] == 1:
            x_tr_jax = jnp.concatenate([x_tr_jax, jnp.zeros_like(x_tr_jax)], axis=1)
            
        # Train Models
        siren, t_s, p_s = train_siren(jax.random.PRNGKey(0), x_tr_jax, y_tr_jax)
        vekua, t_v, p_v = train_vekua_cascade(jax.random.PRNGKey(0), x_tr_jax, y_tr_jax)
        # Grid now works for 3D!
        grid_model, grid_p, t_g, p_g = train_grid(jax.random.PRNGKey(0), x_tr_jax, y_tr_jax)
        
        # --- Evaluation ---
        if "Navier" in exp_name:
            X, Y, Z, t_val = truth_tuple
            flat_x = (X.flatten() - np.pi) / np.pi
            flat_y = (Y.flatten() - np.pi) / np.pi
            flat_t = np.ones_like(flat_x) * ((t_val - 2.5) / 2.5)
            flat_input = jnp.stack([flat_x, flat_y, flat_t], axis=1)
            
            p_s = siren.forward(siren.params, flat_input).reshape(X.shape)
            p_g = grid_model.forward(grid_p, flat_input).reshape(X.shape)
            
            p_v = jnp.zeros((flat_input.shape[0], 1))
            for b, w in zip(vekua.blocks, vekua.scalers):
                p_v += vekua.get_basis(b, flat_input) @ w
            p_v = p_v.reshape(X.shape)
            
            mse_s = jnp.mean((p_s - Z)**2)
            mse_g = jnp.mean((p_g - Z)**2)
            mse_v = jnp.mean((p_v - Z)**2)
            
            axs[i, 0].imshow(Z, cmap='viridis'); axs[i, 0].set_title(f"{exp_name}\nTruth (t={t_val})")
            axs[i, 1].imshow(p_s, cmap='viridis'); axs[i, 1].set_title(f"SIREN\nMSE: {mse_s:.1e}")
            axs[i, 2].imshow(p_g, cmap='viridis'); axs[i, 2].set_title(f"Grid\nMSE: {mse_g:.1e}")
            axs[i, 3].imshow(p_v, cmap='viridis'); axs[i, 3].set_title(f"Vekua\nMSE: {mse_v:.1e}")

        elif "Inverse" in exp_name or "Chirp" in exp_name:
            x_test, k_true, u_true = truth_tuple
            x_test_2d = jnp.stack([x_test, jnp.zeros_like(x_test)], axis=1)
            
            u_s = siren.forward(siren.params, x_test_2d).flatten()
            u_g = grid_model.forward(grid_p, x_test_2d).flatten()
            u_v = jnp.zeros_like(u_s)
            for b, w in zip(vekua.blocks, vekua.scalers):
                u_v += (vekua.get_basis(b, x_test_2d) @ w).flatten()
            
            axs[i, 0].set_title(f"{exp_name}\nGround Truth")
            axs[i, 0].plot(x_test, u_true, 'k-', lw=2, label='True')
            if "Inverse" in exp_name: axs[i, 0].plot(x_test, k_true/np.max(k_true), 'k--', alpha=0.5, label='Param k(x)')
            
            axs[i, 1].set_title(f"SIREN\nMSE:{jnp.mean((u_s-u_true)**2):.1e}")
            axs[i, 1].plot(x_test, u_s, 'r-', alpha=0.8)
            axs[i, 1].scatter(x_tr, y_tr, c='k', s=1, alpha=0.1)
            
            axs[i, 2].set_title(f"Grid\nMSE:{jnp.mean((u_g-u_true)**2):.1e}")
            axs[i, 2].plot(x_test, u_g, 'g-', alpha=0.8)
            
            axs[i, 3].set_title(f"Vekua\nMSE:{jnp.mean((u_v-u_true)**2):.1e}")
            axs[i, 3].plot(x_test, u_v, 'b-', alpha=0.8)
            
            mse_s = jnp.mean((u_s - u_true)**2)
            mse_g = jnp.mean((u_g - u_true)**2)
            mse_v = jnp.mean((u_v - u_true)**2)

        else:
            X, Y, Z = truth_tuple
            flat_grid = jnp.stack([X.flatten(), Y.flatten()], axis=1)
            
            p_s = siren.forward(siren.params, flat_grid).reshape(X.shape)
            p_g = grid_model.forward(grid_p, flat_grid).reshape(X.shape)
            p_v = jnp.zeros((flat_grid.shape[0], 1))
            for b, w in zip(vekua.blocks, vekua.scalers):
                p_v += vekua.get_basis(b, flat_grid) @ w
            p_v = p_v.reshape(X.shape)
            
            mse_s = jnp.mean((p_s - Z)**2)
            mse_g = jnp.mean((p_g - Z)**2)
            mse_v = jnp.mean((p_v - Z)**2)
            
            axs[i, 0].imshow(Z, cmap='viridis'); axs[i, 0].set_title(f"{exp_name}\nTruth")
            axs[i, 1].imshow(p_s, cmap='viridis'); axs[i, 1].set_title(f"SIREN\nMSE: {mse_s:.1e}")
            axs[i, 2].imshow(p_g, cmap='viridis'); axs[i, 2].set_title(f"Grid\nMSE: {mse_g:.1e}")
            axs[i, 3].imshow(p_v, cmap='viridis'); axs[i, 3].set_title(f"Vekua\nMSE: {mse_v:.1e}")

        results.append({"Exp": exp_name, "Method": "SIREN", "MSE": float(mse_s), "Time": t_s, "Params": p_s})
        results.append({"Exp": exp_name, "Method": "Grid", "MSE": float(mse_g), "Time": t_g, "Params": p_g})
        results.append({"Exp": exp_name, "Method": "Vekua", "MSE": float(mse_v), "Time": t_v, "Params": p_v})

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL BENCHMARK REPORT")
    print("="*60)
    print("\n--- MSE (Lower is Better) ---")
    print(df.pivot(index='Exp', columns='Method', values='MSE'))
    print("\n--- Training Time (Seconds) ---")
    print(df.pivot(index='Exp', columns='Method', values='Time'))
    print("\n--- Parameter Count (Model Size) ---")
    print(df.pivot(index='Exp', columns='Method', values='Params'))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()