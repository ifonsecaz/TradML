import os

def np_linalg_svd(*args, **kwargs):
    SVD_BACKEND = os.environ.get('SVD_BACKEND', None)
    SVD_VERBOSITY = os.environ.get('SVD_VERBOSITY', False)
    try:
        import torch
        TORCH_INSTALLED = True
    except:
        TORCH_INSTALLED = False
    PROPER_TORCH_THREADS = os.environ.get('PROPER_TORCH_THREADS', 2)
    
    assert SVD_BACKEND in (None, 'numpy', 'scipy', 'pytorch_cpu', 'pytorch_gpu'), f'Unknown SVD_BACKEND: {SVD_BACKEND}'
    verbose = SVD_VERBOSITY
    if SVD_BACKEND is None:
        if TORCH_INSTALLED:
            SVD_BACKEND = 'pytorch_gpu' if torch.cuda.is_available() else 'pytorch_cpu'
        else:
            SVD_BACKEND = 'numpy'
    
    assert not SVD_BACKEND.startswith('pytorch') or TORCH_INSTALLED, 'You do not have torch installed, so stop asking for it!'
    if TORCH_INSTALLED:
        assert SVD_BACKEND != 'pytorch_gpu' or torch.cuda.is_available(), 'No GPU is available, so stop asking for it!'
    
    if SVD_BACKEND == 'numpy':
        if verbose:
            import time
            start_time = time.time()
        output = np.linalg.svd(*args, **kwargs)
        if verbose:
            msg = f'  (The SVD operation took %.3f seconds using the {SVD_BACKEND} backend)  '%(time.time() - start_time)
            print(msg, end='', flush=True)
        return output
    
    elif SVD_BACKEND == 'scipy':
        import scipy.linalg
        scipy_lapack_default_driver = 'gesdd'
        kwargs.setdefault('lapack_driver', scipy_lapack_default_driver) 
        if verbose:
            import time
            start_time = time.time()
        output = scipy.linalg.svd(*args, **kwargs)
        if verbose:
            msg = f'  (The SVD operation took %.3f seconds using the {SVD_BACKEND} backend)  '%(time.time() - start_time)
            print(msg, end='', flush=True)
        return output
    
    elif SVD_BACKEND.startswith('pytorch'):
        device_str = 'gpu' if SVD_BACKEND == 'pytorch_gpu' else 'cpu'
        if device_str == 'cpu':
            default_torch_threads = torch.get_num_threads()
            torch.set_num_threads(PROPER_TORCH_THREADS)
            
        device_ = torch.device(device_str)
        
        if len(args) > 0:
            input_np = args[0]
        else:
            input_np = kwargs.get('a', None)
            assert input_np is not None, 'The input matrix is not specified'
        
        full_matrices = kwargs.get('full_matrices', True)
        compute_uv = kwargs.get('compute_uv', True)
        
        if verbose:
            import time
            start_time = time.time()
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_np).to(device=device_)
            U_tensor, S_tensor, V_tensor = torch.svd(input_tensor, some=full_matrices, compute_uv=compute_uv, out=None)

            if not compute_uv:
                output = S_tensor.detach().cpu().numpy()
            else:
                output = (U_tensor.detach().cpu().numpy(), S_tensor.detach().cpu().numpy(), V_tensor.detach().cpu().numpy())
        
        if device_str == 'cpu':
            torch.set_num_threads(default_torch_threads)
            
        if verbose:
            msg = f'  (The SVD operation took %.3f seconds using the {SVD_BACKEND} backend)  '%(time.time() - start_time)
            print(msg, end='', flush=True)
            
        return output
    
    else:
        raise Exception(f'Unknown SVD_BACKEND: {SVD_BACKEND}')