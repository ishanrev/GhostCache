from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='GhostCache',
    version='0.1.0',
    ext_modules=[
        CppExtension(
            name='offload_manager',
            sources=['offload_manager.cpp'],
            extra_compile_args=['-O3'],  # optional optimization flag
        ),
        CppExtension(
            name='chunked_sdpa',
            sources=['chunked_sdpa.cpp'],
            extra_compile_args=['-O3'],  # optional optimization flag
        ),
        
    ],
    install_requires=[
        'torch',  # Ensure PyTorch is installed
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
