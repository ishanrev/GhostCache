from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='GhostCache',
    version='0.1.0',
    ext_modules=[
        # CppExtension(
        #     name='offload_manager',
        #     sources=['offload_manager.cpp'],
        #     extra_compile_args=['-O3'],  # optional optimization flag
        # ),
        CppExtension(
            name='ghost',
            # sources=["offload_manager.cpp", "async.cpp"],
            sources = ["offload_manager.cpp", "async.cpp", "chunked_sdpa.cpp", "ghost.cpp"],
            # threads.cpp
            extra_compile_args=['-O3'],  # optional optimization flag
        ),
        CppExtension(
            name='normal_sdpa',
            sources=['normal_sdpa.cpp'],
            extra_compile_args=['-O3'],  # optional optimization flag
        ),
        
    ],
    install_requires=[
        'torch',  # Ensure PyTorch is installed
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    include_dirs=['/opt/apps/cuda/12.2/include'],

)
