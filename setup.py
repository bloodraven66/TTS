import setuptools


setuptools.setup(
    name='TTS',
    version='0.0.1',
    author='Sathvik Udupa',
    author_email='sathvikudupa66@gmail.com',
    packages=['TTS'],
    install_requires=['requests',
                        'unidecode',
                        'inflect',
                        'torch',
                        'librosa',
                        'numpy',
                        'scipy',
                        'cython'],
)