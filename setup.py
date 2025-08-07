from setuptools import setup

setup(
    name='dmp_barcode',
    version='0.1.1',    
    description='Package encapsulating barcode reading- and transformation logic.',
    url='https://github.com/computationalpathologygroup/dmp-barcode',
    author='Thomas Mulder',
    packages=['dmp_barcode'],
    install_requires=[
        'zxing-cpp>=2.3.0,<2.4.0',
        'numpy>=2.0.2,<2.1.0',
        'opencv-python>=4.11.0,<4.12.0',
        'pyzbar>=0.1.9,<0.2.0',
        'pillow>=10.0.0,<11.2.0'
    ]
)
