from setuptools import setup, find_packages, Extension

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

if __name__ == "__main__":
    setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='dnn-inference',
        url='https://github.com/statmlben/dnn-inference',
        author='Ben Dai',
        author_email='bendai@cuhk.edu.hk',
        # Needed to actually package something
        packages=['dnn_inference'],
        # Needed for dependencies
        install_requires=['emoji==1.7.0', 'hachibee_sphinx_theme==0.2.5', 
                            'keras==2.9.0', 'matplotlib==3.5.2', 'numpy==1.23.0', 
                            'pandas==1.4.3', 'scikit_learn==1.1.1', 'scipy==1.8.1', 
                            'seaborn==0.11.2', 'setuptools==59.6.0', 'tensorflow==2.9.1'],
        # *strongly* suggested for sharing
        version='0.15',
        # The license can be anything you like
        license='MIT',
        description='Dnn-Inference is a Python module for hypothesis testing based on deep neural networks.',
        #cmdclass={"build_ext": build_ext},
        # We will also need a readme eventually (there will be a warning)
        long_description_content_type='text/x-rst',
        long_description=LONG_DESCRIPTION,
    )
