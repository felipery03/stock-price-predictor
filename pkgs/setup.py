from setuptools import setup

setup(name='pred_stock_price',
      version='0.1',
      description='''Package to consolidade functions as read data from API,
      train models and utils functions used in Stock price predictor project''',
      url='https://github.com/felipery03/stock-price-predictor/pkgs',
      author='felipery03',
      author_email='feliperigo_yoshimura@hotmail.com',
      license='MIT',
      packages=['move_data', 'feat_metrics', 'process_modules'],
      zip_safe=False)