Installation
============
The most straightforward way of installing ``toppred`` is via pip:

.. code::

  pip3 install toppred

From source
^^^^^^^^^^^
To install this library from source, simply clone the repository:

.. code::

   git clone https://github.com/Thijsvanede/toppred.git

Next, you can install the ``toppred`` from source:

.. code::

  pip install -e <path/to/toppred/>

.. _source code: https://github.com/Thijsvanede/toppred

Dependencies
------------
``toppred`` requires the following python packages to be installed:

- Numpy: https://numpy.org
- Pandas: https://pandas.pydata.org/
- Scikit-learn: https://scikit-learn.org/stable/index.html

All dependencies should be automatically downloaded if you install ``toppred`` via pip. However, should you want to install these libraries manually, you can install the dependencies using the requirements.txt file

.. code::

  pip install -r requirements.txt

Or you can install these libraries yourself

.. code::

  pip install -U numpy pandas sklearn
