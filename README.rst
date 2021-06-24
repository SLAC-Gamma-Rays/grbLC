======
adsgrb
======


.. image:: https://img.shields.io/pypi/v/adsgrb.svg
        :target: https://pypi.python.org/pypi/adsgrb

.. image:: https://img.shields.io/travis/youngsm/adsgrb.svg
        :target: https://travis-ci.com/youngsm/adsgrb

.. image:: https://readthedocs.org/projects/adsgrb/badge/?version=latest
        :target: https://adsgrb.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




A simple way to scour the ADS for GRB data.


* Free software: MIT license
* Documentation: https://adsgrb.readthedocs.io.

Quickstart
--------
```python
>>> import adsgrb
API key not found in /Users/youngsam/.ads/dev_key. Either set adsgrb.config.token manually or consider
calling adsgrb.set_apikey() to save the API key onto your system, bypassing the need to set your API key after
each import. Your key can be found here: https://ui.adsabs.harvard.edu/user/settings/token.
>>> adsgrb.set_key('secret api key')
>>> articles = adsgrb.litSearch('011211')
[011211] 32 entries found.
>>> pdfs = adsgrb.getArticles(articles)
[ADSGRB] 31/32 papers grabbed.
>>> adsgrb.savePDF(pdfs)
```

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

This package heavily relies on Andy Casey's `ads`_ Python package for dealing with search queries.



.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`ads`: https://github.com/andycasey/ads
