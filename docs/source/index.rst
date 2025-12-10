LEGEND Data Objects
===================

|legend-pydataobj| is a Python implementation of the `LEGEND Data Format Specification <1_>`_.


Getting started
---------------

|legend-pydataobj| is published on the `Python Package Index <2_>`_. Install on
local systems with `pip <3_>`_:

.. tab:: Stable release

    .. code-block:: console

        $ pip install legend-pydataobj

.. tab:: Unstable (``main`` branch)

    .. code-block:: console

        $ pip install legend-pydataobj@git+https://github.com/legend-exp/legend-pydataobj@main

.. tab:: Linux Containers

    Get a LEGEND container with |legend-pydataobj| pre-installed on `Docker hub
    <https://hub.docker.com/r/legendexp/legend-software>`_ or follow
    instructions on the `LEGEND wiki
    <https://legend-exp.atlassian.net/l/cp/nF1ww5KH>`_.

If you plan to develop |legend-pydataobj|, refer to the :doc:`developer's guide
<developer>`.

.. attention::

    If installing in a user directory (typically when invoking pip as a normal
    user), make sure ``~/.local/bin`` is appended to ``PATH``. The ``lh5ls``
    executable is installed there.


Next steps
----------

.. toctree::
   :maxdepth: 1

   manual/index
   Package API reference <api/modules>

.. toctree::
   :maxdepth: 1
   :caption: Related projects

   Decoding Digitizer Data <https://legend-daq2lh5.readthedocs.io>
   Digital Signal Processing <https://dspeed.readthedocs.io>
   pygama <https://pygama.readthedocs.io>

.. toctree::
   :maxdepth: 1
   :caption: Development

   Source Code <https://github.com/legend-exp/legend-pydataobj>
   License <https://github.com/legend-exp/legend-pydataobj/blob/main/LICENSE>
   Citation <https://doi.org/10.5281/zenodo.10592107>
   Changelog <https://github.com/legend-exp/legend-pydataobj/releases>
   developer


.. _1: https://legend-exp.github.io/legend-data-format-specs
.. _2: https://pypi.org/project/legend-pydataobj
.. _3: https://pip.pypa.io/en/stable/getting-started
.. |legend-pydataobj| replace:: *legend-pydataobj*
