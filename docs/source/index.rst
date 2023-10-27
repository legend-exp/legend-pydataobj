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

Inspecting LH5 files from the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``lh5ls`` shell command uses :func:`.lh5_store.show` to print a pretty
representation of a LH5 file's contents

.. code-block:: console

    $ lh5ls -a legend-testdata/data/lh5/LDQTA_r117_20200110T105115Z_cal_geds_raw.lh5
    /
    └── geds · HDF5 group
        └── raw · table{packet_id,ievt,timestamp,numtraces,tracelist,baseline,energy,channel,wf_max,wf_std,waveform}
            ├── baseline · array<1>{real}
            ├── channel · array<1>{real}
            ├── energy · array<1>{real}
            ├── ievt · array<1>{real}
            ├── numtraces · array<1>{real}
            ├── packet_id · array<1>{real}
            ├── timestamp · array<1>{real} ── {'units': 's'}
            ├── tracelist · array<1>{array<1>{real}}
            │   ├── cumulative_length · array<1>{real}
            │   └── flattened_data · array<1>{real}
            ├── waveform · table{t0,dt,values}
            │   ├── dt · array<1>{real} ── {'units': 'ns'}
            │   ├── t0 · array<1>{real} ── {'units': 'ns'}
            │   └── values · array_of_equalsized_arrays<1,1>{real}
            ├── wf_max · array<1>{real}
            └── wf_std · array<1>{real}

For more information, have a look at the command's help section:

.. code-block:: console

    $ lh5ls --help

Next steps
----------

.. toctree::
   :maxdepth: 1

   tutorials
   Package API reference <api/modules>

.. _1: https://legend-exp.github.io/legend-data-format-specs
.. _2: https://pypi.org/project/legend-pydataobj
.. _3: https://pip.pypa.io/en/stable/getting-started
.. |legend-pydataobj| replace:: *legend-pydataobj*
