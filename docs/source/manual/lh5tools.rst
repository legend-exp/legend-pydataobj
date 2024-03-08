LH5 Command Line Tools
======================

*legend-pydataobj* provides some command line utilities to deal with LH5 files.


Inspecting LH5 files with ``lh5ls``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``lh5ls`` command uses :func:`.lh5.tools.show` to print a pretty
representation of a LH5 file's contents:

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

For more information, have a look at the command's help section: ::

  lh5ls --help


Concatenating LGDOs with ``lh5concat``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``lh5concat`` command can be used to concatenate LGDO
:class:`~.types.array.Array`, :class:`~.types.vectorofvector.VectorOfVectors`
and :class:`~.types.table.Table` into an output LH5 file.

Concatenate all eligible objects in file{1,2].lh5 into concat.lh5: ::

  lh5concat -o concat.lh5 file1.lh5 file2.lh5

Include only the /data/table1 Table: ::

  lh5concat -o concat.lh5 -i '/data/table1/*' file1.lh5 file2.lh5

Exclude the /data/table1/col1 Table column: ::

  lh5concat -o concat.lh5 -e /data/table1/col1 file1.lh5 file2.lh5

For more information, have a look at the command's help section: ::

  lh5concat --help
