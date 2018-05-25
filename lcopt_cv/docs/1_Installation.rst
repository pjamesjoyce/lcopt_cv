============
Installation
============

.. highlight:: console

.. note:: Note - `lcopt-cv` requires the `lcopt` and `brightway2` packages to be installed, and for `lcopt` to be set up with ecoinvent 3.3 cutoff

The best way to install lcopt-cv is to use the conda package. The command is::

	conda install -y -q -c conda-forge -c cmutel -c haasad -c pjamesjoyce lcopt-cv

One additional dependency isn't available as a conda package and needs to be installed separately using pip. Here is the command::


	pip install opencv-python

If you already had lcopt installed and set up - that's it. If not you need to set up lcopt to talk to brightway.

Full instructions on how to do this are in the `lcopt documentation <http://lcopt.readthedocs.io/en/latest/1_installation.html#step-2a-lcopt-bw2-setup-at-the-command-line>`_

The short version is 

- Download the file called `ecoinvent 3.3_cutoff_ecoSpold02.7z` from the `ecoinvent website <http://ecoinvent.org>`_
- Unzip the file using `7zip <https://www.7-zip.org/download.html>`_ and make a note of the path of the datasets folder
- Run the following command::

	lcopt-bw2-setup path/to/ecospold/files # use "" if there are spaces in your path

This will generate the lcopt template databases in brightway2 so that you can analyse your LCA models.