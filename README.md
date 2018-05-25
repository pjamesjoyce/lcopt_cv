# lcopt-cv
lcopt-cv: Create fully functional LCA models from hand drawn pictures of process diagrams
-----------------------------------------------------------------------------------------

Lcopt-cv is a python module for creating LCA foreground models from hand drawn pictures of process flow diagrams developed by [James Joyce](https://pjamesjoyce.github.io/)

Pretty much every LCA starts with drawing a process flow diagram. The difficult bit is turning that diagram into an LCA model which can be analysed.

What if you could just take a picture of the diagram you've just drawn and have it instantly turned into an LCA model?

Well now you can - introducing lcopt-cv, computer vision for LCA.

![lcopt-cv workflow](https://raw.githubusercontent.com/pjamesjoyce/lcopt_cv/master/lcopt_cv/docs/assets/lcopt_cv_workflow.jpg)


Features
--------

- Uses computer vision to generate an LCA model from a photograph of a process flow diagram
- Exports model directly to `lcopt <http://lcopt.rtfd.io>`_, allowing models to be analysed using `Brightway <http://www.brightwaylca.org>`_
- Links directly to the `ecoinvent <http://www.ecoinvent.org>`_ or `FORWAST <http://forwast.brgm.fr/>`_ databases
# Installation

---
**NOTE**

`lcopt-cv` requires the `lcopt` and `brightway2` packages to be installed, and for `lcopt` to be set up with ecoinvent 3.3 cutoff

---

The best way to install lcopt-cv is to use the conda package. The command is

```
conda install -y -q -c conda-forge -c cmutel -c haasad -c pjamesjoyce lcopt-cv
```

One additional dependency isn't available as a conda package and needs to be installed separately using pip. Here is the command

```
pip install opencv-python
```

If you already had lcopt installed and set up - that's it. If not you need to set up lcopt to talk to brightway.

Full instructions on how to do this are in the [lcopt documentation](http://lcopt.readthedocs.io/en/latest/1_installation.html#step-2a-lcopt-bw2-setup-at-the-command-line)

The short version is 

- Download the file called `ecoinvent 3.3_cutoff_ecoSpold02.7z` from the [ecoinvent website](http://ecoinvent.org)
- Unzip the file using [7zip](https://www.7-zip.org/download.html) and make a note of the path of the datasets folder
- Run the following command:

```
lcopt-bw2-setup path/to/ecospold/files # use "" if there are spaces in your path
```

This will generate the lcopt template databases in brightway2 so that you can analyse your LCA models.

# Use

To launch lcopt-cv at the command line type:

```
lcopt-cv
```

This will launch the lcopt-cv GUI.

![lcopt-cv gui](https://raw.githubusercontent.com/pjamesjoyce/lcopt_cv/master/lcopt_cv/docs/assets/gui.jpg)

More detailed documentation is available in the [online documentation](http://lcopt_cv.rtfd.io)