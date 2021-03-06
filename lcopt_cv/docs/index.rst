.. lcopt-cv documentation master file, created by
   sphinx-quickstart on Fri Apr 13 15:53:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========
Overview
========

lcopt-cv: Create fully functional LCA models from hand drawn pictures of process diagrams
-----------------------------------------------------------------------------------------

Lcopt-cv is a python module for creating LCA foreground models from hand drawn pictures of process flow diagrams developed by `James Joyce <https://pjamesjoyce.github.io/>`_.

Pretty much every LCA starts with drawing a process flow diagram. The difficult bit is turning that diagram into an LCA model which can be analysed.

What if you could just take a picture of the diagram you've just drawn and have it instantly turned into an LCA model?

Well now you can - introducing lcopt-cv, computer vision for LCA.

.. raw:: html

	<video height="480" poster="_static/simple_snap.jpg" controls>
	<source src="_static/lcopt_cv.mp4" type="video/mp4">
	</video>

.. ".. image:: assets/lcopt_cv_workflow.jpg"



Features
--------

- Uses computer vision to generate an LCA model from a photograph of a process flow diagram
- Exports model directly to `lcopt <http://lcopt.rtfd.io>`_, allowing models to be analysed using `Brightway <http://www.brightwaylca.org>`_
- Links directly to the `ecoinvent <http://www.ecoinvent.org>`_ or `FORWAST <http://forwast.brgm.fr/>`_ databases

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   *