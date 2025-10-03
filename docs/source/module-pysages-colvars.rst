Collective Variables
--------------------

.. rubric:: Overview

Collective Variables available in PySAGES

.. autosummary::

   pysages.colvars.angles.Angle
   pysages.colvars.angles.DihedralAngle

   pysages.colvars.shape.RadiusOfGyration
   pysages.colvars.shape.PrincipalMoment
   pysages.colvars.shape.Asphericity
   pysages.colvars.shape.Acylindricity
   pysages.colvars.shape.ShapeAnisotropy

   pysages.colvars.coordinates.Component
   pysages.colvars.coordinates.Distance

   pysages.colvars.orientation.ERMSD
   pysages.colvars.orientation.ERMSDCG

   pysages.colvars.contacts.NativeContactFraction

Abstract base classes

.. autosummary::

   pysages.colvars.core.CollectiveVariable
   pysages.colvars.core.AxisCV
   pysages.colvars.core.TwoPointCV
   pysages.colvars.core.ThreePointCV
   pysages.colvars.core.FourPointCV

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   module-pysages-colvars-angles
   module-pysages-colvars-shape
   module-pysages-colvars-coordinates
   module-pysages-colvars-core
   module-pysages-colvars-orientation
   module-pysages-colvars-contacts

.. automodule:: pysages.colvars
    :synopsis: Python classes for collective variables.
    :members:
