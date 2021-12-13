PySAGES Documentation
=====================

PySAGES' documentation can be found in the [online manual](https://pysages.readthedocs.io).
Alternatively, you can build it locally with the help of [Sphinx](https://www.sphinx-doc.org)

## Requirements

Make sure you have [GNU make](https://www.gpu.org/software/make/) installed.

## Building

You can build the documentation from you local copy of the PySAGES repository as follows:

```
$ pip install -r docs/requirements.txt
$ make html
```

## Viewing the documentation

The final result can be explored with a browser by opening `PySAGES/docs/build/html/index.html`

## Notes

Do not build in the parent directory of PySAGES.
In that case sphinx tries to use the local `pysages` directory as the python module, but that will fail.
Always install PySAGES first and build the documentation based on that installation.

*To developers:* Sphinx builds the documentation from the installation the python interpreter finds.
If you have locally changed the documentation in the source code, install pysages first before rebuilding the documentation.
