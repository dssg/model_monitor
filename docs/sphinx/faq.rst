
Frequently Asked Questions
=============================================


What kinds of projects are supported by ``model_monitor``?
--------------------------------------------------------------

``model_monitor`` was originally developed for use in conjunction with ``triage``, another DSaPP tool. However, many
deployed longitudinal supervised machine learning projects have setups similar to that of a standard ``triage`` project,
so ``model_monitor`` generically accepts data from a common schema.

We define an 'entity' as any statistical unit that undergoes repeated measurement. Models that are supported by
``model_monitor`` attempt to characterize  :math:`Y_{n,t} | X_{n,t}` where :math:`Y_{n,t} \in \{0, 1\}` and
:math:`X_{n,t} \in \mathbb{R}^k`. This implies the following major constraints:

- The response variable must be a binary indicator variable.
- Each feature must be numeric, or encoded as a numeric variable.
- The number of observable entities does not change significantly in small time intervals.
- The number of observable features does not change significantly in small time intervals.


What tools can I use to import data into ``model_monitor``?
--------------------------------------------------------------

Data imports in ``model_monitor`` are based off of ``triage``'s data storage practices, which typically means two things:

- Results (including model and model group metadata, predictions, and feature importances) are stored in a ``postgres`` database
- Feature matrices are stored in HDF5 files, either locally or on an AWS ``s3`` instance.

If your project already follows these practices, great! If not, don't worry; ``model_monitor`` implements readers for
nonstandard data imports, and also allows you to implement your own if necessary. You can find more information about
data imports HERE.


How do I prepare my data for use in ``model_monitor``?
------------------------------------------------------------

``model_monitor`` imports data from a common data schema. This allows for standardized analysis of different deployment
systems generically. For existing data that uses ``triage`` components (such as ``results-schema``), importing data
is handled automatically after ``model_monitor`` is configured.

You can find a description of the common data schema HERE.


How should I select which metrics to calculate?
--------------------------------------------------
``model_monitor`` does not choose which metrics to calculate for you; every project is different, and every data set
is different. To help with this crucial process, we've written a WHITE PAPER that describes the statistical methodology
behind the metrics, and can help you make informed decisions about how to monitor your data.



