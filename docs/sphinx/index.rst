``model_monitor``
=========================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   project_setup/index
   data_schemas/index
   pipelines/index
   math_methods/index
   dev_features/index
   api/index
   faq


----------------
Introduction
----------------

``model_monitor`` is an automated tool for stability analysis and change detection
in deployed longitudinal machine learning models. It can be used to assess a deployment
model for both desirable and undesirable attributes:

- **Stability**: changes in distributions, either pointwise or continuous in time, are measurable and contribute to model form in a meaningful way
- **Model structure consistency**: changes in model structure are the result of changes in distribution, not algorithmic overfitting
- **Bias**: model outcomes and performance are equitable among all entities
- **Generalizability**: modeling results are locally robust, and can accommodate larger training and testing sizes

--------------------
How does it work?
--------------------

``model_monitor`` analyzes the output of a deployed supervised machine learning model over time by interfacing directly
with the model inputs and outputs. The results of its analysis are databased for reference, and used to generate a report
of the performed analyses. This report can illustrate the features of your model defined above.
