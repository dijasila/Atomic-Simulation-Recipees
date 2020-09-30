{{ fullname | escape | underline}}

The main step of this recipe is:

.. autofunction:: {{ fullname }}.main
   :noindex:

Invoke this recipe with

.. code-block:: console

   $ python3 -m {{ fullname }}

or

.. code-block:: console

   $ asr run {{ fullname }}


Documentation
-------------

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   {% for item in functions %}
   .. autofunction:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   {% for item in classes %}
   .. autoclass:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   {% for item in exceptions %}
   .. autoexception:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
