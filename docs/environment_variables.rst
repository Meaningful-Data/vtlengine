Environment Variables
#####################

VTL Engine uses environment variables to configure behavior for number handling and S3 connectivity.
These variables are optional and have sensible defaults.

Number Handling
***************

These variables control how VTL Engine handles floating-point precision in comparison operators and output formatting.

.. important::
    IEEE 754 float64 guarantees **15 significant decimal digits** (DBL_DIG = 15).
    The valid range of 6-15 reflects the practical precision limits of double-precision floating point.

``COMPARISON_ABSOLUTE_THRESHOLD``
=================================

Controls the significant digits used for Number comparison operations (``=``, ``<>``, ``>=``, ``<=``, ``between``).

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Behavior
   * - Not defined
     - Uses default value of **15** significant digits
   * - ``6`` to ``15``
     - Uses the specified number of significant digits
   * - ``-1``
     - Disables tolerance (uses Python's default exact comparison)

The tolerance is calculated as: ``0.5 * 10^(-(N-1))`` where N is the number of significant digits.

For the default of 15, this gives a relative tolerance of ``5e-15``, which filters floating-point
arithmetic artifacts while preserving meaningful differences.

``OUTPUT_NUMBER_SIGNIFICANT_DIGITS``
====================================

Controls the significant digits used when formatting Number values in CSV output.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Behavior
   * - Not defined
     - Uses default value of **15** significant digits
   * - ``6`` to ``15``
     - Uses the specified number of significant digits
   * - ``-1``
     - Disables formatting (uses pandas default behavior)

This variable controls the ``float_format`` parameter in pandas ``to_csv``, using the general format
specifier (e.g., ``%.15g``) which automatically switches between fixed and exponential notation.

S3 Configuration
****************

The following AWS environment variables are used when working with S3 URIs.
This requires the ``vtlengine[s3]`` extra to be installed:

.. code-block:: bash

    pip install vtlengine[s3]

``AWS_ACCESS_KEY_ID``
=====================

The access key ID for AWS authentication.

``AWS_SECRET_ACCESS_KEY``
=========================

The secret access key for AWS authentication.

``AWS_SESSION_TOKEN``
=====================

(Optional) Session token for temporary AWS credentials.

``AWS_DEFAULT_REGION``
======================

(Optional) Default AWS region for S3 operations.

``AWS_ENDPOINT_URL``
====================

(Optional) Custom endpoint URL for S3-compatible storage services (e.g., MinIO, LocalStack).

For more details on AWS configuration, see the
`boto3 documentation <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables>`_.

Examples
********

Setting comparison threshold
============================

.. code-block:: bash

    # Use 10 significant digits for more lenient comparisons (tolerance ~5e-10)
    export COMPARISON_ABSOLUTE_THRESHOLD=10

    # Use maximum precision (default, tolerance ~5e-15)
    export COMPARISON_ABSOLUTE_THRESHOLD=15

    # Disable tolerance-based comparison (exact floating-point comparison)
    export COMPARISON_ABSOLUTE_THRESHOLD=-1

Controlling output precision
=============================

.. code-block:: bash

    # Format output with 10 significant digits
    export OUTPUT_NUMBER_SIGNIFICANT_DIGITS=10

    # Disable output formatting (use pandas defaults)
    export OUTPUT_NUMBER_SIGNIFICANT_DIGITS=-1

Using S3 with environment variables
====================================

.. code-block:: bash

    # Set AWS credentials
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export AWS_DEFAULT_REGION=eu-west-1

.. code-block:: python

    from vtlengine import run

    result = run(
        script="DS_r := DS_1;",
        data_structures=data_structures,
        datapoints="s3://my-bucket/input/DS_1.csv",
        output="s3://my-bucket/output/",
    )

Using a custom S3 endpoint
===========================

.. code-block:: bash

    # For S3-compatible services (MinIO, LocalStack)
    export AWS_ENDPOINT_URL=http://localhost:9000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
