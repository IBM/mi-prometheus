Helpers Explained
===================
`@author: Tomasz Kornuta & Vincent Marois`

Helper is an application useful from the point of view of experiment running, but independent/external to the Workers.
Currently MI-Prometheus offers two type of helpers:

    - **Problem Initializer**, responsible for initialization of a problem (i.e. download of required data from internet or generation of all samples) in advance, before the real experiment starts.
    - **Index Splitter**, responsible for generation of files with indices splitting given dataset (in fact set of indices) into two. The resulting files can later be used in training/verification testing when using ``SubsetRandomSampler``.

We expect this list to grow soon.
