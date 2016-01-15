#! /bin/bash
./run_pipeline_distributed_2015.py filter MD592 -b 46 -e 185
./run_pipeline_distributed_2015.py filter MD593 -b 41 -e 176
./run_pipeline_distributed_2015.py filter MD594 -b 47 -e 186
./run_pipeline_distributed_2015.py filter MD595 -b 35 -e 164
./run_pipeline_distributed_2015.py compute_texmap MD589 -b 49 -e 186
./run_pipeline_distributed_2015.py compute_texmap MD592 -b 46 -e 185
./run_pipeline_distributed_2015.py compute_texmap MD593 -b 41 -e 176
./run_pipeline_distributed_2015.py compute_texmap MD594 -b 47 -e 186
./run_pipeline_distributed_2015.py compute_texmap MD595 -b 35 -e 164
