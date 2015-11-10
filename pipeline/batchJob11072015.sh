./run_pipeline_distributed_2015.py filter MD592 -b 46 -e 185
./run_pipeline_distributed_2015.py compute_texmap MD592 -b 46 -e 185
./run_pipeline_distributed_2015.py segment MD592 -b 46 -e 185 -s tSLIC200
./run_pipeline_distributed_2015.py compute_histograms MD592 -b 46 -e 185 -s tSLIC200
./run_pipeline_distributed_2015.py filter MD589 -b 49 -e 186
./run_pipeline_distributed_2015.py compute_texmap MD589 -b 49 -e 186
./run_pipeline_distributed_2015.py segment MD589 -b 49 -e 186 -s tSLIC200
./run_pipeline_distributed_2015.py compute_histograms MD589 -b 49 -e 186 -s tSLIC200

