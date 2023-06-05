## NOTICE 

- I checked this coco caption repo in the link -> https://github.com/bckim92/coco-caption-py3 
- First, you will neet to download stanford coreNLP 3.6.0 code and models for use by SPICE, To install this, run ./get_models_for_metric.sh 
- SPICE will try to create a cache of parsed sentences in ./pycocoevalcap/spice/cache/.
This dramatically speeds up repeated evaluations. 
The cache directory can be moved by setting 'CACHE_DIR' in ./pycocoevalcap/spice. 
In the same file, caching can be turned off by removing the '-cache' argument to 'spice_cmd'.
