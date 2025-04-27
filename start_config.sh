bash run.sh train RelatE wn18rr 4 WN18RR_N3adv_RT2 1024 1024 768 10.0 1.0 0.00003 300000 16 -dr -de -adv --warm_up_steps 10000 --type_map_path /home/ad.asu.edu/achakr40/TransEE/data/WN18RR/wn18rr_entity_type_map.json --type_lambda 0.2

bash run.sh train RelatE FB15k-237 4 FB15k-237_N3adv_RT5 1024 1024 768 14.0 0.9 0.00002 300000 16 -dr -de -adv --warm_up_steps 10000 --type_map_path /home/ad.asu.edu/achakr40/TransEE/data/FB15k-237/fb15k237_entity_type_map.json --type_lambda 0.15

bash run.sh train RelatE YAGO3-10 4 YAGO310_N3adv_RT2 1024 1024 768 14.0 0.8 0.00004 300000 16 -dr -de -adv --warm_up_steps 10000 --type_map_path /home/ad.asu.edu/achakr40/TransEE/data/YAGO3-10/yago3_10_entity_type_map.json --type_lambda 0.15
