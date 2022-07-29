This folder contains scripts for elastic training for HelloDeepSpeed example.  

DeepSpeed branch: https://github.com/microsoft/DeepSpeed-internal/tree/arpan/elasticity/deepspeed/  
PyTorch version: v1.11.x

Currently, we only support relaunching of training using script method in DeepSpeed (i.e. same script will be ran again when we either scale up or scale down)

Script method: train_bert_ds_el_script.py  
Functional method (not supported directly): train_bert_ds_in_elastic.py

On ITP cluster, there is a need to add hostname (webxt..) to every worker line in "/etc/hosts" file on each worker.

'hosts' file should like something like this

````
# Kubernetes-managed hosts file (host network).
127.0.0.1 localhost

# The following lines are desirable for IPv6 capable hosts
::1 ip6-localhost ip6-loopback
fe00::0 ip6-localnet
ff00::0 ip6-mcastprefix
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
ff02::3 ip6-allhosts
192.168.0.46    ps-0
192.168.0.46    worker-0 webxt7c7400003X
192.168.0.60    worker-1 webxt7c7400004A
````


Once /etc/hosts file is updated, elastic training can be launched using following command. 

````
deepspeed --num_nodes 2 --max_num_nodes 4 --min_num_nodes 1 \
          --num_gpus 8 --force_multi --hostfile hosts_el \
          --elastic_training --master_port 49091 --master_addr worker-0  \
          train_bert_ds_el_script.py --checkpoint_dir ./checks_el
````
Above command runs elastic training on two nodes given in 'hosts_el' file with min_nodes 1 and max_nodes 4. 

To add more nodes to elastic training just run above command on new workers on a new hostfile. 

To test fault tolerance, just kill one of the training process on any node. Elastic agent will restart training on all nodes. 

When an elastic agent dies, it is considered a scale down event




