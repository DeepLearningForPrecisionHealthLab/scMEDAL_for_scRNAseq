#!/bin/bash
#SBATCH --job-name=AE
#SBATCH --time=14-00:00:00
# Speficy type of nodes, number of nodes, and number of CPUs, GPUs per node
#SBATCH -p GPUp100
#SBATCH --nodes=2
# Same number as Num_GPU
#SBATCH --gres=gpu:2
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --exclusive
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH -o /archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics/Synthetic/run_models/OtherSim/1DGMM/20ct-20donor/7-1-6/AE_conv/scripts/job_%j.out
#SBATCH -e /archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics/Synthetic/run_models/OtherSim/1DGMM/20ct-20donor/7-1-6/AE_conv/scripts/job_%j.err
#SBATCH --mail-user=AixaXiuhyolotzin.AndradeHernandez@utsouthwestern.edu                     # specify an email address
#SBATCH --mail-type=ALL   
job_id=$SLURM_JOB_ID
#activate your conda environemnt here
#export CUDA_VISIBLE_DEVICES=0,1
module load cuda118
module load cuda118/toolkit/11.8.0
module load python/3.8.x-anaconda 
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_ARMED_2
Num_CPU=56
Num_GPU=2
TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_FORCE_GPU_ALLOW_GROWTH
# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port 
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --num-cpus $Num_CPU --num-gpus $Num_GPU --block &
sleep 15

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
  ray start --address "$ip_head"  --num-cpus $Num_CPU --num-gpus $Num_GPU --block &
  sleep 5
done

# ===== Call your code below =====
python run_AE_conv_allfolds_Raytune.py
