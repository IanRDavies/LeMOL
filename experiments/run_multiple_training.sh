cd ..
echo $PWD
separator="_vs_"
for agent1 in "maddpg" "mmmaddpg"
do 
for agent2 in "maddpg" "mmmaddpg"
do
for run in {1..3}
do
filename=./tmp/logging/outfiles/run_$run.txt
echo $filename
logdir=./tmp/$agent1$separator$agent2/$run
echo $logdir
savedir=./tmp/policy/$agent1$separator$agent2/$run
echo $savedir
eval python ./experiments/train.py --log-dir=$logdir --scenario=simple_push --good-policy=$agent1 --bad-policy=$agent2 --save-dir=$savedir
done
done
done