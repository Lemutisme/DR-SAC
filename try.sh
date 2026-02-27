# I implement Q-ensemble critic, almost the same, but different performance under pertuabtion.

run_dir1=./try/1
run_dir2=./try/2

python train_sac.py env/sac=cartpole \
 critic_ensemble=true \
 data_path=./old_dataset/CPV0/no_noise_eps0.5 \
 ++hydra.run.dir=$run_dir1

python train_sac.py env/sac=cartpole \
 critic_ensemble=false \
 data_path=./old_dataset/CPV0/no_noise_eps0.5 \
 ++hydra.run.dir=$run_dir2

 #These two models both have perferct score in standard env. However, they have different performances under pertuation.

 python train_sac.py env/sac=cartpole \
  eval_model=true \
  critic_ensemble=true \
  load_model=true load_path=$run_dir1 \
  ++env_mods.use_mods=true \
  ++env_mods.param_shift.enabled=true \
  ++env_mods.param_shift.len_factor=2.6

 python train_sac.py env/sac=cartpole \
  eval_model=true \
  critic_ensemble=false \
  load_model=true load_path=$run_dir2 \
  ++env_mods.use_mods=true \
  ++env_mods.param_shift.enabled=true \
  ++env_mods.param_shift.len_factor=2.6