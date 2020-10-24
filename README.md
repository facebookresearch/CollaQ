## Overview
We propose **CollaQ**, a novel way to decompose Q function for decentralized policy in multi-agent modeling. In StarCraft II Multi-Agent Challenge, CollaQ outperforms existing state-of-the-art techniques (i.e., QMIX, QTRAN, and VDN) by improving the win rate by 40% with the same number of samples. In the more challenging ad hoc team play setting (i.e., reweight/add/remove units without re-training or finetuning), CollaQ outperforms previous SoTA by over 30%. 

Please check our [website](https://sites.google.com/view/collaq-starcraft) for comprehensive results.

3-min [video](http://yuandong-tian.com/collaQ.mp4) for paper introduction. 

Please cite our arXiv [paper](https://arxiv.org/abs/2010.08531) if you use this codebase:

```
@article{zhang2020multi,
  title={Multi-Agent Collaboration via Reward Attribution Decomposition},
  author={Zhang, Tianjun and Xu, Huazhe and Wang, Xiaolong and Wu, Yi and Keutzer, Kurt and Gonzalez, Joseph E and Tian, Yuandong},
  journal={arXiv preprint arXiv:2010.08531},
  year={2020}
}
```

## Installation instructions
```
git clone git@github.com:facebookresearch/CollaQ.git
```
The requirements.txt file can be used to install the necessary packages into a virtual environment with python == 3.6.0 (not recomended).

Install smac and sacred:
```
git submodule sync && git submodule update --init --recursive
cd third_party/sacred
git apply ../sacred.patch
cd ../smac
git apply ../smac.patch
```

## Results
![sc2_standard](/figures/sc2_standard.png?raw=true)
![sc2_vip](/figures/sc2_vip.png?raw=true)
![sc2_sar](/figures/sc2_sar.png?raw=true)


## Run an experiment 
SC2PATH=.../pymarl/StarCraftII

### QMIX
```
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2,
```

### CollaQ
```
python src/main.py --config=qmix_interactive_reg --env-config=sc2 with env_args.map_name=MMM2,
```

### CollaQ with Attn
```
python src/main.py --config=qmix_interactive_reg_attn --env-config=sc2 with env_args.map_name=MMM2,
```

### CollaQ Removing Agents
```
python src/main.py --config=qmix_interactive_reg_attn --env-config=sc2 with env_args.map_name=29m_vs_30m,28m_vs_30m, obs_agent_id=False
```

### CollaQ Removing Agents
```
python src/main.py --config=qmix_interactive_reg_attn --env-config=sc2 with env_args.map_name=27m_vs_30m,28m_vs_30m, obs_agent_id=False
```

### CollaQ Swapping Agents
```
python src/main.py --config=qmix_interactive_reg_attn --env-config=sc2 with env_args.map_name=3s1z_vs_zg_easy, 1s3z_vs_zg_easy,2s2z_vs_zg_easy, obs_agent_id=False
```

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

### Watching Replay
```
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m, evaluate=True checkpoint_path=results/models/5m_vs_6m/... save_replay=True
```

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Acknowledgement

Our vanilla RL algorithm is based on [PyMARL](https://github.com/oxwhirl/pymarl), which is an open source implementation of algorithms in StarCraft II.

## License

This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
