# Test

For image classification task and image retrieval task, you could test your model after training.

## Test with your PC

You can use `tools/test.py` to test a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:

```shell
python tools/neck.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

````{note}
By default, MMPretrain prefers GPU to CPU. If you want to test a model on CPU, please empty `CUDA_VISIBLE_DEVICES` or set it to -1 to make GPU invisible to the program.

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```
````

| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | The path to the config file.                                                                                                                                        |
| `CHECKPOINT_FILE`                     | The path to the checkpoint file (It can be a http link, and you can find checkpoints [here](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html)). |
| `--work-dir WORK_DIR`                 | The directory to save the file containing evaluation metrics.                                                                                                       |
| `--out OUT`                           | The path to save the file containing test results.                                                                                                                  |
| `--out-item OUT_ITEM`                 | To specify the content of the test results file, and it can be "pred" or "metrics". If "pred", save the outputs of the model for offline evaluation. If "metrics", save the evaluation metrics. Defaults to "pred". |
| `--cfg-options CFG_OPTIONS`           | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either `key="[a,b]"` or `key=a,b`. The argument also allows nested list/tuple values, e.g. `key="[(a,b),(c,d)]"`. Note that the quotation marks are necessary and that no white space is allowed. |
| `--show-dir SHOW_DIR`                 | The directory to save the result visualization images.                                                                                                              |
| `--show`                              | Visualize the prediction result in a window.                                                                                                                        |
| `--interval INTERVAL`                 | The interval of samples to visualize.                                                                                                                               |
| `--wait-time WAIT_TIME`               | The display time of every window (in seconds). Defaults to 1.                                                                                                       |
| `--no-pin-memory`                     | Whether to disable the `pin_memory` option in dataloaders.                                                                                                          |
| `--tta`                               | Whether to enable the Test-Time-Aug (TTA). If the config file has `tta_pipeline` and `tta_model` fields, use them to determine the TTA transforms and how to merge the TTA results. Otherwise, use flip TTA by averaging classification score. |
| `--launcher {none,pytorch,slurm,mpi}` | Options for job launcher.                                                                                                                                           |

## Test with multiple GPUs

We provide a shell script to start a multi-GPUs task with `torch.distributed.launch`.

```shell
bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

| ARGS              | Description                                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`     | The path to the config file.                                                                                                                                        |
| `CHECKPOINT_FILE` | The path to the checkpoint file (It can be a http link, and you can find checkpoints [here](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html)). |
| `GPU_NUM`         | The number of GPUs to be used.                                                                                                                                      |
| `[PY_ARGS]`       | The other optional arguments of `tools/test.py`, see [here](#test-with-your-pc).                                                                                    |

You can also specify extra arguments of the launcher by environment variables. For example, change the
communication port of the launcher to 29666 by the below command:

```shell
PORT=29666 bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

If you want to startup multiple test jobs and use different GPUs, you can launch them by specifying
different port and visible devices.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_test.sh ${CONFIG_FILE1} ${CHECKPOINT_FILE} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash ./tools/dist_test.sh ${CONFIG_FILE2} ${CHECKPOINT_FILE} 4 [PY_ARGS]
```

## Test with multiple machines

### Multiple machines in the same network

If you launch a test job with multiple machines connected with ethernet, you can run the following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

Comparing with multi-GPUs in a single machine, you need to specify some extra environment variables:

| ENV_VARS      | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `NNODES`      | The total number of machines.                                                |
| `NODE_RANK`   | The index of the local machine.                                              |
| `PORT`        | The communication port, it should be the same in all machines.               |
| `MASTER_ADDR` | The IP address of the master machine, it should be the same in all machines. |

Usually it is slow if you do not have high speed networking like InfiniBand.

### Multiple machines managed with slurm

If you run MMPretrain on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `tools/slurm_test.sh`.

```shell
[ENV_VARS] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]
```

Here are the arguments description of the script.

| ARGS              | Description                                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PARTITION`       | The partition to use in your cluster.                                                                                                                               |
| `JOB_NAME`        | The name of your job, you can name it as you like.                                                                                                                  |
| `CONFIG_FILE`     | The path to the config file.                                                                                                                                        |
| `CHECKPOINT_FILE` | The path to the checkpoint file (It can be a http link, and you can find checkpoints [here](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html)). |
| `[PY_ARGS]`       | The other optional arguments of `tools/test.py`, see [here](#test-with-your-pc).                                                                                    |

Here are the environment variables can be used to configure the slurm job.

| ENV_VARS        | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `GPUS`          | The number of GPUs to be used. Defaults to 8.                                                              |
| `GPUS_PER_NODE` | The number of GPUs to be allocated per node.                                                               |
| `CPUS_PER_TASK` | The number of CPUs to be allocated per task (Usually one GPU corresponds to one task). Defaults to 5.      |
| `SRUN_ARGS`     | The other arguments of `srun`. Available options can be found [here](https://slurm.schedmd.com/srun.html). |
