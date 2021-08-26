import argparse
import os
import uuid
from pathlib import Path

import train_3D_DINO_transformer_distributed
from train_3D_DINO_transformer_distributed import adjust_paths
from scripts.helper import load_from_json
import submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit for DINO", parents=[train_3D_DINO_transformer_distributed.get_args()])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/scratch/").is_dir():
        p = Path(f"/scratch/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import train_3D_DINO_transformer_distributed

        self._setup_gpu_args()
        train_3D_DINO_transformer_distributed.train_net(self.args.cat_to_idx, self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.cp_dir = Path(str(self.args.cp_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    adjust_paths(args, exceptions=['dist_url'])

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.num_class = len(cat_to_idx)
    args.cat_to_idx = cat_to_idx

    if args.cp_dir == "":
        args.cp_dir = get_shared_folder() / "%j"
    Path(args.cp_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.cp_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    #partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=11 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        additional_parameters={'account': 'rrg-msavva'},
        # Below are cluster dependent parameters
        #slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="3D_pc_embedding")

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.cp_dir}")


if __name__ == "__main__":
    main()
