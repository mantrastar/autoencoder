# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_pegasus.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

from transformers import PegasusForCausalLM, PegasusConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(pegasus, nranks):
    layers_per_rank = pegasus.config.num_hidden_layers // nranks
    for i in range(1, nranks):
        annotate_split_points(
            pegasus, {f"model.decoder.layers.{i * layers_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING})


def run(args):
    # Model configs
    config = PegasusConfig()
    print("Using device:", args.device)

    # Create model
    model_class = PegasusForCausalLM
    model_name = "PegasusForCausalLM"
    pegasus = model_class(config)
    pegasus.to(args.device)
    pegasus.eval()
    if args.rank == 0:
        print(pegasus.config)
        print(f"Total number of params = {get_number_of_params(pegasus) // 10 ** 6}M")
        print(pegasus)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, pegasus, model_name, args.batch_size, args.device)
    input_ids = example_inputs["input_ids"]

    # Annotate split points
    add_split_points(pegasus, args.world_size)

    # Create pipeline
    pegasus_pipe = Pipe.from_tracing(
        pegasus,
        num_chunks=args.chunks,
        example_args=(input_ids, ),
    )
    nstages = len(list(pegasus_pipe.split_gm.children()))
    assert nstages == args.world_size, f"nstages = {nstages} nranks = {args.world_size}"
    if args.rank == 0:
        for i, sm in enumerate(pegasus_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        pegasus_pipe,
        args.rank,
        device=args.device,
    )

    # Run
    if args.rank == 0:
        stage(input_ids)
    elif args.rank == args.world_size - 1:
        out = stage()
    else:
        stage()

    print(f"Rank {args.rank} completes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=1)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)
