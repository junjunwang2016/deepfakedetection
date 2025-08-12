import os, torch
import torch.distributed as dist

def init_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank = 0; args.world_size = 1; args.local_rank = 0
    args.distributed = args.world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        dist.barrier()

def is_main():
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or (torch.distributed.get_rank()==0)

def ddp_print(*msg, **kw):
    if is_main(): print(*msg, **kw)

def all_reduce_sum(t):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t

def broadcast_scalar(x: float, device):
    t = torch.tensor([float(x)], device=device)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(t, src=0)
    return float(t.item())

def all_gather_list(pylist):
    if not torch.distributed.is_initialized():
        return [pylist]
    gather = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gather, pylist)
    return gather
