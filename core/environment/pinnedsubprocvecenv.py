import psutil
import re
import stable_baselines


class PinnedSubprocVecEnv(stable_baselines.common.vec_env.SubprocVecEnv):

    def __init__(self, env_fns, *, start_method=None, cpu_cores=None):
        if cpu_cores is None:
            cpu_cores = available_cpu_cores()
        else:
            cpu_cores = [cpu_core.__index__() for cpu_core in cpu_cores]

        super().__init__(env_fns=env_fns, start_method=start_method)

        if len(cpu_cores) < len(self.processes):
            # or raise error?
            cpu_partitions = [[cpu_core] for cpu_core in
                              (len(self.processes) // len(cpu_cores)) * cpu_cores + cpu_cores[
                                                                                    :len(self.processes) % len(
                                                                                        cpu_cores)]]
        else:
            cpu_partitions = []
            start = 0
            for worker_idx, _ in enumerate(self.processes):
                end = start + len(cpu_cores) // len(self.processes)
                if worker_idx < len(cpu_cores) % len(self.processes):
                    end += 1
                cpu_partitions.append(cpu_cores[start:end])
                start = end

        assert len(self.processes) == len(cpu_partitions)

        for worker_proc, cpu_partition in zip(self.processes, cpu_partitions):
            psutil.Process(worker_proc.pid).cpu_affinity(cpu_partition)


def available_cpu_cores():
    try:
        proc_status_file = open('/proc/%d/status' % psutil.Process().pid)
    except FileNotFoundError:
        raise OSError('system does not support procfs')
    else:
        for line in proc_status_file.readlines():
            match = re.search(
                r'^\s*Cpus_allowed_list\s*:(\s*[0-9]+(\s*\-\s*[0-9]+)?\s*(,\s*[0-9]+(\s*\-\s*[0-9]+)?\s*)?)$',
                line
            )

            if match:
                cpu_cores = []
                for part in match.group(1).split(','):
                    part = [int(n) for n in part.split('-')]
                    if len(part) == 1:
                        cpu_cores.extend(part)
                    elif len(part) == 2:
                        a, b = part
                        cpu_cores.extend(range(a, b + 1))
                    else:
                        raise RuntimeError
                return cpu_cores

    raise RuntimeError