"""
Helper to run commands on the HPC cluster via SSH jump host.
Usage:
  Set HPC_USER and HPC_PASS environment variables, then:
  python hpc/remote_runner.py "command to run"
"""
import os
import sys
import paramiko

JUMP_HOST = "ssh.iesci.tech"
HPC_HOST = "10.205.20.10"
USERNAME = os.environ.get("HPC_USER", "")
PASSWORD = os.environ.get("HPC_PASS", "")


def run_remote(command: str, timeout: int = 600) -> tuple[str, str, int]:
    """Execute a command on the HPC cluster and return (stdout, stderr, exit_code)."""
    if not USERNAME or not PASSWORD:
        raise RuntimeError("Set HPC_USER and HPC_PASS environment variables")

    jump = paramiko.SSHClient()
    jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jump.connect(JUMP_HOST, username=USERNAME, password=PASSWORD, timeout=15)

    transport = jump.get_transport()
    channel = transport.open_channel("direct-tcpip", (HPC_HOST, 22), ("127.0.0.1", 0))

    hpc = paramiko.SSHClient()
    hpc.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    hpc.connect(HPC_HOST, username=USERNAME, password=PASSWORD, sock=channel, timeout=15)

    stdin, stdout, stderr = hpc.exec_command(command, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    exit_code = stdout.channel.recv_exit_status()

    hpc.close()
    jump.close()
    return out, err, exit_code


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hostname && whoami"
    out, err, code = run_remote(cmd)
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    sys.exit(code)
