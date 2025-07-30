import sys
import os

# daemonizing the agent
def daemonize():
    # forking the process, killing the parent process, and then working within a child only
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f'First fork failed: {e}\n')
        sys.exit(1)

    # creates a new session
    os.setsid()

    # again forking the process
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f'Second fork failed: {e}\n')
        sys.exit(1)

    os.umask(0)

    # redictering all outputs
    sys.stdout.flush()
    sys.stderr.flush()

    with open(os.devnull, 'rb') as dev_null:
        os.dup2(dev_null.fileno(), sys.stdin.fileno())

    with open(f'{rc("path")}/{rc("daemon_log")}', 'ab', buffering=0) as log_file:
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

    sys.stdout = open(f'{rc("path")}/{rc("daemon_log")}', 'a')
    sys.stderr = open(f'{rc("path")}/{rc("daemon_log")}', 'a')

    # the agent
    os.umask(0o077)
    start_agent()

# start a semi-sleeping process
def start_agent():
    while True:
        write_agent('iteration_start', None)

        update_files('aut_update')

        rs.restart_zombies()

        write_agent('iteration_end', None)

        time.sleep(rc('period'))

# kill agent-associated processes; TO-DO: improve, sometimes does not work
def stop_agent():
    pids = os.popen("ps aux | grep 'EvoMD' | grep -v grep | awk '{print $2}'").read().split()
    for pid in pids:
        os.system(f'kill -9 {pid}')

if __name__ == '__main__':
    pass
