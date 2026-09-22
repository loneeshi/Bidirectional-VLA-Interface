"""Run a frozen C2 panel in at most five two-episode chunks; stop on any fault."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    args = sys.argv[1:]
    output = Path(args[args.index('--output') + 1])
    for _ in range(5):
        subprocess.run([sys.executable, str(Path(__file__).with_name('run_c2_feedback.py')),
                        *args], check=True)
        state = json.loads((output / 'panel-status.json').read_text())
        rows = state['episodes']
        if any(r['status'] not in ('completed', 'not_run') for r in rows):
            raise SystemExit('Stopped: inspect preserved non-completed attempt before resuming')
        if all(r['status'] == 'completed' for r in rows):
            return
    raise SystemExit('Chunk limit reached with unfinished rows')


if __name__ == '__main__':
    main()
