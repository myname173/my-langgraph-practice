"""Periodically poll thread 030b4b82 state and append progress to _monitor_030b.log.

Run in background; safe to leave unattended. Detects node transitions, terminal
state (final_movie_path / error_log), and run-level status.
"""
import time, json, sys, os
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
TID = "030b4b82-ea0d-48e4-a1b9-38e9d7af239b"
LOG = os.path.join(os.path.dirname(__file__), "..", "workspace", "_monitor_030b.log")
LOG = os.path.abspath(LOG)
LAST_NODE = None


def get(path):
    url = "http://127.0.0.1:2024" + path
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return "ERR", str(e)[:120]


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    log("monitor start")
    while True:
        s, st = get(f"/threads/{TID}/state")
        if s != 200:
            log(f"state err {s}: {st}")
            time.sleep(30)
            continue
        vals = st.get("values", {})
        nextn = st.get("next")
        tasks = st.get("tasks", [])
        final = vals.get("final_movie_path")
        err = vals.get("error_log")
        cur_node = nextn[0] if nextn else ("<terminal>" if (final or err) else "<none>")
        if cur_node != LAST_NODE:
            LAST_NODE = cur_node
            log(f"node -> {cur_node} | step={st.get('metadata', {}).get('step')}")
        # terminal
        if final:
            log(f"DONE final_movie_path={final}")
            break
        if err:
            log(f"ERROR error_log={err}")
            break
        # detect running runs
        s2, runs = get(f"/threads/{TID}/runs")
        if s2 == 200:
            statuses = [r.get("status") for r in runs]
            if not any(x in ("running", "pending") for x in statuses) and not nextn:
                log(f"no active run and no next node; statuses={statuses}")
                time.sleep(15)
                # re-check once more before concluding
                s3, st3 = get(f"/threads/{TID}/state")
                if s3 == 200 and not st3.get("values", {}).get("final_movie_path") and not st3.get("values", {}).get("error_log") and not st3.get("next"):
                    log("STALLED: no active run, no next node, no terminal. Needs manual resume.")
                    break
        time.sleep(30)
    log("monitor stop")
