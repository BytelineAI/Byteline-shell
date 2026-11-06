"""Run from the shell folder with: python tests/simulate_feed.py
Runs the agents' real decision code for hours on a fake clock, sharing one slow model, to see whether
they get stuck, keep starting new posts and use the whole feed. Needs no server or model, and posts
nothing. It proves the decisions, not the wording: only real posts on the live site can prove that."""
import contextlib, io, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    from test_shell import app

app.log = lambda message: None
# what the live bots run with
app.POST_INTERVAL_MIN, app.POST_INTERVAL_MAX = 30.0, 70.0
LOOKBACK = app.CONTEXT_LOOKBACK
NAMES = ["Smoke1", "Smoke2.1", "Smoke3.1", "Smoke4.1", "Smoke5", "Smoke6", "Smoke7", "Smoke8"]


class Agent:
    def __init__(self, name, rng):
        self.name = name
        self.answered = set()
        self.engagement = app.ENGAGEMENT_START
        self.mode = "rest"
        self.rest_until = rng.uniform(0, 30)
        self.next_glance = 8.0
        self.job = None

    def rest(self, t, seconds):
        self.mode, self.rest_until, self.next_glance = "rest", t + seconds, t + app.ENGAGEMENT_POLL_SECONDS


def simulate(count, hours, seed, wipe_every=None):
    random.seed(seed)
    rng = random.Random(seed)
    agents = [Agent(n, rng) for n in NAMES[:count]]
    feed, history, queue, writing = [], [], [], None
    next_id, next_uid = 1, 1
    silences_reset = []
    model_waits = []

    def window():
        return feed[-LOOKBACK:]

    def tick(a, t):
        recent = window()
        if a.answered and (not recent or max(a.answered) > recent[-1]["id"]):
            a.answered.clear()
        if recent and recent[-1]["agent_name"] == a.name and t - recent[-1]["created_at"] < app.SELF_REPLY_WAIT:
            a.rest(t, random.uniform(app.POST_INTERVAL_MIN, app.POST_INTERVAL_MAX))
            return
        pulling = app.pulled_by(recent, a.name, a.answered)
        a.engagement = app.next_engagement(a.engagement, bool(pulling))
        action, target = app.choose_action(recent, a.name, a.answered, pulling, t)
        by_id = {p["id"]: p for p in recent}
        target_thread = app.thread_id(target, by_id) if target else None
        a.answered.update(p["id"] for p in pulling if action != "reply" or app.thread_id(p, by_id) == target_thread)
        if action == "idle":
            a.rest(t, app.pick_pause(a.engagement)[0])
            return
        if target:
            a.answered.add(target["id"])
        a.mode, a.job = "queued", {"agent": a, "action": action, "target": target, "read_at": t}
        queue.append(a.job)

    for t in range(0, int(hours * 3600)):
        if wipe_every and t and t % wipe_every == 0:
            feed.clear()
            next_id = 1
            silences_reset.append(t)
        for a in agents:
            if a.mode == "rest":
                if t >= a.rest_until:
                    tick(a, t)
                elif t >= a.next_glance:
                    a.next_glance = t + app.ENGAGEMENT_POLL_SECONDS
                    if app.pulled_by(window(), a.name, a.answered):
                        tick(a, t)
        if writing is None and queue:
            # the platform's queue: random order, except that anyone kept waiting too long goes first
            overdue = [j for j in queue if t - j["read_at"] >= 90]
            writing = min(overdue, key=lambda j: j["read_at"]) if overdue else rng.choice(queue)
            queue.remove(writing)
            model_waits.append(t - writing["read_at"])
            writing["done"] = t + rng.uniform(20, 30)
        if writing and t >= writing["done"]:
            job, writing = writing, None
            a = job["agent"]
            # about what the live logs show: the odd empty reply or one blocked by the filter
            if rng.random() < 0.05 or (job["action"] == "new" and app.fresh_post_is_up(window(), t)):
                a.rest(t, rng.uniform(5, 10))
            else:
                target = job["target"]
                named = f"{target['agent_name']}, " if target and rng.random() < 0.65 else ""
                job["text"] = named + "word " * rng.randint(12, 30)
                job["post_at"] = t + app.typing_hold(job["text"], t - job["read_at"])
                a.mode, a.job = "typing", job
        for a in agents:
            if a.mode == "typing" and t >= a.job["post_at"]:
                job = a.job
                target = job["target"]
                parent = None
                if target:
                    parent = next((p for p in feed if p["id"] == target["id"] and p["uid"] == target["uid"]), None)
                own_posts = [p for p in feed if p["agent_name"] == a.name and p["parent_id"] is None]
                # the platform's own rules, so a slip in the agent's choices can't hide behind them
                closed = bool(parent) and t - next(p for p in feed if p["uid"] == parent["root_uid"])["created_at"] >= app.POST_OPEN_SECONDS
                refused = closed or (target and parent is None) or (not target and own_posts and t - own_posts[-1]["created_at"] < app.NEW_POST_COOLDOWN_SECONDS)
                if not refused:
                    post = {
                        "id": next_id, "uid": next_uid, "agent_name": a.name, "parent_id": parent["id"] if parent else None,
                        "created_at": float(t), "content": job["text"], "vote_score": 0,
                        "root_uid": parent["root_uid"] if parent else next_uid,
                    }
                    feed.append(post)
                    history.append(post)
                    next_id += 1
                    next_uid += 1
                a.rest(t, app.pick_pause(a.engagement)[0])
    return history, silences_reset, model_waits


def report(label, count, hours, wipe_every=None):
    failures = []
    all_comments, all_alive, all_gaps, all_silences, per_hour, top_shares, unanswered, spreads = [], [], [], [], [], [], [], []
    worst_cap = worst_cooldown_gap = worst_new_gap = worst_old = 0
    worst_cooldown_gap = 10 ** 9
    all_started, open_at_once, all_waits = [], [], []
    worst_new_gap = 10 ** 9
    for seed in range(3):
        history, wipes, waits = simulate(count, hours, seed, wipe_every)
        all_waits += waits
        roots = [p for p in history if p["parent_id"] is None]
        threads = {r["uid"]: [p for p in history if p["root_uid"] == r["uid"] and p["parent_id"] is not None] for r in roots}
        per_hour.append((len(history) / hours, len(roots) / hours))
        for r in roots:
            comments = threads[r["uid"]]
            all_comments.append(len(comments))
            if not any(c["agent_name"] != r["agent_name"] for c in comments) and r["created_at"] < hours * 3600 - 900:
                unanswered.append(r)
            if comments:
                all_alive.append(comments[-1]["created_at"] - r["created_at"])
                by = {}
                for c in comments:
                    by[c["agent_name"]] = by.get(c["agent_name"], 0) + 1
                worst_cap = max(worst_cap, max(by.values()))
        for name in NAMES[:count]:
            mine = [r["created_at"] for r in roots if r["agent_name"] == name]
            for a, b in zip(mine, mine[1:]):
                if not any(a < w <= b for w in wipes):
                    worst_cooldown_gap = min(worst_cooldown_gap, b - a)
        # how soon every agent has made a subject of its own, and how many posts are open at a time
        firsts = {}
        for r in roots:
            firsts.setdefault(r["agent_name"], r["created_at"])
        all_started.append(max(firsts.values()) / 60 if len(firsts) == count else float("inf"))
        for sample in range(1800, int(hours * 3600), 300):
            alive = [w for w in wipes if w <= sample]
            since = alive[-1] if alive else 0
            open_at_once.append(sum(1 for r in roots if since <= r["created_at"] <= sample and sample - r["created_at"] < app.POST_OPEN_SECONDS))
        times = [r["created_at"] for r in roots]
        new_gaps = [b - a for a, b in zip(times, times[1:]) if not any(a < w <= b for w in wipes)]
        all_gaps += new_gaps
        if new_gaps:
            worst_new_gap = min(worst_new_gap, min(new_gaps))
        stamps = [p["created_at"] for p in history]
        all_silences += [b - a for a, b in zip(stamps, stamps[1:]) if not any(a < w <= b for w in wipes)]
        # in each 20 minutes, how many posts were being talked in, and how much went to the busiest one
        for start in range(0, int(hours * 3600), 1200):
            inside = [p for p in history if start <= p["created_at"] < start + 1200 and p["parent_id"] is not None]
            if len(inside) >= 5:
                by = {}
                for c in inside:
                    by[c["root_uid"]] = by.get(c["root_uid"], 0) + 1
                top_shares.append(max(by.values()) / len(inside))
                spreads.append(len(by))
        for r in history:
            if r["parent_id"] is not None:
                root = next(x for x in roots if x["uid"] == r["root_uid"])
                worst_old = max(worst_old, r["created_at"] - root["created_at"])

    print(f"\n== {label}: {count} agents, {hours} hours, 3 runs")
    print(f"   posts an hour {statistics.mean(x[0] for x in per_hour):.0f}, of which new posts {statistics.mean(x[1] for x in per_hour):.1f}")
    print(f"   every agent has made a subject after: typically {statistics.median(all_started):.0f} min, {max(all_started):.0f} min at worst. Posts open at once: median {statistics.median(open_at_once):.0f}, fewest {min(open_at_once)}, most {max(open_at_once)}")
    print(f"   waiting in line for the model: median {statistics.median(all_waits):.0f}s, slowest {max(all_waits):.0f}s")
    print(f"   comments per post: median {statistics.median(all_comments):.0f}, most {max(all_comments)}, posts nobody else answered {len(unanswered)} of {len(all_comments)}")
    print(f"   a post stays talked about: median {statistics.median(all_alive) / 60:.0f} min, longest {max(all_alive) / 60:.0f} min")
    print(f"   time between new posts: median {statistics.median(all_gaps) / 60:.0f} min, shortest {worst_new_gap / 60:.1f} min, longest {max(all_gaps) / 60:.0f} min")
    print(f"   gaps between any posts: median {statistics.median(all_silences):.0f}s, under 40s {100 * sum(s < 40 for s in all_silences) / len(all_silences):.0f}%, over 2 min {100 * sum(s > 120 for s in all_silences) / len(all_silences):.0f}%, longest silence {max(all_silences) / 60:.1f} min")
    print(f"   in any 20 minutes: {statistics.median(spreads):.0f} posts being talked in, busiest gets {100 * statistics.median(top_shares):.0f}% of comments")
    checks = [
        ("every agent makes a subject soon after starting", max(all_started) <= max(15, count * 4), f"{max(all_started):.0f} min at worst"),
        ("several subjects are up at once", statistics.median(open_at_once) >= min(count, 3), f"median {statistics.median(open_at_once):.0f}"),
        ("no agent goes over its limit in one post", worst_cap <= app.THREAD_STAY_REPLIES, f"most {worst_cap}"),
        ("no post can trap everyone", max(all_comments) <= count * app.THREAD_STAY_REPLIES, f"most {max(all_comments)}, limit {count * app.THREAD_STAY_REPLIES}"),
        ("nothing lands on a post after its 30 minutes", worst_old < app.POST_OPEN_SECONDS, f"latest {worst_old / 60:.1f} min"),
        ("no agent posts inside its 30 minutes", worst_cooldown_gap >= app.NEW_POST_COOLDOWN_SECONDS, f"shortest {worst_cooldown_gap / 60:.1f} min"),
        ("new posts keep coming", max(all_gaps) <= 40 * 60, f"longest gap {max(all_gaps) / 60:.0f} min"),
        ("no burst of new posts", worst_new_gap >= app.FRESH_POST_SECONDS, f"shortest gap {worst_new_gap / 60:.1f} min"),
        # with few agents, the 30-minute limit on new posts is itself what makes the quiet spells
        ("the feed never goes dead", max(all_silences) <= (20 if count >= 4 else 30) * 60, f"longest silence {max(all_silences) / 60:.1f} min"),
    ]
    for name, ok, detail in checks:
        print(("   PASS " if ok else "   FAIL ") + f"{name}  [{detail}]")
        if not ok:
            failures.append(name)
    return failures


if __name__ == "__main__":
    failed = []
    if "quick" in sys.argv:
        report("Normal", 4, 2)
        sys.exit(0)
    failed += report("Normal", 4, 4)
    failed += report("Posts cleared every hour", 4, 4, wipe_every=3600)
    failed += report("Only two agents", 2, 4)
    failed += report("Full site", 8, 4)
    print("\nRESULT:", "ALL PASSED" if not failed else f"{len(failed)} FAILED")
