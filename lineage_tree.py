# lineage_tree.py
import csv
from collections import defaultdict, deque

LINEAGE_EDGES = "lineage_edges.csv"   # tick,parent_id,child_id
AGENT_LIFE    = "agent_life.csv"      # optional: agent_id,born_tick,death_tick,...

TOP_K_ROOTS = 5       # "prominent dynasties"
MIN_SUBTREE = 999999999         # OR keep nodes with >= this many descendants
USE_PLOTLY  = True        # else matplotlib

def load_edges(path):
    parent = {}
    born_tick = {}  # child -> tick from edges
    children = defaultdict(list)

    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["tick"])
            p = int(row["parent_id"])
            c = int(row["child_id"])
            parent[c] = p
            born_tick[c] = t
            children[p].append(c)
    return parent, children, born_tick

def load_life(path):
    life = {}
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                aid = int(row["agent_id"])
                bt = row.get("born_tick", "")
                dt = row.get("death_tick", "")
                life[aid] = {
                    "born_tick": int(float(bt)) if bt != "" else None,
                    "death_tick": int(float(dt)) if dt != "" else None,
                    "kills_total": int(float(row.get("kills_total","0") or "0")),
                }
    except FileNotFoundError:
        return None
    return life

def compute_subtree_sizes(children, nodes):
    # postorder without recursion (safer for big trees)
    size = {n: 1 for n in nodes}
    order = []
    stack = [n for n in nodes if n in children]  # not perfect, but ok

    # Build a reverse-topological order using explicit stack
    seen = set()
    st = [(n, 0) for n in nodes]  # (node, state 0=enter 1=exit)
    while st:
        n, state = st.pop()
        if state == 0:
            if n in seen: 
                continue
            seen.add(n)
            st.append((n, 1))
            for c in children.get(n, []):
                st.append((c, 0))
        else:
            order.append(n)

    for n in order:
        s = 1
        for c in children.get(n, []):
            if c in size:
                s += size[c]
        size[n] = s
    return size

def roots_from_edges(parent_map):
    # nodes that are parents but not known as children, plus parents whose parent not known
    all_children = set(parent_map.keys())
    all_parents = set(parent_map.values())
    roots = [p for p in all_parents if p not in all_children]
    return roots

def select_prominent(children, parent_map, subtree_size):
    roots = roots_from_edges(parent_map)
    roots_sorted = sorted(roots, key=lambda r: subtree_size.get(r, 1), reverse=True)

    keep = set()

    # Strategy 1: top K roots
    for r in roots_sorted[:TOP_K_ROOTS]:
        keep.add(r)

    # Strategy 2: any node with big subtree
    for n, s in subtree_size.items():
        if s >= MIN_SUBTREE:
            keep.add(n)

    # Expand: include descendants of kept nodes
    q = deque(list(keep))
    while q:
        n = q.popleft()
        for c in children.get(n, []):
            if c not in keep:
                keep.add(c)
                q.append(c)

    # Expand: include ancestors so structure stays connected
    expanded = set(keep)
    for n in list(keep):
        cur = n
        while cur in parent_map:
            cur = parent_map[cur]
            expanded.add(cur)
    return expanded

def assign_y(children, roots, keep):
    y = {}
    cur = 0
    stack = []
    for r in roots:
        if r in keep:
            stack.append(r)
            while stack:
                n = stack.pop()
                if n in y: 
                    continue
                y[n] = cur
                cur += 1
                # push children reverse for stable ordering
                cs = [c for c in children.get(n, []) if c in keep]
                for c in reversed(cs):
                    stack.append(c)
    return y

def main():
    parent_map, children, born_from_edges = load_edges(LINEAGE_EDGES)
    life = load_life(AGENT_LIFE)

    nodes = set(children.keys()) | set(parent_map.keys()) | set(parent_map.values())
    subtree_size = compute_subtree_sizes(children, nodes)

    keep = select_prominent(children, parent_map, subtree_size)
    roots = roots_from_edges(parent_map)
    y = assign_y(children, roots, keep)

    # Determine born/death ticks
    max_tick_seen = max(born_from_edges.values()) if born_from_edges else 0

    def born(a):
        if life and a in life and life[a]["born_tick"] is not None:
            return life[a]["born_tick"]
        return born_from_edges.get(a, None)

    def death(a):
        if life and a in life and life[a]["death_tick"] is not None:
            return life[a]["death_tick"]
        # fallback: alive until end of run (approx)
        return max_tick_seen

    # Render (choose Plotly for huge; Matplotlib ok for medium)
    if USE_PLOTLY:
        import plotly.graph_objects as go
        bars_x0, bars_x1, bars_y, text = [], [], [], []
        for a in keep:
            if a not in y: 
                continue
            bt = born(a)
            if bt is None:
                continue
            dt = death(a)
            bars_x0.append(bt); bars_x1.append(dt); bars_y.append(y[a])
            text.append(f"agent {a}: {bt}–{dt} | subtree={subtree_size.get(a,1)}")

        fig = go.Figure()

        # lifespans as thick line segments
        fig.add_trace(go.Scattergl(
            x=[v for pair in zip(bars_x0, bars_x1, [None]*len(bars_x0)) for v in pair],
            y=[v for pair in zip(bars_y,  bars_y,  [None]*len(bars_y))  for v in pair],
            mode="lines",
            hoverinfo="text",
            text=[t for t in text for _ in (0,1,2)],
        ))

        # parent->child connectors at birth time
        xs, ys = [], []
        for c in keep:
            if c not in y or c not in parent_map:
                continue
            p = parent_map[c]
            if p not in y or p not in keep:
                continue
            bt = born(c)
            if bt is None:
                continue
            xs += [bt, bt, None]
            ys += [y[p], y[c], None]
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", hoverinfo="skip"))

        fig.update_layout(
            title="Prominent lineage time-tree",
            xaxis_title="tick",
            yaxis_title="lineage order",
            height=max(600, int(len(y) * 2.5)),
        )
        fig.write_html("lineage_time_tree.html")
        print("Wrote lineage_time_tree.html")
    else:
        print("Implement matplotlib rendering here (Plotly recommended for big trees).")

if __name__ == "__main__":
    main()
