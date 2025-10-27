# csp_timetable.py
# Solves the timetable CSP using:
#  - Method A: Backtracking with MRV + LCV
#  - Method B: Backtracking with Forward Checking (+ MRV+LCV ordering)
#
# Usage:
#   python csp_timetable.py --out csp_metrics.csv
#
# Produces csp_metrics.csv summarizing both modes.

import time
import csv
import itertools
import random
from collections import defaultdict, namedtuple

Course = namedtuple("Course", ["id","teacher","groups","size"])
Room = namedtuple("Room", ["id","capacity"])

# Sample dataset (you can expand or replace)
def sample_problem():
    # groups: G1,G2,G3
    courses = [
        Course("CS101","T1", ["G1","G2"], 40),
        Course("MA101","T2", ["G1"], 30),
        Course("PH101","T3", ["G2"], 25),
        Course("CS102","T1", ["G3"], 35),
        Course("EE101","T2", ["G3","G1"], 45),
        Course("HS101","T4", ["G2"], 20),
        Course("ME201","T5", ["G1","G3"], 50),
        Course("CH201","T6", ["G2","G3"], 40),
        Course("CS201","T1", ["G1"], 30),
        Course("MA201","T2", ["G2"], 35),
        Course("PH201","T3", ["G3"], 25),
        Course("EE201","T2", ["G1","G2"], 45),
        Course("CS301","T1", ["G2","G3"], 40),
        Course("HS201","T4", ["G1"], 20),
        Course("ME301","T5", ["G3"], 50)
    ]
    timeslots = ["Mon_9","Mon_11","Tue_9","Tue_11","Wed_9"]
    rooms = [Room("R1",50), Room("R2",40), Room("R3",30)]
    return courses, timeslots, rooms

# Build initial domains: all (timeslot, room) pairs where room.capacity >= course.size
def build_domains(courses, timeslots, rooms):
    domains = {}
    for c in courses:
        dom = []
        for ts in timeslots:
            for r in rooms:
                if r.capacity >= c.size:
                    dom.append((ts,r.id))
        domains[c.id] = dom
    return domains

# Constraint check: given partial assignment, check if assigning (ts,room) to course_id violates constraints
def violates(assignment, course_by_id, course_id, value):
    ts, room = value
    c = course_by_id[course_id]
    for other_id, (ots, oroom) in assignment.items():
        if ots != ts:
            continue
        other = course_by_id[other_id]
        # same teacher at same timeslot?
        if other.teacher == c.teacher:
            return True
        # group overlap?
        if set(other.groups).intersection(set(c.groups)):
            return True
        # same room at same timeslot?
        if oroom == room:
            return True
    return False

# MRV: variable with smallest domain (consider current domains)
def select_mrv(unassigned, domains):
    best = None
    best_size = 10**9
    for var in unassigned:
        s = len(domains[var])
        if s < best_size:
            best_size = s
            best = var
    return best

# LCV: order values by number of eliminations they cause (ascending)
def lcv_order(var, domains, unassigned, courses_by_id):
    vals = domains[var]
    scored = []
    for v in vals:
        elim = 0
        for other in unassigned:
            if other == var: continue
            for ov in domains[other]:
                # if ov would conflict with v w.r.t. timeslot/teacher/group/room, count elimination
                if ov[0] != v[0]:
                    continue
                # same room?
                if ov[1] == v[1]:
                    elim += 1; break
                # same teacher or group => can't count without knowing teachers/groups
                # but we can approximate conservatively:
                co = courses_by_id[var]
                co2 = courses_by_id[other]
                if co.teacher == co2.teacher or set(co.groups).intersection(set(co2.groups)):
                    elim += 1; break
        scored.append((elim, v))
    scored.sort(key=lambda x: x[0])
    return [v for _,v in scored]

# Backtracking search (can enable forward_checking)
def backtrack_search(courses, domains_init, forward_checking=False, time_limit_ms=20000):
    start = time.perf_counter()
    course_ids = [c.id for c in courses]
    courses_by_id = {c.id:c for c in courses}
    domains = {k:list(v) for k,v in domains_init.items()}
    assignment = {}
    backtracks = 0
    tries = 0
    time_limit_s = time_limit_ms/1000.0

    def timeout():
        return (time.perf_counter() - start) > time_limit_s

    def recurse():
        nonlocal backtracks, tries
        if timeout():
            return False
        if len(assignment) == len(course_ids):
            return True
        unassigned = [v for v in course_ids if v not in assignment]
        var = select_mrv(unassigned, domains)
        if var is None:
            return False
        ordered_vals = lcv_order(var, domains, unassigned, courses_by_id)
        if not ordered_vals:
            backtracks += 1
            return False
        for val in ordered_vals:
            tries += 1
            if violates(assignment, courses_by_id, var, val):
                continue
            # assign
            assignment[var] = val
            # forward check: prune domains of neighbors
            domains_backup = None
            failure = False
            if forward_checking:
                domains_backup = {k:list(v) for k,v in domains.items()}
                for other in unassigned:
                    if other == var: continue
                    newdom = []
                    for ov in domains[other]:
                        if not violates({**assignment}, courses_by_id, other, ov):
                            newdom.append(ov)
                    domains[other] = newdom
                    if len(newdom) == 0:
                        failure = True
                        break
            if not failure:
                if recurse():
                    return True
            if forward_checking and domains_backup is not None:
                domains = domains_backup
                # To handle reliably, we'll mutate back:
                for k in domains_backup:
                    domains[k] = domains_backup[k]
            assignment.pop(var, None)
        backtracks += 1
        return False

    assignment.clear()
    domains = {k:list(v) for k,v in domains_init.items()}
    backtracks = 0
    tries = 0

    def recurse2():
        nonlocal backtracks, tries, domains
        if timeout():
            return False
        if len(assignment) == len(course_ids):
            return True
        unassigned = [v for v in course_ids if v not in assignment]
        var = select_mrv(unassigned, domains)
        if var is None:
            return False
        ordered_vals = lcv_order(var, domains, unassigned, courses_by_id)
        if not ordered_vals:
            backtracks += 1
            return False
        for val in ordered_vals:
            tries += 1
            if violates(assignment, courses_by_id, var, val):
                continue
            # assign
            assignment[var] = val
            # forward check: save and prune
            saved = {}
            failure = False
            if forward_checking:
                for other in unassigned:
                    if other == var: continue
                    saved[other] = list(domains[other])
                    newdom = []
                    for ov in domains[other]:
                        if not violates({**assignment}, courses_by_id, other, ov):
                            newdom.append(ov)
                    domains[other] = newdom
                    if len(newdom) == 0:
                        failure = True
                        break
            if not failure:
                if recurse2():
                    return True
            # restore
            for k,v in saved.items():
                domains[k] = v
            assignment.pop(var, None)
        backtracks += 1
        return False

    ok = recurse2()
    t_ms = (time.perf_counter() - start) * 1000.0
    return {"found": ok, "time_ms": t_ms, "backtracks": backtracks, "assignments_tried": tries, "solution": dict(assignment)}

def run_csp_experiments(out_csv="csp_metrics.csv"):
    courses, timeslots, rooms = sample_problem()
    domains = build_domains(courses, timeslots, rooms)
    # Mode 0: MRV+LCV (no forward checking)
    r0 = backtrack_search(courses, domains, forward_checking=False)
    # Mode 1: Forward Checking (+ MRV+LCV)
    r1 = backtrack_search(courses, domains, forward_checking=True)
    with open(out_csv,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode","found","time_ms","backtracks","assignments_tried"])
        writer.writeheader()
        writer.writerow({"mode":"MRV+LCV","found":int(r0["found"]),"time_ms":"{:.4f}".format(r0["time_ms"]),"backtracks":r0["backtracks"],"assignments_tried":r0["assignments_tried"]})
        writer.writerow({"mode":"ForwardChecking","found":int(r1["found"]),"time_ms":"{:.4f}".format(r1["time_ms"]),"backtracks":r1["backtracks"],"assignments_tried":r1["assignments_tried"]})
    print(f"Wrote CSP metrics to {out_csv}")
    if r0["found"]:
        print("Sample solution (MRV+LCV):")
        for k,v in r0["solution"].items():
            print(f"  {k} -> {v}")
    if r1["found"]:
        print("Sample solution (ForwardChecking):")
        for k,v in r1["solution"].items():
            print(f"  {k} -> {v}")

if __name__ == "__main__":
    run_csp_experiments()