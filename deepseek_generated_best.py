import sys
from collections import defaultdict
import math

def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        candidates.append((score, task_id_list_str.strip(), courier_id.strip(), willingness))

    group_to_candidates = defaultdict(list)
    for score, task_id_list_str, courier_id, willingness in candidates:
        group_to_candidates[task_id_list_str].append((score, courier_id, willingness))

    all_tasks = set()
    for _, task_id_list_str, _, _ in candidates:
        for t in task_id_list_str.split(","):
            all_tasks.add(t.strip())

    def expected_penalty(selected):
        if not selected:
            return float('inf')
        miss = 1.0
        sum_w = 0.0
        weighted_score = 0.0
        for score, _, willingness in selected:
            miss *= (1.0 - willingness)
            sum_w += willingness
            weighted_score += willingness * score
        accept = 1.0 - miss
        if sum_w > 0:
            accepted_score = weighted_score / sum_w
        else:
            accepted_score = 0.0
        bundle_size = len(selected[0][1].split(","))
        return accept * accepted_score + miss * 100.0 * bundle_size

    def build_solution_with_sort_key(sort_key):
        assigned_couriers = set()
        assigned_tasks = set()
        result = []
        
        groups_sorted = sorted(group_to_candidates.keys(), key=sort_key)
        
        for group in groups_sorted:
            task_ids = [t.strip() for t in group.split(",")]
            if any(t in assigned_tasks for t in task_ids):
                continue
            
            cands = group_to_candidates[group]
            best_pen = float('inf')
            best_courier = None
            
            for score, courier_id, willingness in cands:
                if courier_id in assigned_couriers:
                    continue
                pen = expected_penalty([(score, group, willingness)])
                if pen < best_pen:
                    best_pen = pen
                    best_courier = courier_id
            
            if best_courier:
                assigned_couriers.add(best_courier)
                for t in task_ids:
                    assigned_tasks.add(t)
                result.append((group, [best_courier]))
        
        return result, assigned_couriers, assigned_tasks

    def global_extra_offer_phase(result, assigned_couriers, assigned_tasks):
        changed = True
        while changed:
            changed = False
            best_gain = 0.0
            best_group_idx = -1
            best_courier = None
            best_candidate = None
            
            for idx, (group, couriers) in enumerate(result):
                cands = group_to_candidates[group]
                current_selected = []
                for score, courier_id, willingness in cands:
                    if courier_id in couriers:
                        current_selected.append((score, courier_id, willingness))
                
                for score, courier_id, willingness in cands:
                    if courier_id in assigned_couriers:
                        continue
                    new_selected = current_selected + [(score, courier_id, willingness)]
                    old_pen = expected_penalty(current_selected)
                    new_pen = expected_penalty(new_selected)
                    gain = old_pen - new_pen
                    if gain > best_gain:
                        best_gain = gain
                        best_group_idx = idx
                        best_courier = courier_id
                        best_candidate = (score, courier_id, willingness)
            
            if best_gain > 1e-6 and best_group_idx >= 0:
                group, couriers = result[best_group_idx]
                couriers.append(best_courier)
                assigned_couriers.add(best_courier)
                changed = True
        
        return result

    def cover_remaining_tasks(result, assigned_couriers, assigned_tasks):
        remaining = all_tasks - assigned_tasks
        if not remaining:
            return result
        
        for group in group_to_candidates:
            task_ids = [t.strip() for t in group.split(",")]
            if not any(t in remaining for t in task_ids):
                continue
            if all(t in assigned_tasks for t in task_ids):
                continue
            
            cands = group_to_candidates[group]
            best_pen = float('inf')
            best_courier = None
            
            for score, courier_id, willingness in cands:
                if courier_id in assigned_couriers:
                    continue
                pen = expected_penalty([(score, group, willingness)])
                if pen < best_pen:
                    best_pen = pen
                    best_courier = courier_id
            
            if best_courier:
                assigned_couriers.add(best_courier)
                for t in task_ids:
                    assigned_tasks.add(t)
                result.append((group, [best_courier]))
        
        return result

    def score_solution(solution):
        total = 0.0
        for group, couriers in solution:
            selected = []
            for score, courier_id, willingness in group_to_candidates[group]:
                if courier_id in couriers:
                    selected.append((score, courier_id, willingness))
            total += expected_penalty(selected)
        
        covered_tasks = set()
        for group, _ in solution:
            for t in group.split(","):
                covered_tasks.add(t.strip())
        total += 100.0 * (len(all_tasks) - len(covered_tasks))
        
        return total

    best_result = None
    best_score = float('inf')

    sort_keys = [
        lambda g: min(c[0] for c in group_to_candidates[g]),
        lambda g: expected_penalty([(c[0], g, c[2]) for c in group_to_candidates[g][:1]]),
        lambda g: min(c[2] for c in group_to_candidates[g]),
        lambda g: -max(c[2] for c in group_to_candidates[g]),
        lambda g: min(c[0] for c in group_to_candidates[g]) / len(g.split(",")),
        lambda g: expected_penalty([(c[0], g, c[2]) for c in group_to_candidates[g][:1]]) / len(g.split(",")),
        lambda g: min(c[0] * c[2] for c in group_to_candidates[g]),
        lambda g: min(c[0] / (c[2] + 0.001) for c in group_to_candidates[g]),
        lambda g: len(group_to_candidates[g]),
        lambda g: -len(group_to_candidates[g]),
        lambda g: sum(c[0] for c in group_to_candidates[g]) / len(group_to_candidates[g]),
        lambda g: min(c[0] for c in group_to_candidates[g]) * (1.0 - max(c[2] for c in group_to_candidates[g])),
    ]

    for sort_key in sort_keys:
        result, assigned_couriers, assigned_tasks = build_solution_with_sort_key(sort_key)
        result = global_extra_offer_phase(result, assigned_couriers, assigned_tasks)
        result = cover_remaining_tasks(result, assigned_couriers, assigned_tasks)
        s = score_solution(result)
        if s < best_score:
            best_score = s
            best_result = result

    return best_result
