import sys
from collections import defaultdict
import math
import copy
import time
from itertools import combinations

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

    def solve_matching():
        task_list = sorted(all_tasks)
        courier_set = set()
        for _, _, cid, _ in candidates:
            courier_set.add(cid)
        courier_list = sorted(courier_set)
        n_tasks = len(task_list)
        n_couriers = len(courier_list)

        task_to_idx = {t: i for i, t in enumerate(task_list)}
        courier_to_idx = {c: j for j, c in enumerate(courier_list)}

        cost_matrix = [[100.0] * n_couriers for _ in range(n_tasks)]
        for score, task_id_list_str, courier_id, willingness in candidates:
            task_ids = [t.strip() for t in task_id_list_str.split(",")]
            if len(task_ids) == 1:
                t_idx = task_to_idx[task_ids[0]]
                c_idx = courier_to_idx[courier_id]
                pen = expected_penalty([(score, task_id_list_str, willingness)])
                cost_matrix[t_idx][c_idx] = min(cost_matrix[t_idx][c_idx], pen)

        u = [0.0] * (n_tasks + 1)
        v = [0.0] * (n_couriers + 1)
        p = [0] * (n_couriers + 1)
        way = [0] * (n_couriers + 1)

        for i in range(1, n_tasks + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n_couriers + 1)
            used = [False] * (n_couriers + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, n_couriers + 1):
                    if not used[j]:
                        cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(n_couriers + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = [-1] * n_tasks
        for j in range(1, n_couriers + 1):
            if p[j] > 0:
                assignment[p[j] - 1] = j - 1

        result = []
        assigned_couriers = set()
        assigned_tasks = set()

        for t_idx, c_idx in enumerate(assignment):
            if c_idx >= 0 and cost_matrix[t_idx][c_idx] < 100.0:
                task_id = task_list[t_idx]
                courier_id = courier_list[c_idx]
                group = task_id
                result.append((group, [courier_id]))
                assigned_couriers.add(courier_id)
                assigned_tasks.add(task_id)

        return result, assigned_couriers, assigned_tasks

    def global_extra_offer_phase(result, assigned_couriers, assigned_tasks):
        changed = True
        while changed:
            changed = False
            best_gain = 0.0
            best_group_idx = -1
            best_courier = None

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

            if best_gain > 1e-6:
                group, couriers = result[best_group_idx]
                couriers.append(best_courier)
                assigned_couriers.add(best_courier)
                changed = True

        return result

    def local_swap_phase(result, assigned_couriers, assigned_tasks):
        improved = True
        while improved:
            improved = False
            for i in range(len(result)):
                group1, couriers1 = result[i]
                for j in range(i + 1, len(result)):
                    group2, couriers2 = result[j]
                    for c1 in couriers1:
                        for c2 in couriers2:
                            cands1 = group_to_candidates[group1]
                            cands2 = group_to_candidates[group2]

                            old_selected1 = [(s, cid, w) for s, cid, w in cands1 if cid in couriers1]
                            old_selected2 = [(s, cid, w) for s, cid, w in cands2 if cid in couriers2]
                            old_pen = expected_penalty(old_selected1) + expected_penalty(old_selected2)

                            new_couriers1 = [c for c in couriers1 if c != c1] + [c2]
                            new_couriers2 = [c for c in couriers2 if c != c2] + [c1]

                            new_selected1 = [(s, cid, w) for s, cid, w in cands1 if cid in new_couriers1]
                            new_selected2 = [(s, cid, w) for s, cid, w in cands2 if cid in new_couriers2]

                            if not new_selected1 or not new_selected2:
                                continue

                            new_pen = expected_penalty(new_selected1) + expected_penalty(new_selected2)

                            if new_pen < old_pen - 1e-6:
                                result[i] = (group1, new_couriers1)
                                result[j] = (group2, new_couriers2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return result

    def cover_remaining_tasks(result, assigned_couriers, assigned_tasks):
        uncovered = [t for t in all_tasks if t not in assigned_tasks]
        for task in uncovered:
            best_pen = float('inf')
            best_courier = None
            best_group = None
            for group, cands in group_to_candidates.items():
                task_ids = [t.strip() for t in group.split(",")]
                if task in task_ids and all(t not in assigned_tasks for t in task_ids):
                    for score, courier_id, willingness in cands:
                        if courier_id in assigned_couriers:
                            continue
                        pen = expected_penalty([(score, group, willingness)])
                        if pen < best_pen:
                            best_pen = pen
                            best_courier = courier_id
                            best_group = group
            if best_courier:
                result.append((best_group, [best_courier]))
                assigned_couriers.add(best_courier)
                for t in best_group.split(","):
                    assigned_tasks.add(t.strip())
        return result

    def courier_reduction_phase(result, assigned_couriers):
        improved = True
        while improved:
            improved = False
            for idx, (group, couriers) in enumerate(result):
                if len(couriers) <= 1:
                    continue
                cands = group_to_candidates[group]
                current_selected = [(s, cid, w) for s, cid, w in cands if cid in couriers]
                old_pen = expected_penalty(current_selected)
                for c in couriers:
                    new_couriers = [x for x in couriers if x != c]
                    new_selected = [(s, cid, w) for s, cid, w in cands if cid in new_couriers]
                    if not new_selected:
                        continue
                    new_pen = expected_penalty(new_selected)
                    if new_pen < old_pen - 1e-6:
                        result[idx] = (group, new_couriers)
                        assigned_couriers.discard(c)
                        improved = True
                        break
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

    def three_task_subset_improvement(result, assigned_couriers):
        best_solution = copy.deepcopy(result)
        best_score = score_solution(best_solution)

        # Get all single-task groups from the solution
        single_task_groups = []
        for idx, (group, couriers) in enumerate(result):
            task_ids = [t.strip() for t in group.split(",")]
            if len(task_ids) == 1:
                single_task_groups.append((idx, task_ids[0], group, couriers))

        if len(single_task_groups) < 3:
            return best_solution

        # Build task to top couriers mapping (single-task only)
        task_to_top_couriers = {}
        for task in all_tasks:
            cands = []
            for score, task_id_list_str, courier_id, willingness in candidates:
                task_ids = [t.strip() for t in task_id_list_str.split(",")]
                if len(task_ids) == 1 and task_ids[0] == task:
                    cands.append((score, task_id_list_str, courier_id, willingness))
            cands.sort(key=lambda x: expected_penalty([(x[0], x[1], x[3])]))
            task_to_top_couriers[task] = cands[:10]

        start_time = time.time()
        improved = True
        while improved and time.time() - start_time < 8.5:
            improved = False

            # Try all triples of single-task groups
            for i in range(len(single_task_groups)):
                for j in range(i + 1, len(single_task_groups)):
                    for k in range(j + 1, len(single_task_groups)):
                        if time.time() - start_time > 8.5:
                            return best_solution

                        idx1, task1, group1, couriers1 = single_task_groups[i]
                        idx2, task2, group2, couriers2 = single_task_groups[j]
                        idx3, task3, group3, couriers3 = single_task_groups[k]

                        # Get current couriers for these groups
                        current_couriers = set(couriers1 + couriers2 + couriers3)

                        # Get candidate subsets for each task
                        cands1 = task_to_top_couriers.get(task1, [])
                        cands2 = task_to_top_couriers.get(task2, [])
                        cands3 = task_to_top_couriers.get(task3, [])

                        if not cands1 or not cands2 or not cands3:
                            continue

                        # Generate top 12 subsets for each task (sizes 1..3)
                        subsets1 = []
                        for size in range(1, min(4, len(cands1) + 1)):
                            for combo in combinations(range(len(cands1)), size):
                                subset = [cands1[x] for x in combo]
                                subset_couriers = [x[2] for x in subset]
                                pen = expected_penalty([(x[0], x[1], x[3]) for x in subset])
                                subsets1.append((subset_couriers, subset, pen))
                        subsets1.sort(key=lambda x: x[2])
                        subsets1 = subsets1[:12]

                        subsets2 = []
                        for size in range(1, min(4, len(cands2) + 1)):
                            for combo in combinations(range(len(cands2)), size):
                                subset = [cands2[x] for x in combo]
                                subset_couriers = [x[2] for x in subset]
                                pen = expected_penalty([(x[0], x[1], x[3]) for x in subset])
                                subsets2.append((subset_couriers, subset, pen))
                        subsets2.sort(key=lambda x: x[2])
                        subsets2 = subsets2[:12]

                        subsets3 = []
                        for size in range(1, min(4, len(cands3) + 1)):
                            for combo in combinations(range(len(cands3)), size):
                                subset = [cands3[x] for x in combo]
                                subset_couriers = [x[2] for x in subset]
                                pen = expected_penalty([(x[0], x[1], x[3]) for x in subset])
                                subsets3.append((subset_couriers, subset, pen))
                        subsets3.sort(key=lambda x: x[2])
                        subsets3 = subsets3[:12]

                        # Try combinations of subsets
                        for sub1_couriers, sub1_data, pen1 in subsets1:
                            for sub2_couriers, sub2_data, pen2 in subsets2:
                                for sub3_couriers, sub3_data, pen3 in subsets3:
                                    all_couriers = set(sub1_couriers + sub2_couriers + sub3_couriers)
                                    if len(all_couriers) != len(sub1_couriers) + len(sub2_couriers) + len(sub3_couriers):
                                        continue
                                    if any(c in assigned_couriers and c not in current_couriers for c in all_couriers):
                                        continue

                                    # Check if this improves the solution
                                    temp_result = copy.deepcopy(result)
                                    temp_result[idx1] = (group1, list(sub1_couriers))
                                    temp_result[idx2] = (group2, list(sub2_couriers))
                                    temp_result[idx3] = (group3, list(sub3_couriers))

                                    temp_assigned = set(assigned_couriers)
                                    for c in current_couriers:
                                        if c not in all_couriers:
                                            temp_assigned.discard(c)
                                    for c in all_couriers:
                                        temp_assigned.add(c)

                                    temp_score = score_solution(temp_result)
                                    if temp_score < best_score - 1e-7:
                                        best_score = temp_score
                                        best_solution = temp_result
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return best_solution

    result, assigned_couriers, assigned_tasks = solve_matching()
    result = cover_remaining_tasks(result, assigned_couriers, assigned_tasks)
    result = global_extra_offer_phase(result, assigned_couriers, assigned_tasks)
    result = local_swap_phase(result, assigned_couriers, assigned_tasks)
    result = courier_reduction_phase(result, assigned_couriers)
    result = global_extra_offer_phase(result, assigned_couriers, assigned_tasks)
    result = local_swap_phase(result, assigned_couriers, assigned_tasks)
    result = courier_reduction_phase(result, assigned_couriers)

    # New three-task subset improvement phase
    result = three_task_subset_improvement(result, assigned_couriers)

    return result
