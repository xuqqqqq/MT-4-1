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
                current_pen = expected_penalty(current_selected)
                for c in couriers:
                    new_couriers = [x for x in couriers if x != c]
                    if not new_couriers:
                        continue
                    new_selected = [(s, cid, w) for s, cid, w in cands if cid in new_couriers]
                    new_pen = expected_penalty(new_selected)
                    if new_pen < current_pen - 1e-6:
                        result[idx] = (group, new_couriers)
                        assigned_couriers.discard(c)
                        improved = True
                        break
                if improved:
                    break
        return result

    def one_courier_relocation(result, assigned_couriers):
        best_score = score_solution(result)
        improved = True
        start_time = time.time()
        while improved and time.time() - start_time < 6.5:
            improved = False
            best_move = None
            best_new_score = best_score
            for idx_a, (group_a, couriers_a) in enumerate(result):
                task_ids_a = [t.strip() for t in group_a.split(",")]
                if len(task_ids_a) != 1:
                    continue
                if len(couriers_a) < 2:
                    continue
                for c in couriers_a:
                    for idx_b, (group_b, couriers_b) in enumerate(result):
                        if idx_a == idx_b:
                            continue
                        task_ids_b = [t.strip() for t in group_b.split(",")]
                        if len(task_ids_b) != 1:
                            continue
                        if c in couriers_b:
                            continue
                        cands_b = group_to_candidates[group_b]
                        found = False
                        for _, cid, _ in cands_b:
                            if cid == c:
                                found = True
                                break
                        if not found:
                            continue
                        new_couriers_a = [x for x in couriers_a if x != c]
                        new_couriers_b = list(couriers_b) + [c]
                        if not new_couriers_a:
                            continue
                        selected_a = []
                        for score, cid, w in group_to_candidates[group_a]:
                            if cid in new_couriers_a:
                                selected_a.append((score, cid, w))
                        selected_b = []
                        for score, cid, w in group_to_candidates[group_b]:
                            if cid in new_couriers_b:
                                selected_b.append((score, cid, w))
                        if not selected_a or not selected_b:
                            continue
                        new_pen_a = expected_penalty(selected_a)
                        new_pen_b = expected_penalty(selected_b)
                        old_selected_a = []
                        for score, cid, w in group_to_candidates[group_a]:
                            if cid in couriers_a:
                                old_selected_a.append((score, cid, w))
                        old_selected_b = []
                        for score, cid, w in group_to_candidates[group_b]:
                            if cid in couriers_b:
                                old_selected_b.append((score, cid, w))
                        old_pen_a = expected_penalty(old_selected_a)
                        old_pen_b = expected_penalty(old_selected_b)
                        old_total = old_pen_a + old_pen_b
                        new_total = new_pen_a + new_pen_b
                        if new_total < old_total - 1e-7:
                            temp_result = copy.deepcopy(result)
                            temp_result[idx_a] = (group_a, new_couriers_a)
                            temp_result[idx_b] = (group_b, new_couriers_b)
                            temp_score = score_solution(temp_result)
                            if temp_score < best_new_score - 1e-7:
                                best_new_score = temp_score
                                best_move = (idx_a, idx_b, c, new_couriers_a, new_couriers_b)
            if best_move is not None:
                idx_a, idx_b, c, new_couriers_a, new_couriers_b = best_move
                result[idx_a] = (result[idx_a][0], new_couriers_a)
                result[idx_b] = (result[idx_b][0], new_couriers_b)
                assigned_couriers.discard(c)
                assigned_couriers.add(c)
                best_score = best_new_score
                improved = True
        return result

    def three_group_redistribution_fast(result, assigned_couriers):
        start_time = time.time()
        n = len(result)
        single_task_indices = []
        for i in range(n):
            group, couriers = result[i]
            if len(group.split(",")) == 1 and 2 <= len(couriers) <= 3:
                single_task_indices.append(i)
        if len(single_task_indices) < 3:
            return result

        improved = True
        while improved and time.time() - start_time < 8.2:
            improved = False
            best_triple = None
            best_improvement = 0.0

            for i in range(len(single_task_indices)):
                for j in range(i + 1, len(single_task_indices)):
                    for k in range(j + 1, len(single_task_indices)):
                        if time.time() - start_time > 8.2:
                            return result
                        idx_a = single_task_indices[i]
                        idx_b = single_task_indices[j]
                        idx_c = single_task_indices[k]
                        group_a = result[idx_a][0]
                        group_b = result[idx_b][0]
                        group_c = result[idx_c][0]
                        couriers_a = result[idx_a][1]
                        couriers_b = result[idx_b][1]
                        couriers_c = result[idx_c][1]
                        pool = list(set(couriers_a + couriers_b + couriers_c))
                        if len(pool) > 7:
                            continue
                        cands_a = group_to_candidates[group_a]
                        cands_b = group_to_candidates[group_b]
                        cands_c = group_to_candidates[group_c]

                        old_selected_a = [(s, cid, w) for s, cid, w in cands_a if cid in couriers_a]
                        old_selected_b = [(s, cid, w) for s, cid, w in cands_b if cid in couriers_b]
                        old_selected_c = [(s, cid, w) for s, cid, w in cands_c if cid in couriers_c]
                        old_pen = expected_penalty(old_selected_a) + expected_penalty(old_selected_b) + expected_penalty(old_selected_c)

                        subsets_a = []
                        for r in range(1, min(4, len(pool) + 1)):
                            for subset in combinations(pool, r):
                                subset_set = set(subset)
                                selected = [(s, cid, w) for s, cid, w in cands_a if cid in subset_set]
                                if len(selected) == r:
                                    pen = expected_penalty(selected)
                                    subsets_a.append((list(subset), pen))
                        subsets_a.sort(key=lambda x: x[1])
                        subsets_a = subsets_a[:12]

                        subsets_b = []
                        for r in range(1, min(4, len(pool) + 1)):
                            for subset in combinations(pool, r):
                                subset_set = set(subset)
                                selected = [(s, cid, w) for s, cid, w in cands_b if cid in subset_set]
                                if len(selected) == r:
                                    pen = expected_penalty(selected)
                                    subsets_b.append((list(subset), pen))
                        subsets_b.sort(key=lambda x: x[1])
                        subsets_b = subsets_b[:12]

                        subsets_c = []
                        for r in range(1, min(4, len(pool) + 1)):
                            for subset in combinations(pool, r):
                                subset_set = set(subset)
                                selected = [(s, cid, w) for s, cid, w in cands_c if cid in subset_set]
                                if len(selected) == r:
                                    pen = expected_penalty(selected)
                                    subsets_c.append((list(subset), pen))
                        subsets_c.sort(key=lambda x: x[1])
                        subsets_c = subsets_c[:12]

                        if not subsets_a or not subsets_b or not subsets_c:
                            continue

                        for sub_a, pen_a in subsets_a:
                            set_a = set(sub_a)
                            for sub_b, pen_b in subsets_b:
                                if set_a & set(sub_b):
                                    continue
                                set_b = set(sub_b)
                                for sub_c, pen_c in subsets_c:
                                    if set_a & set(sub_c) or set_b & set(sub_c):
                                        continue
                                    new_pen = pen_a + pen_b + pen_c
                                    improvement = old_pen - new_pen
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_triple = (idx_a, idx_b, idx_c, sub_a, sub_b, sub_c)

            if best_triple is not None and best_improvement > 1e-7:
                idx_a, idx_b, idx_c, sub_a, sub_b, sub_c = best_triple
                result[idx_a] = (result[idx_a][0], sub_a)
                result[idx_b] = (result[idx_b][0], sub_b)
                result[idx_c] = (result[idx_c][0], sub_c)
                assigned_couriers.clear()
                for _, couriers in result:
                    for c in couriers:
                        assigned_couriers.add(c)
                improved = True
                single_task_indices = []
                for i in range(len(result)):
                    group, couriers = result[i]
                    if len(group.split(",")) == 1 and 2 <= len(couriers) <= 3:
                        single_task_indices.append(i)
                if len(single_task_indices) < 3:
                    break

        return result

    def additional_three_group_redistribution_fast(result, assigned_couriers):
        start_time = time.time()
        n = len(result)
        single_task_indices = []
        for i in range(n):
            group, couriers = result[i]
            if len(group.split(",")) == 1 and 2 <= len(couriers) <= 3:
                single_task_indices.append(i)
        if len(single_task_indices) < 3:
            return result

        for i in range(len(single_task_indices)):
            for j in range(i + 1, len(single_task_indices)):
                for k in range(j + 1, len(single_task_indices)):
                    if time.time() - start_time > 7.5:
                        return result
                    idx_a = single_task_indices[i]
                    idx_b = single_task_indices[j]
                    idx_c = single_task_indices[k]
                    group_a = result[idx_a][0]
                    group_b = result[idx_b][0]
                    group_c = result[idx_c][0]
                    couriers_a = result[idx_a][1]
                    couriers_b = result[idx_b][1]
                    couriers_c = result[idx_c][1]
                    pool = list(set(couriers_a + couriers_b + couriers_c))
                    if len(pool) > 7:
                        continue
                    cands_a = group_to_candidates[group_a]
                    cands_b = group_to_candidates[group_b]
                    cands_c = group_to_candidates[group_c]

                    old_selected_a = [(s, cid, w) for s, cid, w in cands_a if cid in couriers_a]
                    old_selected_b = [(s, cid, w) for s, cid, w in cands_b if cid in couriers_b]
                    old_selected_c = [(s, cid, w) for s, cid, w in cands_c if cid in couriers_c]
                    old_pen = expected_penalty(old_selected_a) + expected_penalty(old_selected_b) + expected_penalty(old_selected_c)

                    subsets_a = []
                    for r in range(1, min(4, len(pool) + 1)):
                        for subset in combinations(pool, r):
                            subset_set = set(subset)
                            selected = [(s, cid, w) for s, cid, w in cands_a if cid in subset_set]
                            if len(selected) == r:
                                pen = expected_penalty(selected)
                                subsets_a.append((list(subset), pen))
                    subsets_a.sort(key=lambda x: x[1])
                    subsets_a = subsets_a[:12]

                    subsets_b = []
                    for r in range(1, min(4, len(pool) + 1)):
                        for subset in combinations(pool, r):
                            subset_set = set(subset)
                            selected = [(s, cid, w) for s, cid, w in cands_b if cid in subset_set]
                            if len(selected) == r:
                                pen = expected_penalty(selected)
                                subsets_b.append((list(subset), pen))
                    subsets_b.sort(key=lambda x: x[1])
                    subsets_b = subsets_b[:12]

                    subsets_c = []
                    for r in range(1, min(4, len(pool) + 1)):
                        for subset in combinations(pool, r):
                            subset_set = set(subset)
                            selected = [(s, cid, w) for s, cid, w in cands_c if cid in subset_set]
                            if len(selected) == r:
                                pen = expected_penalty(selected)
                                subsets_c.append((list(subset), pen))
                    subsets_c.sort(key=lambda x: x[1])
                    subsets_c = subsets_c[:12]

                    if not subsets_a or not subsets_b or not subsets_c:
                        continue

                    for sub_a, pen_a in subsets_a:
                        set_a = set(sub_a)
                        for sub_b, pen_b in subsets_b:
                            if set_a & set(sub_b):
                                continue
                            set_b = set(sub_b)
                            for sub_c, pen_c in subsets_c:
                                if set_a & set(sub_c) or set_b & set(sub_c):
                                    continue
                                new_pen = pen_a + pen_b + pen_c
                                if new_pen < old_pen - 1e-7:
                                    result[idx_a] = (group_a, sub_a)
                                    result[idx_b] = (group_b, sub_b)
                                    result[idx_c] = (group_c, sub_c)
                                    assigned_couriers.clear()
                                    for _, couriers in result:
                                        for c in couriers:
                                            assigned_couriers.add(c)
                                    return result
        return result

    result, assigned_couriers, assigned_tasks = solve_matching()
    result = cover_remaining_tasks(result, assigned_couriers, assigned_tasks)
    result = global_extra_offer_phase(result, assigned_couriers, assigned_tasks)
    result = local_swap_phase(result, assigned_couriers, assigned_tasks)
    result = courier_reduction_phase(result, assigned_couriers)
    result = global_extra_offer_phase(result, assigned_couriers, assigned_tasks)
    result = local_swap_phase(result, assigned_couriers, assigned_tasks)
    result = courier_reduction_phase(result, assigned_couriers)
    result = one_courier_relocation(result, assigned_couriers)
    result = three_group_redistribution_fast(result, assigned_couriers)
    result = additional_three_group_redistribution_fast(result, assigned_couriers)

    return result
