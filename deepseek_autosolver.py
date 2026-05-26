#!/usr/bin/env python3
"""
deepseek_autosolver.py - EOH/REEVO-like autosolver controller for DeepSeek.
Generates, evaluates, submits, and iterates on contest solver code.
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.parse
import urllib.error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path("outputs/deepseek_autosolver")
STATE_PATH = BASE_DIR / "state.json"
LOG_PATH = BASE_DIR / "log.jsonl"
CANDIDATES_DIR = BASE_DIR / "candidates"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
HACKATHON_LOGIN_URL = "https://hackathon.mykeeta.com/login"
HACKATHON_JUDGE_URL = "https://hackathon.mykeeta.com/judge"
HACKATHON_RESULT_URL = "https://hackathon.mykeeta.com/result"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create required directories."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    """Load state from JSON file, return empty dict if not exists."""
    if STATE_PATH.exists():
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save state to JSON file."""
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def append_log(entry: dict):
    """Append a JSON line to the log file."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_file(path: Path) -> str:
    """Read file content as string."""
    with open(path, "r") as f:
        return f.read()


def write_file(path: Path, content: str):
    """Write string content to file."""
    with open(path, "w") as f:
        f.write(content)


def strip_code_fences(text):
    # If text contains fenced code blocks, return the largest python/plain block.
    # If no fence exists, remove leading/trailing whitespace and append newline.
    lines = text.split('\n')
    fences = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            lang = line[3:].strip().lower()
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                j += 1
            if j < len(lines):
                fences.append((i, j, lang))
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    if not fences:
        return text.strip() + '\n'
    allowed = []
    for fence in fences:
        if fence[2] in ('', 'python', 'py'):
            allowed.append(fence)
    if not allowed:
        return text.strip() + '\n'
    best = max(allowed, key=lambda f: f[1] - f[0] - 1)
    start, end, _ = best
    code_lines = lines[start + 1:end]
    code = '\n'.join(code_lines).strip()
    if code.startswith('python\n'):
        code = code[7:].strip()
    return code + '\n'


def call_deepseek(prompt: str, model: str = "deepseek-chat", max_tokens: int = 4096) -> str:
    """Call DeepSeek API with given prompt and return response text."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(api_key),
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("DeepSeek API call failed: {}".format(e))
        raise


# ---------------------------------------------------------------------------
# Local evaluator
# ---------------------------------------------------------------------------
def parse_tsv(tsv_content: str) -> Tuple[Dict, List[str], List[str]]:
    """
    Parse TSV with columns: task_id_list, courier_id, total_score, willingness.
    Returns (rows_by_key, all_task_ids, all_couriers).
    rows_by_key: {(task_id_list_str, courier_id): {task_id_list, courier, score, willingness, task_ids}}
    """
    lines = tsv_content.strip().split("\n")
    if len(lines) < 2:
        raise ValueError("TSV must have header and at least one data row")
    header = lines[0].strip().split("\t")
    expected = ["task_id_list", "courier_id", "total_score", "willingness"]
    if header != expected:
        raise ValueError("TSV header mismatch: expected {}, got {}".format(expected, header))

    rows_by_key = {}
    all_task_ids = set()
    all_couriers = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) != 4:
            raise ValueError("Invalid TSV row: {}".format(line))
        task_id_list_str, courier, score_str, willingness_str = parts
        score = float(score_str)
        willingness = float(willingness_str)
        task_ids = [tid.strip() for tid in task_id_list_str.split(",") if tid.strip()]
        key = (task_id_list_str, courier)
        rows_by_key[key] = {
            "task_id_list": task_id_list_str,
            "courier": courier,
            "score": score,
            "willingness": willingness,
            "task_ids": task_ids,
        }
        all_task_ids.update(task_ids)
        all_couriers.add(courier)
    return rows_by_key, sorted(all_task_ids), sorted(all_couriers)


def evaluate_solution(tsv_content: str, solution: list) -> Dict[str, Any]:
    """
    Evaluate a solution against the TSV problem definition.
    solution: list of (task_id_list_string, courier_id_list) tuples/lists
    Returns dict with cost, missing_tasks, duplicate_tasks, etc.
    """
    rows_by_key, all_task_ids, all_couriers = parse_tsv(tsv_content)

    # Normalize solution into groups: each group is (task_id_list_str, [courier1, courier2, ...])
    groups = []
    for item in solution:
        if isinstance(item, dict):
            task_id_list_str = item.get("task_id_list", "")
            courier_list = item.get("courier", [])
        elif isinstance(item, (tuple, list)):
            if len(item) != 2:
                continue
            task_id_list_str, courier_list = item
        else:
            continue
        
        if isinstance(courier_list, str):
            courier_list = [courier_list]
        elif isinstance(courier_list, (tuple, list)):
            courier_list = list(courier_list)
        else:
            continue
        
        groups.append((task_id_list_str, courier_list))

    # Validate each courier in each group exists in TSV
    valid_groups = []  # list of (task_id_list_str, [valid_couriers])
    invalid_pairs = []
    for task_id_list_str, courier_list in groups:
        valid_couriers = []
        for courier in courier_list:
            key = (task_id_list_str, courier)
            if key in rows_by_key:
                valid_couriers.append(courier)
            else:
                invalid_pairs.append((task_id_list_str, courier))
        if valid_couriers:
            valid_groups.append((task_id_list_str, valid_couriers))

    # Track coverage by groups
    covered_tasks = set()
    covered_couriers = set()
    duplicate_tasks = []
    duplicate_couriers = []
    used_task_id_list_strs = set()  # track which exact task_id_list_str groups have been used

    for task_id_list_str, courier_list in valid_groups:
        # Check if this exact task_id_list_str group has already been used
        if task_id_list_str in used_task_id_list_strs:
            # This is a duplicate group - all its tasks are duplicates
            row = rows_by_key[(task_id_list_str, courier_list[0])]
            for tid in row["task_ids"]:
                if tid not in duplicate_tasks:
                    duplicate_tasks.append(tid)
            continue
        
        # Check for duplicate couriers in this group
        for courier in courier_list:
            if courier in covered_couriers:
                duplicate_couriers.append(courier)
            else:
                covered_couriers.add(courier)
        
        # Check for task overlap with previously covered tasks
        row = rows_by_key[(task_id_list_str, courier_list[0])]
        task_ids = row["task_ids"]
        new_tasks = []
        for tid in task_ids:
            if tid in covered_tasks:
                if tid not in duplicate_tasks:
                    duplicate_tasks.append(tid)
            else:
                new_tasks.append(tid)
                covered_tasks.add(tid)
        
        used_task_id_list_strs.add(task_id_list_str)

    # Find missing tasks
    missing_tasks = [tid for tid in all_task_ids if tid not in covered_tasks]

    # Compute cost
    total_cost = 0.0

    # Penalty for missing tasks: 100 per task
    total_cost += len(missing_tasks) * 100.0

    # Penalty for duplicate tasks: 100 per duplicate
    total_cost += len(duplicate_tasks) * 100.0

    # Penalty for duplicate couriers: 100 per duplicate
    total_cost += len(duplicate_couriers) * 100.0

    # Penalty for invalid pairs: 100 per invalid
    total_cost += len(invalid_pairs) * 100.0

    # Calculate expected penalty for each used group (one per task_id_list_str)
    for task_id_list_str in used_task_id_list_strs:
        # Get all valid couriers for this group (from valid_groups)
        group_couriers = []
        for t, c_list in valid_groups:
            if t == task_id_list_str:
                group_couriers.extend(c_list)
        if not group_couriers:
            continue
        
        # Get the row data for first courier (all have same task_ids, score, willingness)
        first_key = (task_id_list_str, group_couriers[0])
        row = rows_by_key[first_key]
        task_ids = row["task_ids"]
        size = len(task_ids)
        
        # Check if all tasks in this group are actually covered (not duplicated)
        all_covered = all(tid in covered_tasks for tid in task_ids)
        if not all_covered:
            continue
        
        # Calculate combined expected penalty for all couriers in this group
        # miss = product(1-w_i) for all couriers in group
        miss = 1.0
        for courier in group_couriers:
            key = (task_id_list_str, courier)
            w = rows_by_key[key]["willingness"]
            miss *= (1.0 - w)
        
        accept = 1.0 - miss
        
        # Calculate weighted accepted score
        total_w = 0.0
        total_ws = 0.0
        for courier in group_couriers:
            key = (task_id_list_str, courier)
            w = rows_by_key[key]["willingness"]
            s = rows_by_key[key]["score"]
            total_w += w
            total_ws += w * s
        
        if total_w > 0:
            accepted_score = total_ws / total_w
        else:
            accepted_score = 100.0 * size
        
        expected_penalty = accept * accepted_score + miss * 100.0 * size
        total_cost += expected_penalty

    return {
        "cost": total_cost,
        "missing_tasks": missing_tasks,
        "duplicate_tasks": duplicate_tasks,
        "duplicate_couriers": duplicate_couriers,
        "invalid_pairs": invalid_pairs,
        "num_valid_pairs": sum(len(c) for _, c in valid_groups),
        "num_invalid_pairs": len(invalid_pairs),
        "total_tasks": len(all_task_ids),
        "covered_tasks": len(covered_tasks),
    }


# ---------------------------------------------------------------------------
# Candidate solver execution
# ---------------------------------------------------------------------------
def run_candidate_solver(code: str, input_text: str, timeout: int = 30) -> Optional[list]:
    """
    Execute candidate solver code in a subprocess with timeout.
    Returns parsed solution list or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        solver_file = tmp_path / "solver.py"
        write_file(solver_file, code)

        # Create wrapper script that imports and calls solve
        # Pass input_text as a raw string literal using repr() to preserve exact content
        wrapper = """
import sys
sys.path.insert(0, {tmpdir!r})
import json
import solver

input_text = {input_text_repr}
try:
    result = solver.solve(input_text)
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
""".format(tmpdir=tmpdir, input_text_repr=repr(input_text))
        wrapper_file = tmp_path / "wrapper.py"
        write_file(wrapper_file, wrapper)

        try:
            proc = subprocess.Popen(
                [sys.executable, str(wrapper_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            if proc.returncode != 0:
                logger.error("Solver subprocess failed: {}".format(stderr.decode("utf-8")))
                return None
            output = stdout.decode("utf-8").strip()
            if not output:
                logger.error("No output from solver")
                return None
            result = json.loads(output)
            if isinstance(result, list):
                return result
            else:
                logger.error("Unexpected result format: {}".format(result))
                return None
        except subprocess.TimeoutExpired:
            logger.error("Solver timed out after {}s".format(timeout))
            return None
        except json.JSONDecodeError as e:
            logger.error("Failed to parse solver output: {}".format(e))
            return None
        except Exception as e:
            logger.error("Unexpected error running solver: {}".format(e))
            return None


# ---------------------------------------------------------------------------
# Generation prompt builder
# ---------------------------------------------------------------------------
def build_generation_prompt(
    problem_info: str = "",
    seed_solver: str = "",
    local_cases: List[Dict] = None,
    previous_feedback: str = "",
) -> str:
    """Build prompt for DeepSeek to generate solver code."""
    prompt_parts = [
        "You are an expert Python programmer. Generate a complete Python 3.6 solver script.",
        "The script must define a function `solve(input_text: str) -> list`.",
        "The function should parse the input text (format described below) and return a list of tuples/lists.",
        "Each item must be a tuple/list `(task_id_list_string, courier_id_list)`.",
        "task_id_list_string must match an exact input TSV task_id_list value (e.g. 'T0000' or 'T0024,T0019').",
        "courier_id_list is a list/tuple of one or more courier IDs offered for that exact task group.",
        "Use only Python 3.6 standard library. No external packages.",
        "Do not use dataclasses, walrus operator, or match/case statements.",
        "Output ONLY the raw Python code, no explanations or markdown.",
        "",
    ]

    if problem_info:
        prompt_parts.append("=== PROBLEM INFORMATION ===")
        prompt_parts.append(problem_info)
        prompt_parts.append("")

    if seed_solver:
        prompt_parts.append("=== SEED SOLVER EXCERPT ===")
        prompt_parts.append(seed_solver)
        prompt_parts.append("")

    if local_cases:
        prompt_parts.append("=== LOCAL EVALUATION CASES ===")
        for i, case in enumerate(local_cases):
            prompt_parts.append("Case {}:".format(i+1))
            prompt_parts.append("  Input TSV: {}".format(case.get('tsv_preview', 'N/A')))
            prompt_parts.append("  Expected cost: {}".format(case.get('expected_cost', 'N/A')))
            prompt_parts.append("  Your last cost: {}".format(case.get('last_cost', 'N/A')))
        prompt_parts.append("")

    if previous_feedback:
        prompt_parts.append("=== PREVIOUS FEEDBACK ===")
        prompt_parts.append(previous_feedback)
        prompt_parts.append("")

    prompt_parts.append("Generate the complete Python solver code now:")
    return "\n".join(prompt_parts)


def build_repair_prompt(previous_code: str, error_message: str) -> str:
    """Build prompt for DeepSeek to repair broken solver code."""
    prompt = """The following Python solver code has an error. Please provide a corrected version.

=== PREVIOUS CODE ===
{previous_code}

=== ERROR MESSAGE ===
{error_message}

Please output ONLY the corrected Python code, no explanations. The code must define solve(input_text: str) -> list.
Use Python 3.6 standard library only. No dataclasses, walrus, or match/case.
""".format(previous_code=previous_code, error_message=error_message)
    return prompt


# ---------------------------------------------------------------------------
# Submission to hackathon
# ---------------------------------------------------------------------------
def login_to_hackathon(team: str, email: str) -> Optional[str]:
    """Login and return auth token."""
    data = json.dumps({"team": team, "email": email}).encode("utf-8")
    req = urllib.request.Request(
        HACKATHON_LOGIN_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("token")
    except Exception as e:
        logger.error("Login failed: {}".format(e))
        return None


def submit_to_hackathon(code: str, token: str) -> Optional[str]:
    """Submit code and return job_id."""
    data = json.dumps({"code": code, "token": token}).encode("utf-8")
    req = urllib.request.Request(
        HACKATHON_JUDGE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("job_id")
    except Exception as e:
        logger.error("Submission failed: {}".format(e))
        return None


def get_hackathon_result(job_id: str) -> Optional[Dict]:
    """Get result for a submitted job."""
    url = "{}/{}".format(HACKATHON_RESULT_URL, job_id)
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error("Failed to get result: {}".format(e))
        return None


# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------
def cmd_generate(args):
    """Generate a new candidate solver using DeepSeek."""
    ensure_dirs()
    state = load_state()

    # Build prompt
    problem_info = args.problem_info or state.get("problem_info", "")
    seed_solver = args.seed_solver or state.get("seed_solver", "")
    local_cases = state.get("local_cases", [])
    previous_feedback = state.get("last_feedback", "")

    prompt = build_generation_prompt(
        problem_info=problem_info,
        seed_solver=seed_solver,
        local_cases=local_cases,
        previous_feedback=previous_feedback,
    )

    logger.info("Calling DeepSeek to generate solver...")
    try:
        code = call_deepseek(prompt, max_tokens=args.max_tokens)
        code = strip_code_fences(code)
    except Exception as e:
        logger.error("Generation failed: {}".format(e))
        return

    # Save candidate
    timestamp = int(time.time())
    candidate_file = CANDIDATES_DIR / "solver_{}.py".format(timestamp)
    write_file(candidate_file, code)
    logger.info("Saved candidate to {}".format(candidate_file))

    # Update state
    state["last_candidate"] = str(candidate_file)
    state["last_candidate_code"] = code
    state["generation_count"] = state.get("generation_count", 0) + 1
    save_state(state)

    # Log
    append_log({
        "event": "generate",
        "timestamp": timestamp,
        "candidate": str(candidate_file),
        "prompt_length": len(prompt),
    })


def cmd_evaluate(args):
    """Evaluate a candidate solver locally."""
    ensure_dirs()
    state = load_state()

    # Determine which code to evaluate
    if args.code_file:
        code = read_file(Path(args.code_file))
    else:
        code = state.get("last_candidate_code")
        if not code:
            logger.error("No candidate code found. Run generate first or specify --code-file")
            return

    # Read TSV input
    if not args.tsv_file:
        logger.error("--tsv-file is required for evaluation")
        return
    tsv_content = read_file(Path(args.tsv_file))

    # Run solver
    logger.info("Running candidate solver...")
    solution = run_candidate_solver(code, tsv_content, timeout=args.timeout)

    if solution is None:
        logger.error("Solver execution failed")
        # Try repair if requested
        if args.repair:
            logger.info("Attempting repair...")
            error_msg = "Solver execution failed or timed out"
            repair_prompt = build_repair_prompt(code, error_msg)
            try:
                new_code = call_deepseek(repair_prompt, max_tokens=args.max_tokens)
                new_code = strip_code_fences(new_code)
                timestamp = int(time.time())
                repair_file = CANDIDATES_DIR / "solver_repair_{}.py".format(timestamp)
                write_file(repair_file, new_code)
                logger.info("Repair saved to {}".format(repair_file))
                state["last_candidate_code"] = new_code
                save_state(state)
                # Re-run with repaired code
                solution = run_candidate_solver(new_code, tsv_content, timeout=args.timeout)
                if solution is None:
                    logger.error("Repair also failed")
                    return
            except Exception as e:
                logger.error("Repair failed: {}".format(e))
                return
        else:
            return

    # Evaluate
    result = evaluate_solution(tsv_content, solution)
    logger.info("Evaluation result: cost={:.2f}, covered={}/{}, missing={}, duplicates={}".format(
        result['cost'], result['covered_tasks'], result['total_tasks'],
        len(result['missing_tasks']), len(result['duplicate_tasks'])))

    # Update state with feedback
    feedback = (
        "Cost: {:.2f}\n"
        "Missing tasks: {}\n"
        "Duplicate tasks: {}\n"
        "Duplicate couriers: {}\n"
        "Invalid pairs: {}\n"
    ).format(result['cost'], result['missing_tasks'], result['duplicate_tasks'], 
             result['duplicate_couriers'], result['invalid_pairs'])
    state["last_feedback"] = feedback
    state["last_evaluation"] = result
    save_state(state)

    # Log
    append_log({
        "event": "evaluate",
        "timestamp": int(time.time()),
        "cost": result["cost"],
        "covered_tasks": result["covered_tasks"],
        "total_tasks": result["total_tasks"],
        "missing": len(result["missing_tasks"]),
        "duplicates": len(result["duplicate_tasks"]),
    })

    # Print detailed results
    if args.verbose:
        print(json.dumps(result, indent=2))


def cmd_submit(args):
    """Submit a candidate to the hackathon."""
    ensure_dirs()
    state = load_state()

    # Get code
    if args.code_file:
        code = read_file(Path(args.code_file))
    else:
        code = state.get("last_candidate_code")
        if not code:
            logger.error("No candidate code found. Run generate first or specify --code-file")
            return

    # Login
    team = args.team or os.environ.get("HACKATHON_TEAM")
    email = args.email or os.environ.get("HACKATHON_EMAIL")
    if not team or not email:
        logger.error("Team and email required. Set HACKATHON_TEAM/HACKATHON_EMAIL env vars or use --team/--email")
        return

    token = login_to_hackathon(team, email)
    if not token:
        logger.error("Login failed")
        return

    # Submit
    job_id = submit_to_hackathon(code, token)
    if not job_id:
        logger.error("Submission failed")
        return

    logger.info("Submitted successfully. Job ID: {}".format(job_id))
    state["last_job_id"] = job_id
    save_state(state)

    # Wait for result if requested
    if args.wait:
        logger.info("Waiting for result...")
        for _ in range(30):  # Wait up to 5 minutes
            time.sleep(10)
            result = get_hackathon_result(job_id)
            if result:
                logger.info("Result: {}".format(json.dumps(result, indent=2)))
                state["last_submission_result"] = result
                save_state(state)
                append_log({
                    "event": "submit_result",
                    "timestamp": int(time.time()),
                    "job_id": job_id,
                    "result": result,
                })
                return
        logger.warning("Timed out waiting for result")


def cmd_loop(args):
    """Main loop: generate, evaluate, submit, repeat."""
    ensure_dirs()
    state = load_state()

    # Set defaults from args
    if args.problem_info:
        state["problem_info"] = args.problem_info
    if args.seed_solver:
        state["seed_solver"] = args.seed_solver
    if args.tsv_file:
        state["tsv_file"] = args.tsv_file
    save_state(state)

    tsv_file = state.get("tsv_file")
    if not tsv_file:
        logger.error("No TSV file specified. Use --tsv-file or set in state")
        return

    tsv_content = read_file(Path(tsv_file))

    max_iterations = args.max_iterations or 10
    for iteration in range(1, max_iterations + 1):
        logger.info("\n=== Iteration {}/{} ===".format(iteration, max_iterations))

        # Generate
        logger.info("Generating new candidate...")
        prompt = build_generation_prompt(
            problem_info=state.get("problem_info", ""),
            seed_solver=state.get("seed_solver", ""),
            local_cases=state.get("local_cases", []),
            previous_feedback=state.get("last_feedback", ""),
        )
        try:
            code = call_deepseek(prompt, max_tokens=args.max_tokens)
            code = strip_code_fences(code)
        except Exception as e:
            logger.error("Generation failed: {}".format(e))
            continue

        timestamp = int(time.time())
        candidate_file = CANDIDATES_DIR / "solver_iter{}_{}.py".format(iteration, timestamp)
        write_file(candidate_file, code)
        state["last_candidate_code"] = code
        state["generation_count"] = state.get("generation_count", 0) + 1
        save_state(state)

        # Evaluate
        logger.info("Evaluating...")
        solution = run_candidate_solver(code, tsv_content, timeout=args.timeout)
        if solution is None:
            logger.warning("Solver failed, attempting repair...")
            repair_prompt = build_repair_prompt(code, "Execution failed")
            try:
                code = call_deepseek(repair_prompt, max_tokens=args.max_tokens)
                code = strip_code_fences(code)
                repair_file = CANDIDATES_DIR / "solver_repair_iter{}_{}.py".format(iteration, timestamp)
                write_file(repair_file, code)
                state["last_candidate_code"] = code
                save_state(state)
                solution = run_candidate_solver(code, tsv_content, timeout=args.timeout)
            except Exception as e:
                logger.error("Repair failed: {}".format(e))
                continue

        if solution is None:
            logger.error("Skipping iteration due to solver failure")
            continue

        result = evaluate_solution(tsv_content, solution)
        logger.info("Cost: {:.2f}, Covered: {}/{}".format(result['cost'], result['covered_tasks'], result['total_tasks']))

        # Update feedback
        feedback = (
            "Iteration {}:\n"
            "Cost: {:.2f}\n"
            "Missing: {}\n"
            "Duplicates: {}\n"
        ).format(iteration, result['cost'], result['missing_tasks'], result['duplicate_tasks'])
        state["last_feedback"] = feedback
        state["last_evaluation"] = result
        save_state(state)

        append_log({
            "event": "loop_iteration",
            "iteration": iteration,
            "timestamp": timestamp,
            "cost": result["cost"],
            "covered_tasks": result["covered_tasks"],
            "total_tasks": result["total_tasks"],
        })

        # Submit if credentials available
        team = args.team or os.environ.get("HACKATHON_TEAM")
        email = args.email or os.environ.get("HACKATHON_EMAIL")
        if team and email and args.auto_submit:
            logger.info("Submitting to hackathon...")
            token = login_to_hackathon(team, email)
            if token:
                job_id = submit_to_hackathon(code, token)
                if job_id:
                    logger.info("Submitted, job_id: {}".format(job_id))
                    state["last_job_id"] = job_id
                    save_state(state)
                    # Wait briefly for result
                    time.sleep(5)
                    result_data = get_hackathon_result(job_id)
                    if result_data:
                        logger.info("Hackathon result: {}".format(result_data))
                        append_log({
                            "event": "hackathon_result",
                            "iteration": iteration,
                            "job_id": job_id,
                            "result": result_data,
                        })

        # Check if we should stop (e.g., cost below threshold)
        if args.target_cost and result["cost"] <= args.target_cost:
            logger.info("Target cost {} achieved! Stopping.".format(args.target_cost))
            break

    logger.info("Loop completed.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek Autosolver Controller (EOH/REEVO-like)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for DeepSeek")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate a new candidate solver")
    gen_parser.add_argument("--problem-info", help="Problem description")
    gen_parser.add_argument("--seed-solver", help="Seed solver code excerpt")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a candidate solver")
    eval_parser.add_argument("--code-file", help="Path to solver code file")
    eval_parser.add_argument("--tsv-file", required=True, help="Path to TSV input file")
    eval_parser.add_argument("--timeout", type=int, default=30, help="Solver timeout in seconds")
    eval_parser.add_argument("--repair", action="store_true", help="Attempt to repair broken code")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")

    # Submit
    sub_parser = subparsers.add_parser("submit", help="Submit to hackathon")
    sub_parser.add_argument("--code-file", help="Path to solver code file")
    sub_parser.add_argument("--team", help="Team name")
    sub_parser.add_argument("--email", help="Email address")
    sub_parser.add_argument("--wait", action="store_true", help="Wait for result")

    # Loop
    loop_parser = subparsers.add_parser("loop", help="Run generation-evaluation-submission loop")
    loop_parser.add_argument("--tsv-file", help="Path to TSV input file")
    loop_parser.add_argument("--problem-info", help="Problem description")
    loop_parser.add_argument("--seed-solver", help="Seed solver code excerpt")
    loop_parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    loop_parser.add_argument("--target-cost", type=float, help="Stop when cost <= target")
    loop_parser.add_argument("--auto-submit", action="store_true", help="Auto-submit each iteration")
    loop_parser.add_argument("--team", help="Team name for auto-submit")
    loop_parser.add_argument("--email", help="Email for auto-submit")
    loop_parser.add_argument("--timeout", type=int, default=30, help="Solver timeout in seconds")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Dispatch
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "loop":
        cmd_loop(args)


if __name__ == "__main__":
    main()
