import argparse
import copy
import json
import os
import random
import re
import sys
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.adas_ext import (
    BudgetController,
    Instrumentation,
    SandboxError,
    SandboxPolicy,
    SandboxTimeout,
    classify_failure,
    get_instrumentation,
    run_sandboxed,
    set_instrumentation,
)
from experiments.adas_ext.model_router import (
    AGENT_DESIGN_SCHEMA,
    ContextOverflowError,
    build_role_clients,
    role_client_from_config,
    worker_client_config,
)
from experiments.adas_ext.recombination import random_recombination

from mgsm_prompt import get_init_archive, get_prompt, get_reflexion_prompt, set_worker_documentation

from utils import get_all_examples, random_id, bootstrap_confidence_interval, score_mgsm

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

# Name under which the sandbox child re-imports this module.
_ENTRY_MODULE = __name__ if __name__ != "__main__" else os.path.splitext(os.path.basename(__file__))[0]

# Per-role endpoints. WORKER_CLIENT executes discovered agents, META_CLIENT
# writes them. They are distinct vLLM servers on distinct ports.
WORKER_CLIENT = None
META_CLIENT = None


def configure_clients(worker=None, meta=None):
    global WORKER_CLIENT, META_CLIENT
    if worker is not None:
        WORKER_CLIENT = worker
    if meta is not None:
        META_CLIENT = meta


def _as_text(value):
    """Worker fields reach designs as text, as GPT-3.5 always returned them in stock ADAS.

    Open workers sometimes nest objects or emit null, and the seed agents call
    string methods on every field.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    """Worker-role call. ``model`` is accepted for signature compatibility with
    the utility code shown to the meta agent, but routing is by role."""
    if WORKER_CLIENT is None:
        raise RuntimeError("worker client is not configured")
    return WORKER_CLIENT.chat_json(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
    )


def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    """Meta-role call."""
    if META_CLIENT is None:
        raise RuntimeError("meta client is not configured")
    return META_CLIENT.chat_json(msg_list, temperature=temperature, schema=AGENT_DESIGN_SCHEMA)


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model=None, temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model or (WORKER_CLIENT.model if WORKER_CLIENT else 'gpt-3.5-turbo-0125')
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        # Provenance of every input is what makes the runtime dataflow graph
        # (analysis Layer 2) reconstructible after the fact.
        input_authors = [i.author for i in input_infos if isinstance(i, Info)]
        input_names = [i.name for i in input_infos if isinstance(i, Info)]
        call_error = None
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            call_error = f"{type(e).__name__}: {e}"
            if "maximum context length" in str(e) and SEARCHING_MODE:
                get_instrumentation().log_call(
                    agent_name=self.agent_name, agent_id=self.id, role=self.role,
                    output_fields=self.output_fields, iteration_idx=iteration_idx,
                    input_authors=input_authors, input_names=input_names,
                    temperature=self.temperature, error=call_error,
                )
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        get_instrumentation().log_call(
            agent_name=self.agent_name, agent_id=self.id, role=self.role,
            output_fields=self.output_fields, iteration_idx=iteration_idx,
            input_authors=input_authors, input_names=input_names,
            temperature=self.temperature, error=call_error,
        )
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), _as_text(value), iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


class DeadlineReached(Exception):
    """The job's walltime deadline, not the design, ended an evaluation."""


def _point_accuracy(acc_list):
    """Point estimate of accuracy: the value the bootstrap median converges to.

    Not the median of acc_list. That vector is per-question 0/1, so its median is
    a step function -- 0 below 50% correct and 1 above -- which reported 22.3% as
    0.0 and 86.7% as 100.0.
    """
    return float(np.mean(acc_list)) if len(acc_list) else 0.0


def _meta_reasoning():
    """Trace from the meta agent's last call, capped so archives stay readable."""
    trace = getattr(META_CLIENT, "last_reasoning", None) if META_CLIENT else None
    return trace[:20000] if trace else None


def _seed_cache_path(args):
    """Seed-archive fitness is cached per (worker, domain, split).

    Stock ADAS re-evaluates the seven seed agents on every run, which injects
    variance into the starting point of every search.
    """
    key = "__".join([
        str(args.worker_model).replace("/", "_"),
        str(getattr(args, "domain", "mgsm")),
        f"valid{args.valid_size}",
        f"shuffle{args.shuffle_seed}",
        f"rep{args.n_repreat}",
        str(getattr(args, "worker_reasoning_effort", None)),
    ])
    return os.path.join(args.seed_cache_dir or os.path.join(args.save_dir, "seed_cache"), f"{key}.json")


def _load_seed_cache(args):
    path = _seed_cache_path(args)
    if os.path.exists(path):
        with open(path, 'r') as handle:
            return json.load(handle)
    return {}


def _save_seed_cache(args, cache):
    path = _seed_cache_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic, because cells sharing a cache directory may read it while it is written.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, 'w') as handle:
        json.dump(cache, handle, indent=2)
    os.replace(tmp, path)


def _attach_fitness(args, solution, result, generation):
    """Write fitness plus every instrumented field onto an archive entry."""
    acc_list = result["acc_list"]
    solution['fitness'] = bootstrap_confidence_interval(acc_list)
    solution['raw_median_accuracy'] = _point_accuracy(acc_list)
    # The raw per-question binary vector powers the whole behavioural layer.
    solution['valid_per_question'] = acc_list

    usage = result.get("usage", {})
    n_questions = max(len(acc_list), 1)
    worker_usage = usage.get("worker", {})
    n_calls = worker_usage.get("n_calls", 0)
    total_tokens = worker_usage.get("prompt_tokens", 0) + worker_usage.get("completion_tokens", 0)
    solution['resource'] = {
        "generation": generation,
        "n_questions": n_questions,
        "worker_calls": n_calls,
        "calls_per_question": n_calls / n_questions,
        "prompt_tokens": worker_usage.get("prompt_tokens", 0),
        "completion_tokens": worker_usage.get("completion_tokens", 0),
        "reasoning_tokens": worker_usage.get("reasoning_tokens", 0),
        "truncated_calls": worker_usage.get("n_truncated", 0),
        "tokens_per_question": total_tokens / n_questions,
        "wallclock_s": result.get("wallclock_s"),
    }

    if getattr(args, "cost_lambda", 0.0) > 0:
        cost = solution['resource']["tokens_per_question"] / 1000.0
        penalized = solution['raw_median_accuracy'] - args.cost_lambda * cost
        solution['cost_penalized_fitness'] = penalized
        solution['fitness'] = (
            f"{solution['fitness']} | cost-penalized utility: {penalized:.4f} "
            f"(lambda={args.cost_lambda}, {solution['resource']['calls_per_question']:.2f} calls/question, "
            f"{cost:.2f}k tokens/question)"
        )
    return solution


def _propose_solution(args, archive, msg_list, generation):
    """One meta-agent proposal, including the two Reflexion rounds."""
    if args.search_mode == "random":
        # Reference condition: sample and splice archive entries with no meta
        # reasoning, to test whether reasoning or sampling drives any gain.
        return random_recombination(archive, rng=random.Random())

    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
    if args.no_reflection:
        return next_solution

    Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if archive and generation > 0 else None)
    msg_list.append({"role": "assistant", "content": str(next_solution)})
    msg_list.append({"role": "user", "content": Reflexion_prompt_1})
    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
    msg_list.append({"role": "assistant", "content": str(next_solution)})
    msg_list.append({"role": "user", "content": Reflexion_prompt_2})
    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
    return next_solution


def search(args):
    instrumentation = get_instrumentation()
    instrumentation.set_context(phase="search", generation="initial")

    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if archive and "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = [] if args.init_archive == "empty" else get_init_archive()
        start = 0

    seed_cache = _load_seed_cache(args)
    for solution in archive:
        if solution.get('generation', 'initial') == 'initial':
            solution.setdefault('design_id', f"{args.run_id}/seed/{solution['name']}")
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        cached = seed_cache.get(solution['name'])
        if cached is not None:
            print(f"============Initial Archive (cached): {solution['name']}=================")
            solution.update(copy.deepcopy(cached))
            continue
        if args.require_seed_cache:
            # Designers compared side by side must start from one evaluated archive.
            raise SystemExit(f"seed '{solution['name']}' is not in {_seed_cache_path(args)}; "
                             f"run the seed-evaluation cell first")

        print(f"============Initial Archive: {solution['name']}=================")
        instrumentation.set_context(generation=f"initial:{solution['name']}")
        try:
            result = run_evaluation(args, solution["code"])
        except DeadlineReached as e:
            # Unevaluated seeds are evaluated by the resubmitted job.
            print(f"stopping seed evaluation: {e}")
            break
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            instrumentation.record_failure(classify_failure(e), str(e), stage="initial_archive")
            continue

        _attach_fitness(args, solution, result, "initial")
        seed_cache[solution['name']] = {
            k: solution[k] for k in ('fitness', 'raw_median_accuracy', 'valid_per_question', 'resource')
            if k in solution
        }
        _save_seed_cache(args, seed_cache)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    if args.seeds_only:
        missing = [s['name'] for s in archive if 'fitness' not in s]
        if missing:
            raise SystemExit(f"seed evaluation failed for {missing}")
        print(f"seed archive evaluated -> {_seed_cache_path(args)}")
        return

    # Equal successes, not equal attempts. See experiments/adas_ext/budget.py --
    # the stock `for n in range(...)` with `n -= 1` silently skips failed
    # generations, making search budget correlate with meta-agent capability.
    budget = BudgetController(args.target_generations, args.max_attempts, completed=start)
    stopped_on_deadline = False
    while budget.should_continue():
        # Stop between generations rather than being SIGKILLed inside one. The
        # archive is written after every success, so a clean stop resumes.
        if args.deadline_epoch and time.time() >= args.deadline_epoch:
            print(f"stopping before generation {budget.successes + 1}: within "
                  f"the walltime safety margin")
            stopped_on_deadline = True
            break
        generation, attempt = budget.begin_attempt()
        print(f"============Generation {generation + 1} (attempt {attempt}/{budget.max_attempts})=================")
        instrumentation.set_context(generation=generation + 1, attempt=attempt, phase="search")
        failure_trace = []

        system_prompt, prompt = get_prompt(archive)
        design_id = f"{args.run_id}/g{generation + 1}/a{attempt}"
        # Exactly what the designer could see, so archive utilisation is measured
        # rather than inferred from structural similarity.
        archive_context = [e.get('design_id') or e.get('name') for e in archive]
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = _propose_solution(args, archive, msg_list, generation)
        except Exception as e:
            category = classify_failure(e)
            print(f"During LLM generate new solution [{category}]:")
            print(e)
            instrumentation.record_failure(category, str(e), stage="propose")
            failure_trace.append({"stage": "propose", "category": category, "message": str(e)[:2000]})
            budget.record_failure(category, debug_attempts=0, failure_trace=failure_trace)
            instrumentation.record_abandoned(generation + 1, attempt, category, 0, None,
                                             failure_trace, design_id=design_id,
                                             archive_context=archive_context,
                                             meta_reasoning=_meta_reasoning())
            continue

        result = None
        debug_attempts = 0
        terminal_failure = None
        failed_solution = None
        deadline_hit = None
        for debug_round in range(args.debug_max):
            try:
                result = run_evaluation(args, next_solution["code"])
                if np.mean(result["acc_list"]) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except DeadlineReached as e:
                result, deadline_hit = None, e
                break
            except Exception as e:
                result = None
                debug_attempts += 1
                terminal_failure = classify_failure(e)
                # Keep the source that actually failed. next_solution is about to
                # be replaced by a revision, and pairing a failure category with
                # code that was never run would corrupt Q6 and Q9.
                failed_solution = copy.deepcopy(next_solution)
                print(f"During evaluation [{terminal_failure}]:")
                print(e)
                instrumentation.record_failure(terminal_failure, str(e), stage="evaluate")
                failure_trace.append({"stage": "evaluate", "category": terminal_failure, "message": str(e)[:2000]})
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                if args.search_mode == "random":
                    break
                if debug_round == args.debug_max - 1:
                    break  # nothing would evaluate the revision
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    terminal_failure = classify_failure(e)
                    print(f"During LLM generate new solution [{terminal_failure}]:")
                    print(e)
                    instrumentation.record_failure(terminal_failure, str(e), stage="debug")
                    failure_trace.append({"stage": "debug", "category": terminal_failure, "message": str(e)[:2000]})
                    break

        if deadline_hit is not None:
            # Not counted against the designer; the resubmitted job redoes this generation.
            print(f"stopping during generation {generation + 1}: {deadline_hit}")
            budget.void_attempt()
            stopped_on_deadline = True
            break

        if result is None:
            budget.record_failure(terminal_failure or "runtime_exception", debug_attempts, failure_trace)
            # Failed designs are kept. Discarding them biases every downstream
            # metric toward the subset each model could successfully produce.
            instrumentation.record_abandoned(generation + 1, attempt,
                                             terminal_failure or "runtime_exception",
                                             debug_attempts,
                                             failed_solution or next_solution, failure_trace,
                                             design_id=design_id,
                                             archive_context=archive_context,
                                             meta_reasoning=_meta_reasoning())
            continue

        outcome = budget.record_success(debug_attempts, failure_trace)
        _attach_fitness(args, next_solution, result, generation + 1)
        next_solution['generation'] = generation + 1
        next_solution['attempt'] = attempt
        next_solution['debug_attempts'] = debug_attempts
        next_solution['failure_trace'] = failure_trace
        next_solution['meta_model'] = args.model
        next_solution['worker_model'] = args.worker_model
        next_solution['run_id'] = args.run_id
        next_solution['design_id'] = design_id
        next_solution['archive_context'] = archive_context
        next_solution['meta_reasoning'] = _meta_reasoning()
        next_solution['meta_usage'] = instrumentation.usage_for(generation + 1).get("meta", {})

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)
        instrumentation.write(os.path.join(args.save_dir, args.run_id))

    summary = budget.summary()
    summary["stopped_on_deadline"] = stopped_on_deadline
    print(f"search budget summary: {json.dumps(summary)}")
    os.makedirs(os.path.join(args.save_dir, args.run_id), exist_ok=True)
    with open(os.path.join(args.save_dir, args.run_id, "budget_summary.json"), 'w') as handle:
        json.dump(summary, handle, indent=2)
    instrumentation.write(os.path.join(args.save_dir, args.run_id))


# Stage 0.5 reference conditions. The bare agent is the floor for Q1 and the
# anchor for the Pareto front; solo CoT is what the informed meta agent is told
# its worker already achieves.
BARE_FORWARD = """def forward(self, taskInfo):
    agent = LLMAgentBase(['answer'], 'Bare Agent')
    answer = agent([taskInfo], "Solve the task. Reply with the final answer only.")[0]
    return answer
"""


def run_baseline(args):
    """Measure the worker with no searched scaffold.

    Routed through run_evaluation so the baseline is produced by exactly the
    machinery that scores every searched agent; a separately written evaluation
    path would not be comparable to the thing it anchors.
    """
    instrumentation = get_instrumentation()

    if args.baseline == "cot":
        entry = next(e for e in get_init_archive() if e.get("name") == "Chain-of-Thought")
        code, name = entry["code"], entry["name"]
    else:
        code, name = BARE_FORWARD, "Bare Model"

    medians, per_question, resources = [], [], []
    samples = []
    for rep in range(max(1, args.baseline_repeats)):
        instrumentation.set_context(phase="baseline",
                                    generation=f"{args.baseline}:rep{rep + 1}",
                                    attempt=rep + 1)
        result = run_evaluation(args, code)
        acc_list = result["acc_list"]
        medians.append(_point_accuracy(acc_list))
        per_question.append(acc_list)
        resources.append(result.get("usage", {}).get("worker", {}))
        if rep == 0:
            samples = result.get("samples") or []
            for s in samples[:5]:
                print(f"    target={s['target']!r} prediction={str(s.get('prediction'))[:70]!r} "
                      f"correct={s['correct']}")
        print(f"  {args.baseline} rep {rep + 1}/{args.baseline_repeats}: "
              f"accuracy {medians[-1]:.4f}")

    pooled = [outcome for rep in per_question for outcome in rep]
    payload = {
        "condition": args.baseline,
        "name": name,
        "worker_model": args.worker_model,
        "worker_reasoning_effort": args.worker_reasoning_effort,
        "domain": args.domain,
        "valid_size": args.valid_size,
        "shuffle_seed": args.shuffle_seed,
        "repeats": len(medians),
        # per_question is laid out as valid_size questions x n_repreat passes.
        "n_repreat": args.n_repreat,
        # Mean across repeats: this single number is what the meta prompt quotes.
        "accuracy": sum(medians) / len(medians),
        "accuracy_per_repeat": medians,
        "fitness": bootstrap_confidence_interval(pooled),
        "per_question": per_question,
        "resource": resources,
        "samples": samples,
        "code": code,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{args.baseline}_baseline.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"{name} baseline: {payload['accuracy']:.4f} over {payload['repeats']} repeats "
          f"-> {out_path}")
    instrumentation.write(os.path.join(args.save_dir, args.run_id))


def evaluate(args):
    instrumentation = get_instrumentation()
    instrumentation.set_context(phase="test")
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    if not os.path.exists(file_path):
        # Otherwise a search that produced nothing ends in a bare FileNotFoundError
        # here, hiding whatever actually went wrong upstream.
        raise SystemExit(
            f"no archive at {file_path}: the search produced no designs. "
            f"Check {os.path.join(args.save_dir, args.run_id)} for the failure taxonomy."
        )
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    stopped_on_deadline = False
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        sol = archive[current_idx]
        cap = _test_timeout(args, sol)
        # Designs whose test run failed are retried in place; results are matched by position.
        if current_idx < len(eval_archive) and not _should_retest(eval_archive[current_idx], cap):
            current_idx += 1
            continue
        if args.deadline_epoch and time.time() >= args.deadline_epoch:
            print(f"stopping test evaluation before design {current_idx}: within the walltime safety margin")
            stopped_on_deadline = True
            break
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}, test timeout: {cap:.0f}s")
        instrumentation.set_context(generation=sol['generation'], phase="test")
        idx = current_idx
        current_idx += 1
        try:
            result = run_evaluation(args, sol["code"], timeout=cap)
        except DeadlineReached as e:
            # Left unrecorded, so the resubmitted job runs it again with its full cap.
            print(f"stopping test evaluation at design {idx}: {e}")
            stopped_on_deadline = True
            break
        except Exception as e:
            print(e)
            instrumentation.record_failure(classify_failure(e), str(e), stage="test")
            sol['test_error'] = f"{type(e).__name__}: {e}"[:500]
            result = None
        sol['test_timeout_s'] = cap
        if result is not None:
            sol.pop('test_error', None)
            acc_list = result["acc_list"]
            sol['test_fitness'] = bootstrap_confidence_interval(acc_list)
            sol['test_median_accuracy'] = _point_accuracy(acc_list)
            sol['test_per_question'] = acc_list
            sol['test_crashed_questions'] = result.get("n_crashed", 0)
            if 'raw_median_accuracy' in sol:
                sol['generalization_gap'] = sol['raw_median_accuracy'] - sol['test_median_accuracy']
            sol['test_resource'] = result.get("usage", {}).get("worker", {})
        if idx < len(eval_archive):
            eval_archive[idx] = sol
        else:
            eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)
        instrumentation.write(os.path.join(args.save_dir, args.run_id))
    _write_evaluation_summary(args, eval_archive, len(archive), stopped_on_deadline)


def _test_timeout(args, sol):
    """Wall-clock cap for one design's test run.

    The validation cap does not carry over to a test split 6x larger: designs
    that validated in 17-55 minutes were cut off on test. A design gets its own
    validation time scaled by the split sizes, with 2x headroom, so one that ran
    on validation gets what it needs on test while one that hangs still stops.
    Never below the validation cap.
    """
    if args.test_timeout:
        return float(args.test_timeout)
    valid_wallclock = (sol.get('resource') or {}).get('wallclock_s') or 0.0
    return max(args.sandbox_timeout, 2.0 * valid_wallclock * args.test_size / max(args.valid_size, 1))


def _should_retest(entry, cap):
    """Retry a failed test run, unless it already timed out under at least this cap."""
    error = entry.get('test_error')
    if not error:
        return False
    if not error.startswith("SandboxTimeout"):
        return True
    used = entry.get('test_timeout_s')
    if used is None:
        # Records from before the cap was stored carry it only in the message.
        match = re.search(r"exceeded (\d+)s wall clock", error)
        used = float(match.group(1)) if match else 0.0
    return cap > used + 1


def _write_evaluation_summary(args, eval_archive, n_designs, stopped_on_deadline):
    """Read by maybe_resubmit: a test evaluation stopped by the deadline needs another job."""
    summary = {
        "designs": n_designs,
        "tested": sum(1 for e in eval_archive if not e.get('test_error')),
        "test_errors": sum(1 for e in eval_archive if e.get('test_error')),
        "untested": n_designs - len(eval_archive),
        "stopped_on_deadline": stopped_on_deadline,
    }
    print(f"test evaluation summary: {json.dumps(summary)}")
    run_dir = os.path.join(args.save_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "evaluation_summary.json"), 'w') as handle:
        json.dump(summary, handle, indent=2)


def run_evaluation(args, forward_str, timeout=None):
    """Evaluate one agent, in a sandbox unless explicitly disabled.

    The child returns plain data only, so nothing produced by the untrusted code
    is revived as a live object in the parent.

    ``timeout`` replaces --sandbox_timeout for this run. Under a deadline the run
    gets at most the time left before it, and a run the deadline cuts raises
    DeadlineReached instead of failing the design.
    """
    limit = float(timeout or args.sandbox_timeout)
    cut_by_deadline = False
    if args.deadline_epoch:
        left = args.deadline_epoch - time.time()
        if left <= 0:
            raise DeadlineReached("the walltime deadline has passed")
        if left < limit:
            limit, cut_by_deadline = left, True
    kwargs = {
        "args": args,
        "forward_str": forward_str,
        "worker_config": worker_client_config(build_role_clients(args)),
        "searching_mode": SEARCHING_MODE,
        "run_id": args.run_id,
        "generation": get_instrumentation().generation,
        "attempt": get_instrumentation().attempt,
        "phase": get_instrumentation().phase,
    }
    if args.no_sandbox:
        result = _sandboxed_evaluate(**kwargs)
    else:
        policy = SandboxPolicy.from_args(args, [args.worker_base_url, args.meta_base_url])
        policy.timeout_s = limit
        try:
            result = run_sandboxed(_ENTRY_MODULE, "_sandboxed_evaluate", kwargs, policy)
        except SandboxTimeout:
            if cut_by_deadline:
                raise DeadlineReached(f"cut after {limit:.0f}s at the walltime deadline") from None
            raise

    get_instrumentation().merge(result.pop("instrumentation", {}))
    return result


def _sandboxed_evaluate(args, forward_str, worker_config, searching_mode, run_id,
                        generation, attempt, phase):
    """Entry point executed inside the sandbox child."""
    global SEARCHING_MODE
    SEARCHING_MODE = searching_mode

    instrumentation = set_instrumentation(Instrumentation(run_id=run_id))
    instrumentation.set_context(generation=generation, attempt=attempt, phase=phase)
    configure_clients(worker=role_client_from_config(worker_config))

    started = time.time()
    acc_list, samples, n_crashed = evaluate_forward_fn(args, forward_str)
    return {
        "acc_list": acc_list,
        "samples": samples,
        "n_crashed": n_crashed,
        "wallclock_s": time.time() - started,
        "usage": instrumentation.usage_for(generation, phase=phase),
        "instrumentation": instrumentation.snapshot(),
    }


def _forward_for_mode(agent_system, crashes):
    """During search a crash propagates, so the debug loop sees it, as in ADAS.

    At test time the design is fixed: one crashing question is scored wrong
    instead of voiding the design's whole test score.
    """
    if SEARCHING_MODE:
        return agent_system.forward

    def forward(task):
        try:
            return agent_system.forward(task)
        except Exception as e:
            crashes.append(f"{type(e).__name__}: {e}"[:200])
            return None
    return forward


def evaluate_forward_fn(args, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # set seed 0 for valid set
    examples = get_all_examples()
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()
    crashes = []

    acc_list = []
    # A handful of scored pairs, so a zero score can be attributed to the model
    # being wrong rather than to the answer never being extracted.
    samples = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_forward_for_mode(agentSystem, crashes), task_queue), total=len(task_queue)))

    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            correct_answer = answers[q_idx]
            correct = score_mgsm(correct_answer, extracted_answer)
        except Exception as e:
            acc_list.append(0)
            if len(samples) < 8:
                samples.append({"target": answers[q_idx], "prediction": None,
                                "error": f"{type(e).__name__}: {e}"[:200], "correct": 0})
            continue

        if len(samples) < 8:
            samples.append({"target": correct_answer,
                            "prediction": str(extracted_answer)[:300],
                            "correct": int(bool(correct))})
        acc_list.append(1 if correct else 0)
    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    if crashes:
        print(f"crashed questions: {len(crashes)}, e.g. {crashes[0]}")
    return acc_list, samples, len(crashes)


def build_parser():
    """Exposed so the SLURM config layer's generated argument list can be checked offline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    # Tuned for a rate-limited hosted API in stock ADAS. Against local vLLM this
    # should match --max-num-seqs; it is the highest-leverage wall-clock knob.
    parser.add_argument('--max_workers', type=int, default=192)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default="mgsm_gpt3.5_results")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--domain', type=str, default='mgsm')

    # --- model routing ---
    # No hosted-API defaults: every model here is served locally, so falling back
    # to a GPT name would produce a confusing 404 rather than a clear omission.
    parser.add_argument('--model', type=str,
                        default=os.environ.get('ADAS_META_MODEL', os.environ.get('ADAS_MODEL')),
                        help="Meta agent: writes, reflects on and debugs agent code. "
                             "Omitted for baseline and random-recombination cells.")
    parser.add_argument('--worker_model', type=str,
                        default=os.environ.get('ADAS_WORKER_MODEL'),
                        help="Worker: executes inside the discovered agent.")
    parser.add_argument('--meta_base_url', type=str,
                        default=os.environ.get('ADAS_META_BASE_URL', 'http://127.0.0.1:8001/v1'))
    parser.add_argument('--worker_base_url', type=str,
                        default=os.environ.get('ADAS_WORKER_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--meta_temperature', type=float, default=0.8)
    parser.add_argument('--worker_temperature', type=float, default=0.5)
    # Completion budgets are per-model, not shared: reasoning traces draw on the
    # same allowance as the code, so a fixed limit is not a fairness control.
    parser.add_argument('--meta_max_tokens', type=int, default=8192)
    parser.add_argument('--worker_max_tokens', type=int, default=4096)
    parser.add_argument('--meta_reasoning_effort', type=str, default=None,
                        choices=[None, 'low', 'medium', 'high'])
    parser.add_argument('--worker_reasoning_effort', type=str, default=None,
                        choices=[None, 'low', 'medium', 'high'])
    parser.add_argument('--request_timeout', type=float, default=900.0)

    # --- budget control ---
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--target_generations', type=int, default=None,
                        help="Successful archive entries to collect. Defaults to --n_generation.")
    parser.add_argument('--max_attempts', type=int, default=None,
                        help="Total tries allowed. Defaults to 3x --target_generations.")

    # --- conditions ---
    parser.add_argument('--init_archive', type=str, default='seed', choices=['seed', 'empty'])
    parser.add_argument('--search_mode', type=str, default='meta', choices=['meta', 'random'])
    parser.add_argument('--no_reflection', action='store_true', default=False)
    parser.add_argument('--cost_lambda', type=float, default=0.0)
    parser.add_argument('--informed_meta', dest='informed_meta', action='store_true', default=True)
    parser.add_argument('--blind_meta', dest='informed_meta', action='store_false')
    parser.add_argument('--worker_cot_accuracy', type=float, default=None,
                        help="Measured solo-CoT accuracy of the worker, shown to an informed meta agent.")

    # --- sandbox ---
    parser.add_argument('--no_sandbox', action='store_true', default=False)
    parser.add_argument('--sandbox_timeout', type=float, default=3600.0)
    parser.add_argument('--test_timeout', type=float, default=None,
                        help="Flat cap per test run; by default each design's validation "
                             "time sets it (see _test_timeout).")
    parser.add_argument('--sandbox_memory_bytes', type=int, default=None)

    parser.add_argument('--search_only', action='store_true', default=False)
    parser.add_argument('--evaluate_only', action='store_true', default=False)
    parser.add_argument('--seed_cache_dir', type=str, default=None,
                        help="Shared seed-fitness cache; default is per cell under --save_dir.")
    parser.add_argument('--require_seed_cache', action='store_true', default=False)
    parser.add_argument('--seeds_only', action='store_true', default=False,
                        help="Evaluate the seed archive into the cache, then stop.")
    parser.add_argument('--deadline_epoch', type=float, default=None,
                        help="Unix time to stop starting new generations, so the job "
                             "hands off cleanly instead of being killed mid-generation.")

    # --- stage 0.5 ---
    parser.add_argument('--baseline', choices=['bare', 'cot'], default=None,
                        help="Measure the worker alone instead of searching. "
                             "'cot' writes cot_baseline.json, which the informed meta prompt quotes.")
    parser.add_argument('--baseline_repeats', type=int, default=1,
                        help="Independent re-evaluations. Stage 0.5 replicates via --n_repreat "
                             "instead, so raising both multiplies the cost.")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.run_id is None:
        args.run_id = args.expr_name
    if args.target_generations is None:
        args.target_generations = args.n_generation
    if args.seeds_only:
        args.search_only = True

    clients = build_role_clients(args)
    configure_clients(worker=clients.worker, meta=clients.meta)
    set_instrumentation(Instrumentation(run_id=args.run_id,
                                        output_dir=os.path.join(args.save_dir, args.run_id)))
    set_worker_documentation(
        worker_model=args.worker_model,
        worker_cot_accuracy=args.worker_cot_accuracy,
        worker_reasoning_effort=args.worker_reasoning_effort,
        informed=args.informed_meta,
    )

    if args.baseline:
        # Measured on the validation split, so the number quoted to the meta
        # agent is on the same questions the search will optimise against.
        SEARCHING_MODE = True
        run_baseline(args)
    else:
        if not args.evaluate_only:
            SEARCHING_MODE = True
            search(args)

        if not args.search_only:
            SEARCHING_MODE = False
            evaluate(args)

    get_instrumentation().write(os.path.join(args.save_dir, args.run_id))
