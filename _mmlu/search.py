import argparse
import copy
import json
import os
import random
import sys
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.adas_ext import (
    BudgetController,
    Instrumentation,
    SandboxError,
    SandboxPolicy,
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

from mmlu_prompt import get_init_archive, get_prompt, get_reflexion_prompt, set_worker_documentation

from utils import format_multichoice_question, random_id, bootstrap_confidence_interval

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
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
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
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


def _median(acc_list):
    return float(np.median(acc_list)) if len(acc_list) else 0.0


def _seed_cache_path(args):
    """Seed-archive fitness is cached per (worker, domain, split).

    Stock ADAS re-evaluates the seven seed agents on every run, which injects
    variance into the starting point of every search.
    """
    key = "__".join([
        str(args.worker_model).replace("/", "_"),
        str(getattr(args, "domain", "mmlu")),
        f"valid{args.valid_size}",
        f"shuffle{args.shuffle_seed}",
        f"rep{args.n_repreat}",
        str(getattr(args, "worker_reasoning_effort", None)),
    ])
    return os.path.join(args.save_dir, "seed_cache", f"{key}.json")


def _load_seed_cache(args):
    path = _seed_cache_path(args)
    if os.path.exists(path):
        with open(path, 'r') as handle:
            return json.load(handle)
    return {}


def _save_seed_cache(args, cache):
    path = _seed_cache_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(cache, handle, indent=2)


def _attach_fitness(args, solution, result, generation):
    """Write fitness plus every instrumented field onto an archive entry."""
    acc_list = result["acc_list"]
    solution['fitness'] = bootstrap_confidence_interval(acc_list)
    solution['raw_median_accuracy'] = _median(acc_list)
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
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        cached = seed_cache.get(solution['name'])
        if cached is not None:
            print(f"============Initial Archive (cached): {solution['name']}=================")
            solution.update(copy.deepcopy(cached))
            continue

        print(f"============Initial Archive: {solution['name']}=================")
        instrumentation.set_context(generation=f"initial:{solution['name']}")
        try:
            result = run_evaluation(args, solution["code"])
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

    # Equal successes, not equal attempts. See experiments/adas_ext/budget.py --
    # the stock `for n in range(...)` with `n -= 1` silently skips failed
    # generations, making search budget correlate with meta-agent capability.
    budget = BudgetController(args.target_generations, args.max_attempts, completed=start)
    while budget.should_continue():
        generation, attempt = budget.begin_attempt()
        print(f"============Generation {generation + 1} (attempt {attempt}/{budget.max_attempts})=================")
        instrumentation.set_context(generation=generation + 1, attempt=attempt, phase="search")
        failure_trace = []

        system_prompt, prompt = get_prompt(archive)
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
            instrumentation.record_abandoned(generation + 1, attempt, category, 0, None, failure_trace)
            continue

        result = None
        debug_attempts = 0
        terminal_failure = None
        for _ in range(args.debug_max):
            try:
                result = run_evaluation(args, next_solution["code"])
                if np.mean(result["acc_list"]) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                result = None
                debug_attempts += 1
                terminal_failure = classify_failure(e)
                print(f"During evaluation [{terminal_failure}]:")
                print(e)
                instrumentation.record_failure(terminal_failure, str(e), stage="evaluate")
                failure_trace.append({"stage": "evaluate", "category": terminal_failure, "message": str(e)[:2000]})
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                if args.search_mode == "random":
                    break
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    terminal_failure = classify_failure(e)
                    print(f"During LLM generate new solution [{terminal_failure}]:")
                    print(e)
                    instrumentation.record_failure(terminal_failure, str(e), stage="debug")
                    failure_trace.append({"stage": "debug", "category": terminal_failure, "message": str(e)[:2000]})
                    break

        if result is None:
            budget.record_failure(terminal_failure or "runtime_exception", debug_attempts, failure_trace)
            # Failed designs are kept. Discarding them biases every downstream
            # metric toward the subset each model could successfully produce.
            instrumentation.record_abandoned(generation + 1, attempt,
                                             terminal_failure or "runtime_exception",
                                             debug_attempts, next_solution, failure_trace)
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
    print(f"search budget summary: {json.dumps(summary)}")
    os.makedirs(os.path.join(args.save_dir, args.run_id), exist_ok=True)
    with open(os.path.join(args.save_dir, args.run_id, "budget_summary.json"), 'w') as handle:
        json.dump(summary, handle, indent=2)
    instrumentation.write(os.path.join(args.save_dir, args.run_id))


def evaluate(args):
    instrumentation = get_instrumentation()
    instrumentation.set_context(phase="test")

    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        instrumentation.set_context(generation=sol['generation'], phase="test")
        current_idx += 1
        try:
            result = run_evaluation(args, sol["code"])
        except Exception as e:
            print(e)
            instrumentation.record_failure(classify_failure(e), str(e), stage="test")
            continue
        acc_list = result["acc_list"]
        sol['test_fitness'] = bootstrap_confidence_interval(acc_list)
        sol['test_median_accuracy'] = _median(acc_list)
        sol['test_per_question'] = acc_list
        if 'raw_median_accuracy' in sol:
            sol['generalization_gap'] = sol['raw_median_accuracy'] - sol['test_median_accuracy']
        sol['test_resource'] = result.get("usage", {}).get("worker", {})
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)
        instrumentation.write(os.path.join(args.save_dir, args.run_id))


def run_evaluation(args, forward_str):
    """Evaluate one agent, in a sandbox unless explicitly disabled.

    The child returns plain data only, so nothing produced by the untrusted code
    is revived as a live object in the parent.
    """
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
        result = run_sandboxed(_ENTRY_MODULE, "_sandboxed_evaluate", kwargs, policy)

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
    acc_list = evaluate_forward_fn(args, forward_str)
    return {
        "acc_list": acc_list,
        "wallclock_s": time.time() - started,
        "usage": instrumentation.usage_for(generation, phase=phase),
        "instrumentation": instrumentation.snapshot(),
    }


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

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # set seed 0 for valid set
    df = pandas.read_csv(args.data_filename)
    random.seed(args.shuffle_seed)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    questions = [format_multichoice_question(example) for example in examples]
    answers = [LETTER_TO_INDEX[example['Answer']] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, str) and res in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res]
            elif 'A)' in res:
                predicted_idx = 0
            elif 'B)' in res:
                predicted_idx = 1
            elif 'C)' in res:
                predicted_idx = 2
            elif 'D)' in res:
                predicted_idx = 3
            elif isinstance(res, list):
                try_res = res[1]
                predicted_idx = LETTER_TO_INDEX[try_res.content]
            elif res.content in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res.content]
            elif 'A)' in res.content:
                predicted_idx = 0
            elif 'B)' in res.content:
                predicted_idx = 1
            elif 'C)' in res.content:
                predicted_idx = 2
            elif 'D)' in res.content:
                predicted_idx = 3
            else:
                print(f"error in q {q_idx}")
                acc_list.append(0)
                continue
        except Exception as e:
            acc_list.append(0)
            continue

        if predicted_idx == answers[q_idx]:
            acc_list.append(1)
        else:
            acc_list.append(0)
    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="dataset/mmlu.csv")
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
    parser.add_argument('--expr_name', type=str, default="mmlu_gpt3.5_results")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--domain', type=str, default='mmlu')

    # --- model routing ---
    parser.add_argument('--model', type=str,
                        default=os.environ.get('ADAS_META_MODEL', os.environ.get('ADAS_MODEL', 'gpt-4o-2024-05-13')),
                        help="Meta agent: writes, reflects on and debugs agent code.")
    parser.add_argument('--worker_model', type=str,
                        default=os.environ.get('ADAS_WORKER_MODEL', 'gpt-3.5-turbo-0125'),
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
    parser.add_argument('--sandbox_memory_bytes', type=int, default=None)

    parser.add_argument('--search_only', action='store_true', default=False)
    parser.add_argument('--evaluate_only', action='store_true', default=False)

    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = args.expr_name
    if args.target_generations is None:
        args.target_generations = args.n_generation

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

    if not args.evaluate_only:
        SEARCHING_MODE = True
        search(args)

    if not args.search_only:
        SEARCHING_MODE = False
        evaluate(args)

    get_instrumentation().write(os.path.join(args.save_dir, args.run_id))
