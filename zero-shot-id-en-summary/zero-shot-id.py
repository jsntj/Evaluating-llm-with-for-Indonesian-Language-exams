"""
Zero-shot evaluation script for IndoMMLU dataset (multi-model, Indonesian prompt).
Evaluates questions using SeaLLMs, Aya, and Sailor models with Indonesian prompts.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from tqdm import tqdm
import time

# Set up environment variables for model caching
# Note: These should be set by the shell script (run_zero_shot.sh)
# If running directly, ensure USER_ID and HF_TOKEN are set in environment
user_id = os.environ.get('USER_ID', '')
if not user_id:
    raise ValueError(
        "USER_ID environment variable is not set. "
        "Please run the script using run_zero_shot.sh or set USER_ID in your environment."
    )

# Set cache directories based on USER_ID
work_dir = f'/work/{user_id}'
os.environ['HF_HOME'] = f'{work_dir}/huggingface'
os.environ['HF_HUB_CACHE'] = f'{work_dir}/huggingface/hub'
os.environ['VLLM_CACHE_ROOT'] = f'{work_dir}/'
os.environ['XDG_CACHE_HOME'] = f'{work_dir}/'

# Check if HF_TOKEN is set
if 'HF_TOKEN' not in os.environ or not os.environ['HF_TOKEN']:
    raise ValueError(
        "HF_TOKEN environment variable is not set. "
        "Please set your Hugging Face token in the shell script or environment."
    )

def get_subject_level_groups(csv_path, num_groups=20):
    """
    Get top N subject-level groups from the IndoMMLU dataset.

    Args:
        csv_path: Path to the IndoMMLU dataset CSV
        num_groups: Number of subject-level groups to select (default: 20)

    Returns:
        List of tuples (subject, level, count) sorted by count
    """
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    groups = df.groupby(['subject', 'level']).size().reset_index(name='count')
    groups = groups.sort_values('count', ascending=False)
    return groups.head(num_groups)[['subject', 'level', 'count']].values.tolist()

def load_indommlu_dataset(csv_path, subject=None, level=None, num_questions=10):
    """
    Load questions from the IndoMMLU dataset CSV (columns: id, subject, level, soal, jawaban, kunci).

    Args:
        csv_path: Path to the IndoMMLU dataset CSV
        subject: Filter by subject (optional)
        level: Filter by level (optional)
        num_questions: Number of questions to load per group (default: 10)

    Returns:
        List of dictionaries containing question data
    """
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    
    # Filter by subject and level if provided
    if subject is not None:
        df = df[df['subject'] == subject]
    if level is not None:
        df = df[df['level'] == level]
    
    # Sample questions (up to num_questions)
    num_to_load = min(num_questions, len(df))
    df_sample = df.sample(n=num_to_load, random_state=42) if len(df) > num_to_load else df
    
    questions = []
    for idx, row in df_sample.iterrows():
        questions.append({
            'id': row['id'],
            'subject': row['subject'],
            'level': row['level'],
            'question': row['soal'],
            'choices': row['jawaban'],
            'correct_answer': row['kunci']
        })
    
    return questions

def format_prompt(question, choices):
    """Format the question and choices into a prompt for the model"""
    return f"""Pertanyaan: {question}

Pilihan jawaban:
{choices}

Jawablah dengan memilih salah satu huruf (A, B, C, D, atau E) yang paling tepat. Jawaban:"""

def extract_answer(response):
    """Extract answer letter (A-E) from model response"""
    response_upper = response.upper()
    
    # Look for common answer patterns
    patterns = [
        r'\b([A-E])\b',  # Standalone letter
        r'jawaban[:\s]+([A-E])',
        r'pilihan[:\s]+([A-E])',
        r'option[:\s]+([A-E])'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
    
    # Check first few words for letter
    words = response_upper.split()[:10]
    for word in words:
        if word in ['A', 'B', 'C', 'D', 'E']:
            return word
    
    return None

def evaluate_zero_shot(model, tokenizer, questions, model_name=""):
    """Evaluate questions using zero-shot approach (Indonesian prompt)"""
    results = []
    total_start = time.time()

    desc = f"Evaluating questions ({model_name})" if model_name else "Evaluating questions"
    for q in tqdm(questions, desc=desc, unit="question"):
        prompt_text = format_prompt(q['question'], q['choices'])
        messages = [{"role": "user", "content": prompt_text}]

        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            input_text = prompt_text
        inputs = tokenizer(input_text, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]

        response = tokenizer.decode(generation, skip_special_tokens=True)
        prediction = extract_answer(response)
        is_correct = (prediction == q['correct_answer'])
        results.append({
            'id': q['id'],
            'subject': q['subject'],
            'level': q.get('level', 'N/A'),
            'correct_answer': q['correct_answer'],
            'prediction': prediction,
            'is_correct': is_correct,
            'response': response
        })

    total_time = time.time() - total_start
    if len(questions) > 0:
        print(f"\n  Timing: {len(questions)} questions in {total_time:.2f}s (avg {total_time/len(questions):.2f}s/q)")
    return results

# Models to evaluate (zero-shot, Indonesian prompt)
MODEL_NAMES = [
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "CohereLabs/aya-expanse-8b",
    "sail/Sailor-7B-Chat",
    "google/gemma-3-12b-it",
]


def load_and_evaluate_model(model_name, groups, csv_path, num_questions_per_group):
    """Load a model and evaluate it on all groups (zero-shot, Indonesian prompt)."""
    print(f"\n{'=' * 80}")
    print(f"Loading and evaluating (zero-shot ID): {model_name}")
    print(f"{'=' * 80}")
    print("This may take a while on first run as the model will be downloaded...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ['HF_TOKEN'],
            cache_dir=os.environ['HF_HUB_CACHE'],
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ['HF_TOKEN'],
            cache_dir=os.environ['HF_HUB_CACHE'],
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()

        print("Model loaded successfully!")
        model_device = next(model.parameters()).device
        print(f"Model device: {model_device}")
        if model_device.type == 'cpu':
            print("⚠ WARNING: Model is on CPU! This will be very slow. Check GPU availability.")
        else:
            print(f"✓ Model is on {model_device}")

        all_results = []
        print("\n" + "=" * 80)
        print(f"Evaluating with {model_name} (zero-shot, Indonesian)")
        print("=" * 80)

        for group_idx, (subject, level, total_available) in enumerate(groups, 1):
            print(f"\n[{group_idx}/{len(groups)}] Evaluating: {subject} ({level})")
            print(f"  Available: {total_available}, Sampling: {num_questions_per_group}")

            try:
                questions = load_indommlu_dataset(
                    csv_path, subject=subject, level=level, num_questions=num_questions_per_group
                )
            except Exception as e:
                print(f"  ❌ ERROR loading questions: {e}")
                continue

            if len(questions) == 0:
                print(f"  ⚠ No questions for {subject} ({level})")
                continue
            print(f"  ✓ Loaded {len(questions)} questions")

            group_results = evaluate_zero_shot(model, tokenizer, questions, model_name)
            for r in group_results:
                r['model'] = model_name
            all_results.extend(group_results)

            if group_results:
                c = sum(1 for r in group_results if r['is_correct'])
                print(f"  Results: {c}/{len(group_results)} correct ({c/len(group_results)*100:.2f}%)")

        total = len(all_results)
        correct = sum(1 for r in all_results if r['is_correct'])
        acc = correct / total * 100 if total > 0 else 0

        print(f"\n  Cleaning up {model_name} from memory...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ✓ Cleanup complete.")

        return {
            'model_name': model_name,
            'results': all_results,
            'total_questions': total,
            'correct_count': correct,
            'accuracy': acc
        }
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR loading/evaluating {model_name}: {e}")
        traceback.print_exc()
        return {
            'model_name': model_name,
            'results': [],
            'total_questions': 0,
            'correct_count': 0,
            'accuracy': 0.0,
            'error': str(e)
        }


def main():
    """Main function to run zero-shot evaluation on SeaLLMs, Aya, and Sailor."""
    print("=" * 80)
    print("Zero-Shot Evaluation (Indonesian Prompt) - Multi-Model")
    print("Models: SeaLLMs-v3-7B, Aya-Expanse-8B, Sailor-7B")
    print("Evaluating 20 Subject-Level Groups")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "IndoMMLU.csv")
    print(f"Dataset: {csv_path}")

    num_groups = 20
    num_questions_per_group = 10
    print(f"\nModels to evaluate ({len(MODEL_NAMES)}):")
    for i, n in enumerate(MODEL_NAMES, 1):
        print(f"  {i}. {n}")

    print(f"\nAnalyzing dataset for top {num_groups} subject-level groups...")
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV not found at {csv_path}")
        return

    groups = get_subject_level_groups(csv_path, num_groups=num_groups)
    if not groups:
        print("❌ ERROR: No subject-level groups found. Check CSV columns 'subject' and 'level'.")
        return

    print(f"Selected {len(groups)} groups:")
    for i, (s, lv, cnt) in enumerate(groups, 1):
        print(f"  {i}. {s} ({lv}) - {cnt} questions")

    all_model_results = []
    for idx, model_name in enumerate(MODEL_NAMES, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# MODEL {idx}/{len(MODEL_NAMES)}: {model_name}")
        print(f"{'#' * 80}")
        res = load_and_evaluate_model(model_name, groups, csv_path, num_questions_per_group)
        all_model_results.append(res)
        if 'error' not in res:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY: {model_name}")
            print(f"  Total: {res['total_questions']}, Correct: {res['correct_count']}, Accuracy: {res['accuracy']:.2f}%")
            print(f"{'=' * 80}")

    total_eval = sum(r['total_questions'] for r in all_model_results if 'error' not in r)
    if total_eval == 0:
        print("\n❌ NO RESULTS. Check errors above, CSV format, and HF_TOKEN.")
        return

    summary_lines = []

    def out(s):
        print(s)
        summary_lines.append(s)

    out("\n\n" + "=" * 80)
    out("COMPARATIVE SUMMARY (Zero-Shot, Indonesian)")
    out("=" * 80)
    out(f"{'Model':<50} {'Accuracy':<12} {'Correct/Total':<16}")
    out("-" * 80)
    for r in all_model_results:
        name = r['model_name']
        if 'error' in r:
            out(f"{name:<50} {'ERROR':<12} {str(r.get('error', ''))[:40]}")
        elif r['total_questions'] > 0:
            out(f"{name:<50} {r['accuracy']:>6.2f}%     {r['correct_count']}/{r['total_questions']}")
        else:
            out(f"{name:<50} {'NO DATA':<12} 0/0")
    out("=" * 80)

    valid = [x for x in all_model_results if 'error' not in x and x['total_questions'] > 0]
    if valid:
        best = max(valid, key=lambda x: x['accuracy'])
        out(f"\n🏆 Best: {best['model_name']} — {best['accuracy']:.2f}%")
    else:
        out("\n⚠ No valid results to pick best model.")

    out("\n\n" + "=" * 80)
    out("DETAILED STATISTICS BY MODEL (Zero-Shot ID)")
    out("=" * 80)
    for r in all_model_results:
        if 'error' in r or r['total_questions'] == 0:
            continue
        out(f"\n{'-' * 80}\nModel: {r['model_name']}\n{'-' * 80}")
        by_subj = {}
        for x in r['results']:
            s = x['subject']
            if s not in by_subj:
                by_subj[s] = {'total': 0, 'correct': 0}
            by_subj[s]['total'] += 1
            if x['is_correct']:
                by_subj[s]['correct'] += 1
        out(f"{'Subject':<40} {'Accuracy':<20}")
        out("-" * 60)
        for s in sorted(by_subj.keys()):
            st = by_subj[s]
            acc = st['correct'] / st['total'] * 100
            out(f"{s:<40} {st['correct']}/{st['total']} ({acc:.2f}%)")

    summary_path = os.path.join(script_dir, "results_zero_shot_id_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n✓ Result summary written to {summary_path}")


if __name__ == "__main__":
    main()
