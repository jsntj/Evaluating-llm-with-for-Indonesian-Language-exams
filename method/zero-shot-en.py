"""
Zero-shot evaluation script for IndoMMLU dataset (multi-model, English prompt).
Evaluates questions using SeaLLMs, Aya, and Sailor models with English prompts.
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

def get_subject_level_groups(csv_path, num_groups=20, ensure_subject_diversity=True):
    """
    Get top N subject-level groups from the dataset, ensuring subject diversity
    
    Args:
        csv_path: Path to the CSV file
        num_groups: Number of subject-level groups to select (default: 20)
        ensure_subject_diversity: If True, ensures we get diverse subjects (default: True)
    
    Returns:
        List of tuples (subject, level, count) sorted by count
    """
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    groups = df.groupby(['subject', 'level']).size().reset_index(name='count')
    
    if ensure_subject_diversity:
        # First, get unique subjects and their total question counts
        subject_totals = df.groupby('subject').size().reset_index(name='total_count')
        subject_totals = subject_totals.sort_values('total_count', ascending=False)
        
        # Select top subjects (aim for diversity, but ensure we can get num_groups total)
        # Try to get at least num_groups/3 unique subjects, but cap at reasonable number
        num_subjects = min(len(subject_totals), max(10, num_groups // 3))
        top_subjects = subject_totals.head(num_subjects)['subject'].tolist()
        
        # For each subject, get its top groups
        selected_groups = []
        groups_per_subject = max(1, (num_groups + num_subjects - 1) // num_subjects)  # Ceiling division to ensure we get enough
        
        for subject in top_subjects:
            subject_groups = groups[groups['subject'] == subject].sort_values('count', ascending=False)
            # Take top groups for this subject
            selected = subject_groups.head(groups_per_subject)
            selected_groups.append(selected)
            
            # Stop if we have enough groups (count total groups collected so far)
            total_collected = sum(len(df) for df in selected_groups)
            if total_collected >= num_groups:
                break
        
        # Combine and sort by count
        if selected_groups:
            result = pd.concat(selected_groups, ignore_index=True)
            result = result.sort_values('count', ascending=False)
            # Take exactly num_groups
            result = result.head(num_groups)
            return result[['subject', 'level', 'count']].values.tolist()
        else:
            # Fallback to original method
            groups = groups.sort_values('count', ascending=False)
            return groups.head(num_groups)[['subject', 'level', 'count']].values.tolist()
    else:
        # Original method: just get top N groups
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
    return f"""question: {question}

select the answers:
{choices}
Answer by choosing the most appropriate letter (A, B, C, D, or E). Answer::"""

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

# Models to evaluate (zero-shot, English prompt)
MODEL_NAMES = [
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "CohereLabs/aya-expanse-8b",
    "sail/Sailor-7B-Chat",
    "google/gemma-3-12b-it",

]


def evaluate_zero_shot(model, tokenizer, questions, model_name=""):
    """Evaluate questions using zero-shot approach (English prompt)"""
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
            'question': q.get('question', ''),
            'choices': q.get('choices', ''),
            'correct_answer': q['correct_answer'],
            'prediction': prediction,
            'is_correct': is_correct,
            'response': response
        })

    total_time = time.time() - total_start
    if len(questions) > 0:
        print(f"\n  Timing: {len(questions)} questions in {total_time:.2f}s (avg {total_time/len(questions):.2f}s/q)")
    return results

def load_and_evaluate_model(model_name, groups, csv_path, num_questions_per_group):
    """Load a model and evaluate it on all groups (zero-shot, English prompt)."""
    print(f"\n{'=' * 80}")
    print(f"Loading and evaluating (zero-shot EN): {model_name}")
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
        print(f"Evaluating with {model_name} (zero-shot, English)")
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
    print("Zero-Shot Evaluation (English Prompt) - Multi-Model")
    print("Models: SeaLLMs-v3-7B, Aya-Expanse-8B, Sailor-7B")
    print("Evaluating 20 Subject-Level Groups")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "IndoMMLU.csv")
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
    out("COMPARATIVE SUMMARY (Zero-Shot, English)")
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


def save_results_to_csv(all_model_results, output_path):
    """Save zero-shot results to CSV file with question, ground_truth, prediction, response"""
    csv_rows = []
    for model_result in all_model_results:
        if 'error' in model_result:
            continue
        for r in model_result.get('results', []):
            csv_rows.append({
                'question': r.get('question', ''),
                'ground_truth': r.get('correct_answer', 'N/A'),
                'prediction': r.get('prediction', 'N/A'),
                'response': r.get('response', '')
            })
    
    if csv_rows:
        df_out = pd.DataFrame(csv_rows)
        df_out.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Saved zero-shot results to CSV: {output_path}")
        print(f"  Columns: question, ground_truth, prediction, response")
        print(f"  Total rows: {len(csv_rows)}")
        return True
    else:
        print(f"⚠ No results available to save to CSV.")
        return False


def escape_latex(text):
    """Escape LaTeX special characters in text"""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '\\': '\\textbackslash{}',
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def save_results_to_latex(all_model_results, output_path):
    """Save results summary to LaTeX format"""
    latex_content = []
    latex_content.append("\\documentclass{article}")
    latex_content.append("\\usepackage{booktabs}")
    latex_content.append("\\usepackage{longtable}")
    latex_content.append("\\usepackage{array}")
    latex_content.append("\\usepackage{multirow}")
    latex_content.append("\\usepackage{geometry}")
    latex_content.append("\\geometry{a4paper, margin=1in}")
    latex_content.append("\\begin{document}")
    latex_content.append("")
    latex_content.append("\\title{Zero-Shot Evaluation Results (English)}")
    latex_content.append("\\author{IndoMMLU Dataset}")
    latex_content.append("\\maketitle")
    latex_content.append("")
    
    latex_content.append("\\section{Comparative Summary}")
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\begin{tabular}{lcc}")
    latex_content.append("\\toprule")
    latex_content.append("Model & Accuracy (\\%) & Correct/Total \\\\")
    latex_content.append("\\midrule")
    
    for model_result in all_model_results:
        model_name = model_result['model_name']
        model_name_escaped = escape_latex(model_name)
        
        if 'error' in model_result:
            latex_content.append(f"{model_name_escaped} & ERROR & -- \\\\")
        else:
            accuracy = model_result['accuracy']
            correct = model_result['correct_count']
            total = model_result['total_questions']
            if total > 0:
                latex_content.append(f"{model_name_escaped} & {accuracy:.2f} & {correct}/{total} \\\\")
            else:
                latex_content.append(f"{model_name_escaped} & -- & 0/0 \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\caption{Overall accuracy comparison across all models}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    valid_results = [r for r in all_model_results if 'error' not in r and r['total_questions'] > 0]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x['accuracy'])
        best_name_escaped = escape_latex(best_model['model_name'])
        latex_content.append(f"\\textbf{{Best Model:}} {best_name_escaped} with {best_model['accuracy']:.2f}\\% accuracy.")
        latex_content.append("")
    
    latex_content.append("\\section{Detailed Statistics by Model}")
    latex_content.append("")
    
    for model_result in all_model_results:
        if 'error' in model_result or model_result['total_questions'] == 0:
            continue
        
        model_name = model_result['model_name']
        model_name_escaped = escape_latex(model_name)
        results = model_result['results']
        
        latex_content.append(f"\\subsection{{{model_name_escaped}}}")
        latex_content.append("")
        
        subject_stats = {}
        for r in results:
            subject = r['subject']
            if subject not in subject_stats:
                subject_stats[subject] = {'total': 0, 'correct': 0}
            subject_stats[subject]['total'] += 1
            if r['is_correct']:
                subject_stats[subject]['correct'] += 1
        
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\begin{tabular}{lcc}")
        latex_content.append("\\toprule")
        latex_content.append("Subject & Correct/Total & Accuracy (\\%) \\\\")
        latex_content.append("\\midrule")
        
        for subject in sorted(subject_stats.keys()):
            stats = subject_stats[subject]
            subj_accuracy = stats['correct'] / stats['total'] * 100
            subject_escaped = escape_latex(subject)
            latex_content.append(f"{subject_escaped} & {stats['correct']}/{stats['total']} & {subj_accuracy:.2f} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append(f"\\caption{{Accuracy by subject for {model_name_escaped}}}")
        latex_content.append("\\end{table}")
        latex_content.append("")
    
    latex_content.append("\\end{document}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Saved LaTeX results to: {output_path}")
    return True

    out("\n\n" + "=" * 80)
    out("DETAILED STATISTICS BY MODEL (Zero-Shot EN)")
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

    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "results_zero_shot_en_summary.txt")
    csv_path = os.path.join(results_dir, "results_zero_shot_en_results.csv")
    latex_path = os.path.join(results_dir, "results_zero_shot_en_results.tex")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n✓ Result summary written to {summary_path}")
    
    # Save CSV results
    save_results_to_csv(all_model_results, csv_path)
    
    # Save LaTeX results
    save_results_to_latex(all_model_results, latex_path)


if __name__ == "__main__":
    main()
