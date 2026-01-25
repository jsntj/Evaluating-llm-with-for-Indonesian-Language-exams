"""
Chain of Thought (COT) with Simple Reasoning and Example evaluation script for IndoMMLU dataset.
This script evaluates questions using a simple reasoning prompt that requests an example (English version).

Uses IndoMMLU.csv dataset with hardcoded models.

Outputs: Summary TXT, CSV with reasoning results, and LaTeX tables.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from tqdm import tqdm
import time

# Set up environment variables for model caching
user_id = os.environ.get('USER_ID', '')
if not user_id:
    raise ValueError(
        "USER_ID environment variable is not set. "
        "Please run the script using run_cot_reason_eg.sh or set USER_ID in your environment."
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
    """Get top N subject-level groups from the dataset, ensuring subject diversity"""
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    groups = df.groupby(['subject', 'level']).size().reset_index(name='count')
    
    if ensure_subject_diversity:
        subject_totals = df.groupby('subject').size().reset_index(name='total_count')
        subject_totals = subject_totals.sort_values('total_count', ascending=False)
        num_subjects = max(15, min(len(subject_totals), num_groups // 2))
        top_subjects = subject_totals.head(num_subjects)['subject'].tolist()
        
        selected_groups = []
        groups_per_subject = max(1, num_groups // num_subjects)
        
        for subject in top_subjects:
            subject_groups = groups[groups['subject'] == subject].sort_values('count', ascending=False)
            selected = subject_groups.head(groups_per_subject)
            selected_groups.append(selected)
            
            total_collected = sum(len(df) for df in selected_groups)
            if total_collected >= num_groups:
                break
        
        if selected_groups:
            result = pd.concat(selected_groups, ignore_index=True)
            result = result.sort_values('count', ascending=False)
            result = result.head(num_groups)
            return result[['subject', 'level', 'count']].values.tolist()
        else:
            groups = groups.sort_values('count', ascending=False)
            return groups.head(num_groups)[['subject', 'level', 'count']].values.tolist()
    else:
        groups = groups.sort_values('count', ascending=False)
        return groups.head(num_groups)[['subject', 'level', 'count']].values.tolist()

def load_indommlu_dataset(csv_path, subject=None, level=None, num_questions=10):
    """Load questions from IndoMMLU.csv, optionally filtered by subject and level"""
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    
    if subject is not None:
        df = df[df['subject'] == subject]
    if level is not None:
        df = df[df['level'] == level]
    
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
    """Format the question with a prompt requesting detailed thought process reasoning with examples."""
    return f"""Question: {question}

Choices:
{choices}

Please provide your thought process and reasoning for this question. Explain step by step how you analyze the question, evaluate each choice option, and use examples or analogies to support your reasoning. After explaining your thought process with examples, provide your final answer in the format "Final Answer: X" where X is the letter (A, B, C, D, or E)."""

def extract_answer(response):
    """Extract answer letter (A-E) from model response, prioritizing 'Final Answer:' pattern"""
    response_upper = response.upper()
    
    patterns = [
        r'final\s+answer[:\s]+([A-E])',
        r'answer[:\s]+([A-E])',
        r'jawaban[:\s]+([A-E])',
        r'pilihan[:\s]+([A-E])',
        r'option[:\s]+([A-E])',
        r'\b([A-E])\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
    
    lines = response_upper.split('\n')
    for line in reversed(lines[-5:]):
        words = line.split()
        for word in words:
            if word in ['A', 'B', 'C', 'D', 'E']:
                return word
    
    words = response_upper.split()[:10]
    for word in words:
        if word in ['A', 'B', 'C', 'D', 'E']:
            return word
    
    return None

def extract_reasoning(response):
    """Extract the reasoning chain from the model response"""
    return response.strip()

# Models to evaluate (hardcoded)
MODEL_NAMES = [
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "CohereLabs/aya-expanse-8b",
    "sail/Sailor-7B-Chat",
    "google/gemma-3-12b-it",
]

def load_model_names(seallm_file_path):
    """Load model names from seallm.txt file (DEPRECATED - using MODEL_NAMES instead)"""
    model_mapping = {
        "SeaLLMs-v3-7B-Chat": "SeaLLMs/SeaLLMs-v3-7B-Chat",
        "CohereLabs/aya-expanse-8b": "CohereLabs/aya-expanse-8b",
        "Sailor-7B-Chat": "sail/Sailor-7B-Chat",
        "google/gemma-3-12b-it": "google/gemma-3-12b-it"
    }
    
    excluded_models = []
    
    try:
        with open(seallm_file_path, 'r') as f:
            model_names = [line.strip() for line in f if line.strip()]
        
        model_names = [name for name in model_names if name not in excluded_models]
        
        full_model_names = []
        for name in model_names:
            if name in model_mapping:
                mapped_name = model_mapping[name]
                if mapped_name not in excluded_models:
                    full_model_names.append(mapped_name)
            else:
                if name not in excluded_models:
                    full_model_names.append(name)
        
        if len(model_names) != len(full_model_names):
            excluded_count = len(model_names) - len(full_model_names)
            print(f"  Note: Excluded {excluded_count} model(s)")
        
        return full_model_names
    except FileNotFoundError:
        print(f"Warning: {seallm_file_path} not found. Using default models.")
        default_models = [
            "SeaLLMs/SeaLLMs-v3-7B-Chat",
            "sail/Sailor-7B-Chat",
            "google/gemma-3-12b-it",
            "CohereLabs/aya-expanse-8b"
        ]
        return default_models

def evaluate_cot(model, tokenizer, questions, model_name=""):
    """Evaluate questions using Chain of Thought approach"""
    if len(questions) == 0:
        print("  ⚠ Warning: No questions provided to evaluate_cot")
        return []
    
    results = []
    total_start = time.time()
    tokenize_time = 0
    generate_time = 0
    decode_time = 0
    
    for q_idx, q in enumerate(tqdm(questions, desc="Evaluating questions", unit="question"), 1):
        try:
            prompt_text = format_prompt(q['question'], q['choices'])
            messages = [{"role": "user", "content": prompt_text}]
            
            t0 = time.time()
            try:
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"\n  ⚠ Warning: Chat template failed for question {q_idx}, using direct prompt: {str(e)}")
                input_text = prompt_text
            
            inputs = tokenizer(input_text, return_tensors="pt")
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            tokenize_time += time.time() - t0
            
            t1 = time.time()
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                generation = generation[0][input_len:]
            generate_time += time.time() - t1
            
            t2 = time.time()
            response = tokenizer.decode(generation, skip_special_tokens=True)
            decode_time += time.time() - t2
            
            prediction = extract_answer(response)
            reasoning = extract_reasoning(response)
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
                'response': response,
                'reasoning': reasoning
            })
        
        except Exception as e:
            print(f"\n  ❌ ERROR evaluating question {q_idx} (ID: {q.get('id', 'unknown')}): {str(e)}")
            results.append({
                'id': q.get('id', 'unknown'),
                'subject': q.get('subject', 'N/A'),
                'level': q.get('level', 'N/A'),
                'question': q.get('question', ''),
                'choices': q.get('choices', ''),
                'correct_answer': q.get('correct_answer', 'N/A'),
                'prediction': None,
                'is_correct': False,
                'response': f"ERROR: {str(e)}",
                'reasoning': f"ERROR: {str(e)}"
            })
            continue
    
    total_time = time.time() - total_start
    if len(questions) > 0:
        avg_time = total_time / len(questions)
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        print(f"\n  Evaluation complete: {len(results)}/{len(questions)} questions processed")
        if len(results) > 0:
            print(f"  Correct answers: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
        print(f"  Timing stats: {len(questions)} questions in {total_time:.2f}s")
        print(f"    Avg per question: {avg_time:.2f}s")
    
    return results

def load_and_evaluate_model(model_name, groups, csv_path, num_questions_per_group):
    """Load a model and evaluate it on all groups"""
    print(f"\n{'=' * 80}")
    print(f"Loading and evaluating model: {model_name}")
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
            torch_dtype=torch.bfloat16,
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
        print(f"Evaluating questions with {model_name}")
        print("=" * 80)
        
        for group_idx, (subject, level, total_available) in enumerate(groups, 1):
            print(f"\n[{group_idx}/{len(groups)}] Evaluating: {subject} ({level})")
            print(f"  Available questions: {total_available}, Sampling: {num_questions_per_group}")
            
            try:
                questions = load_indommlu_dataset(
                    csv_path, 
                    subject=subject, 
                    level=level, 
                    num_questions=num_questions_per_group
                )
            except Exception as e:
                print(f"  ❌ ERROR loading questions for {subject} ({level}): {str(e)}")
                continue
            
            if len(questions) == 0:
                print(f"  ⚠ Warning: No questions found for {subject} ({level})")
                continue
            
            print(f"  ✓ Loaded {len(questions)} questions")
            
            group_results = evaluate_cot(model, tokenizer, questions, model_name)
            for r in group_results:
                r['model'] = model_name
            all_results.extend(group_results)
            
            if len(group_results) > 0:
                group_correct = sum(1 for r in group_results if r['is_correct'])
                group_accuracy = group_correct / len(group_results) * 100
                print(f"  Results: {group_correct}/{len(group_results)} correct ({group_accuracy:.2f}%)")
        
        total_questions = len(all_results)
        correct_count = sum(1 for r in all_results if r['is_correct'])
        accuracy = correct_count / total_questions * 100 if total_questions > 0 else 0
        
        print(f"\n  Cleaning up {model_name} from memory...")
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  ✓ Cleanup complete. Ready for next model.")
        
        return {
            'model_name': model_name,
            'results': all_results,
            'total_questions': total_questions,
            'correct_count': correct_count,
            'accuracy': accuracy
        }
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR loading/evaluating model {model_name}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        traceback.print_exc()
        return {
            'model_name': model_name,
            'results': [],
            'total_questions': 0,
            'correct_count': 0,
            'accuracy': 0.0,
            'error': str(e)
        }

def save_results_to_csv(all_model_results, output_path):
    """Save reasoning results to CSV file with question, ground_truth, prediction, reasoning"""
    csv_rows = []
    for model_result in all_model_results:
        if 'error' in model_result:
            continue
        for r in model_result.get('results', []):
            csv_rows.append({
                'question': r.get('question', ''),
                'ground_truth': r.get('correct_answer', 'N/A'),
                'prediction': r.get('prediction', 'N/A'),
                'reasoning': r.get('reasoning', r.get('response', ''))
            })
    
    if csv_rows:
        df_out = pd.DataFrame(csv_rows)
        df_out.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Saved reasoning results to CSV: {output_path}")
        print(f"  Columns: question, ground_truth, prediction, reasoning")
        print(f"  Total rows: {len(csv_rows)}")
        return True
    else:
        print(f"⚠ No results available to save to CSV.")
        return False

def save_results_to_txt(all_model_results, output_path):
    """Save summary results to TXT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHAIN OF THOUGHT REASONING EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("COMPARATIVE SUMMARY - ALL MODELS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<50} {'Accuracy':<15} {'Correct/Total':<20}\n")
        f.write("-" * 80 + "\n")
        
        for model_result in all_model_results:
            model_name = model_result['model_name']
            if 'error' in model_result:
                error_msg = model_result.get('error', 'Unknown error')[:30]
                f.write(f"{model_name:<50} {'ERROR':<15} {error_msg:<20}\n")
            else:
                accuracy = model_result['accuracy']
                correct = model_result['correct_count']
                total = model_result['total_questions']
                if total > 0:
                    f.write(f"{model_name:<50} {accuracy:>6.2f}%{'':<8} {correct}/{total}\n")
                else:
                    f.write(f"{model_name:<50} {'NO DATA':<15} {'0/0':<20}\n")
        
        f.write("=" * 80 + "\n\n")
        
        valid_results = [r for r in all_model_results if 'error' not in r and r['total_questions'] > 0]
        if valid_results:
            best_model = max(valid_results, key=lambda x: x['accuracy'])
            f.write(f"Best Model: {best_model['model_name']} with {best_model['accuracy']:.2f}% accuracy.\n\n")
        
        f.write("DETAILED STATISTICS BY MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        for model_result in all_model_results:
            if 'error' in model_result or model_result['total_questions'] == 0:
                continue
            
            model_name = model_result['model_name']
            results = model_result['results']
            
            f.write(f"\n{'-' * 80}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'-' * 80}\n\n")
            
            subject_stats = {}
            for r in results:
                subject = r['subject']
                if subject not in subject_stats:
                    subject_stats[subject] = {'total': 0, 'correct': 0}
                subject_stats[subject]['total'] += 1
                if r['is_correct']:
                    subject_stats[subject]['correct'] += 1
            
            f.write("Accuracy by Subject:\n")
            f.write(f"{'Subject':<40} {'Accuracy':<20}\n")
            f.write("-" * 60 + "\n")
            for subject in sorted(subject_stats.keys()):
                stats = subject_stats[subject]
                subj_accuracy = stats['correct'] / stats['total'] * 100
                f.write(f"{subject:<40} {stats['correct']}/{stats['total']} ({subj_accuracy:.2f}%)\n")
            f.write("\n")
    
    print(f"✓ Saved summary results to TXT: {output_path}")
    return True

def save_reasoning_examples_by_subject(all_model_results, output_path):
    """Save reasoning examples grouped by subject for analysis"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REASONING EXAMPLES BY SUBJECT\n")
        f.write("=" * 80 + "\n\n")
        f.write("This file contains sample reasoning responses from models, grouped by subject.\n")
        f.write("Each entry shows the question, model's reasoning, and whether it was correct.\n\n")
        
        # Group results by subject
        subject_data = {}
        
        for model_result in all_model_results:
            if 'error' in model_result or model_result['total_questions'] == 0:
                continue
            
            model_name = model_result['model_name']
            results = model_result['results']
            
            for r in results:
                subject = r.get('subject', 'Unknown')
                if subject not in subject_data:
                    subject_data[subject] = []
                
                subject_data[subject].append({
                    'model': model_name,
                    'question': r.get('question', ''),
                    'choices': r.get('choices', ''),
                    'ground_truth': r.get('correct_answer', 'N/A'),
                    'prediction': r.get('prediction', 'N/A'),
                    'is_correct': r.get('is_correct', False),
                    'reasoning': r.get('reasoning', r.get('response', ''))
                })
        
        # Write by subject
        for subject in sorted(subject_data.keys()):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"SUBJECT: {subject}\n")
            f.write("=" * 80 + "\n\n")
            
            examples = subject_data[subject]
            # Show a few examples per subject (up to 5)
            examples_to_show = examples[:5]
            
            for idx, ex in enumerate(examples_to_show, 1):
                f.write(f"\nExample {idx}:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Model: {ex['model']}\n")
                f.write(f"Question: {ex['question']}\n")
                f.write(f"Choices: {ex['choices']}\n")
                f.write(f"Ground Truth: {ex['ground_truth']}\n")
                f.write(f"Prediction: {ex['prediction']}\n")
                f.write(f"Correct: {'✓ YES' if ex['is_correct'] else '✗ NO'}\n")
                f.write(f"\nReasoning/Thought Process:\n{ex['reasoning']}\n")
                f.write("-" * 80 + "\n")
            
            if len(examples) > 5:
                f.write(f"\n... ({len(examples) - 5} more examples for this subject)\n")
            
            f.write(f"\nTotal examples for {subject}: {len(examples)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REASONING EXAMPLES\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved reasoning examples by subject to: {output_path}")
    return True

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
    latex_content.append("\\title{Chain of Thought Reasoning with Example Evaluation Results}")
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

def main():
    """Main function to run Chain of Thought evaluation on multiple models"""
    print("=" * 80)
    print("Chain of Thought (COT) Evaluation - Simple Reasoning with Example (English)")
    print("Evaluating 20 Subject-Level Groups")
    print("Output: TXT Summary, CSV, and LaTeX")
    print("=" * 80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "IndoMMLU.csv")
    
    # Use hardcoded model names
    model_names = MODEL_NAMES
    print(f"\nModels to evaluate ({len(model_names)}):")
    for i, model_name in enumerate(model_names, 1):
        print(f"  {i}. {model_name}")
    
    num_groups = 20
    num_questions_per_group = 10
    print(f"\nAnalyzing dataset to find top {num_groups} subject-level groups...")
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file not found at {csv_path}")
        return
    
    groups = get_subject_level_groups(csv_path, num_groups=num_groups)
    
    if len(groups) == 0:
        print("❌ ERROR: No subject-level groups found in the dataset!")
        return
    
    print(f"Selected {len(groups)} groups:")
    for i, (subject, level, count) in enumerate(groups, 1):
        print(f"  {i}. {subject} ({level}) - {count} questions available")
    
    all_model_results = []
    for model_idx, model_name in enumerate(model_names, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# MODEL {model_idx}/{len(model_names)}: {model_name}")
        print(f"{'#' * 80}")
        
        model_results = load_and_evaluate_model(
            model_name, 
            groups, 
            csv_path, 
            num_questions_per_group
        )
        all_model_results.append(model_results)
        
        if 'error' not in model_results:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY for {model_name}")
            print(f"{'=' * 80}")
            print(f"Total Questions: {model_results['total_questions']}")
            print(f"Correct Answers: {model_results['correct_count']}")
            print(f"Overall Accuracy: {model_results['accuracy']:.2f}%")
            print(f"{'=' * 80}")
    
    total_evaluated = sum(r['total_questions'] for r in all_model_results if 'error' not in r)
    
    if total_evaluated == 0:
        print("\n\n" + "=" * 80)
        print("❌ NO RESULTS GENERATED")
        print("=" * 80)
        return
    
    print("\n\n" + "=" * 80)
    print("COMPARATIVE SUMMARY - ALL MODELS")
    print("=" * 80)
    print(f"{'Model':<50} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-" * 80)
    
    for model_result in all_model_results:
        model_name = model_result['model_name']
        if 'error' in model_result:
            error_msg = model_result.get('error', 'Unknown error')[:30]
            print(f"{model_name:<50} {'ERROR':<15} {error_msg:<20}")
        else:
            accuracy = model_result['accuracy']
            correct = model_result['correct_count']
            total = model_result['total_questions']
            if total > 0:
                print(f"{model_name:<50} {accuracy:>6.2f}%{'':<8} {correct}/{total}")
    
    print("=" * 80)
    
    valid_results = [r for r in all_model_results if 'error' not in r and r['total_questions'] > 0]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best Model: {best_model['model_name']} with {best_model['accuracy']:.2f}% accuracy")
    
    print("\n\n" + "=" * 80)
    print("DETAILED STATISTICS BY MODEL")
    print("=" * 80)
    
    for model_result in all_model_results:
        if 'error' in model_result or model_result['total_questions'] == 0:
            continue
        
        model_name = model_result['model_name']
        results = model_result['results']
        
        print(f"\n{'-' * 80}")
        print(f"Model: {model_name}")
        print(f"{'-' * 80}")
        
        subject_stats = {}
        for r in results:
            subject = r['subject']
            if subject not in subject_stats:
                subject_stats[subject] = {'total': 0, 'correct': 0}
            subject_stats[subject]['total'] += 1
            if r['is_correct']:
                subject_stats[subject]['correct'] += 1
        
        print("\nAccuracy by Subject:")
        print(f"{'Subject':<40} {'Accuracy':<20}")
        print("-" * 60)
        for subject in sorted(subject_stats.keys()):
            stats = subject_stats[subject]
            subj_accuracy = stats['correct'] / stats['total'] * 100
            print(f"{subject:<40} {stats['correct']}/{stats['total']} ({subj_accuracy:.2f}%)")
    
    print("\n\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    txt_output_path = os.path.join(results_dir, "COT_en_reason_eg_summary.txt")
    csv_output_path = os.path.join(results_dir, "COT_en_reason_eg_results.csv")
    latex_output_path = os.path.join(results_dir, "COT_en_reason_eg_results.tex")
    reasoning_examples_path = os.path.join(results_dir, "reasoningexample_en.txt")
    
    save_results_to_txt(all_model_results, txt_output_path)
    save_results_to_csv(all_model_results, csv_output_path)
    save_results_to_latex(all_model_results, latex_output_path)
    save_reasoning_examples_by_subject(all_model_results, reasoning_examples_path)
    
    print(f"\n✓ All exports completed!")
    print(f"  TXT Summary: {txt_output_path}")
    print(f"  CSV: {csv_output_path}")
    print(f"  LaTeX: {latex_output_path}")
    print(f"  Reasoning Examples: {reasoning_examples_path}")

if __name__ == "__main__":
    main()
