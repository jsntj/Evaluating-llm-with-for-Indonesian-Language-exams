"""
Chain of Thought (COT) Step-by-Step evaluation script for IndoMMLU dataset.
This script evaluates questions using a step-by-step reasoning approach.

Uses IndoMMLU.csv dataset with hardcoded models.

This version (COT-id-step-by-step.py) outputs a simplified CSV with only:
- question
- ground_truth
- prediction
- reasoning
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
        csv_path: Path to the IndoMMLU_samples.csv file
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
        
        # Select top subjects (aim for at least num_groups/2 unique subjects, but at least 15)
        num_subjects = max(15, min(len(subject_totals), num_groups // 2))
        top_subjects = subject_totals.head(num_subjects)['subject'].tolist()
        
        # For each subject, get its top groups
        selected_groups = []
        groups_per_subject = max(1, num_groups // num_subjects)  # Distribute groups across subjects
        
        for subject in top_subjects:
            subject_groups = groups[groups['subject'] == subject].sort_values('count', ascending=False)
            # Take top groups for this subject
            selected = subject_groups.head(groups_per_subject)
            selected_groups.append(selected)
            
            # Stop if we have enough groups (count total rows collected so far)
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
    Load questions from IndoMMLU_samples.csv, optionally filtered by subject and level
    
    Args:
        csv_path: Path to the IndoMMLU_samples.csv file
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
    """Format the question and choices with a Chain of Thought prompt (5 sentences)."""
    
    return f"""Pertanyaan: {question}

Pilihan:
{choices}

Silakan pikirkan langkah demi langkah dan jelaskan alasan Anda untuk pertanyaan ini. Pertama, baca dan pahami dengan seksama apa yang ditanyakan dalam pertanyaan. Kemudian, analisis setiap pilihan jawaban secara sistematis dan evaluasi seberapa baik setiap pilihan menjawab pertanyaan. Pertimbangkan konsep kunci, fakta, atau prinsip yang berkaitan dengan setiap pilihan. Setelah menyelesaikan analisis langkah demi langkah Anda, berikan jawaban akhir Anda dalam format "Jawaban Akhir: X" di mana X adalah huruf (A, B, C, D, atau E)."""

def extract_answer(response):
    """Ekstrak surat jawaban (A-E) dari contoh jawaban, dengan memprioritaskan pola ‘Jawaban Akhir:’"""
    response_upper = response.upper()
    
    # Look for common answer patterns, prioritizing "Jawaban Akhir:" pattern for Indonesian
    patterns = [
        r'jawaban\s+akhir[:\s]+([A-E])',  # Indonesian "Jawaban Akhir: A" - highest priority
        r'final\s+answer[:\s]+([A-E])',  # "Final Answer: A"
        r'answer[:\s]+([A-E])',  # "Answer: A"
        r'jawaban[:\s]+([A-E])',  # Indonesian "jawaban: A"
        r'pilihan[:\s]+([A-E])',  # Indonesian "pilihan: A"
        r'option[:\s]+([A-E])',  # "Option: A"
        r'\b([A-E])\b'  # Standalone letter (fallback)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
    
    # Check last few lines for letter (reasoning often ends with answer)
    lines = response_upper.split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        words = line.split()
        for word in words:
            if word in ['A', 'B', 'C', 'D', 'E']:
                return word
    
    # Fallback: check first few words for letter
    words = response_upper.split()[:10]
    for word in words:
        if word in ['A', 'B', 'C', 'D', 'E']:
            return word
    
    return None

def extract_reasoning(response):
    """
    Extract the reasoning chain from the model response.
    
    The reasoning is the full response text, which contains the step-by-step
    reasoning process. This function can be used to clean or format the 
    reasoning if needed, but by default returns the full response as-is.
    
    Args:
        response: Full model response
    
    Returns:
        Reasoning text (full response)
    """
    # For now, return the full response as the reasoning
    # This can be modified to extract specific sections if needed
    return response.strip()

# Models to evaluate (hardcoded)
MODEL_NAMES = [
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "CohereLabs/aya-expanse-8b",
    "sail/Sailor-7B-Chat",
    "google/gemma-3-12b-it",
]

def load_model_names(seallm_file_path):
    """
    Load model names from seallm.txt file (DEPRECATED - using MODEL_NAMES instead)
    
    Args:
        seallm_file_path: Path to seallm.txt file
    
    Returns:
        List of model names with proper formatting
    """
    model_mapping = {
        "SeaLLMs-v3-7B-Chat": "SeaLLMs/SeaLLMs-v3-7B-Chat",
        "CohereLabs/aya-expanse-8b": "CohereLabs/aya-expanse-8b",
        "Sailor-7B-Chat": "sail/Sailor-7B-Chat",
        "google/gemma-3-12b-it": "google/gemma-3-12b-it"
    }
    
    # Models to exclude (currently none)
    excluded_models = []
    
    try:
        with open(seallm_file_path, 'r') as f:
            model_names = [line.strip() for line in f if line.strip()]
        
        # Filter out excluded models
        model_names = [name for name in model_names if name not in excluded_models]
        
        # Map to full model paths
        full_model_names = []
        for name in model_names:
            if name in model_mapping:
                mapped_name = model_mapping[name]
                # Double check: also exclude if mapped name is in excluded list
                if mapped_name not in excluded_models:
                    full_model_names.append(mapped_name)
            else:
                # Use as-is if not in mapping, but check if it should be excluded
                if name not in excluded_models:
                    full_model_names.append(name)
        
        if len(model_names) != len(full_model_names):
            excluded_count = len(model_names) - len(full_model_names)
            print(f"  Note: Excluded {excluded_count} model(s)")
        
        return full_model_names
    except FileNotFoundError:
        # Fallback to default models if file not found
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
    
    # Track timing
    total_start = time.time()
    tokenize_time = 0
    generate_time = 0
    decode_time = 0
    
    # Use tqdm for progress bar
    for q_idx, q in enumerate(tqdm(questions, desc="Evaluating questions", unit="question"), 1):
        try:
            # Format prompt
            prompt_text = format_prompt(q['question'], q['choices'])
            messages = [{"role": "user", "content": prompt_text}]
            
            # Apply chat template and tokenize
            t0 = time.time()
            try:
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback: use prompt directly if chat template fails
                print(f"\n  ⚠ Warning: Chat template failed for question {q_idx}, using direct prompt: {str(e)}")
                input_text = prompt_text
            
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Move to model device
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            input_len = inputs["input_ids"].shape[-1]
            tokenize_time += time.time() - t0
            
            # Generate response with optimizations
            # Increased max_new_tokens to allow for full reasoning chains
            t1 = time.time()
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=400,  # Increased from 50 to allow full reasoning chains
                    do_sample=False,  # Deterministic generation for reproducible reasoning
                    pad_token_id=tokenizer.eos_token_id
                )
                generation = generation[0][input_len:]
            generate_time += time.time() - t1
            
            # Decode response
            t2 = time.time()
            response = tokenizer.decode(generation, skip_special_tokens=True)
            decode_time += time.time() - t2
            
            # Extract answer and reasoning from the output
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
                'response': response,  # Keep full response for internal use
                'reasoning': reasoning  # Extracted reasoning for CSV export
            })
        
        except Exception as e:
            print(f"\n  ❌ ERROR evaluating question {q_idx} (ID: {q.get('id', 'unknown')}): {str(e)}")
            # Add error result
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
        else:
            print(f"  Correct answers: {correct_count}/{len(results)} (N/A - no results)")
        print(f"  Timing stats: {len(questions)} questions in {total_time:.2f}s")
        print(f"    Avg per question: {avg_time:.2f}s")
        if total_time > 0:
            print(f"    Tokenize: {tokenize_time:.2f}s ({tokenize_time/total_time*100:.1f}%)")
            print(f"    Generate: {generate_time:.2f}s ({generate_time/total_time*100:.1f}%)")
            print(f"    Decode: {decode_time:.2f}s ({decode_time/total_time*100:.1f}%)")
    else:
        print(f"\n  ⚠ Warning: No questions were evaluated (questions list was empty)")
    
    return results

def load_and_evaluate_model(model_name, groups, csv_path, num_questions_per_group):
    """
    Load a model and evaluate it on all groups
    
    Args:
        model_name: Name of the model to load
        groups: List of (subject, level, count) tuples
        csv_path: Path to CSV file
        num_questions_per_group: Number of questions per group
    
    Returns:
        Dictionary with model results and statistics
    """
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
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ['HF_TOKEN'],
            cache_dir=os.environ['HF_HUB_CACHE'],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        
        print("Model loaded successfully!")
        
        # Check device placement
        model_device = next(model.parameters()).device
        print(f"Model device: {model_device}")
        if model_device.type == 'cpu':
            print("⚠ WARNING: Model is on CPU! This will be very slow. Check GPU availability.")
        else:
            print(f"✓ Model is on {model_device}")
        
        # Evaluate each group
        all_results = []
        print("\n" + "=" * 80)
        print(f"Evaluating questions with {model_name}")
        print("=" * 80)
        
        for group_idx, (subject, level, total_available) in enumerate(groups, 1):
            print(f"\n[{group_idx}/{len(groups)}] Evaluating: {subject} ({level})")
            print(f"  Available questions: {total_available}, Sampling: {num_questions_per_group}")
            
            # Load questions for this group
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
                print(f"     This might indicate a data mismatch. Check CSV file.")
                continue
            
            print(f"  ✓ Loaded {len(questions)} questions")
            
            # Evaluate this group
            group_results = evaluate_cot(model, tokenizer, questions, model_name)
            # Add model name to each result
            for r in group_results:
                r['model'] = model_name
            all_results.extend(group_results)
            
            # Print group summary
            if len(group_results) > 0:
                group_correct = sum(1 for r in group_results if r['is_correct'])
                group_accuracy = group_correct / len(group_results) * 100
                print(f"  Results: {group_correct}/{len(group_results)} correct ({group_accuracy:.2f}%)")
            else:
                print(f"  Results: 0/0 (No results generated)")
        
        # Calculate overall statistics
        total_questions = len(all_results)
        correct_count = sum(1 for r in all_results if r['is_correct'])
        accuracy = correct_count / total_questions * 100 if total_questions > 0 else 0
        
        # Clean up model from memory before moving to next model
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
        print(f"\n   Full traceback:")
        traceback.print_exc()
        return {
            'model_name': model_name,
            'results': [],
            'total_questions': 0,
            'correct_count': 0,
            'accuracy': 0.0,
            'error': str(e)
        }

def save_results_to_txt(all_model_results, output_path):
    """Save summary results to TXT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHAIN OF THOUGHT STEP-BY-STEP EVALUATION SUMMARY\n")
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


def save_results_to_csv(all_model_results, output_path):
    """
    Save simplified reasoning results to CSV file with only 4 columns:
    - question
    - ground_truth
    - prediction
    - reasoning
    
    This function extracts the reasoning and answers from each model output
    and saves them in a clean, simplified format.
    
    Args:
        all_model_results: List of model result dictionaries
        output_path: Path to save CSV file
    """
    csv_rows = []
    for model_result in all_model_results:
        if 'error' in model_result:
            continue
        for r in model_result.get('results', []):
            csv_rows.append({
                'question': r.get('question', ''),
                'ground_truth': r.get('correct_answer', 'N/A'),
                'prediction': r.get('prediction', 'N/A'),
                'reasoning': r.get('reasoning', r.get('response', ''))  # Use extracted reasoning or fallback to response
            })
    
    if csv_rows:
        df_out = pd.DataFrame(csv_rows)
        df_out.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Saved simplified reasoning results to CSV: {output_path}")
        print(f"  Columns: question, ground_truth, prediction, reasoning")
        print(f"  Total rows: {len(csv_rows)}")
        return True
    else:
        print(f"⚠ No results available to save to CSV.")
        return False


def escape_latex(text):
    """
    Escape LaTeX special characters in text.
    
    Args:
        text: String to escape
    
    Returns:
        Escaped string safe for LaTeX
    """
    if not isinstance(text, str):
        text = str(text)
    # Escape LaTeX special characters
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
    """
    Save results summary to LaTeX format (comparative table and per-model statistics).
    
    Args:
        all_model_results: List of model result dictionaries
        output_path: Path to save LaTeX file
    """
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
    latex_content.append("\\title{Chain of Thought Reasoning Evaluation Results}")
    latex_content.append("\\author{IndoMMLU Dataset}")
    latex_content.append("\\maketitle")
    latex_content.append("")
    
    # Comparative Summary Table
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
    
    # Best Model
    valid_results = [r for r in all_model_results if 'error' not in r and r['total_questions'] > 0]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x['accuracy'])
        best_name_escaped = escape_latex(best_model['model_name'])
        latex_content.append(f"\\textbf{{Best Model:}} {best_name_escaped} with {best_model['accuracy']:.2f}\\% accuracy.")
        latex_content.append("")
    
    # Detailed Statistics by Model
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
        
        # Subject-level statistics
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
    
    # Write LaTeX file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Saved LaTeX results to: {output_path}")
    return True


def main():
    """Main function to run Chain of Thought evaluation on multiple models"""
    print("=" * 80)
    print("Chain of Thought (COT) Evaluation - Multi-Model Comparison")
    print("Evaluating 20 Subject-Level Groups")
    print("=" * 80)
    
    # Get script directory to find CSV file and model list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "IndoMMLU.csv")
    
    # Use hardcoded model names
    model_names = MODEL_NAMES
    print(f"\nModels to evaluate ({len(model_names)}):")
    for i, model_name in enumerate(model_names, 1):
        print(f"  {i}. {model_name}")
    
    print("Chain of Thought (COT) Evaluation - Multi-Model Comparison")
    print("Evaluating 20 Subject-Level Groups")
    print("Output: CSV and LaTeX (question, ground_truth, prediction, reasoning)")
    print("=" * 80)
    
    # Get top 20 subject-level groups
    num_groups = 20
    num_questions_per_group = 10
    print(f"\nAnalyzing dataset to find top {num_groups} subject-level groups...")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file not found at {csv_path}")
        return
    
    groups = get_subject_level_groups(csv_path, num_groups=num_groups)
    
    if len(groups) == 0:
        print("❌ ERROR: No subject-level groups found in the dataset!")
        print(f"   CSV file: {csv_path}")
        print("   Please check that the CSV file has 'subject' and 'level' columns with data.")
        return
    
    print(f"Selected {len(groups)} groups:")
    for i, (subject, level, count) in enumerate(groups, 1):
        print(f"  {i}. {subject} ({level}) - {count} questions available")
    
    # Evaluate each model SEQUENTIALLY (one at a time)
    # Each model is fully loaded, evaluated, and cleaned up before moving to the next
    all_model_results = []
    for model_idx, model_name in enumerate(model_names, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# MODEL {model_idx}/{len(model_names)}: {model_name}")
        print(f"# Processing model {model_idx} of {len(model_names)} - Sequential execution")
        print(f"{'#' * 80}")
        
        # Load, evaluate, and cleanup this model before moving to next
        model_results = load_and_evaluate_model(
            model_name, 
            groups, 
            csv_path, 
            num_questions_per_group
        )
        all_model_results.append(model_results)
        
        # Model is cleaned up in load_and_evaluate_model() before next iteration
        
        # Print model summary
        if 'error' not in model_results:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY for {model_name}")
            print(f"{'=' * 80}")
            print(f"Total Questions: {model_results['total_questions']}")
            print(f"Correct Answers: {model_results['correct_count']}")
            print(f"Overall Accuracy: {model_results['accuracy']:.2f}%")
            print(f"{'=' * 80}")
    
    # Check if we have any results at all
    total_evaluated = sum(r['total_questions'] for r in all_model_results if 'error' not in r)
    
    if total_evaluated == 0:
        print("\n\n" + "=" * 80)
        print("❌ NO RESULTS GENERATED")
        print("=" * 80)
        print("\nPossible reasons:")
        print("  1. All models failed to load (check errors above)")
        print("  2. No questions were found in the dataset")
        print("  3. CSV file format issues")
        print("  4. Model loading errors (check HuggingFace access)")
        print("\nCheck the error messages above for details.")
        print("=" * 80)
        return
    
    # Print comparative summary
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
            else:
                print(f"{model_name:<50} {'NO DATA':<15} {'0/0':<20}")
    
    print("=" * 80)
    
    # Find best model
    valid_results = [r for r in all_model_results if 'error' not in r and r['total_questions'] > 0]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best Model: {best_model['model_name']} with {best_model['accuracy']:.2f}% accuracy")
    else:
        print(f"\n⚠ No valid results to determine best model")
    
    # Print detailed statistics by subject for each model
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
        
        # Accuracy by subject
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
    
    # Save results to CSV and LaTeX
    print("\n\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    txt_output_path = os.path.join(results_dir, "COT_id_step_by_step_summary.txt")
    csv_output_path = os.path.join(results_dir, "COT_id_step_by_step_results.csv")
    latex_output_path = os.path.join(results_dir, "COT_id_step_by_step_results.tex")
    
    save_results_to_txt(all_model_results, txt_output_path)
    save_results_to_csv(all_model_results, csv_output_path)
    save_results_to_latex(all_model_results, latex_output_path)
    
    print(f"\n✓ All exports completed!")
    print(f"  TXT Summary: {txt_output_path}")
    print(f"  CSV: {csv_output_path}")
    print(f"  LaTeX: {latex_output_path}")
    print(f"  CSV Format: question, ground_truth, prediction, reasoning")

if __name__ == "__main__":
    main()
