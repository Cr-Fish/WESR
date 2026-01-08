import re
from pypinyin import lazy_pinyin
import difflib
import json
from pathlib import Path
import os
from typing import List, Dict, Union, Tuple, Optional, Any
from dataclasses import dataclass

# Configuration
tags_1226_21 = ["singing", "shouting", "inhale", "whispering", "cough", "laughs", "laughing", "crowd_laughter", "exhale", "chuckle",
            "shout", "sobbing", "crying", "cry", "giggle", "sigh", "clear_throat", "roar", "scream", "panting", "breathing"]

tags_mapping_1226 = {
    "LAUGH": ["laughs", "laughing", "chuckle", "giggle", "chortle", "crowd_laughter"],
    "SHOUT": ["scream", "roar", "shout", "shouting"],
    "WHISPERING": ["whispering"],
    "SINGING": ["singing"],
    "BREATHING": ["inhale", "exhale", "panting", "sigh", "breathing", "sniff"],
    "COUGH": ["cough", "clear_throat"],
    "CRYING": ["crying", "sobbing", "cry"],
}

@dataclass
class EvaluationConfig:
    include_tags: Optional[List[str]] = None
    big: bool = False
    tags_mapping: Optional[Dict[str, List[str]]] = None
    use_classification: bool = False

class TextProcessor:
    
    @staticmethod
    def remove_speaker_tags(text: str) -> str:
        return re.sub(r'\[S\d+\]', '', text).strip()

    @staticmethod
    def remove_punct_except_brackets(text: str) -> str:
        allowed = r'\[\]<>\/'
        return re.sub(rf'[^\w\s{allowed}]', '', text)

    @staticmethod
    def split_sentence(text: str) -> List[str]:
        token_pattern = r'\[[^\]]+\]|<[^>/]+>|</[^>]+>'
        pattern = re.compile(f'{token_pattern}|[\u4e00-\u9fff]|[a-zA-Z0-9]+')
        return pattern.findall(text)

    @staticmethod
    def is_token(x: str) -> bool:
        return bool(re.match(r'\[[^\]]+\]|<[^>/]+>|</[^>]+>', x))

    @classmethod
    def to_pinyin(cls, mixed: List[str]) -> List[str]:
        result = []
        for x in mixed:
            if cls.is_token(x):
                result.append(x)
            elif re.match(r'[\u4e00-\u9fff]', x):
                result.append(''.join(lazy_pinyin(x)))
            else:
                result.append(x.lower())
        return result

    @classmethod
    def replace_tags(cls, tag_list: List[str], tags_mapping: Dict[str, List[str]]) -> List[str]:
        reverse_mapping = {}
        for category, tags in tags_mapping.items():
            for tag in tags:
                reverse_mapping[tag] = category
        
        result = []
        for item in tag_list:
            if item.startswith('[') and item.endswith(']'):
                tag_content = item[1:-1]
                if tag_content in reverse_mapping:
                    result.append(f"[{reverse_mapping[tag_content]}]")
                else:
                    result.append(item)
            elif item.startswith('<') and item.endswith('>') and not item.startswith('</'):
                tag_content = item[1:-1]
                if tag_content in reverse_mapping:
                    result.append(f"<{reverse_mapping[tag_content]}>")
                else:
                    result.append(item)
            elif item.startswith('</') and item.endswith('>'):
                tag_content = item[2:-1]
                if tag_content in reverse_mapping:
                    result.append(f"</{reverse_mapping[tag_content]}>")
                else:
                    result.append(item)
            else:
                result.append(item)
        return result

    @classmethod
    def preprocess_text(cls, text: str, config: EvaluationConfig, use_pinyin: bool = True) -> List[str]:
        processed = cls.remove_punct_except_brackets(text)
        if use_pinyin:
            processed = cls.remove_speaker_tags(processed)
        
        mixed = cls.split_sentence(processed)
        
        if use_pinyin:
            mixed = cls.to_pinyin(mixed)
            
        if config.big and config.tags_mapping:
            mixed = cls.replace_tags(mixed, config.tags_mapping)
            
        return mixed

    @classmethod
    def separate_tag_types(cls, text: str) -> Tuple[str, str]:
        clean_text = cls.remove_punct_except_brackets(text)
        tokens = cls.split_sentence(clean_text)
        
        bracket_tokens = []
        angle_tokens = []
        
        for token in tokens:
            if re.match(r'\[[^\]]+\]', token):
                bracket_tokens.append(token)
                angle_tokens.append('')
            elif re.match(r'<[^>]*>', token):
                angle_tokens.append(token)
                bracket_tokens.append('')
            else:
                bracket_tokens.append(token)
                angle_tokens.append(token)
        
        bracket_text = ''.join([t for t in bracket_tokens if t])
        angle_text = ''.join([t for t in angle_tokens if t])
        
        return bracket_text, angle_text

import os

class FileManager:
    
    @staticmethod
    def read_jsonl_files(file_paths: Union[str, List[str]]) -> Dict[str, str]:
        result = {}
        
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if "audio" in obj and isinstance(obj["audio"], dict) and "path" in obj["audio"]:
                        full_path = obj["audio"]["path"]
                    elif "audio_path" in obj:
                        full_path = obj["audio_path"]
                    else:
                        raise KeyError("Neither 'audio.path' nor 'audio_path' found")
                    
                    key = os.path.basename(full_path)
                    result[key] = obj["sentence"]
        
        return result

    @staticmethod
    def read_jsonl_dir(dir_path: str) -> Dict[str, str]:
        result = {}
        
        if os.path.isdir(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                obj = json.loads(line)
                                if "audio" in obj and isinstance(obj["audio"], dict) and "path" in obj["audio"]:
                                    full_path = obj["audio"]["path"]
                                elif "audio_path" in obj:
                                    full_path = obj["audio_path"]
                                else:
                                    raise KeyError("Neither 'audio.path' nor 'audio_path' found")
                                
                                key = os.path.basename(full_path)
                                
                                if isinstance(obj["sentence"], dict) and "tagged_asr" in obj["sentence"]:
                                    result[key] = obj["sentence"]["tagged_asr"]
                                elif isinstance(obj["sentence"], str):
                                    result[key] = obj["sentence"]
        else:
            # Single file
            with open(dir_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if "audio" in obj and isinstance(obj["audio"], dict) and "path" in obj["audio"]:
                        full_path = obj["audio"]["path"]
                    elif "audio_path" in obj:
                        full_path = obj["audio_path"]
                    else:
                        raise KeyError("Neither 'audio.path' nor 'audio_path' found")
                    
                    key = os.path.basename(full_path)
                    
                    if isinstance(obj["sentence"], dict) and "tagged_asr" in obj["sentence"]:
                        result[key] = obj["sentence"]["tagged_asr"]
                    elif isinstance(obj["sentence"], str):
                        result[key] = obj["sentence"]
        
        return result

    def load_data(cls, true_path: Union[str, List[str]], pred_path: str) -> Tuple[List[str], List[str]]:
        pred_path_obj = Path(pred_path)
        
        # Determine format automatically
        if pred_path_obj.is_file():
            # If it's a file, determine format by extension
            if pred_path_obj.suffix == '.json':
                format_type = "json"
            else:
                format_type = "jsonl"
        elif pred_path_obj.is_dir():
            # If it's a directory, check what files are inside
            json_files = list(pred_path_obj.glob("*.json"))
            jsonl_files = list(pred_path_obj.glob("*.jsonl"))
            
            if json_files:
                format_type = "json"
            elif jsonl_files:
                format_type = "jsonl"
            else:
                # Default to jsonl if no files found
                format_type = "jsonl"
        else:
            # Default to jsonl if path doesn't exist
            format_type = "jsonl"
        
        if format_type == "json":
            if isinstance(true_path, (str, list)):
                # Read true from jsonl, pred from json
                true_dict = cls.read_jsonl_files(true_path)
                pred_dict = {}
                
                path = Path(pred_path)
                if path.is_file() and path.suffix == '.json':
                    json_files = [path]
                elif path.is_dir():
                    json_files = list(path.glob("*.json"))
                else:
                    json_files = [Path(pred_path)]
                
                for json_file in json_files:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    for sample in data["samples"]:
                        if "audio_path" in sample:
                            full_path = sample["audio_path"]
                            key = os.path.basename(full_path)
                            pred_dict[key] = sample["prediction"]
                
                # Get common keys
                shared_paths = set(true_dict.keys()) & set(pred_dict.keys())
                true_list = [true_dict[path] for path in shared_paths]
                pred_list = [pred_dict[path] for path in shared_paths]
                
                return true_list, pred_list
            else:
                # Pure json format
                return cls.read_json_files(pred_path)
        
        else:  # jsonl format
            true_dict = cls.read_jsonl_files(true_path)
            pred_dict = cls.read_jsonl_dir(pred_path)
            
            shared_paths = set(true_dict.keys()) & set(pred_dict.keys())
            true_list = [true_dict[path] for path in shared_paths]
            pred_list = [pred_dict[path] for path in shared_paths]
            
            return true_list, pred_list

class MetricsCalculator:
    
    @staticmethod
    def calculate_similarity(char1: str, char2: str) -> float:
        return difflib.SequenceMatcher(None, char1, char2).ratio()

    @classmethod
    def compute_tp_fp_fn(cls, ref_tags: Dict, pred_tags: Dict, all_positions=None) -> Tuple[Dict, Dict]:
        results = {}
        
        all_tags = set()
        for tags_list in ref_tags.values():
            all_tags.update(tags_list)
        for tags_list in pred_tags.values():
            all_tags.update(tags_list)
        
        if all_positions is None:
            all_positions = set(ref_tags.keys()) | set(pred_tags.keys())

        for tag in all_tags:
            results[tag] = {'tp': 0, 'fp': 0, 'fn': 0}

        for pos in all_positions:
            ref_set = set(ref_tags.get(pos, []))
            pred_set = set(pred_tags.get(pos, []))

            for tag in all_tags:
                ref_has = tag in ref_set
                pred_has = tag in pred_set

                if ref_has and pred_has:
                    results[tag]['tp'] += 1
                elif not ref_has and pred_has:
                    results[tag]['fp'] += 1
                elif ref_has and not pred_has:
                    results[tag]['fn'] += 1

        total_tp = sum(r['tp'] for r in results.values())
        total_fp = sum(r['fp'] for r in results.values())
        total_fn = sum(r['fn'] for r in results.values())
        
        overall = {'tp': total_tp, 'fp': total_fp, 'fn': total_fn}

        return results, overall

    @staticmethod
    def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    @classmethod
    def aggregate_metrics(cls, tag2tp: Dict, tag2fp: Dict, tag2fn: Dict, include_tags: Optional[List[str]] = None, tags_mapping: Optional[Dict] = None) -> Dict:
        if include_tags is None:
            include_tags = []
            
        if tags_mapping is not None:
            include_tags.extend(list(tags_mapping.keys()))
        
        if include_tags:  # If list is not empty
            total_tp_filtered = sum(tp for tag, tp in tag2tp.items() if tag in include_tags)
            total_fp_filtered = sum(fp for tag, fp in tag2fp.items() if tag in include_tags)
            total_fn_filtered = sum(fn for tag, fn in tag2fn.items() if tag in include_tags)
        else:
            total_tp_filtered = sum(tag2tp.values())
            total_fp_filtered = sum(tag2fp.values())
            total_fn_filtered = sum(tag2fn.values())

        # macro
        p_list_filtered, r_list_filtered, f1_list_filtered = [], [], []
        per_tag_stats = {}

        for tag in sorted(set(list(tag2tp.keys()) + list(tag2fp.keys()) + list(tag2fn.keys()))):
            tp = tag2tp.get(tag, 0)
            fp = tag2fp.get(tag, 0)
            fn = tag2fn.get(tag, 0)
            p, r, f1 = cls.calculate_metrics(tp, fp, fn)

            if include_tags is None or tag in include_tags:
                p_list_filtered.append(p)
                r_list_filtered.append(r)
                f1_list_filtered.append(f1)
                per_tag_stats[tag] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}

        macro_p_filtered = sum(p_list_filtered) / len(p_list_filtered) if p_list_filtered else 0.0
        macro_r_filtered = sum(r_list_filtered) / len(r_list_filtered) if r_list_filtered else 0.0
        macro_f1_filtered = sum(f1_list_filtered) / len(f1_list_filtered) if f1_list_filtered else 0.0

        # micro
        micro_p_filtered, micro_r_filtered, micro_f1_filtered = cls.calculate_metrics(
            total_tp_filtered, total_fp_filtered, total_fn_filtered)

        return {
            "micro": {"p": micro_p_filtered, "r": micro_r_filtered, "f1": micro_f1_filtered},
            "macro": {"p": macro_p_filtered, "r": macro_r_filtered, "f1": macro_f1_filtered},
            "per_tag": per_tag_stats,
            "overall": {"tp": total_tp_filtered, "fp": total_fp_filtered, "fn": total_fn_filtered}
        }

class TagExtractor:
    
    @staticmethod
    def extract_tags_from_text(mixed: List[str], include_tags: Optional[List[str]] = None) -> List[str]:
        tags = []
        for item in mixed:
            if TextProcessor.is_token(item):
                if item.startswith('[') and item.endswith(']'):
                    tag = item[1:-1]
                    if include_tags is None or tag in include_tags:
                        tags.append(tag)
                elif item.startswith('<') and item.endswith('>'):
                    if item.startswith('</'):
                        tag = item[2:-1]
                    else:
                        tag = item[1:-1]
                    if include_tags is None or tag in include_tags:
                        tags.append(tag)
        return tags

    @staticmethod
    def extract_tag_set_from_text(mixed: List[str], include_tags: Optional[List[str]] = None) -> set:
        tags = set()
        for item in mixed:
            if TextProcessor.is_token(item):
                if item.startswith('[') and item.endswith(']'):
                    tag = item[1:-1]
                    if include_tags is None or tag in include_tags:
                        tags.add(tag)
                elif item.startswith('<') and item.endswith('>'):
                    if item.startswith('</'):
                        tag = item[2:-1]
                    else:
                        tag = item[1:-1]
                    if include_tags is None or tag in include_tags:
                        tags.add(tag)
        return tags

# Preserve original complex functions but simplify calls
def force_align_pred(pred_mixed, true_mixed):
    true_clean = [x for x in true_mixed if not TextProcessor.is_token(x)]
    sm = difflib.SequenceMatcher(a=pred_mixed, b=true_clean)
    opcodes = sm.get_opcodes()
    result = pred_mixed.copy()
    offset = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        curr_i1 = i1 + offset
        curr_i2 = i2 + offset
        
        if tag == 'replace':
            if i2-i1 == 1 and TextProcessor.is_token(pred_mixed[i1]):
                result[curr_i1:curr_i1] = true_clean[j1:j2]
                offset += (j2 - j1)
                continue
                
            result[curr_i1:curr_i2] = true_clean[j1:j2]
            offset += (j2 - j1) - (curr_i2 - curr_i1)
            
            pred_tokens = [(idx, t) for idx, t in enumerate(pred_mixed[i1:i2], i1) if TextProcessor.is_token(t)]
            
            for token_idx, token in pred_tokens:
                relative_pos = token_idx - i1
                
                if relative_pos == 0 and i2-i1 > 1:
                    next_char = pred_mixed[token_idx+1]
                    max_sim = -1
                    best_pos = curr_i1
                    
                    for j in range(curr_i1, min(curr_i1 + j2-j1, len(result))):
                        sim = MetricsCalculator.calculate_similarity(next_char, result[j])
                        if sim > max_sim:
                            max_sim = sim
                            best_pos = j
                    
                    result.insert(best_pos, token)
                    offset += 1
                    
                else:
                    prev_char = pred_mixed[token_idx-1]
                    max_sim = -1
                    best_pos = curr_i1
                    
                    for j in range(curr_i1, min(curr_i1 + j2-j1, len(result))):
                        sim = MetricsCalculator.calculate_similarity(prev_char, result[j])
                        if sim > max_sim:
                            max_sim = sim
                            best_pos = j
                    
                    result.insert(best_pos + 1, token)
                    offset += 1
        
        elif tag == 'insert':
            result[curr_i1:curr_i1] = true_clean[j1:j2]
            offset += (j2 - j1)
            
        elif tag == 'delete':
            delete_count = 0
            for idx in range(curr_i1, curr_i2):
                if not TextProcessor.is_token(pred_mixed[i1 + (idx-curr_i1)]):
                    result[idx] = ''
                    delete_count += 1
            offset -= delete_count
    
    result = [x for x in result if x != '']
    return result

def split_with_tag_positions(mixed_list):
    result = {}
    word_idx = 0     
    tag_stack = {}   
    
    for item in mixed_list:
        m1 = re.match(r'\[([^\]]+)\]', item)
        m2 = re.match(r'<([a-zA-Z][^>/\s]*)>', item)
        m3 = re.match(r'</([a-zA-Z][^>\s]*)>', item)

        if m1:
            tag = m1.group(1)
            pos = word_idx * 2
            result.setdefault(pos, []).append(tag)
        elif m2:
            tag = m2.group(1)
            if tag in tag_stack and tag_stack[tag] != []:  
                positions_list = tag_stack[tag][:-1]
                for pos in positions_list:
                    result.setdefault(pos, []).append(tag)
                tag_stack[tag] = []
            else:  
                tag_stack.setdefault(tag, []).append(word_idx * 2 + 1)
        elif m3:
            tag = m3.group(1)
            if tag in tag_stack and tag_stack[tag] != []:  
                positions_list = tag_stack[tag][:-1]
                for pos in positions_list:
                    result.setdefault(pos, []).append(tag)
                tag_stack[tag] = []
            else:  
                pos = word_idx * 2
                result.setdefault(pos, []).append(tag)
        else:
            for tag in tag_stack:
                if tag_stack[tag] != []:
                    tag_stack[tag].append(word_idx * 2 + 1)
            word_idx += 1

    for tag in tag_stack:
        if tag_stack[tag] != []:
            pos = tag_stack[tag][0]
            result.setdefault(pos, []).append(tag)

    return result

# Refactored main evaluation functions
class Evaluator:
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.text_processor = TextProcessor()
        self.file_manager = FileManager()
        self.metrics_calculator = MetricsCalculator()
        self.tag_extractor = TagExtractor()

    def evaluate_batch(self, true_list: List[str], pred_list: List[str]) -> Dict:
        tag2tp = {}
        tag2fp = {}
        tag2fn = {}
        
        for true_text, pred_text in zip(true_list, pred_list):
            true_mixed = self.text_processor.preprocess_text(true_text, self.config, use_pinyin=True)
            pred_mixed = self.text_processor.preprocess_text(pred_text, self.config, use_pinyin=True)
            
            aligned_pred = force_align_pred(pred_mixed, true_mixed)
            ref_tags = split_with_tag_positions(true_mixed)
            pred_tags = split_with_tag_positions(aligned_pred)

            results, overall = self.metrics_calculator.compute_tp_fp_fn(ref_tags, pred_tags)
            
            for tag, metrics in results.items():
                tag2tp[tag] = tag2tp.get(tag, 0) + metrics['tp']
                tag2fp[tag] = tag2fp.get(tag, 0) + metrics['fp']
                tag2fn[tag] = tag2fn.get(tag, 0) + metrics['fn']
        
        return self.metrics_calculator.aggregate_metrics(tag2tp, tag2fp, tag2fn, self.config.include_tags, self.config.tags_mapping)

    def evaluate_by_tag_type(self, true_list: List[str], pred_list: List[str]) -> Dict:
        true_bracket_list = []
        true_angle_list = []
        pred_bracket_list = []
        pred_angle_list = []
        
        for true_text, pred_text in zip(true_list, pred_list):
            true_bracket, true_angle = self.text_processor.separate_tag_types(true_text)
            pred_bracket, pred_angle = self.text_processor.separate_tag_types(pred_text)
            
            true_bracket_list.append(true_bracket)
            true_angle_list.append(true_angle)
            pred_bracket_list.append(pred_bracket)
            pred_angle_list.append(pred_angle)
        
        # Bracket tag evaluation
        bracket_results = self.evaluate_batch(true_bracket_list, pred_bracket_list)
        
        # Angle bracket tag evaluation
        include_tags_cont = ["crying", "laughing", "panting", "shouting", "singing", "whispering"]
        angle_config = EvaluationConfig(include_tags=include_tags_cont, big=False, tags_mapping=None)
        angle_evaluator = Evaluator(angle_config)
        angle_results = angle_evaluator.evaluate_batch(true_angle_list, pred_angle_list)
        
        # Combined evaluation
        combined_results = self.evaluate_batch(true_list, pred_list)
        
        return {
            'bracket': bracket_results,
            'angle': angle_results, 
            'combined': combined_results
        }

    def evaluate_classification_accuracy(self, true_list: List[str], pred_list: List[str]) -> Dict:
        tag2tp = {}
        tag2fp = {}
        tag2fn = {}
        
        for true_text, pred_text in zip(true_list, pred_list):
            true_mixed = self.text_processor.preprocess_text(true_text, self.config, use_pinyin=False)
            pred_mixed = self.text_processor.preprocess_text(pred_text, self.config, use_pinyin=False)
            
            true_tags = self.tag_extractor.extract_tag_set_from_text(true_mixed, self.config.include_tags)
            pred_tags = self.tag_extractor.extract_tag_set_from_text(pred_mixed, self.config.include_tags)
            
            all_tags = true_tags | pred_tags
            
            for tag in all_tags:
                true_has = tag in true_tags
                pred_has = tag in pred_tags
                
                if true_has and pred_has:
                    tag2tp[tag] = tag2tp.get(tag, 0) + 1
                elif not true_has and pred_has:
                    tag2fp[tag] = tag2fp.get(tag, 0) + 1
                elif true_has and not pred_has:
                    tag2fn[tag] = tag2fn.get(tag, 0) + 1
        
        return self.metrics_calculator.aggregate_metrics(tag2tp, tag2fp, tag2fn, self.config.include_tags, self.config.tags_mapping)

# Unified API function
def evaluate(true_path: Union[str, List[str]], pred_path: str, 
            config: EvaluationConfig, 
            eval_type: str = "sequence", 
            tag_type: str = "combined") -> Dict:
    evaluator = Evaluator(config)
    true_list, pred_list = evaluator.file_manager.load_data(true_path, pred_path)
    
    if tag_type == "by_type":
        return evaluator.evaluate_by_tag_type(true_list, pred_list)
    if eval_type == "sequence":
        return evaluator.evaluate_batch(true_list, pred_list)
    elif eval_type == "classification":
        return evaluator.evaluate_classification_accuracy(true_list, pred_list)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

# Output formatting functions
def format_output(results: Dict) -> str:
    table = "| Tag | Precision | Recall | F1 |\n"
    table += "|-----|-----------|--------|----|\n"

    per_tag = results['per_tag']
    for tag, metrics in per_tag.items():
        table += f"|{tag}|{metrics['p']:.3f}|{metrics['r']:.3f}|{metrics['f1']:.3f}|\n"

    table += f"|Micro|{results['micro']['p']:.3f}|{results['micro']['r']:.3f}|{results['micro']['f1']:.3f}|\n"
    table += f"|Macro|{results['macro']['p']:.3f}|{results['macro']['r']:.3f}|{results['macro']['f1']:.3f}|"

    return table

def format_output_by_tag_type(results_by_type: Dict) -> str:
    output = []
    
    for tag_type in ['bracket', 'angle', 'combined']:
        type_name = {
            'bracket': 'Discrete tags [tag]',
            'angle': 'Continuous tags <tag>',
            'combined': 'Combined'
        }[tag_type]
        
        results = results_by_type[tag_type]
        output.append(f"\n## {type_name}")
        output.append(format_output(results))

    return '\n'.join(output)

if __name__ == "__main__":
    pred_path = "your_predictions.jsonl"
    true_path = ["wesr_bench.jsonl"] # download wesr-bench from hf, or use download.py
    
    TAG_CONFIG = tags_1226_21
    TAG_MAPPING_CONFIG = tags_mapping_1226
    
    config = EvaluationConfig(include_tags=TAG_CONFIG, big=False)
    results = evaluate(true_path, pred_path, config, eval_type="sequence", tag_type="by_type")
    print(format_output_by_tag_type(results))
    
    # Aggregated results
    big_config = EvaluationConfig(include_tags=TAG_CONFIG, big=True, tags_mapping=TAG_MAPPING_CONFIG)
    big_results = evaluate(true_path, pred_path, big_config, eval_type="sequence", tag_type="by_type")
    print(format_output_by_tag_type(big_results))
