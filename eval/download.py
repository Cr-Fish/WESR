# download.py
import json
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_audio(args):
    audio_data, path = args
    sf.write(str(path), audio_data['array'], audio_data['sampling_rate'])

def download_wesr_bench(dataset_name, output_dir=".", audio_subdir="audio"):
    output_path = Path(output_dir)
    audio_path = output_path / audio_subdir
    audio_path.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(dataset_name)
    print("Splits:", list(dataset.keys()))
    
    all_entries = []
    audio_tasks = []
    
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        
        for item in split_data:
            audio_filename = item['audio']['path']
            audio_file_path = audio_path / audio_filename
            
            audio_tasks.append((item['audio'], audio_file_path))
            
            all_entries.append({
                "audio": {"path": audio_filename},
                "sentence": item['sentence'],
                "duration": item['duration'],
                "language": item['language']
            })
    
    print(f"Saving {len(audio_tasks)} audio files...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(save_audio, audio_tasks), total=len(audio_tasks)))
    
    jsonl_path = output_path / "wesr_bench.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Done! {len(all_entries)} samples")

if __name__ == "__main__":
    download_wesr_bench("yfish/WESR-Bench")