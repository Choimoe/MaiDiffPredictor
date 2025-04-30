import json
import os
import argparse

def safe_json_load(file_path):
    """安全读取JSON文件，处理常见格式问题"""
    try:
        # 检测文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检测文件大小
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"空文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8-sig') as f:  # 处理BOM头
            content = f.read().strip()
            
            # 处理尾随逗号问题
            content = content.replace(',]', ']').replace(',}', '}')
            return json.loads(content)
            
    except UnicodeDecodeError as e:
        print(f"编码错误 {os.path.basename(file_path)}: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON格式错误 {os.path.basename(file_path)}:")
        print(f"• 错误位置: 第{e.lineno}行第{e.colno}列")
        print(f"• 错误上下文: {e.doc[e.pos-30:e.pos+30]}")
        return None

def needs_adjustment(data):
    """检查文件是否需要处理"""
    for entry in data:
        base_time = entry["Time"]
        for note in entry.get("Notes", []):
            if note["noteType"] == "Slide" and note["slideStartTime"] >= base_time:
                return True
    return False

def process_single_file(input_path, output_dir=None):
    """
    处理单个JSON文件，调整Slide音符的slideStartTime
    :param input_path: 输入文件路径
    :param output_dir: 输出目录（None表示覆盖原文件）
    """
    data = safe_json_load(input_path)
    if data is None:
        print(f"❌ 跳过 {os.path.basename(input_path)}")
        return
    
    # 先检查是否需要处理
    if not needs_adjustment(data):
        has_slide = any(note["noteType"] == "Slide" for entry in data for note in entry.get("Notes", []))
        status = "ℹ️ 已处理" if has_slide else "⚠️ 无Slide"
        print(f"{status} {os.path.basename(input_path)}")
        return

    modified = False
    for entry in data:
        base_time = entry["Time"]
        for note in entry.get("Notes", []):
            if note["noteType"] == "Slide":
                original = note["slideStartTime"]
                if original >= base_time:  # 防重复处理
                    note["slideStartTime"] = original - base_time
                    modified = True

    # 确定输出路径
    output_path = input_path if not output_dir else os.path.join(output_dir, os.path.basename(input_path))
    
    if output_dir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 已处理 {os.path.basename(input_path)} -> {output_path}")


def process_directory(input_dir, output_dir):
    """处理目录下的所有JSON文件"""
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        try:
            process_single_file(input_path, output_dir)
        except Exception as e:
            print(f"处理 {filename} 时发生未捕获异常: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='调整Slide音符的slideStartTime为相对时间',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', help='输出目录（默认覆盖原文件）')
    
    args = parser.parse_args()

    if os.path.isfile(args.input):
        process_single_file(args.input, args.output)
    elif os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        print(f"错误：路径 {args.input} 不存在或不可访问")