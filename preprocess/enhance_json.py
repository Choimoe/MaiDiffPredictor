import sys
import os
import json
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from tools.inference.src.enhance_json_with_features import process_json_file

def safe_json_load(file_path):
    """安全读取JSON文件，处理常见格式问题"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"空文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read().strip()
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

def process_single_file(input_path, output_dir=None, original_filename=None):
    """增强版单文件处理"""
    output_filename = original_filename or os.path.basename(input_path)
    output_path = os.path.join(output_dir, output_filename) if output_dir else input_path
    
    data = safe_json_load(input_path)
    if data is None:
        print(f"❌ 跳过 {output_filename}")
        return

    try:
        # 直接使用导入的process_json_file处理文件
        process_json_file(input_path, output_path)
        print(f"✅ 已增强 {output_filename}")
    except Exception as e:
        print(f"❌ 处理失败 {output_filename}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)  # 清理不完整输出
        raise

def process_directory(input_dir, output_dir):
    """处理目录下的所有JSON文件（最终修复版）"""
    # 严格过滤临时文件
    file_list = [
        f for f in os.listdir(input_dir) 
        if f.endswith('.json') 
        and not f.startswith('_processing_')
        and not f.startswith('_processed_')
    ]
    
    same_directory = (os.path.abspath(input_dir) == os.path.abspath(output_dir or input_dir))
    
    for filename in file_list:
        original_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename) if output_dir else original_path
        
        # 当需要覆盖原文件时
        if same_directory:
            temp_path = os.path.join(input_dir, f"_processing_{filename}")
            try:
                os.rename(original_path, temp_path)
                process_single_file(temp_path, output_dir, original_filename=filename)
            except Exception as e:
                print(f"处理 {filename} 失败: {str(e)}")
                if os.path.exists(temp_path):
                    os.rename(temp_path, original_path)  # 恢复原文件
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        else:
            process_single_file(original_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='增强JSON文件中的note特征',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', help='输出目录（默认覆盖原文件）')
    
    args = parser.parse_args()

    # 规范化路径处理
    args.input = os.path.abspath(args.input)
    if args.output:
        args.output = os.path.abspath(args.output)
        os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        process_single_file(args.input, args.output)
    elif os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        print(f"错误：路径 {args.input} 不存在或不可访问") 