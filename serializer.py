import subprocess
import os
import glob
from tqdm import tqdm  # 新增进度条库

def run_csharp_program(input_path, output_directory, csv_path):
    """
    使用 .NET CLI 运行 C# 程序，并传递输入路径和输出目录。
    
    :param input_path: 输入文件夹路径（例如：./data/niconicoボーカロイド/44_ハツヒイシンセサイサ/）
    :param output_directory: 输出目录路径（例如：./serialized_data/）
    :param csv_path: CSV文件路径
    :return: C# 程序的输出结果
    """
    try:
        # 确保输入路径和输出目录存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入路径 {input_path} 不存在。")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # 构建命令
        cs_project_path = os.path.join(os.getcwd(), "tools", "serializer", "src")
        command = [
            "dotnet",
            "run",
            "--project", cs_project_path,
            input_path,
            output_directory,
            csv_path
        ]

        # print(f"Running command: {' '.join(command)}")
        
        # 执行命令并捕获输出，添加超时判断
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=20
        )
        
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def load_processed_log(log_file):
    """加载已处理成功的日志文件"""
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    return set()

if __name__ == "__main__":
    try:
        # 检查tqdm是否安装
        from tqdm import tqdm
    except ImportError:
        print("请先安装进度条库：pip install tqdm")
        exit(1)

    output_directory = "./serialized_data/"
    success_log = "success.log"
    error_log = "error.log"
    csv_path = os.path.join(output_directory, "chart_info.csv")  # CSV文件路径

    # 加载已处理记录
    processed = load_processed_log(success_log)
    errors = load_processed_log(error_log)

    # 查找所有maidata.txt路径
    maidat_paths = glob.glob('./data/**/*/maidata.txt', recursive=True)
    target_dirs = {os.path.abspath(os.path.dirname(p)) for p in maidat_paths}

    # 过滤未处理的目录
    todo_dirs = [d for d in target_dirs if d not in processed and d not in errors]
    print(f"总目录数: {len(target_dirs)} | 待处理: {len(todo_dirs)} | 已成功: {len(processed)} | 失败: {len(errors)}")

    # 创建进度条
    with tqdm(todo_dirs, unit="dir", desc="处理进度") as pbar:
        for input_dir in pbar:
            pbar.set_postfix(file=os.path.basename(input_dir))
            try:
                result = run_csharp_program(input_dir, output_directory, csv_path)
                if result:
                    # 记录成功
                    with open(success_log, 'a', encoding='utf-8') as f:
                        f.write(f"{input_dir}\n")
                else:
                    # 记录失败
                    with open(error_log, 'a', encoding='utf-8') as f:
                        f.write(f"{input_dir}\n")
            except Exception as e:
                print(f"\n处理异常: {input_dir} - {str(e)}")
                with open(error_log, 'a', encoding='utf-8') as f:
                    f.write(f"{input_dir}\n")

    print("\n处理完成！结果汇总：")
    print(f"成功: {len(load_processed_log(success_log))}")
    print(f"失败: {len(load_processed_log(error_log))}")