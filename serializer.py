import subprocess
import os

def run_csharp_program(input_path, output_directory):
    """
    使用 .NET CLI 运行 C# 程序，并传递输入路径和输出目录。
    
    :param input_path: 输入文件夹路径（例如：./data/niconicoボーカロイド/44_ハツヒイシンセサイサ/）
    :param output_directory: 输出目录路径（例如：./serialized_data/）
    :return: C# 程序的输出结果
    """
    try:
        # 确保输入路径和输出目录存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入路径 {input_path} 不存在。")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # 构建命令
        cs_project_path = os.path.join(os.getcwd(), "tools", "serializer", "src")
        command = [
            "dotnet",
            "run",
            "--project", cs_project_path,
            input_path,
            output_directory
        ]

        print(f"Running command: {' '.join(command)}")
        
        # 执行命令并捕获输出
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    # 示例输入路径和输出目录
    input_path = "./data/niconicoボーカロイド/44_ハツヒイシンセサイサ/"  # 替换为实际路径
    output_directory = "./serialized_data/"  # 替换为实际路径

    # 调用 C# 程序
    result = run_csharp_program(input_path, output_directory)
    
    if result:
        print("C# 程序输出：")
        print(result)
    else:
        print("程序执行失败，请检查输入路径和输出目录是否正确。")