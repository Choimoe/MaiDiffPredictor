import json
import requests

def fetch_chart_stats(stat_output_dir, file_name='chart_stat_diff.json', config_path='config/api.json'):
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
            api_url = config['diff_api_url']
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到，请确保文件存在！")
        return
    except KeyError:
        print(f"错误：配置文件缺少 'diff_api_url' 字段！")
        return

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        data = response.json()

        import os
        output_dir = stat_output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if output_dir[-1] != '/':
            output_dir += '/'
        output_path = f'{output_dir}{file_name}'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"数据已成功保存至：{output_path}")

    except requests.exceptions.RequestException as e:
        print(f"网络请求失败：{str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON解析失败：{str(e)}")
    except Exception as e:
        print(f"未知错误：{str(e)}")

if __name__ == "__main__":
    fetch_chart_stats('./info')