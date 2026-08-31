import argparse
import requests

from colorama import Fore

from model.cli_funcs import cli_env
from model import __version__

PYPI_API_URL = 'https://pypi.org/pypi/dg-force/json'

def version_and_check_for_updates():
    print(f'DG-Force codebase version: {__version__}')
    try:
        response = requests.get(PYPI_API_URL, timeout=5)
        response.raise_for_status()  # 如果响应码不是200，则抛出异常
        pypi_data = response.json()
        cloud_version = pypi_data['info']['version']  # 获取最新版本号
        # cloud_version = '0.1.21'   # for debug & test
        print("\tChecking for updates...")
        print("\tLocal version: ", __version__)
        print("\tPyPI newest version: ", cloud_version)
        local_version = __version__

        if cloud_version != local_version:
            print(Fore.YELLOW + f"New version available: {cloud_version}. Your version: {local_version}.")
            print("Run 'pip install --upgrade .' to upgrade.")
        else:
            print(Fore.GREEN + f"You are using the latest version: {local_version}.")
    except requests.exceptions.RequestException as e:
        print(Fore.RED + "Failed to check for updates from PyPI. Please check your internet connection.")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='DG-Force command line interface, with codebase version: ' + __version__)
    
    parser.add_argument('-v', '--version', action='store_true', help="Show the version of the tool")
    
    subparsers = parser.add_subparsers(dest='command', required=False)
    subparsers.add_parser('env', help='Show environment information')

    args = parser.parse_args()
    if args.version:
        version_and_check_for_updates()
    elif args.command == 'env':
        cli_env(None)

if __name__ == '__main__':
    main()
