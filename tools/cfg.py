import os
import sys
from pathlib import Path


def get_executable_path():
    # 这个函数会返回可执行文件所在的目录
    if getattr(sys, 'frozen', False):
        # 如果程序是被“冻结”打包的，使用这个路径
        return Path(sys.executable).parent.as_posix()
    else:
        return Path.cwd().as_posix()


ROOT_DIR = get_executable_path()

MODEL_DIR_PATH = Path(ROOT_DIR + "/asset")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
MODEL_DIR = MODEL_DIR_PATH.as_posix()

WAVS_DIR_PATH = Path(ROOT_DIR + "/static/wavs")
WAVS_DIR_PATH.mkdir(parents=True, exist_ok=True)
WAVS_DIR = WAVS_DIR_PATH.as_posix()


SPEAKER_DIR_PATH = Path(ROOT_DIR + "/speaker")
SPEAKER_DIR_PATH.mkdir(parents=True, exist_ok=True)
SPEAKER_DIR = SPEAKER_DIR_PATH.as_posix()

# ffmpeg
if sys.platform == 'win32':
    os.environ['PATH'] = ROOT_DIR + f';{ROOT_DIR}/ffmpeg;' + os.environ['PATH']

else:
    os.environ['PATH'] = ROOT_DIR + f':{ROOT_DIR}/ffmpeg:' + os.environ['PATH']
