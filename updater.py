# -*- coding: utf-8 -*-
"""
=============================================================================
설치 · 업데이트 도우미 (파이썬 표준 라이브러리만 사용)
=============================================================================
  python updater.py --bootstrap   런처(CellCounter.exe)가 호출: 라이브러리·AI 모델 설치 확인 → 프로그램 실행
  python updater.py --check       GitHub에 새 버전이 있는지 확인
  python updater.py --update      새 버전으로 프로그램 파일 교체 (이후 --bootstrap 으로 재시작)

GUI의 [업데이트 확인] 버튼도 이 모듈의 함수를 사용합니다.
코드는 공개 GitHub 저장소에서 받으므로 다른 PC에서 로그인할 필요가 없습니다.
"""

import os
import sys
import shutil
import hashlib
import zipfile
import tempfile
import subprocess
import urllib.error
import urllib.request

REPO = "nthusualcoder/cell-counter"
BRANCH = "main"
VERSION_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/VERSION"
ZIP_URL = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"
GET_PIP_URLS = ["https://bootstrap.pypa.io/pip/3.10/get-pip.py",  # pip 가 3.10 지원을 끊은 뒤에도 동작
                "https://bootstrap.pypa.io/get-pip.py"]
MODEL_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
TORCH_VERSION, TORCHVISION_VERSION = "2.7.1", "0.22.1"
TORCH_INDEX = "https://download.pytorch.org/whl/{}"  # cu118 (NVIDIA GPU) 또는 cpu
FREE_GB_NEEDED = 10  # 설치 중 필요한 여유 공간 (torch GPU 버전 내려받기 + 설치)

# 이 PC의 사용자 폴더(%APPDATA%\Python)에 깔린 다른 패키지가 섞이지 않게 함 (pip·프로그램 등 하위 프로세스에도 적용)
os.environ["PYTHONNOUSERSITE"] = "1"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(APP_DIR)  # CellCounter.exe 설치본이면 %LOCALAPPDATA%\CellCounter
PYTHON_DIR = os.path.dirname(sys.executable)
GUI_SCRIPT = os.path.join(APP_DIR, "01_cellpose_gui.py")
INSTALL_STAMP = os.path.join(PYTHON_DIR, "cellcounter_installed.txt")


# ---------------------------------------------------------------------------
# 버전 확인
# ---------------------------------------------------------------------------
def local_version():
    with open(os.path.join(APP_DIR, "VERSION"), encoding="utf-8") as f:
        return f.read().strip()


def _vtuple(v):
    return tuple(int(x) for x in v.strip().split("."))


def _open(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "cell-counter-updater"})
    return urllib.request.urlopen(req, timeout=timeout)


def remote_version(timeout=10):
    # GitHub 이 이 파일을 최대 5분 캐시하므로, 방금 올린 버전은 몇 분 뒤에 보일 수 있음
    with _open(VERSION_URL, timeout) as r:
        return r.read().decode("utf-8").strip()


def check_for_update(timeout=10):
    """(현재 버전, 최신 버전, 새 버전 있음 여부). 네트워크 오류는 예외로 전달됩니다."""
    cur, new = local_version(), remote_version(timeout)
    return cur, new, _vtuple(new) > _vtuple(cur)


def is_dev_checkout():
    """개발용 git 폴더면 True (이 경우 파일을 덮어쓰지 않고 git pull 을 쓰도록 안내)"""
    return os.path.isdir(os.path.join(APP_DIR, ".git"))


# ---------------------------------------------------------------------------
# 다운로드 / 업데이트 적용
# ---------------------------------------------------------------------------
def _download(url, dst, progress=print, label=""):
    """url → dst (중간에 끊기면 .part 파일만 남고 dst 는 그대로)"""
    part = dst + ".part"
    with _open(url, 60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if pct != last and (pct % 5 == 0 or pct < 0):
                last = pct
                size = f"{done >> 20} / {total >> 20} MB" if total else f"{done >> 20} MB"
                progress(f"  {label} {pct}% ({size})" if total else f"  {label} {size}")
    if total and done != total:
        raise IOError(f"다운로드가 중간에 끊겼습니다 ({done}/{total} bytes): {url}")
    os.replace(part, dst)


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""


def download_and_apply(progress=print):
    """GitHub의 최신 코드를 받아 APP_DIR 의 프로그램 파일을 교체합니다.
    반환: (적용된 버전, requirements.txt 변경 여부)"""
    if is_dev_checkout():
        raise RuntimeError("개발용 git 폴더입니다. 이 폴더는 'git pull' 로 업데이트하세요.")
    old_req = _read(os.path.join(APP_DIR, "requirements.txt"))
    tmp = tempfile.mkdtemp(prefix="cellcounter_update_")
    try:
        zpath = os.path.join(tmp, "src.zip")
        progress("최신 버전 내려받는 중...")
        _download(ZIP_URL, zpath, progress, "프로그램")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(tmp, "src"))
        root = os.path.join(tmp, "src", os.listdir(os.path.join(tmp, "src"))[0])  # cell-counter-main\
        for dirpath, _, files in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            os.makedirs(os.path.join(APP_DIR, rel), exist_ok=True)
            for name in files:
                shutil.copy2(os.path.join(dirpath, name), os.path.join(APP_DIR, rel, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return local_version(), _read(os.path.join(APP_DIR, "requirements.txt")) != old_req


def restart_via_bootstrap():
    """새 콘솔 창에서 --bootstrap 실행 (필요하면 라이브러리 설치 후 프로그램을 다시 띄움)"""
    python = os.path.join(PYTHON_DIR, "python.exe")
    if not os.path.exists(python):
        python = sys.executable
    subprocess.Popen([python, "-s", os.path.abspath(__file__), "--bootstrap"], cwd=APP_DIR,
                     creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))


# ---------------------------------------------------------------------------
# 라이브러리 / 모델 설치
# ---------------------------------------------------------------------------
def has_nvidia_gpu():
    """NVIDIA 드라이버(CUDA)가 있는지: nvcuda.dll 또는 nvidia-smi 로 확인"""
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    if os.path.isfile(os.path.join(system32, "nvcuda.dll")):
        return True
    for exe in (shutil.which("nvidia-smi"), os.path.join(system32, "nvidia-smi.exe"),
                os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                             "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")):
        try:
            if exe and subprocess.run([exe, "-L"], capture_output=True, text=True,
                                      timeout=20).stdout.count("GPU"):
                return True
        except (OSError, subprocess.SubprocessError):
            pass
    return False


def _pip(*args):
    cmd = [sys.executable, "-s", "-m", "pip", *args, "--no-cache-dir",
           "--disable-pip-version-check", "--no-warn-script-location"]
    if subprocess.call(cmd) != 0:
        raise RuntimeError("라이브러리 설치에 실패했습니다: pip " + " ".join(args))


def _installed_version(dist):
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _ensure_pip(progress):
    try:
        import pip  # noqa: F401
        return
    except ImportError:
        pass
    progress("pip 설치 중...")
    get_pip = os.path.join(tempfile.gettempdir(), "get-pip.py")
    for url in GET_PIP_URLS:
        try:
            _download(url, get_pip, progress, "pip")
        except urllib.error.HTTPError:
            continue  # 해당 주소가 아직 없으면 다음 주소
        if subprocess.call([sys.executable, "-s", get_pip, "--no-warn-script-location"]) == 0:
            return
    raise RuntimeError("pip 설치에 실패했습니다.")


def ensure_requirements(progress=print):
    """필요할 때만 pip 설치: requirements.txt 가 바뀌었거나, torch 가 없거나,
    CPU용 torch 가 깔려 있는데 NVIDIA GPU 가 있을 때 (설치 완료 기록은 INSTALL_STAMP)"""
    req = os.path.join(APP_DIR, "requirements.txt")
    want = hashlib.sha256(_read(req) + TORCH_VERSION.encode()).hexdigest()
    torch_ver = _installed_version("torch") or ""
    gpu = has_nvidia_gpu()
    torch_ok = torch_ver.startswith(TORCH_VERSION) and not (gpu and torch_ver.endswith("+cpu"))
    if torch_ok and _read(INSTALL_STAMP).decode(errors="ignore").strip() == want:
        return

    need_gb = FREE_GB_NEEDED if not torch_ok else 1
    free_gb = shutil.disk_usage(PYTHON_DIR).free / 1e9
    if free_gb < need_gb:
        raise RuntimeError(f"디스크 여유 공간이 부족합니다 ({free_gb:.1f}GB). "
                           f"설치에 약 {need_gb}GB가 필요합니다.")
    _ensure_pip(progress)

    if not torch_ok:
        variant = "cu118" if gpu else "cpu"
        progress(f"PyTorch 설치 중 ({'NVIDIA GPU용, 약 2.8GB' if gpu else 'CPU용, 약 0.2GB'} 다운로드)...")
        _pip("install", f"torch=={TORCH_VERSION}+{variant}", f"torchvision=={TORCHVISION_VERSION}+{variant}",
             "--index-url", TORCH_INDEX.format(variant))

    progress("나머지 라이브러리 설치 중...")
    _pip("install", "-r", req)

    with open(INSTALL_STAMP, "w", encoding="utf-8") as f:
        f.write(want)


def model_candidates():
    """cellpose_counter.py 와 같은 순서로 모델 폴더 후보를 반환"""
    return [os.environ.get("CELLPOSE_LOCAL_MODELS_PATH"),
            os.path.join(HOME_DIR, "models"),                      # CellCounter.exe 설치본
            os.path.join(APP_DIR, ".cellpose", "models"),           # 기존 Cellpose_Deploy 폴더
            os.path.join(os.path.expanduser("~"), ".cellpose", "models")]  # Cellpose 기본 위치


def ensure_model(progress=print):
    if any(d and os.path.isfile(os.path.join(d, "cpsam")) for d in model_candidates()):
        return
    dst_dir = os.path.join(HOME_DIR, "models")
    os.makedirs(dst_dir, exist_ok=True)
    progress("AI 모델(Cellpose-SAM, 약 1.2GB) 내려받는 중...")
    _download(MODEL_URL, os.path.join(dst_dir, "cpsam"), progress, "모델")


def launch_gui():
    pythonw = os.path.join(PYTHON_DIR, "pythonw.exe")
    subprocess.Popen([pythonw if os.path.exists(pythonw) else sys.executable, "-s", GUI_SCRIPT], cwd=APP_DIR)


def bootstrap():
    print("=" * 56)
    print(f"  Cellpose 세포 카운팅 프로그램 v{local_version()}")
    print("=" * 56)
    try:
        ensure_requirements()
        ensure_model()
    except Exception as e:
        print(f"\n[오류] {e}\n인터넷 연결과 디스크 공간을 확인한 뒤 다시 실행해 주세요.")
        input("엔터를 누르면 종료합니다...")
        return 1
    if os.environ.get("CELLCOUNTER_NO_LAUNCH"):  # 설치 테스트용
        print("설치 확인 완료 (CELLCOUNTER_NO_LAUNCH: 프로그램 실행 생략)")
        return 0
    print("프로그램을 실행합니다...")
    launch_gui()
    return 0


if __name__ == "__main__":
    if "--bootstrap" in sys.argv:
        sys.exit(bootstrap())
    elif "--update" in sys.argv:
        ver, req_changed = download_and_apply()
        print(f"v{ver} 적용 완료" + (" (라이브러리 변경 있음 → --bootstrap 실행 필요)" if req_changed else ""))
    else:
        cur, new, newer = check_for_update()
        print(f"현재 v{cur} / 최신 v{new}" + (" → 업데이트 가능" if newer else " (최신)"))
