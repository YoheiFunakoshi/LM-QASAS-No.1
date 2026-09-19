"""Start the LM-QASAS interface on this PC only."""
import argparse
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lmqasas.server import Application, LocalHTTPServer


def main() -> int:
    parser = argparse.ArgumentParser(description='LM-QASAS ローカル画面を起動します。')
    parser.add_argument('--port', type=int, default=8765, help='127.0.0.1で使用するポート（既定8765）')
    parser.add_argument('--no-browser', action='store_true', help='ブラウザーを自動で開かない')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    expected_python = ROOT / '.venv/Scripts/python.exe'
    if not expected_python.is_file() or Path(sys.executable).resolve() != expected_python.resolve():
        print('専用環境の .venv\\Scripts\\python.exe で起動してください。日本語解説書に起動手順があります。', file=sys.stderr)
        return 1
    if not (ROOT / 'app/index.html').is_file() or not (ROOT / 'scripts/app_worker.py').is_file():
        print('アプリのファイルが不足しています。プロジェクトの配置を確認してください。', file=sys.stderr)
        return 1
    try:
        server = LocalHTTPServer(Application(ROOT), args.port)
    except (OSError, ValueError):
        print('起動できませんでした。同じアプリが開いていないか、保存先・ポートを確認してください。', file=sys.stderr)
        return 1
    print(f'LM-QASAS: {server.origin}', flush=True)
    print('終了するにはこの端末で Ctrl+C を押してください。', flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(server.origin)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
