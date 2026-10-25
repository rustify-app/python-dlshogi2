import os
import socket
import sys
import tempfile
import threading
import types
import unittest

from cshogi import Board

from pydlshogi2 import csa_client

FAKE_ENGINE = r'''
import sys
from cshogi import Board, move_to_usi
board = Board()
for line in sys.stdin:
    cmd = line.split()
    if not cmd:
        continue
    if cmd[0] == 'usi':
        print('id name fake'); print('usiok', flush=True)
    elif cmd[0] == 'isready':
        print('readyok', flush=True)
    elif cmd[0] == 'position':
        board.set_sfen(' '.join(cmd[2:6]))
        if 'moves' in cmd:
            for m in cmd[cmd.index('moves') + 1:]:
                board.push_usi(m)
    elif cmd[0] == 'go':
        move = move_to_usi(next(iter(board.legal_moves)))
        print('info depth 1 score cp 42 pv ' + move)
        print('bestmove ' + move, flush=True)
    elif cmd[0] == 'quit':
        break
'''

SUMMARY = '''BEGIN Game_Summary
Protocol_Version:1.2
Format:Shogi 1.0
Game_ID:test-game-1
Name+:me
Name-:other
Your_Turn:+
To_Move:+
BEGIN Time
Time_Unit:1sec
Total_Time:300
Byoyomi:0
Increment:10
END Time
BEGIN Position
PI
+
END Position
END Game_Summary
'''


class FakeServer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.received = []

    def run(self):
        conn, _ = self.listener.accept()
        reader = conn.makefile('r')

        def recv():
            while True:
                line = reader.readline()
                if line.strip():
                    self.received.append(line.strip())
                    return line.strip()

        def send(text):
            conn.sendall(text.encode())

        recv()  # LOGIN
        send('LOGIN:me OK\n' + SUMMARY)
        recv()  # AGREE
        send('START:test-game-1\n')
        first = recv().split(',')[0]
        send(first + ',T1\n-3334FU,T2\n')
        second = recv().split(',')[0]
        send(second + ',T1\n#RESIGN\n#WIN\n')
        recv()  # LOGOUT
        conn.close()


class CsaClientTest(unittest.TestCase):
    def test_plays_a_game_against_a_fake_server(self):
        server = FakeServer()
        server.start()
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, 'engine.py')
            with open(script, 'w') as f:
                f.write(FAKE_ENGINE)
            engine = csa_client.UsiEngine([sys.executable, script], {}, tmp)
            args = types.SimpleNamespace(host='127.0.0.1', port=server.port, user='me',
                                         password='secret', eval_comments=True)
            try:
                csa_client.run_once(args, engine)
            finally:
                engine.quit()
        server.join(timeout=10)

        login, agree, move1, move2, logout = server.received
        self.assertEqual(login, 'LOGIN me secret')
        self.assertEqual(agree, 'AGREE test-game-1')
        self.assertEqual(logout, 'LOGOUT')
        self.assertIn(",'* 42", move1)
        board = Board()
        board.push_csa(move1.split(',')[0][1:])
        board.push_csa('3334FU')
        self.assertTrue(board.is_legal(board.move_from_csa(move2.split(',')[0][1:])))
        self.assertTrue(move1.startswith('+') and move2.startswith('+'))

    def test_parse_position_with_moves(self):
        sfen, moves = csa_client.parse_position(['PI', '+', '+7776FU,T3', '-3334FU,T5'])
        self.assertTrue(sfen.startswith('lnsgkgsnl/1r5b1/ppppppppp'))
        self.assertEqual(moves, [('+', '7776FU', 3), ('-', '3334FU', 5)])


if __name__ == '__main__':
    unittest.main()
