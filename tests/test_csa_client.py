import os
import socket
import sys
import tempfile
import threading
import types
import unittest

from cshogi import Board, move_to_csa

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

PONDER_ENGINE = r'''
import sys
from cshogi import Board, move_to_usi
board = Board()
log = open(LOG, 'a')
pending = None
for line in sys.stdin:
    log.write(line); log.flush()
    cmd = line.split()
    if not cmd:
        continue
    if cmd[0] == 'usi':
        print('usiok', flush=True)
    elif cmd[0] == 'isready':
        print('readyok', flush=True)
    elif cmd[0] == 'position':
        board.set_sfen(' '.join(cmd[2:6]))
        if 'moves' in cmd:
            for m in cmd[cmd.index('moves') + 1:]:
                board.push_usi(m)
    elif cmd[0] == 'go':
        move = next(iter(board.legal_moves))
        board.push(move)
        reply = move_to_usi(next(iter(board.legal_moves)))
        board.pop()
        out = 'bestmove ' + move_to_usi(move) + ' ponder ' + reply
        if 'ponder' in cmd[1:2]:
            pending = out
        else:
            print(out, flush=True)
    elif cmd[0] in ('ponderhit', 'stop') and pending:
        print(pending, flush=True); pending = None
    elif cmd[0] == 'quit':
        break
'''


class FakeConn:
    """Echoes our moves. Reply 1 is the engine's predicted move (first legal), reply 2 is not,
    then the game ends while the engine is pondering."""

    def __init__(self):
        self.board, self.sent, self.queue = Board(), [], []

    def send(self, line, log=True):
        self.sent.append(line)
        csa = line.split(',')[0]
        self.board.push_csa(csa[1:])
        self.queue.append(csa + ',T1')
        legal = list(self.board.legal_moves)
        if len(self.sent) == 1:
            reply = legal[0]
        elif len(self.sent) == 2:
            reply = legal[1]
        else:
            self.queue.append('#WIN')
            return
        self.queue.append('-' + move_to_csa(reply) + ',T2')
        self.board.push(reply)

    def readline(self, timeout=None):
        return self.queue.pop(0)


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

    def test_ponder_hit_and_miss(self):
        """Ponder on the predicted reply; ponderhit on a hit, stop on a miss and at game end."""
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, 'engine.py')
            log = os.path.join(tmp, 'usi.log')
            with open(script, 'w') as f:
                f.write(PONDER_ENGINE.replace('LOG', repr(log)))
            engine = csa_client.UsiEngine([sys.executable, script], {'USI_Ponder': 'true'}, tmp)
            conn = FakeConn()
            fields = {'Your_Turn': '+', 'Total_Time': '300', 'Increment': '10'}
            try:
                result = csa_client.play_game(conn, engine, fields, ['PI', '+'],
                                              eval_comments=False, ponder=True)
            finally:
                engine.quit()
            with open(log) as f:
                cmds = [c.split()[0] + (' ponder' if c.startswith('go ponder') else '')
                        for c in f if c.startswith(('go', 'ponderhit', 'stop'))]
        self.assertEqual(result, '#WIN')
        self.assertEqual(cmds, ['go', 'go ponder', 'ponderhit',      # move 1, hit
                                'go ponder', 'stop', 'go',            # move 2, miss
                                'go ponder', 'stop'])                 # move 3, game over
        self.assertEqual(len(conn.sent), 3)

    def test_parse_position_with_moves(self):
        sfen, moves = csa_client.parse_position(['PI', '+', '+7776FU,T3', '-3334FU,T5'])
        self.assertTrue(sfen.startswith('lnsgkgsnl/1r5b1/ppppppppp'))
        self.assertEqual(moves, [('+', '7776FU', 3), ('-', '3334FU', 5)])


if __name__ == '__main__':
    unittest.main()
