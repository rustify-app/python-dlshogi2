"""Connect a USI engine to a CSA protocol server (e.g. floodgate) and keep playing.

The client logs in, accepts each game it is offered, relays moves between the
server and the engine, and reconnects after every game, so it can be left
running as a service. Floodgate pairs logged-in programs at :00 and :30.

Connection settings come from the environment so that credentials stay out
of the command line and the repository:

    CSA_HOST, CSA_PORT, CSA_USER, CSA_PASSWORD

Example:
    CSA_HOST=wdoor.c.u-tokyo.ac.jp CSA_PORT=4081 CSA_USER=myengine \\
    CSA_PASSWORD='floodgate-300-10F,secret' \\
    python -m pydlshogi2.csa_client --engine ./FukauraOu --option DNN_Model=model.onnx
"""
import argparse
import logging
import os
import queue
import re
import socket
import subprocess
import threading
import time

import cshogi
from cshogi import CSA, Board, move_to_csa

MOVE_LINE = re.compile(r'^([+-])(\d{4}[A-Z]{2})(?:,T(\d+))?')
RESULT_LINES = ('#WIN', '#LOSE', '#DRAW', '#CENSORED')
KEEPALIVE_SECONDS = 30


class UsiEngine:
    """Minimal synchronous USI engine wrapper."""

    def __init__(self, command, options, cwd=None):
        self.process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.send('usi')
        self.wait_for('usiok')
        for name, value in options.items():
            self.send('setoption name {} value {}'.format(name, value))
        self.send('isready')
        # Building a TensorRT engine on the first run can take minutes.
        self.wait_for('readyok', timeout=1800)

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip('\n'))
        self.lines.put(None)

    def send(self, command):
        logging.debug('usi> %s', command)
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def readline(self, timeout):
        line = self.lines.get(timeout=timeout)
        if line is None:
            raise RuntimeError('USI engine exited')
        logging.debug('usi< %s', line)
        return line

    def wait_for(self, prefix, timeout=120):
        deadline = time.monotonic() + timeout
        while True:
            line = self.readline(max(0.1, deadline - time.monotonic()))
            if line.startswith(prefix):
                return line

    def go(self, position, btime, wtime, binc, winc, byoyomi):
        """Search ``position`` and return ``(bestmove, score_cp, pv)``."""
        self.send(position)
        command = 'go btime {} wtime {}'.format(btime, wtime)
        if binc or winc:
            command += ' binc {} winc {}'.format(binc, winc)
        if byoyomi:
            command += ' byoyomi {}'.format(byoyomi)
        self.send(command)
        score, pv = None, []
        while True:
            line = self.readline(timeout=3600)
            if line.startswith('info ') and ' pv ' in line:
                tokens = line.split()
                if 'score' in tokens:
                    i = tokens.index('score')
                    if tokens[i + 1] == 'cp':
                        score = int(tokens[i + 2])
                    elif tokens[i + 1] == 'mate':
                        value = tokens[i + 2]
                        score = -100000 if value.startswith('-') else 100000
                pv = tokens[tokens.index('pv') + 1:]
            elif line.startswith('bestmove'):
                return line.split()[1], score, pv

    def quit(self):
        try:
            self.send('quit')
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()


class CsaConnection:
    """Line-based CSA connection that sends keep-alive newlines when idle."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=60)
        self.sock.settimeout(1.0)
        self.buffer = b''
        self.lock = threading.Lock()
        self.last_send = time.monotonic()
        self.closed = False
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _keepalive(self):
        while not self.closed:
            time.sleep(1)
            if time.monotonic() - self.last_send >= KEEPALIVE_SECONDS:
                try:
                    self.send('', log=False)
                except OSError:
                    return

    def send(self, line, log=True):
        if log:
            logging.info('csa> %s', line)
        with self.lock:
            self.sock.sendall((line + '\n').encode('utf-8'))
            self.last_send = time.monotonic()

    def readline(self, timeout=None):
        """Return the next non-empty line, or ``None`` on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if b'\n' in self.buffer:
                raw, self.buffer = self.buffer.split(b'\n', 1)
                line = raw.decode('utf-8', 'replace').rstrip('\r')
                if line:
                    logging.info('csa< %s', line)
                    return line
                continue
            if deadline is not None and time.monotonic() > deadline:
                return None
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError('server closed the connection')
            self.buffer += chunk

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass


def read_game_summary(conn):
    """Read one Game_Summary block into ``(fields, position_lines)``."""
    fields, position, section = {}, [], None
    while True:
        line = conn.readline()
        if line == 'BEGIN Game_Summary':
            fields, position, section = {}, [], None
        elif line == 'END Game_Summary':
            return fields, position
        elif line == 'BEGIN Position':
            section = 'position'
        elif line == 'END Position':
            section = None
        elif line.startswith(('BEGIN ', 'END ')):
            continue
        elif section == 'position':
            position.append(line)
        elif ':' in line:
            key, value = line.split(':', 1)
            fields[key] = value


def parse_position(position_lines):
    """Return ``(start_sfen, [(sign, csa_move, seconds), ...])`` from a Position block."""
    board_lines, moves = [], []
    for line in position_lines:
        match = MOVE_LINE.match(line)
        if match:
            moves.append((match.group(1), match.group(2), int(match.group(3) or 0)))
        else:
            board_lines.append(line)
    kif = CSA.Parser.parse_str('V2.2\n' + '\n'.join(board_lines) + '\n')[0]
    return kif.sfen, moves


def seconds(fields, key, unit):
    value = fields.get(key)
    return int(value) * unit if value else 0


def play_game(conn, engine, fields, position_lines, eval_comments=True):
    """Play one accepted game; return the final result line."""
    my_sign = fields['Your_Turn']
    unit = 60 if fields.get('Time_Unit', '1sec').endswith('min') else 1
    total = seconds(fields, 'Total_Time', unit)
    increment = seconds(fields, 'Increment', unit)
    byoyomi = seconds(fields, 'Byoyomi', unit)
    remaining = {'+': total, '-': total}

    start_sfen, initial_moves = parse_position(position_lines)
    board = Board(start_sfen)
    usi_moves = []

    def apply(sign, csa_move, spent):
        move = board.move_from_csa(csa_move)
        usi_moves.append(cshogi.move_to_usi(move))
        board.push(move)
        remaining[sign] = remaining[sign] - spent + increment

    for sign, csa_move, spent in initial_moves:
        apply(sign, csa_move, spent)

    engine.send('usinewgame')
    sign_of_turn = lambda: '+' if board.turn == cshogi.BLACK else '-'  # noqa: E731
    while True:
        if sign_of_turn() == my_sign:
            position = 'position sfen {}{}'.format(
                start_sfen, (' moves ' + ' '.join(usi_moves)) if usi_moves else '')
            bestmove, score, pv = engine.go(
                position, max(0, remaining['+']) * 1000, max(0, remaining['-']) * 1000,
                increment * 1000, increment * 1000, byoyomi * 1000)
            if bestmove == 'resign':
                conn.send('%TORYO')
            elif bestmove == 'win':
                conn.send('%KACHI')
            else:
                move = board.move_from_usi(bestmove)
                line = my_sign + move_to_csa(move)
                if eval_comments and score is not None:
                    shown = score if my_sign == '+' else -score
                    line += ",'* {}".format(shown)
                conn.send(line)
        # Wait for the server to echo our move, or for the opponent's move.
        while True:
            line = conn.readline()
            match = MOVE_LINE.match(line)
            if match:
                apply(match.group(1), match.group(2), int(match.group(3) or 0))
                break
            if line.startswith(RESULT_LINES):
                return line
            if line.startswith('#'):
                continue  # reason line such as #RESIGN; the result follows


def run_once(args, engine):
    conn = CsaConnection(args.host, args.port)
    try:
        conn.send('LOGIN {} {}'.format(args.user, args.password), log=False)
        logging.info('csa> LOGIN %s ********', args.user)
        reply = conn.readline(timeout=60)
        if not reply or not reply.endswith('OK'):
            raise ConnectionError('login failed: {}'.format(reply))
        fields, position = read_game_summary(conn)
        game_id = fields.get('Game_ID', '')
        conn.send('AGREE ' + game_id)
        while True:
            line = conn.readline()
            if line.startswith('START:'):
                break
            if line.startswith('REJECT:'):
                logging.info('game rejected')
                return
        result = play_game(conn, engine, fields, position, args.eval_comments)
        logging.info('game %s finished: %s', game_id, result)
        engine.send('gameover ' + {'#WIN': 'win', '#LOSE': 'lose'}.get(result, 'draw'))
        try:
            conn.send('LOGOUT')
        except OSError:
            pass
    finally:
        conn.close()


def parse_options(pairs):
    options = {}
    for pair in pairs:
        name, value = pair.split('=', 1)
        options[name] = value
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--engine', required=True, help='USI engine executable')
    parser.add_argument('--engine-dir', default=None, help='working directory of the engine')
    parser.add_argument('--option', action='append', default=[],
                        help='USI option as name=value (repeatable)')
    parser.add_argument('--games', type=int, default=0, help='stop after N games; 0 runs forever')
    parser.add_argument('--no-eval-comments', dest='eval_comments', action='store_false',
                        help="do not send the engine's evaluation as a move comment")
    parser.add_argument('--log', default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.host = os.environ['CSA_HOST']
    args.port = int(os.environ.get('CSA_PORT', '4081'))
    args.user = os.environ['CSA_USER']
    args.password = os.environ['CSA_PASSWORD']

    logging.basicConfig(filename=args.log, level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    engine_dir = args.engine_dir or os.path.dirname(os.path.abspath(args.engine))
    engine = UsiEngine([os.path.abspath(args.engine)], parse_options(args.option), engine_dir)
    played = 0
    try:
        while not args.games or played < args.games:
            try:
                run_once(args, engine)
                played += 1
            except (OSError, ConnectionError) as exc:
                logging.warning('connection problem: %s; retrying in 30s', exc)
                time.sleep(30)
            except RuntimeError as exc:
                logging.error('engine problem: %s; restarting the engine', exc)
                engine.quit()
                engine = UsiEngine([os.path.abspath(args.engine)],
                                   parse_options(args.option), engine_dir)
    finally:
        engine.quit()


if __name__ == '__main__':
    main()
